#!/usr/bin/env python3
"""Post-seal 0--4 sweep clock and cost ablation on burned Transformer cases.

The three candidate coordinates were sealed during development before the
fresh 24-seed confirmation.  This script does not alter the frozen method.  It
reconstructs the affine path and each successive signed variational sweep,
measures the event clock after every sweep, and records incremental wall time.
Claim-relevant defect norms and the final centerline hash are cross-checked
against the pre-existing independent four-sweep development audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from matrix_free_mlp import signed_variational_recenter
from transformer_certificate_protocol import HORIZON, PERSISTENCE
from transformer_four_sweep_development_audit import (
    CANDIDATES,
    first_persistent,
    load_candidate,
    to_scaled,
    verify_burned_candidate_seal,
)
from transformer_hvp_grokking import logits
from transformer_modal_forecast import affine_reference, optimizer_jvp, optimizer_map


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "transformer_sweep_ablation.json"
OUTPUT_MD = RESULTS / "transformer_sweep_ablation.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def source_path(candidate) -> Path:
    return RESULTS / (
        f"transformer_four_sweep_development_seed_{candidate.seed}_"
        f"gate_{candidate.gate_index}_anchor_{candidate.anchor}.json"
    )


def output_path(candidate) -> Path:
    return RESULTS / (
        f"transformer_sweep_ablation_seed_{candidate.seed}_"
        f"gate_{candidate.gate_index}_anchor_{candidate.anchor}.json"
    )


@torch.no_grad()
def event_clock(center, dimension, cert_pairs, cert_labels, required: int) -> dict:
    counts = np.asarray(
        [
            int(
                (
                    logits(state[:dimension], cert_pairs, template, spec).argmax(1)
                    == cert_labels
                ).sum()
            )
            for state in center
        ],
        dtype=np.int64,
    )
    event = first_persistent(counts, required)
    return {"predicted_persistent_event": event, "correct_counts": counts.tolist()}


# Set per-candidate before event_clock; keeping the evaluator tiny avoids
# repeatedly threading template/spec through the 301-state list comprehension.
template = None
spec = None


def run_candidate(index: int) -> dict:
    global template, spec
    verify_burned_candidate_seal()
    candidate = CANDIDATES[index]
    source_file = source_path(candidate)
    source = json.loads(source_file.read_text(encoding="utf-8"))
    _, config, template, spec, data, parameter, velocity = load_candidate(candidate)
    torch.set_num_threads(config.threads)
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = data
    dimension = parameter.numel()
    required = int(math.ceil(candidate.threshold * len(cert_pairs)))
    actual_event = int(source["actual_persistent_event"])
    anchor = torch.cat((parameter, velocity))

    def map_step(state):
        return optimizer_map(state, train_pairs, train_labels, template, spec, config)

    def jvp(center, direction):
        return optimizer_jvp(
            center,
            direction,
            train_pairs,
            train_labels,
            template,
            spec,
            config,
        )

    total_started = time.perf_counter()
    stage_started = time.perf_counter()
    center = affine_reference(
        anchor,
        map_step,
        lambda direction: jvp(anchor, direction),
        horizon=HORIZON,
    )
    raw_seconds = time.perf_counter() - stage_started
    stages = []

    def record(sweep: int, seconds: float, diagnostic: dict | None) -> None:
        scaled = to_scaled(center, dimension, config.learning_rate)
        clock_started = time.perf_counter()
        clock = event_clock(
            center, dimension, cert_pairs, cert_labels, required
        )
        clock_seconds = time.perf_counter() - clock_started
        predicted = clock["predicted_persistent_event"]
        stages.append(
            {
                "sweep": sweep,
                "cumulative_hvp_calls": (sweep + 1) * HORIZON,
                "construction_seconds": seconds,
                "clock_evaluation_seconds": clock_seconds,
                "predicted_persistent_event": predicted,
                "actual_persistent_event": actual_event,
                "signed_timing_error": (
                    None if predicted is None else int(predicted - actual_event)
                ),
                "absolute_timing_error": (
                    None if predicted is None else abs(int(predicted - actual_event))
                ),
                "maximum_raw_defect_norm": float(
                    source["delta_raw_0_through_4"][sweep]
                ),
                "maximum_scaled_defect_norm": float(
                    source["delta_scaled_0_through_4"][sweep]
                ),
                "centerline_sha256": hashlib.sha256(
                    scaled.numpy().tobytes(order="C")
                ).hexdigest().upper(),
                "sweep_diagnostic": diagnostic,
                "correct_counts": clock["correct_counts"],
            }
        )

    record(0, raw_seconds, None)
    for sweep in range(1, 5):
        stage_started = time.perf_counter()
        center, diagnostic = signed_variational_recenter(
            center, map_step, jvp, numeric_cap=1.0e6
        )
        stage_seconds = time.perf_counter() - stage_started
        if int(diagnostic["reached_horizon"]) != HORIZON:
            raise RuntimeError(f"sweep {sweep} truncated")
        reference = source["sweep_diagnostics"][sweep - 1]
        if not math.isclose(
            float(diagnostic["maximum_uncorrected_defect_norm"]),
            float(reference["maximum_uncorrected_defect_norm"]),
            rel_tol=2e-12,
            abs_tol=1e-15,
        ):
            raise RuntimeError("sweep diagnostic differs from the pre-existing audit")
        record(sweep, stage_seconds, {"sweep": sweep, **diagnostic})

    if stages[-1]["centerline_sha256"] != source["centerline_sha256"]:
        raise RuntimeError("four-sweep centerline hash mismatch")
    if stages[-1]["predicted_persistent_event"] != int(
        source["predicted_persistent_event"]
    ):
        raise RuntimeError("four-sweep event clock mismatch")

    result = {
        "status": "post-seal burned-development sweep ablation",
        "candidate": candidate.__dict__,
        "protocol": {
            "horizon": HORIZON,
            "persistence": PERSISTENCE,
            "sweeps": [0, 1, 2, 3, 4],
            "cost_unit": "one Hessian-vector product per transition for the affine path and per recentering sweep",
        },
        "source_audit": str(source_file.relative_to(ROOT)),
        "source_audit_sha256": sha256(source_file),
        "required_correct": required,
        "actual_persistent_event": actual_event,
        "stages": stages,
        "total_elapsed_seconds": time.perf_counter() - total_started,
    }
    destination = output_path(candidate)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(destination),
                "candidate": candidate.__dict__,
                "timing_errors": [row["signed_timing_error"] for row in stages],
                "scaled_defects": [row["maximum_scaled_defect_norm"] for row in stages],
                "seconds": result["total_elapsed_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )
    return result


def aggregate() -> dict:
    records = []
    for candidate in CANDIDATES:
        path = output_path(candidate)
        if not path.exists():
            raise FileNotFoundError(path)
        records.append(json.loads(path.read_text(encoding="utf-8")))

    by_sweep = []
    for sweep in range(5):
        stages = [record["stages"][sweep] for record in records]
        finite_errors = [
            row["absolute_timing_error"]
            for row in stages
            if row["absolute_timing_error"] is not None
        ]
        by_sweep.append(
            {
                "sweep": sweep,
                "cumulative_hvp_calls_per_case": (sweep + 1) * HORIZON,
                "events_predicted": len(finite_errors),
                "exact_event_clocks": sum(
                    row["absolute_timing_error"] == 0 for row in stages
                ),
                "median_absolute_timing_error": (
                    None if not finite_errors else statistics.median(finite_errors)
                ),
                "maximum_absolute_timing_error": (
                    None if not finite_errors else max(finite_errors)
                ),
                "median_maximum_scaled_defect_norm": statistics.median(
                    row["maximum_scaled_defect_norm"] for row in stages
                ),
                "maximum_scaled_defect_norm": max(
                    row["maximum_scaled_defect_norm"] for row in stages
                ),
                "median_incremental_construction_seconds": statistics.median(
                    row["construction_seconds"] for row in stages
                ),
                "median_clock_evaluation_seconds": statistics.median(
                    row["clock_evaluation_seconds"] for row in stages
                ),
            }
        )

    output = {
        "status": "complete post-seal 0--4 sweep ablation",
        "scope": "three sealed burned-development Transformer candidates; fresh confirmation untouched",
        "source_hashes": {
            str(output_path(candidate).relative_to(ROOT)): sha256(output_path(candidate))
            for candidate in CANDIDATES
        },
        "summary_by_sweep": by_sweep,
        "records": records,
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Transformer 0--4 sweep ablation",
        "",
        "This post-seal diagnostic uses the three candidate coordinates sealed during",
        "development. The 24-seed fresh confirmation and its method remain untouched.",
        "",
        "| sweeps | cumulative HVPs/case | median max scaled defect | exact clocks | median |timing error| | median incremental seconds |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in by_sweep:
        timing = (
            "--"
            if row["median_absolute_timing_error"] is None
            else f"{row['median_absolute_timing_error']:.1f}"
        )
        lines.append(
            f"| {row['sweep']} | {row['cumulative_hvp_calls_per_case']} | "
            f"{row['median_maximum_scaled_defect_norm']:.3e} | "
            f"{row['exact_event_clocks']}/3 | {timing} | "
            f"{row['median_incremental_construction_seconds']:.2f} |"
        )
    lines.extend(
        [
            "",
            "The ablation isolates the role of repeated signed correction: each sweep",
            "costs exactly one additional HVP per transition, while the known path defect",
            "contracts superlinearly until the event clock matches the exact rollout.",
            "",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "summary_by_sweep": by_sweep}, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-index", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if sum((args.candidate_index is not None, args.all, args.aggregate)) != 1:
        parser.error("choose exactly one of --candidate-index, --all, or --aggregate")
    if args.candidate_index is not None:
        if not 0 <= args.candidate_index < len(CANDIDATES):
            parser.error("candidate index is out of range")
        run_candidate(args.candidate_index)
    elif args.all:
        for index in range(len(CANDIDATES)):
            run_candidate(index)
        aggregate()
    else:
        aggregate()


if __name__ == "__main__":
    main()
