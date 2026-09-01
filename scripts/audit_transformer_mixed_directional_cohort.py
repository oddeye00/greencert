#!/usr/bin/env python3
"""Independent cohort replay using the linear-cost mixed directional jet."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch

from audit_transformer_adaptive_sweep_cohort import all_reduced_paths, scaled
from audit_transformer_direct_image_green_panel import tensor_sha256
from corrected_path_closure import exact_corrected_path_closure
from transformer_certificate_protocol import Candidate
from transformer_mixed_directional_jet import mixed_directional_objective_fourth_bound
from transformer_modal_forecast import optimizer_map
from transformer_optimizer_probe import make_scaled_optimizer_jvp_vjp
from transformer_v3_certificate import load_candidate, safe_json


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PARENT = RESULTS / "transformer_directional_block_remainder_diagnostic.json"
PROTOCOL = ROOT / "MIXED_DIRECTIONAL_JET_AUDIT_PROTOCOL.md"
MODULE = ROOT / "scripts" / "transformer_mixed_directional_jet.py"
TEST = ROOT / "scripts" / "test_transformer_mixed_directional_jet.py"
OUTPUT = RESULTS / "transformer_mixed_directional_cohort_audit.json"
CACHE = RESULTS / "transformer_mixed_directional_cohort_cache"
SWEEPS = 3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def cache_path(candidate: Candidate) -> Path:
    return CACHE / (
        f"seed_{candidate.seed}_gate_{candidate.gate_index}_"
        f"anchor_{candidate.anchor}.json"
    )


def run_case(task: dict) -> dict:
    parent = task["parent"]
    hashes = task["hashes"]
    candidate = Candidate(**parent["candidate"])
    destination = cache_path(candidate)
    if destination.is_file():
        cached = safe_json(destination)
        if cached.get("source_hashes") == hashes:
            return cached
    started = time.perf_counter()
    config, template, spec, data, parameter, velocity = load_candidate(candidate)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    train_pairs, train_labels, _, _, _, _ = data
    horizon = int(parent["horizon"])
    dimension = int(parameter.numel())
    eta = float(config.learning_rate)
    paths, pipeline = all_reduced_paths(
        config,
        template,
        spec,
        train_pairs,
        train_labels,
        parameter,
        velocity,
        horizon,
    )
    center = paths[SWEEPS - 1]
    scaled_center = scaled(center, dimension, eta)
    mapped = [
        optimizer_map(center[step], train_pairs, train_labels, template, spec, config)
        for step in range(horizon)
    ]
    residual = torch.stack(
        [
            scaled(mapped[step], dimension, eta) - scaled_center[step + 1]
            for step in range(horizon)
        ]
    )
    products = [
        make_scaled_optimizer_jvp_vjp(
            center[step, :dimension],
            train_pairs,
            train_labels,
            template,
            spec,
            config,
        )
        for step in range(horizon)
    ]
    correction_rows = []
    prior = torch.zeros_like(residual[0])
    for step in range(horizon):
        current = products[step][0](prior) + residual[step]
        correction_rows.append(current)
        prior = current
    correction = torch.cat(
        (torch.zeros_like(torch.stack(correction_rows)[:1]), torch.stack(correction_rows)),
        dim=0,
    )
    corrected_hash = tensor_sha256(scaled_center + correction)
    if corrected_hash != parent["corrected_path_sha256"]:
        raise RuntimeError(f"mixed replay corrected-path mismatch for {candidate}")

    terms = []
    maximum_relative_error = 0.0
    for step, expected in zip(range(1, horizon), parent["step_rows"]):
        if int(expected["step"]) != step:
            raise RuntimeError("parent step registry is not contiguous")
        record = mixed_directional_objective_fourth_bound(
            center[step, :dimension], correction[step, :dimension], spec, config
        )
        observed = float(record["gradient_taylor_remainder_upper"])
        reference = float(expected["directional_gradient_remainder_upper"])
        relative = abs(observed - reference) / max(abs(reference), 1.0e-300)
        maximum_relative_error = max(maximum_relative_error, relative)
        if not math.isclose(observed, reference, rel_tol=3.0e-12, abs_tol=1.0e-300):
            raise RuntimeError(
                f"mixed/polynomial local mismatch for {candidate} step {step}: "
                f"{observed} != {reference}"
            )
        terms.append(math.sqrt(2.0) * eta * observed)
    sequence = math.sqrt(sum(value * value for value in terms))
    reference_sequence = float(parent["directional_block_taylor_sequence_upper"])
    if not math.isclose(sequence, reference_sequence, rel_tol=3.0e-12, abs_tol=1.0e-300):
        raise RuntimeError(f"mixed sequence mismatch for {candidate}")
    injection = (
        float(parent["directional_cancellation_safe_injection_upper"])
        - reference_sequence
        + sequence
    )
    response = float(parent["unchanged_green_operator_norm_upper_bound"]) * injection
    closure = exact_corrected_path_closure(
        kappa=float(parent["unchanged_green_operator_norm_upper_bound"]),
        derivative_drift=float(parent["closure"]["derivative_drift"]),
        defect_response_bound=response,
        domain_radius=float(parent["closure"]["domain_radius"]),
    )
    closure_match = closure.closure_passed == bool(parent["closure_passed"])
    if not closure_match:
        raise RuntimeError(f"mixed closure decision mismatch for {candidate}")
    result = {
        "candidate": parent["candidate"],
        "development_row": parent["development_row"],
        "horizon": horizon,
        "pipeline_diagnostics": pipeline,
        "corrected_path_sha256": corrected_hash,
        "mixed_taylor_sequence_upper": sequence,
        "polynomial_taylor_sequence_upper": reference_sequence,
        "maximum_local_relative_error": maximum_relative_error,
        "closure_passed": closure.closure_passed,
        "parent_closure_passed": parent["closure_passed"],
        "closure_decision_matches": closure_match,
        "mixed_closure": closure.as_dict(),
        "polynomial_elapsed_seconds": float(parent["elapsed_seconds"]),
        "mixed_elapsed_seconds": time.perf_counter() - started,
        "source_hashes": hashes,
        "outcome_files_read": 0,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    parent = safe_json(PARENT)
    hashes = {
        "parent": sha256(PARENT),
        "protocol": sha256(PROTOCOL),
        "module": sha256(MODULE),
        "test": sha256(TEST),
        "script": sha256(Path(__file__)),
    }
    tasks = [{"parent": row, "hashes": hashes} for row in parent["rows"]]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_case, task): task for task in tasks}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                json.dumps(
                    {
                        "completed": row["candidate"],
                        "relative_error": row["maximum_local_relative_error"],
                        "closure": row["closure_passed"],
                        "speedup": row["polynomial_elapsed_seconds"]
                        / row["mixed_elapsed_seconds"],
                    }
                ),
                flush=True,
            )
    rows.sort(
        key=lambda row: (
            row["candidate"]["seed"],
            row["candidate"]["threshold"],
            row["candidate"]["anchor"],
        )
    )
    polynomial_median = statistics.median(
        row["polynomial_elapsed_seconds"] for row in rows
    )
    mixed_median = statistics.median(row["mixed_elapsed_seconds"] for row in rows)
    speedup = polynomial_median / mixed_median
    equivalence = (
        all(row["closure_decision_matches"] for row in rows)
        and max(row["maximum_local_relative_error"] for row in rows) <= 3.0e-12
    )
    result = {
        "status": "mixed directional cohort audit complete",
        "evidence_boundary": (
            "Independent linear-cost replay; no outcomes and no randomized Green query."
        ),
        "source_hashes": hashes,
        "cases": len(rows),
        "all_local_and_closure_results_reproduced": equivalence,
        "maximum_local_relative_error": max(
            row["maximum_local_relative_error"] for row in rows
        ),
        "polynomial_median_seconds": polynomial_median,
        "mixed_median_seconds": mixed_median,
        "median_runtime_speedup": speedup,
        "prespecified_speed_gate": 2.0,
        "prespecified_audit_passed": equivalence and speedup >= 2.0,
        "rows": rows,
        "outcome_files_read": 0,
    }
    OUTPUT.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
