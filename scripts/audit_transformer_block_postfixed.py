#!/usr/bin/env python3
"""Audit the frozen Transformer's ball-inflation post-fixed inequalities.

The sealed implementation accepts a relative numerical tolerance.  This
post-seal audit recomputes every issued-v3 envelope and reports both the exact
binary64 comparison and the tolerated comparison.  It changes no certificate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from transformer_block_envelope import ball_valid_envelope
from transformer_block_envelope import objective_hessian_lipschitz
from transformer_certificate_protocol import Candidate
from transformer_green_development_audit import build_frozen_centerline
from transformer_v3_certificate import _logic_slack, _persistent_bracket, load_candidate
from one_shot_recenter_closure import conservative_one_shot_closure
from strict_transformer_block_envelope import strict_ball_valid_envelope


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "transformer_v3_block_postfixed_postseal_audit.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _record_paths() -> list[Path]:
    paths = []
    for path in sorted(RESULTS.glob("transformer_v3_certificate_seed_*_gate_*_anchor_*.json")):
        if bool(_load(path).get("certificate_issued")):
            paths.append(path)
    if len(paths) != 11:
        raise AssertionError(f"expected 11 issued records, found {len(paths)}")
    return paths


def main(*, shortest_only: bool = False) -> None:
    paths = _record_paths()
    output = OUTPUT
    if shortest_only:
        paths = [min(paths, key=lambda path: int(_load(path)["protocol"]["horizon"]))]
        output = RESULTS / "transformer_v3_block_postfixed_shortest_postseal_audit.json"
    records = []
    total_states = 0
    exact_failures = 0
    maximum_absolute_deficit = 0.0
    maximum_relative_deficit = 0.0
    maximum_jet_relative_error = 0.0
    strict_binary64_failures = 0
    maximum_strict_jet_increase = 0.0

    for path in paths:
        payload = _load(path)
        print(
            f"recomputing {path.name} (H={payload['protocol']['horizon']})",
            flush=True,
        )
        raw = payload["candidate"]
        candidate = Candidate(int(raw["seed"]), float(raw["threshold"]), int(raw["anchor"]))
        config, template, spec, data, parameter, velocity = load_candidate(candidate)
        torch.set_num_threads(config.threads)
        torch.use_deterministic_algorithms(True)
        train_pairs, train_labels, *_ = data
        centerline = build_frozen_centerline(
            config,
            template,
            spec,
            train_pairs,
            train_labels,
            parameter,
            velocity,
        )["center"]
        dimension = parameter.numel()
        epsilon = float(payload["outer_domain_radius"])
        row_by_step = {int(row["step"]): row for row in payload["output_rows"]}

        case_exact_failures = 0
        case_max_abs = 0.0
        case_max_rel = 0.0
        case_max_jet_rel = 0.0
        case_strict_failures = 0
        case_max_strict_increase = 0.0
        strict_rows = []
        for step, stored in row_by_step.items():
            block = ball_valid_envelope(
                centerline[step, :dimension],
                spec,
                config,
                epsilon=epsilon,
                exact_values=True,
                sphere=True,
            )
            for key in ("first", "second", "third"):
                expected = float(stored[f"block_{key}"])
                observed = float(block[key])
                rel = abs(observed - expected) / max(abs(expected), 1e-300)
                case_max_jet_rel = max(case_max_jet_rel, rel)
            for name, first in block["stage_first"].items():
                target = float(first) * epsilon
                inflation = float(block["inflation"][name])
                deficit = max(0.0, target - inflation)
                relative = deficit / max(abs(target), 1e-300)
                if deficit > 0.0:
                    case_exact_failures += 1
                case_max_abs = max(case_max_abs, deficit)
                case_max_rel = max(case_max_rel, relative)

            strict = strict_ball_valid_envelope(
                centerline[step, :dimension],
                spec,
                config,
                epsilon=epsilon,
                exact_values=True,
                sphere=True,
            )
            for name, first in strict["stage_first"].items():
                if float(first) * epsilon > float(strict["inflation"][name]):
                    case_strict_failures += 1
            for key in ("first", "second", "third"):
                baseline = float(block[key])
                increase = (float(strict[key]) - baseline) / max(abs(baseline), 1e-300)
                case_max_strict_increase = max(case_max_strict_increase, increase)
            strict_rows.append((step, stored, strict))

        power = int(payload["earliest_issuing_power"])
        power_row = payload["power_rows"][power - 1]
        kappa = float(power_row["kappa_upper"])
        maximum_drift = 0.0
        guarantee_slacks = [-math.inf]
        exclusion_slacks = [math.inf]  # exact anchor is below gate by selection
        for step, stored, strict in strict_rows:
            output_upper = float(
                stored["trace"]["rows"][power - 1]["operator_norm_upper_bound"]
            )
            first_ball = output_upper + float(strict["second"]) * epsilon
            objective_drift = objective_hessian_lipschitz(
                first_ball,
                float(strict["second"]),
                float(strict["third"]),
            )
            if step < int(payload["protocol"]["horizon"]):
                maximum_drift = max(
                    maximum_drift,
                    math.sqrt(2.0) * float(config.learning_rate) * objective_drift,
                )
        strict_closure = conservative_one_shot_closure(
            kappa=kappa,
            derivative_drift=maximum_drift,
            response_sequence_norm=float(payload["signed_response_sequence_norm"]),
            response_max_state_norm=float(payload["signed_response_max_state_norm"]),
            domain_radius=epsilon,
        )
        strict_radius = strict_closure.total_pointwise_radius
        strict_bracket = None
        strict_logic_slack = None
        if strict_closure.closure_passed and strict_radius is not None:
            for _, stored, strict in strict_rows:
                output_upper = float(
                    stored["trace"]["rows"][power - 1]["operator_norm_upper_bound"]
                )
                margin = math.sqrt(2.0) * (
                    output_upper * strict_radius
                    + 0.5 * float(strict["second"]) * strict_radius * strict_radius
                )
                guarantee_slacks.append(float(stored["raw_guarantee_slack"]) - margin)
                exclusion_slacks.append(float(stored["raw_exclusion_slack"]) - margin)
            strict_bracket = _persistent_bracket(guarantee_slacks, exclusion_slacks)
            strict_logic_slack = _logic_slack(
                strict_bracket, guarantee_slacks, exclusion_slacks
            )

        states = len(row_by_step)
        total_states += states
        exact_failures += case_exact_failures
        maximum_absolute_deficit = max(maximum_absolute_deficit, case_max_abs)
        maximum_relative_deficit = max(maximum_relative_deficit, case_max_rel)
        maximum_jet_relative_error = max(maximum_jet_relative_error, case_max_jet_rel)
        strict_binary64_failures += case_strict_failures
        maximum_strict_jet_increase = max(
            maximum_strict_jet_increase, case_max_strict_increase
        )
        records.append(
            {
                "record": str(path.relative_to(ROOT)),
                "candidate": raw,
                "states": states,
                "exact_postfixed_failures": case_exact_failures,
                "maximum_absolute_postfixed_deficit": case_max_abs,
                "maximum_relative_postfixed_deficit": case_max_rel,
                "maximum_stored_jet_relative_error": case_max_jet_rel,
                "strict_binary64_postfixed_failures": case_strict_failures,
                "maximum_strict_jet_relative_increase": case_max_strict_increase,
                "strict_closure_passed": strict_closure.closure_passed,
                "strict_total_pointwise_radius": strict_radius,
                "strict_bracket": strict_bracket,
                "strict_logic_slack": strict_logic_slack,
                "frozen_bracket_reproduced": strict_bracket == payload["certified_bracket"],
            }
        )

    result = {
        "status": "post-seal numerical audit; no prospective count changed",
        "summary": {
            "issued_records": len(records),
            "states_recomputed": total_states,
            "exact_binary64_postfixed_failures": exact_failures,
            "maximum_absolute_postfixed_deficit": maximum_absolute_deficit,
            "maximum_relative_postfixed_deficit": maximum_relative_deficit,
            "maximum_stored_jet_relative_error": maximum_jet_relative_error,
            "frozen_check_relative_tolerance": 1e-9,
            "strict_binary64_postfixed_failures": strict_binary64_failures,
            "maximum_strict_jet_relative_increase": maximum_strict_jet_increase,
            "all_strict_brackets_reproduce_frozen": all(
                row["frozen_bracket_reproduced"] for row in records
            ),
        },
        "interpretation": (
            "An exact binary64 post-fixed comparison is stronger than the frozen "
            "1e-9-relative acceptance check but is still not outward real arithmetic."
        ),
        "records": records,
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest().upper()
    print(json.dumps({"output": str(output), "sha256": digest, **result["summary"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shortest-only",
        action="store_true",
        help="recompute only the shortest issued v3 candidate",
    )
    args = parser.parse_args()
    main(shortest_only=args.shortest_only)
