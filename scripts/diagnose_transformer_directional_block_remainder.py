#!/usr/bin/env python3
"""Frozen, outcome-blind cohort diagnostic for the directional block remainder."""
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

from audit_transformer_adaptive_sweep_cohort import all_reduced_paths, scaled, unscaled
from audit_transformer_direct_image_green_panel import tensor_sha256
from corrected_path_closure import exact_corrected_path_closure
from transformer_certificate_protocol import Candidate
from transformer_directional_fourth_bound import directional_objective_fourth_bound
from transformer_fourth_jet_bound import objective_fourth_derivative_bound
from transformer_modal_forecast import optimizer_map
from transformer_optimizer_probe import make_scaled_optimizer_jvp_vjp
from transformer_v3_certificate import load_candidate, safe_json


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PARENT = RESULTS / "transformer_fully_recentered_three_sweep_audit.json"
PROTOCOL = ROOT / "DIRECTIONAL_BLOCK_REMAINDER_PROTOCOL.md"
THEOREM = ROOT / "DIRECTIONAL_BLOCK_REMAINDER_THEOREM.md"
MODULE = ROOT / "scripts" / "transformer_directional_fourth_bound.py"
TEST = ROOT / "scripts" / "test_transformer_directional_fourth_bound.py"
OUTPUT = RESULTS / "transformer_directional_block_remainder_diagnostic.json"
CACHE = RESULTS / "transformer_directional_block_remainder_cache"
DEVELOPMENT = (366, 0.8, 1120)
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
    candidate = Candidate(**parent["candidate"])
    destination = cache_path(candidate)
    hashes = task["hashes"]
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
    correction_rows_tensor = torch.stack(correction_rows)
    correction = torch.cat(
        (torch.zeros_like(correction_rows_tensor[:1]), correction_rows_tensor), dim=0
    )
    corrected_scaled = scaled_center + correction
    corrected = unscaled(corrected_scaled, dimension, eta)
    if tensor_sha256(corrected_scaled) != parent["corrected_path_sha256"]:
        raise RuntimeError(f"corrected path replay mismatch for {candidate}")

    step_rows = []
    old_terms = []
    new_terms = []
    all_no_larger = True
    for step in range(1, horizon):
        direction = correction[step, :dimension]
        direction_norm = float(torch.linalg.vector_norm(direction))
        old_fourth = objective_fourth_derivative_bound(
            center[step, :dimension],
            template,
            spec,
            config,
            radius=direction_norm,
        )
        old_gradient_remainder = old_fourth * direction_norm**3 / 6.0
        directional = directional_objective_fourth_bound(
            center[step, :dimension], direction, spec, config
        )
        new_gradient_remainder = float(
            directional["gradient_taylor_remainder_upper"]
        )
        old_scaled = math.sqrt(2.0) * eta * old_gradient_remainder
        new_scaled = math.sqrt(2.0) * eta * new_gradient_remainder
        old_terms.append(old_scaled)
        new_terms.append(new_scaled)
        no_larger = new_gradient_remainder <= old_gradient_remainder * (
            1.0 + 2.0e-12
        ) + 1.0e-300
        all_no_larger = all_no_larger and no_larger
        step_rows.append(
            {
                "step": step,
                "direction_norm": direction_norm,
                "old_gradient_remainder_upper": old_gradient_remainder,
                "directional_gradient_remainder_upper": new_gradient_remainder,
                "directional_to_old_ratio": (
                    new_gradient_remainder / old_gradient_remainder
                    if old_gradient_remainder > 0.0
                    else 0.0
                ),
                "directional_no_larger": no_larger,
                "objective_polynomial_terms": directional[
                    "objective_polynomial_terms"
                ],
                "fixed_point_iterations_used": directional[
                    "fixed_point_iterations_used"
                ],
                "maximum_stage_inflation": directional[
                    "maximum_stage_inflation"
                ],
            }
        )

    old_sequence = math.sqrt(sum(value * value for value in old_terms))
    new_sequence = math.sqrt(sum(value * value for value in new_terms))
    recorded_old = float(parent["directional_quadratic_taylor_error_upper"])
    if not math.isclose(old_sequence, recorded_old, rel_tol=3.0e-11, abs_tol=1.0e-300):
        raise RuntimeError(
            f"scalar fourth-order replay mismatch for {candidate}: "
            f"{old_sequence} != {recorded_old}"
        )
    injection = (
        float(parent["response_recurrence_residual_norm"])
        + float(parent["quadratic_surrogate_injection_norm"])
        + new_sequence
    )
    kappa = float(parent["green_operator_norm_upper_bound"])
    response = kappa * injection
    closure = exact_corrected_path_closure(
        kappa=kappa,
        derivative_drift=float(parent["maximum_optimizer_jacobian_drift"]),
        defect_response_bound=response,
        domain_radius=float(parent["domain_radius_about_corrected_path"]),
    )
    result = {
        "candidate": candidate.__dict__,
        "development_row": (
            candidate.seed,
            candidate.threshold,
            candidate.anchor,
        )
        == DEVELOPMENT,
        "horizon": horizon,
        "pipeline_diagnostics": pipeline,
        "corrected_path_sha256": tensor_sha256(corrected_scaled),
        "old_scalar_taylor_sequence_upper": old_sequence,
        "directional_block_taylor_sequence_upper": new_sequence,
        "directional_to_scalar_sequence_ratio": (
            new_sequence / old_sequence if old_sequence > 0.0 else 0.0
        ),
        "every_step_directional_no_larger": all_no_larger,
        "old_cancellation_safe_injection_upper": float(
            parent["cancellation_safe_injection_upper"]
        ),
        "directional_cancellation_safe_injection_upper": injection,
        "unchanged_green_operator_norm_upper_bound": kappa,
        "directional_defect_response_upper": response,
        "closure": closure.as_dict(),
        "closure_passed": closure.closure_passed,
        "parent_closure_passed": bool(parent["closure_passed"]),
        "step_rows": step_rows,
        "source_hashes": hashes,
        "outcome_files_read": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return result


def summarize(rows: list[dict], exclude_development: bool) -> dict:
    selected = [
        row for row in rows if not exclude_development or not row["development_row"]
    ]
    ratios = [row["directional_to_scalar_sequence_ratio"] for row in selected]
    newly_closed = [
        row
        for row in selected
        if row["closure_passed"] and not row["parent_closure_passed"]
    ]
    return {
        "cases": len(selected),
        "every_step_directional_no_larger": all(
            row["every_step_directional_no_larger"] for row in selected
        ),
        "closure_passed": sum(row["closure_passed"] for row in selected),
        "newly_closed": len(newly_closed),
        "newly_closed_candidates": [row["candidate"] for row in newly_closed],
        "median_directional_to_scalar_sequence_ratio": statistics.median(ratios),
        "maximum_directional_to_scalar_sequence_ratio": max(ratios),
        "minimum_directional_to_scalar_sequence_ratio": min(ratios),
        "median_elapsed_seconds": statistics.median(
            row["elapsed_seconds"] for row in selected
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    parent = safe_json(PARENT)
    hashes = {
        "parent": sha256(PARENT),
        "protocol": sha256(PROTOCOL),
        "theorem": sha256(THEOREM),
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
                        "ratio": row["directional_to_scalar_sequence_ratio"],
                        "closure": row["closure_passed"],
                        "seconds": row["elapsed_seconds"],
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
    all_summary = summarize(rows, exclude_development=False)
    holdout_summary = summarize(rows, exclude_development=True)
    promotion_passed = (
        holdout_summary["newly_closed"] >= 3
        and holdout_summary["every_step_directional_no_larger"]
    )
    result = {
        "status": "directional block remainder cohort diagnostic complete",
        "evidence_boundary": (
            "Frozen post-release diagnostic; reuses recorded Green bounds and "
            "reads no future outcomes."
        ),
        "source_hashes": hashes,
        "all_cases": all_summary,
        "nondevelopment_cases": holdout_summary,
        "prespecified_promotion_gate": {
            "required_new_nondevelopment_closures": 3,
            "requires_no_larger_at_every_step": True,
            "passed": promotion_passed,
        },
        "rows": rows,
        "outcome_files_read": 0,
    }
    OUTPUT.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
