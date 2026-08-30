#!/usr/bin/env python3
"""Outcome-blind checkpoint-local fourth-order audit for two responses."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from transformer_certificate_protocol import Candidate
from transformer_four_sweep_development_audit import to_scaled
from transformer_fourth_jet_bound import objective_fourth_derivative_bound
from transformer_green_development_audit import build_frozen_centerline
from transformer_green_operator import make_transformer_green_products
from transformer_v3_certificate import load_candidate, output_path, safe_json


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
POLICY = RESULTS / "transformer_v3_two_response_policy_audit.json"
OUTPUT = RESULTS / "transformer_v3_two_response_local_fourth_audit.json"
CACHE = RESULTS / "transformer_v3_two_response_local_fourth_cache"
VERSION = 1


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def cache_path(candidate: Candidate) -> Path:
    return CACHE / (
        f"seed_{candidate.seed}_gate_{candidate.gate_index}_"
        f"anchor_{candidate.anchor}_v{VERSION}.json"
    )


def relative_error(left: float, right: float) -> float:
    return abs(left - right) / max(abs(right), torch.finfo(torch.float64).tiny)


def audit_row(policy_row: dict) -> dict:
    started = time.perf_counter()
    candidate = Candidate(**policy_row["candidate"])
    certificate_path = output_path(candidate)
    certificate = safe_json(certificate_path)
    config, template, spec, data, parameter, velocity = load_candidate(candidate)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    train_pairs, train_labels, _, _, _, _ = data
    horizon = int(policy_row["horizon"])
    dimension = int(parameter.numel())
    path = build_frozen_centerline(
        config,
        template,
        spec,
        train_pairs,
        train_labels,
        parameter,
        velocity,
    )
    if path["centerline_sha256"] != certificate["centerline_sha256"]:
        raise RuntimeError(f"centerline hash mismatch for {candidate}")
    center = path["center"][: horizon + 1]
    scaled_center = path["scaled_center"][: horizon + 1]
    residual = torch.stack(
        [
            to_scaled(
                path["map_step"](center[step]),
                dimension,
                config.learning_rate,
            )
            - scaled_center[step + 1]
            for step in range(horizon)
        ]
    )
    apply_green, _ = make_transformer_green_products(
        center[:horizon, :dimension],
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )
    signed = apply_green(residual.reshape(-1)).reshape(horizon, -1)
    observed_norm = float(torch.linalg.vector_norm(signed))
    if relative_error(
        observed_norm, float(certificate["signed_response_sequence_norm"])
    ) > 2.0e-12:
        raise RuntimeError(f"signed-response mismatch for {candidate}")

    terms = []
    fourth_bounds = []
    direction_norms = []
    for step in range(1, horizon):
        direction = signed[step - 1, :dimension]
        direction_norm = float(torch.linalg.vector_norm(direction))
        direction_norms.append(direction_norm)
        fourth = objective_fourth_derivative_bound(
            center[step, :dimension],
            template,
            spec,
            config,
            radius=direction_norm,
        )
        fourth_bounds.append(fourth)
        terms.append(
            math.sqrt(2.0)
            * float(config.learning_rate)
            * fourth
            * direction_norm**3
            / 6.0
        )
    taylor_error = math.sqrt(sum(term * term for term in terms))
    total_headroom = float(policy_row["admissible_sigma_q_plus_tau_q"])
    passed = taylor_error < total_headroom
    return {
        "version": VERSION,
        "candidate": policy_row["candidate"],
        "certificate_path": certificate_path.relative_to(ROOT).as_posix(),
        "certificate_sha256": sha256(certificate_path),
        "horizon": horizon,
        "centerline_sha256": path["centerline_sha256"],
        "maximum_parameter_direction_norm": max(direction_norms, default=0.0),
        "maximum_local_objective_fourth_derivative_upper": max(
            fourth_bounds, default=0.0
        ),
        "minimum_local_objective_fourth_derivative_upper": min(
            fourth_bounds, default=0.0
        ),
        "directional_quadratic_taylor_error_upper": taylor_error,
        "admissible_sigma_q_plus_tau_q": total_headroom,
        "remaining_arithmetic_and_recurrence_headroom": (
            total_headroom - taylor_error
        ),
        "headroom_to_taylor_error_ratio": (
            math.inf if taylor_error == 0.0 else total_headroom / taylor_error
        ),
        "local_fourth_order_taylor_gate_passed": passed,
        "local_fourth_jet_evaluations": max(0, horizon - 1),
        "elapsed_seconds": time.perf_counter() - started,
        "outcome_files_read": 0,
    }


def valid_cache(candidate: Candidate, row: dict) -> bool:
    return (
        int(row.get("version", -1)) == VERSION
        and row.get("candidate") == candidate.__dict__
        and row.get("certificate_sha256") == sha256(output_path(candidate))
    )


def write_cache(candidate: Candidate, row: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    destination = cache_path(candidate)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    policy = safe_json(POLICY)
    selected = [
        row for row in policy["rows"] if row["adaptive_second_response_invoked"]
    ]
    rows = []
    pending = []
    for policy_row in selected:
        candidate = Candidate(**policy_row["candidate"])
        destination = cache_path(candidate)
        if not args.refresh and destination.exists():
            cached = safe_json(destination)
            if valid_cache(candidate, cached):
                rows.append(cached)
                print(
                    f"reused seed={candidate.seed} gate={candidate.threshold:.1f} "
                    f"anchor={candidate.anchor}"
                )
                continue
        pending.append(policy_row)
    with ProcessPoolExecutor(
        max_workers=min(args.workers, max(1, len(pending)))
    ) as pool:
        futures = {pool.submit(audit_row, row): row for row in pending}
        for future in as_completed(futures):
            row = future.result()
            candidate = Candidate(**row["candidate"])
            rows.append(row)
            write_cache(candidate, row)
            print(
                f"audited seed={candidate.seed} gate={candidate.threshold:.1f} "
                f"anchor={candidate.anchor} "
                f"pass={row['local_fourth_order_taylor_gate_passed']}"
            )
    rows.sort(
        key=lambda row: (
            int(row["candidate"]["seed"]),
            float(row["candidate"]["threshold"]),
            int(row["candidate"]["anchor"]),
        )
    )
    passed = sum(row["local_fourth_order_taylor_gate_passed"] for row in rows)
    payload = {
        "status": (
            "OUTCOME-BLIND LOCAL FOURTH-ORDER AUDIT PASSED"
            if passed == len(rows)
            else "OUTCOME-BLIND LOCAL FOURTH-ORDER AUDIT FOUND FAILURES"
        ),
        "evidence_boundary": (
            "Post-seal method-development audit. Each fourth-derivative envelope "
            "is evaluated on the exact local segment ball c_j+t z_j. Positive "
            "remaining headroom is reserved for outward directional-product and "
            "response-recurrence arithmetic."
        ),
        "selected_adaptive_cases": len(selected),
        "audited_cases": len(rows),
        "local_fourth_order_taylor_passes": passed,
        "minimum_headroom_to_taylor_error_ratio": min(
            row["headroom_to_taylor_error_ratio"] for row in rows
        ),
        "maximum_local_objective_fourth_derivative_upper": max(
            row["maximum_local_objective_fourth_derivative_upper"] for row in rows
        ),
        "maximum_parameter_direction_norm": max(
            row["maximum_parameter_direction_norm"] for row in rows
        ),
        "aggregate_elapsed_seconds": sum(
            row["elapsed_seconds"] for row in rows
        ),
        "policy_source": POLICY.relative_to(ROOT).as_posix(),
        "policy_source_sha256": sha256(POLICY),
        "outcome_files_read": 0,
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"{payload['status']}: {passed}/{len(rows)} pass")


if __name__ == "__main__":
    main()
