#!/usr/bin/env python3
"""Outcome-blind numerical audit of the amplified-secant response primitive."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from transformer_certificate_protocol import Candidate
from transformer_four_sweep_development_audit import to_scaled
from transformer_fourth_jet_bound import objective_fourth_derivative_bound
from transformer_green_development_audit import build_frozen_centerline
from transformer_green_operator import make_transformer_green_products
from transformer_hvp_grokking import gradient, objective_hvp
from transformer_two_response import optimizer_center_quadratic_defect
from transformer_v3_certificate import load_candidate, output_path, safe_json


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "transformer_v3_amplified_secant_one_step_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denominator = float(torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right))
    if denominator == 0.0:
        return None
    return float(torch.dot(left, right) / denominator)


def run(candidate: Candidate, amplifications: list[float]) -> dict:
    certificate_path = output_path(candidate)
    certificate = safe_json(certificate_path)
    config, template, spec, data, parameter, velocity = load_candidate(candidate)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    train_pairs, train_labels, _, _, _, _ = data
    horizon = int(certificate["protocol"]["horizon"])
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
        raise RuntimeError("centerline hash mismatch")
    center = path["center"][: horizon + 1]
    scaled_center = path["scaled_center"][: horizon + 1]
    residual = torch.stack(
        [
            to_scaled(path["map_step"](center[step]), dimension, config.learning_rate)
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
    parameter_norms = torch.linalg.vector_norm(signed[:, :dimension], dim=1)
    # q_0 is anchored at zero, so select among transition inputs 1,...,H-1.
    selected = int(torch.argmax(parameter_norms[: horizon - 1]).item()) + 1
    direction = signed[selected - 1]
    parameter_direction = direction[:dimension]
    point = center[selected, :dimension]

    started = time.perf_counter()
    q2 = optimizer_center_quadratic_defect(
        point,
        direction,
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )
    q2_seconds = time.perf_counter() - started
    started = time.perf_counter()
    base_gradient = gradient(
        point, train_pairs, train_labels, template, spec, config
    )
    hessian_direction = objective_hvp(
        point,
        parameter_direction,
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )
    setup_seconds = time.perf_counter() - started
    q2_norm = float(torch.linalg.vector_norm(q2))
    direction_norm = float(torch.linalg.vector_norm(parameter_direction))
    rows = []
    for lam in amplifications:
        started = time.perf_counter()
        shifted_gradient = gradient(
            point + lam * parameter_direction,
            train_pairs,
            train_labels,
            template,
            spec,
            config,
        )
        numerator_gradient = (
            shifted_gradient - base_gradient - lam * hessian_direction
        )
        remainder = numerator_gradient / (lam * lam)
        scaled = config.learning_rate * remainder
        secant = torch.cat((-scaled, scaled))
        elapsed = time.perf_counter() - started
        fourth = objective_fourth_derivative_bound(
            point,
            template,
            spec,
            config,
            radius=max(1.0, lam) * direction_norm,
        )
        map_third = math.sqrt(2.0) * config.learning_rate * fourth
        analytic_error = abs(lam - 1.0) * map_third * direction_norm**3 / 6.0
        secant_norm = float(torch.linalg.vector_norm(secant))
        difference = float(torch.linalg.vector_norm(secant - q2))
        rows.append(
            {
                "amplification": lam,
                "shift_radius": lam * direction_norm,
                "numerator_gradient_norm": float(
                    torch.linalg.vector_norm(numerator_gradient)
                ),
                "secant_norm": secant_norm,
                "q2_reference_norm": q2_norm,
                "secant_to_q2_norm_ratio": (
                    secant_norm / q2_norm if q2_norm > 0.0 else None
                ),
                "secant_q2_difference_norm": difference,
                "secant_q2_relative_difference": (
                    difference / q2_norm if q2_norm > 0.0 else None
                ),
                "secant_q2_cosine": cosine(secant, q2),
                "analytic_exact_defect_to_secant_upper": analytic_error,
                "analytic_upper_to_q2_norm_ratio": (
                    analytic_error / q2_norm if q2_norm > 0.0 else None
                ),
                "shifted_gradient_seconds": elapsed,
            }
        )
    payload = {
        "status": "OUTCOME-BLIND AMPLIFIED-SECANT ONE-STEP AUDIT COMPLETED",
        "evidence_boundary": (
            "Post-seal arithmetic diagnostic. The q2 reference and secants are "
            "ordinary float64; analytic discrepancy bounds are theorem-valid, "
            "but no outward gradient/HVP residual is claimed."
        ),
        "candidate": candidate.__dict__,
        "horizon": horizon,
        "selected_transition_input": selected,
        "selection_rule": "largest parameter-component signed-response norm",
        "parameter_direction_norm": direction_norm,
        "q2_reference_norm": q2_norm,
        "q2_reference_seconds": q2_seconds,
        "base_gradient_and_hvp_setup_seconds": setup_seconds,
        "certificate_sha256": sha256(certificate_path),
        "centerline_sha256": path["centerline_sha256"],
        "outcome_files_read": 0,
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=366)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--anchor", type=int, default=1040)
    parser.add_argument(
        "--amplifications",
        type=float,
        nargs="+",
        default=[1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0, 65536.0],
    )
    args = parser.parse_args()
    payload = run(
        Candidate(args.seed, args.threshold, args.anchor),
        [float(value) for value in args.amplifications],
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
