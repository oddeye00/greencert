#!/usr/bin/env python3
"""Outcome-blind full-horizon audit of amplified-secant second responses."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch

from one_shot_recenter_closure import exact_one_shot_closure
from transformer_certificate_protocol import Candidate
from transformer_four_sweep_development_audit import to_scaled
from transformer_fourth_jet_bound import objective_fourth_derivative_bound
from transformer_green_development_audit import build_frozen_centerline
from transformer_green_operator import make_transformer_green_products
from transformer_hvp_grokking import gradient, logits, objective_hvp
from transformer_v3_certificate import (
    _bracket_at_radius,
    _gate_raw_slacks,
    load_candidate,
    output_path,
    safe_json,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "transformer_v3_amplified_secant_full_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def maximum_forcing_for_closure(
    *, kappa: float, derivative_drift: float, response_max: float, domain_radius: float
) -> float:
    b = float(kappa) * float(derivative_drift)
    linear = b * float(response_max)
    if linear >= 1.0:
        return 0.0
    available = float(domain_radius) - float(response_max)
    if available < 0.0:
        return 0.0
    if b == 0.0:
        return math.inf
    radius = min(available, (1.0 - linear) / b)
    return max(0.0, (1.0 - linear) * radius - 0.5 * b * radius * radius)


def run(candidate: Candidate, amplifications: list[float]) -> dict:
    certificate_path = output_path(candidate)
    certificate = safe_json(certificate_path)
    if certificate.get("green_trace") is None:
        raise ValueError("candidate has no sealed Green trace")
    config, template, spec, data, parameter, velocity = load_candidate(candidate)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = data
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
    mapped = [path["map_step"](center[step]) for step in range(horizon)]
    residual = torch.stack(
        [
            to_scaled(mapped[step], dimension, config.learning_rate)
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
    response_norm = float(torch.linalg.vector_norm(signed))
    response_max = float(torch.linalg.vector_norm(signed, dim=1).max())
    output_rows = certificate["output_rows"]
    required = int(certificate["required_correct"])
    raw_zero = _gate_raw_slacks(
        logits(center[0, :dimension], cert_pairs, template, spec),
        cert_labels,
        required,
    )
    power = 1
    power_row = certificate["power_rows"][power - 1]
    kappa = float(power_row["kappa_upper"])
    drift = float(power_row["maximum_optimizer_derivative_drift_upper"])
    domain_radius = float(power_row["one_shot_closure"]["domain_radius"])
    forcing_cap = maximum_forcing_for_closure(
        kappa=kappa,
        derivative_drift=drift,
        response_max=response_max,
        domain_radius=domain_radius,
    )
    injection_cap = forcing_cap / kappa
    output_uppers = [
        float(output["trace"]["rows"][power - 1]["operator_norm_upper_bound"])
        for output in output_rows
    ]

    rows = []
    for lam in amplifications:
        phase = time.perf_counter()
        q_rows = [torch.zeros_like(signed[0])]
        terms = []
        shifted_gradient_times = []
        hvp_times = []
        numerator_norms = []
        for step in range(1, horizon):
            direction = signed[step - 1]
            parameter_direction = direction[:dimension]
            direction_norm = float(torch.linalg.vector_norm(parameter_direction))
            point = center[step, :dimension]
            # The already-evaluated optimizer map stores v_next=mu*v+grad F.
            base_gradient = (
                mapped[step][dimension:]
                - config.momentum * center[step, dimension:]
            )
            started = time.perf_counter()
            hessian_direction = objective_hvp(
                point,
                parameter_direction,
                train_pairs,
                train_labels,
                template,
                spec,
                config,
            )
            hvp_times.append(time.perf_counter() - started)
            started = time.perf_counter()
            shifted_gradient = gradient(
                point + lam * parameter_direction,
                train_pairs,
                train_labels,
                template,
                spec,
                config,
            )
            shifted_gradient_times.append(time.perf_counter() - started)
            numerator = shifted_gradient - base_gradient - lam * hessian_direction
            numerator_norms.append(float(torch.linalg.vector_norm(numerator)))
            remainder = numerator / (lam * lam)
            scaled = config.learning_rate * remainder
            q_rows.append(torch.cat((-scaled, scaled)))
            fourth = objective_fourth_derivative_bound(
                point,
                template,
                spec,
                config,
                radius=max(1.0, lam) * direction_norm,
            )
            map_third = math.sqrt(2.0) * config.learning_rate * fourth
            terms.append(
                abs(lam - 1.0) * map_third * direction_norm**3 / 6.0
            )
        secant = torch.stack(q_rows)
        construction_seconds = time.perf_counter() - phase
        phase = time.perf_counter()
        second_response = apply_green(secant.reshape(-1)).reshape(horizon, -1)
        response_seconds = time.perf_counter() - phase
        secant_norm = float(torch.linalg.vector_norm(secant))
        second_norm = float(torch.linalg.vector_norm(second_response))
        sigma_secant = math.sqrt(sum(term * term for term in terms))
        beta_without_arithmetic = second_norm + kappa * sigma_secant
        closure = exact_one_shot_closure(
            kappa=kappa,
            derivative_drift=drift,
            response_sequence_norm=response_norm,
            response_max_state_norm=response_max,
            corrected_defect_response_bound=beta_without_arithmetic,
            domain_radius=domain_radius,
        )
        bracket = None
        logic_slack = None
        if closure.closure_passed:
            bracket, logic_slack, _ = _bracket_at_radius(
                radius=float(closure.total_pointwise_radius),
                output_uppers=output_uppers,
                output_rows=output_rows,
                raw_zero=raw_zero,
            )
        rows.append(
            {
                "amplification": lam,
                "secant_injection_norm": secant_norm,
                "secant_response_norm": second_norm,
                "analytic_secant_discrepancy_upper": sigma_secant,
                "beta_without_arithmetic": beta_without_arithmetic,
                "admissible_total_injection_error": injection_cap,
                "remaining_arithmetic_and_recurrence_headroom": (
                    injection_cap - sigma_secant
                ),
                "analytic_headroom_ratio": (
                    injection_cap / sigma_secant if sigma_secant > 0.0 else math.inf
                ),
                "maximum_numerator_gradient_norm": max(numerator_norms, default=0.0),
                "median_numerator_gradient_norm": statistics.median(numerator_norms),
                "construction_seconds": construction_seconds,
                "causal_response_seconds": response_seconds,
                "total_branch_seconds": construction_seconds + response_seconds,
                "median_shifted_gradient_seconds": statistics.median(
                    shifted_gradient_times
                ),
                "median_hvp_seconds": statistics.median(hvp_times),
                "closure": closure.as_dict(),
                "bracket": bracket,
                "logic_slack": logic_slack,
                "issued_without_arithmetic_budget": bracket is not None,
            }
        )
    payload = {
        "status": "OUTCOME-BLIND FULL-HORIZON AMPLIFIED-SECANT AUDIT COMPLETED",
        "evidence_boundary": (
            "Post-seal method-development audit. Analytic secant discrepancy is "
            "included exactly as required by the theorem; reported closure still "
            "reserves, but does not instantiate, outward arithmetic and recurrence "
            "error. No future outcome is read."
        ),
        "candidate": candidate.__dict__,
        "horizon": horizon,
        "power": power,
        "response_sequence_norm": response_norm,
        "response_max_state_norm": response_max,
        "admissible_total_injection_error": injection_cap,
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
        "--amplifications", type=float, nargs="+", default=[4096.0, 16384.0]
    )
    args = parser.parse_args()
    payload = run(
        Candidate(args.seed, args.threshold, args.anchor),
        [float(value) for value in args.amplifications],
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
