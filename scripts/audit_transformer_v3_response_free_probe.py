#!/usr/bin/env python3
"""Outcome-blind audit of the response-free amplified-secant probe interface."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from audit_transformer_v3_amplified_secant_full import maximum_forcing_for_closure
from one_shot_recenter_closure import exact_one_shot_closure
from randomized_residual_certificate import response_free_amplified_secant_beta
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
PROTOCOL = ROOT / "AMPLIFIED_SECANT_PROBE_PROTOCOL.md"
OUTPUT = RESULTS / "transformer_v3_response_free_probe_audit.json"
NONCE = "2df178250abfd4272951e2493a1c1b93ddc7d29e73a4359246eee8998f8a0778"
DOMAIN = "greencert-response-free-secant-v1|"
PROBES = 16
DELTA = 1.0e-6
AMPLIFICATION = 4096.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def probe_seed(*, nonce: str = NONCE, domain: str = DOMAIN) -> int:
    digest = hashlib.sha256((domain + nonce).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def run(
    *,
    nonce: str = NONCE,
    domain: str = DOMAIN,
    probes: int = PROBES,
    delta: float = DELTA,
    amplification: float = AMPLIFICATION,
    protocol_path: Path = PROTOCOL,
    output_path_: Path = OUTPUT,
) -> dict:
    candidate = Candidate(366, 0.7, 1040)
    certificate_path = output_path(candidate)
    certificate = safe_json(certificate_path)
    if certificate.get("green_trace") is None:
        raise RuntimeError("sealed candidate has no Green trace")
    config, template, spec, data, parameter, velocity = load_candidate(candidate)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = data
    horizon = 52
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

    q_rows = [torch.zeros_like(signed[0])]
    analytic_terms = []
    for step in range(1, horizon):
        direction = signed[step - 1]
        parameter_direction = direction[:dimension]
        direction_norm = float(torch.linalg.vector_norm(parameter_direction))
        point = center[step, :dimension]
        base_gradient = (
            mapped[step][dimension:] - config.momentum * center[step, dimension:]
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
        shifted_gradient = gradient(
            point + amplification * parameter_direction,
            train_pairs,
            train_labels,
            template,
            spec,
            config,
        )
        numerator = (
            shifted_gradient
            - base_gradient
            - amplification * hessian_direction
        )
        scaled = config.learning_rate * numerator / amplification**2
        q_rows.append(torch.cat((-scaled, scaled)))
        fourth = objective_fourth_derivative_bound(
            point,
            template,
            spec,
            config,
            radius=amplification * direction_norm,
        )
        map_third = math.sqrt(2.0) * config.learning_rate * fourth
        analytic_terms.append(
            (amplification - 1.0)
            * map_third
            * direction_norm**3
            / 6.0
        )
    secant = torch.stack(q_rows)
    secant_norm = float(torch.linalg.vector_norm(secant))
    sigma_secant = math.sqrt(sum(value * value for value in analytic_terms))

    generator = torch.Generator(device="cpu").manual_seed(
        probe_seed(nonce=nonce, domain=domain)
    )
    gaussian = torch.randn(
        probes,
        secant.numel(),
        generator=generator,
        dtype=secant.dtype,
    )
    projections = (gaussian @ secant.reshape(-1)).tolist()
    probe_budget = response_free_amplified_secant_beta(
        projections,
        projections,
        delta=delta,
        green_gain=float(certificate["power_rows"][0]["kappa_upper"]),
        analytic_discrepancy=sigma_secant,
    )

    power_row = certificate["power_rows"][0]
    kappa = float(power_row["kappa_upper"])
    drift = float(power_row["maximum_optimizer_derivative_drift_upper"])
    domain_radius = float(power_row["one_shot_closure"]["domain_radius"])
    closure = exact_one_shot_closure(
        kappa=kappa,
        derivative_drift=drift,
        response_sequence_norm=response_norm,
        response_max_state_norm=response_max,
        corrected_defect_response_bound=probe_budget.beta_upper,
        domain_radius=domain_radius,
    )
    raw_zero = _gate_raw_slacks(
        logits(center[0, :dimension], cert_pairs, template, spec),
        cert_labels,
        int(certificate["required_correct"]),
    )
    output_rows = certificate["output_rows"]
    output_uppers = [
        float(row["trace"]["rows"][0]["operator_norm_upper_bound"])
        for row in output_rows
    ]
    bracket = None
    logic_slack = None
    if closure.closure_passed:
        bracket, logic_slack, _ = _bracket_at_radius(
            radius=float(closure.total_pointwise_radius),
            output_uppers=output_uppers,
            output_rows=output_rows,
            raw_zero=raw_zero,
        )
    forcing_cap = maximum_forcing_for_closure(
        kappa=kappa,
        derivative_drift=drift,
        response_max=response_max,
        domain_radius=domain_radius,
    ) / kappa
    payload = {
        "status": "OUTCOME-BLIND RESPONSE-FREE PROBE AUDIT COMPLETED",
        "evidence_boundary": (
            "Post-seal method-development audit under the ideal-PRNG model. "
            "It reads no future outcome. Float64 point projections test the "
            "response-free geometry but are not outward intervals or an "
            "exact-real computer-assisted proof."
        ),
        "candidate": candidate.__dict__,
        "horizon": horizon,
        "power": 1,
        "amplification": amplification,
        "probes": probes,
        "delta": delta,
        "probe_seed": probe_seed(nonce=nonce, domain=domain),
        "protocol_sha256": sha256(protocol_path),
        "certificate_sha256": sha256(certificate_path),
        "centerline_sha256": path["centerline_sha256"],
        "outcome_files_read": 0,
        "secant_injection_norm_float64": secant_norm,
        "analytic_secant_discrepancy_upper": sigma_secant,
        "projections_float64": projections,
        "projection_absolute_max_float64": (
            probe_budget.projection_certificate.projection_upper
        ),
        "calibration": probe_budget.projection_certificate.calibration,
        "secant_norm_upper_point_projection": (
            probe_budget.projection_certificate.residual_norm_upper
        ),
        "probe_bound_to_observed_norm_ratio": (
            probe_budget.projection_certificate.residual_norm_upper / secant_norm
        ),
        "response_free_beta_upper_point_projection": probe_budget.beta_upper,
        "forcing_cap": forcing_cap,
        "forcing_headroom_ratio": forcing_cap
        / (
            sigma_secant
            + probe_budget.projection_certificate.residual_norm_upper
        ),
        "closure": closure.as_dict(),
        "bracket": bracket,
        "logic_slack": logic_slack,
    }
    output_path_.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
