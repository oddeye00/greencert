#!/usr/bin/env python3
"""Fresh-probe corrected-path closure with cancellation-safe secant forcing."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import torch

from probe_jacobian_bound import ProbeConfig, ProbeRegistry, gram_norm_bound
from relinearized_green_closure import exact_relinearized_closure
from transformer_certificate_protocol import Candidate
from transformer_four_sweep_development_audit import to_scaled
from transformer_green_development_audit import build_frozen_centerline
from transformer_green_operator import make_transformer_green_products
from transformer_hvp_grokking import logits
from transformer_v3_certificate import (
    _gate_raw_slacks,
    _logic_slack,
    _persistent_bracket,
    load_candidate,
    output_path,
    safe_json,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "transformer_v3_relinearized_secant_audit.json"
PROTOCOL = ROOT / "RELINEARIZED_SECANT_AUDIT_PROTOCOL.md"
AMPLIFIED_RESULT = RESULTS / "transformer_v3_amplified_secant_full_audit.json"
OUTWARD_RESULT = RESULTS / "transformer_v3_arb_secant_full_v2_independent_audit.json"

CANDIDATE = Candidate(366, 0.70, 1040)
HORIZON = 52
SWEEPS = 4
PROBES = 16
POWER = 1
DELTA = 4.59896983075791e-11
LAMBDA = 4096.0
MASTER_NONCE = "8d6d17bfe5fb4a861d7673d13a633bfe6e761b0e6755a3d21030c5dea0992066"
IDENTITY = (92, CANDIDATE.seed, CANDIDATE.gate_index, CANDIDATE.anchor, HORIZON, SWEEPS, POWER)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def from_scaled(path: torch.Tensor, dimension: int, eta: float) -> torch.Tensor:
    return torch.cat((path[..., :dimension], path[..., dimension:] / eta), dim=-1)


def maximum_injection_forcing(
    *, kappa: float, derivative_drift: float, correction_max: float, domain: float
) -> dict:
    available = max(0.0, float(domain) - float(correction_max))
    coefficient = float(kappa) * float(derivative_drift)
    if kappa <= 0.0:
        return {"radius": available, "response_cap": math.inf, "injection_cap": math.inf}
    if coefficient <= 0.0:
        return {
            "radius": available,
            "response_cap": available,
            "injection_cap": available / kappa,
        }
    radius = min(available, 1.0 / coefficient)
    response_cap = max(0.0, radius - 0.5 * coefficient * radius * radius)
    return {
        "radius": radius,
        "response_cap": response_cap,
        "injection_cap": response_cap / kappa,
    }


def main() -> None:
    started = time.perf_counter()
    certificate_path = output_path(CANDIDATE)
    certificate = safe_json(certificate_path)
    amplified = safe_json(AMPLIFIED_RESULT)
    outward = safe_json(OUTWARD_RESULT)
    if amplified["candidate"] != CANDIDATE.__dict__:
        raise RuntimeError("amplified-secant candidate changed")
    row = next(
        item for item in amplified["rows"] if float(item["amplification"]) == LAMBDA
    )
    secant_discrepancy = float(row["analytic_secant_discrepancy_upper"])
    outward_secant_norm = float(outward["secant_forcing_norm_upper"])
    if outward["candidate"] != CANDIDATE.__dict__:
        raise RuntimeError("outward secant candidate changed")
    if int(outward["intervals_recomputed"]) != 204:
        raise RuntimeError("outward secant audit no longer contains 204 intervals")
    if int(outward["outcome_files_read"]) != 0:
        raise RuntimeError("outward secant audit crossed the outcome barrier")

    config, template, spec, data, parameter, velocity = load_candidate(CANDIDATE)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = data
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
    center = path["center"][: HORIZON + 1]
    scaled_center = path["scaled_center"][: HORIZON + 1]
    mapped = [path["map_step"](center[step]) for step in range(HORIZON)]
    residual = torch.stack(
        [
            to_scaled(mapped[step], dimension, config.learning_rate)
            - scaled_center[step + 1]
            for step in range(HORIZON)
        ]
    )
    old_apply, _ = make_transformer_green_products(
        center[:HORIZON, :dimension],
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )
    correction_rows = old_apply(residual.reshape(-1)).reshape(HORIZON, -1)
    correction = torch.cat(
        (torch.zeros_like(correction_rows[:1]), correction_rows), dim=0
    )
    corrected_scaled = scaled_center + correction
    corrected = from_scaled(corrected_scaled, dimension, config.learning_rate)
    correction_max = float(torch.linalg.vector_norm(correction, dim=1).max())

    corrected_apply, corrected_transpose = make_transformer_green_products(
        corrected[:HORIZON, :dimension],
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )
    registry = ProbeRegistry([IDENTITY], MASTER_NONCE)
    probe = ProbeConfig(probes=PROBES, power=POWER, delta=DELTA)

    def gram(vector: torch.Tensor) -> torch.Tensor:
        return corrected_transpose(corrected_apply(vector))

    probe_started = time.perf_counter()
    green = gram_norm_bound(
        gram,
        dimension=HORIZON * 2 * dimension,
        dtype=corrected.dtype,
        device=corrected.device,
        config=probe,
        identity=IDENTITY,
        registry=registry,
    )
    probe_seconds = time.perf_counter() - probe_started
    kappa = float(green["operator_norm_upper_bound"])
    old_row = certificate["power_rows"][POWER - 1]
    drift = float(old_row["maximum_optimizer_derivative_drift_upper"])
    domain = float(old_row["one_shot_closure"]["domain_radius"])
    total_injection = secant_discrepancy + outward_secant_norm
    forcing = kappa * total_injection
    closure = exact_relinearized_closure(
        kappa=kappa,
        derivative_drift=drift,
        corrected_defect_response_bound=forcing,
        correction_max_state_norm=correction_max,
        domain_radius=domain,
    )
    forcing_cap = maximum_injection_forcing(
        kappa=kappa,
        derivative_drift=drift,
        correction_max=correction_max,
        domain=domain,
    )

    bracket = None
    logic_slack = None
    maximum_margin = None
    if closure.closure_passed:
        radius = float(closure.remainder_radius)
        required = int(certificate["required_correct"])
        guarantee_slacks = []
        exclusion_slacks = []
        margins = []
        for step in range(HORIZON + 1):
            raw = _gate_raw_slacks(
                logits(corrected[step, :dimension], cert_pairs, template, spec),
                cert_labels,
                required,
            )
            if step == 0:
                margin = 0.0
            else:
                output = certificate["output_rows"][step - 1]
                output_upper = float(
                    output["trace"]["rows"][POWER - 1][
                        "operator_norm_upper_bound"
                    ]
                )
                second = float(output["block_second"])
                parameter_shift = float(
                    torch.linalg.vector_norm(correction[step, :dimension])
                )
                margin = math.sqrt(2.0) * (
                    (output_upper + second * parameter_shift) * radius
                    + 0.5 * second * radius * radius
                )
            margins.append(margin)
            guarantee_slacks.append(raw[0] - margin)
            exclusion_slacks.append(raw[1] - margin)
        bracket = _persistent_bracket(guarantee_slacks, exclusion_slacks)
        logic_slack = _logic_slack(bracket, guarantee_slacks, exclusion_slacks)
        maximum_margin = max(margins)

    payload = {
        "status": "FRESH-PROBE RELINEARIZED AMPLIFIED-SECANT AUDIT COMPLETED",
        "evidence_boundary": (
            "Post-seal, outcome-blind method-development audit frozen after the "
            "literal corrected-defect branch exposed cancellation. The rebuilt "
            "operator receives a new fixed Gaussian block. The secant scalar "
            "forcing is outward conditional on stored dyadic inputs; upstream "
            "Green products and output margins remain float64. Prospective counts "
            "are unchanged."
        ),
        "candidate": CANDIDATE.__dict__,
        "horizon": HORIZON,
        "probe": green,
        "probe_registry": registry.summary(),
        "old_power_one_kappa": float(old_row["kappa_upper"]),
        "relinearized_kappa": kappa,
        "old_mixed_coefficient": float(
            old_row["one_shot_closure"]["linearized_remainder_coefficient"]
        ),
        "new_mixed_coefficient": 0.0,
        "correction_sequence_norm": float(torch.linalg.vector_norm(correction_rows)),
        "correction_max_state_norm": correction_max,
        "amplification": LAMBDA,
        "analytic_secant_discrepancy_upper": secant_discrepancy,
        "outward_secant_forcing_norm_upper": outward_secant_norm,
        "total_injection_forcing_upper": total_injection,
        "corrected_defect_response_bound": forcing,
        "forcing_capacity": forcing_cap,
        "forcing_headroom_ratio": (
            forcing_cap["injection_cap"] / total_injection
            if total_injection > 0.0
            else math.inf
        ),
        "closure": closure.as_dict(),
        "bracket": bracket,
        "logic_slack": logic_slack,
        "maximum_margin_radius": maximum_margin,
        "old_power_one_bracket": old_row["certified_bracket"],
        "probe_seconds": probe_seconds,
        "total_seconds": time.perf_counter() - started,
        "old_power_one_logical_green_gram_applications": int(
            old_row["logical_green_gram_applications"]
        ),
        "new_logical_green_gram_applications": int(green["gram_applications"]),
        "extra_causal_response_sweeps": 0,
        "certificate_sha256": sha256(certificate_path),
        "centerline_sha256": path["centerline_sha256"],
        "amplified_result_sha256": sha256(AMPLIFIED_RESULT),
        "outward_result_sha256": sha256(OUTWARD_RESULT),
        "protocol_sha256": sha256(PROTOCOL),
        "source_sha256": sha256(Path(__file__)),
        "combined_failure_upper": DELTA + 1.0e-6,
        "outcome_files_read": 0,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
