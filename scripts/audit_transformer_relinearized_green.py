#!/usr/bin/env python3
"""Fresh-probe post-seal audit of corrected-path Green closure.

The script reads no revealed future trajectory.  It reconstructs the sealed
four-sweep centerline, computes its deterministic signed response, rebuilds the
Jacobian sequence on that corrected path, and gives the rebuilt Green operator
one fresh, fixed Gaussian block.
"""

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
OUTPUT = ROOT / "results" / "transformer_v3_relinearized_green_audit.json"
PROTOCOL = ROOT / "RELINEARIZED_GREEN_AUDIT_PROTOCOL.md"

CANDIDATE = Candidate(366, 0.70, 1040)
HORIZON = 52
SWEEPS = 4
PROBES = 16
POWER = 1
DELTA = 4.59896983075791e-11
MASTER_NONCE = "f06a2a2711237d39d9a243db6d226de1224f7379947d75fb3870600aeec97886"
IDENTITY = (91, CANDIDATE.seed, CANDIDATE.gate_index, CANDIDATE.anchor, HORIZON, SWEEPS, POWER)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def from_scaled(path: torch.Tensor, dimension: int, eta: float) -> torch.Tensor:
    return torch.cat((path[..., :dimension], path[..., dimension:] / eta), dim=-1)


def main() -> None:
    started = time.perf_counter()
    certificate_path = output_path(CANDIDATE)
    certificate = safe_json(certificate_path)
    if int(certificate["protocol"]["horizon"]) != HORIZON:
        raise RuntimeError("sealed candidate horizon changed")
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
    corrected_mapped = [
        path["map_step"](corrected[step]) for step in range(HORIZON)
    ]
    corrected_defect = torch.stack(
        [
            to_scaled(corrected_mapped[step], dimension, config.learning_rate)
            - corrected_scaled[step + 1]
            for step in range(HORIZON)
        ]
    )

    corrected_apply, corrected_transpose = make_transformer_green_products(
        corrected[:HORIZON, :dimension],
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )
    corrected_response = corrected_apply(corrected_defect.reshape(-1)).reshape(
        HORIZON, -1
    )
    forcing = float(torch.linalg.vector_norm(corrected_response))
    correction_max = float(torch.linalg.vector_norm(correction, dim=1).max())

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
    closure = exact_relinearized_closure(
        kappa=kappa,
        derivative_drift=drift,
        corrected_defect_response_bound=forcing,
        correction_max_state_norm=correction_max,
        domain_radius=domain,
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
        "status": "FRESH-PROBE RELINEARIZED GREEN AUDIT COMPLETED",
        "evidence_boundary": (
            "Post-seal, outcome-blind method-development audit. The rebuilt "
            "operator receives one fresh fixed Gaussian block. Float64 model "
            "products are not an outward exact-real proof, and no prospective "
            "issuance count is changed."
        ),
        "candidate": CANDIDATE.__dict__,
        "horizon": HORIZON,
        "sweeps_before_correction": SWEEPS,
        "probe": green,
        "probe_registry": registry.summary(),
        "old_power_one_kappa": float(old_row["kappa_upper"]),
        "relinearized_kappa": kappa,
        "old_mixed_coefficient": float(
            old_row["one_shot_closure"]["linearized_remainder_coefficient"]
        ),
        "correction_sequence_norm": float(torch.linalg.vector_norm(correction_rows)),
        "correction_max_state_norm": correction_max,
        "corrected_defect_sequence_norm": float(
            torch.linalg.vector_norm(corrected_defect)
        ),
        "corrected_defect_response_norm": forcing,
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
        "extra_causal_response_sweeps": 1,
        "certificate_sha256": sha256(certificate_path),
        "centerline_sha256": path["centerline_sha256"],
        "protocol_sha256": sha256(PROTOCOL),
        "source_sha256": sha256(Path(__file__)),
        "outcome_files_read": 0,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
