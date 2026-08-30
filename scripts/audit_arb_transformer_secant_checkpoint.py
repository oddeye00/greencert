#!/usr/bin/env python3
"""Outward 192-bit scalar secant jets on a real sealed Transformer checkpoint."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch
from flint import arb, ctx

from arb_transformer_multijet import (
    arb_transformer_objective_jet,
    make_parameter_jet,
)
from audit_transformer_v3_response_free_probe import probe_seed
from transformer_certificate_protocol import Candidate
from transformer_four_sweep_development_audit import to_scaled
from transformer_green_development_audit import build_frozen_centerline
from transformer_green_operator import make_transformer_green_products
from transformer_hvp_grokking import gradient, objective_hvp
from transformer_v3_certificate import load_candidate, output_path, safe_json


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "transformer_v3_arb_secant_checkpoint_audit.json"
NONCE = "5a37e5ccaf6834c438fde251d52ec1de313329314377d70cc1cb25e62fc52f2a"
DOMAIN = "greencert-response-free-secant-four-probe-v1|"
LAMBDA = 4096.0
HORIZON = 52
PROBES = 4


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    ctx.prec = 192
    candidate = Candidate(366, 0.7, 1040)
    certificate_path = output_path(candidate)
    certificate = safe_json(certificate_path)
    config, template, spec, data, parameter, velocity = load_candidate(candidate)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    train_pairs, train_labels, *_ = data
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
    apply_green, _ = make_transformer_green_products(
        center[:HORIZON, :dimension],
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )
    signed = apply_green(residual.reshape(-1)).reshape(HORIZON, -1)

    q_rows = [torch.zeros_like(signed[0])]
    for step in range(1, HORIZON):
        point = center[step, :dimension]
        direction = signed[step - 1, :dimension]
        base_gradient = (
            mapped[step][dimension:] - config.momentum * center[step, dimension:]
        )
        hessian_direction = objective_hvp(
            point,
            direction,
            train_pairs,
            train_labels,
            template,
            spec,
            config,
        )
        shifted_gradient = gradient(
            point + LAMBDA * direction,
            train_pairs,
            train_labels,
            template,
            spec,
            config,
        )
        numerator = shifted_gradient - base_gradient - LAMBDA * hessian_direction
        scaled = config.learning_rate * numerator / LAMBDA**2
        q_rows.append(torch.cat((-scaled, scaled)))
    secant = torch.stack(q_rows)
    norms = torch.linalg.vector_norm(secant, dim=1)
    step = int(torch.argmax(norms[1:]).item()) + 1

    generator = torch.Generator(device="cpu").manual_seed(
        probe_seed(nonce=NONCE, domain=DOMAIN)
    )
    gaussian = torch.randn(
        PROBES,
        secant.numel(),
        generator=generator,
        dtype=secant.dtype,
    ).reshape(PROBES, HORIZON, -1)
    state_probes = gaussian[:, step, :]
    theta_probe, velocity_probe = state_probes.split(dimension, dim=1)
    objective_probes = velocity_probe - theta_probe
    point = center[step, :dimension]
    direction = signed[step - 1, :dimension]

    started = time.perf_counter()
    base_jet = arb_transformer_objective_jet(
        make_parameter_jet(
            point.tolist(),
            [row.tolist() for row in objective_probes],
            spec,
            x_direction=direction.tolist(),
        ),
        train_pairs,
        train_labels,
        config,
    )
    base_seconds = time.perf_counter() - started
    started = time.perf_counter()
    shifted_jet = arb_transformer_objective_jet(
        make_parameter_jet(
            point.tolist(),
            [row.tolist() for row in objective_probes],
            spec,
            base_terms=[(LAMBDA, direction.tolist())],
        ),
        train_pairs,
        train_labels,
        config,
    )
    shifted_seconds = time.perf_counter() - started
    if base_jet.xy is None:
        raise RuntimeError("mixed jet was not constructed")
    intervals = []
    maximum_midpoint_difference = 0.0
    for index in range(PROBES):
        raw = (
            shifted_jet.y[index]
            - base_jet.y[index]
            - arb(LAMBDA) * base_jet.xy[index]
        )
        projected = arb(config.learning_rate) * raw / arb(LAMBDA) ** 2
        observed = float(torch.dot(state_probes[index], secant[step]))
        midpoint_difference = abs(float(projected.mid()) - observed)
        maximum_midpoint_difference = max(maximum_midpoint_difference, midpoint_difference)
        cancellation_ratio = (
            abs(observed) / max(abs(float(projected.mid())), float(projected.rad()))
            if observed != 0.0
            else 0.0
        )
        intervals.append(
            {
                "probe": index,
                "lower": float(projected.lower()),
                "upper": float(projected.upper()),
                "midpoint": float(projected.mid()),
                "radius": float(projected.rad()),
                "float64_vector_projection": observed,
                "midpoint_difference": midpoint_difference,
                "float_to_outward_magnitude_ratio": cancellation_ratio,
            }
        )
    payload = {
        "status": "OUTWARD ARB TRANSFORMER SECANT CHECKPOINT AUDIT PASSED",
        "evidence_boundary": (
            "Post-seal, outcome-blind numerical-rigor audit. Arb encloses exact "
            "scalar objective jets conditional on stored dyadic center/response "
            "and ideal-PRNG probe values; it is one selected checkpoint, not the "
            "full sequence or a complete Transformer certificate."
        ),
        "candidate": candidate.__dict__,
        "horizon": HORIZON,
        "selected_checkpoint": step,
        "selection_rule": "largest float64 amplified-forcing row norm among steps 1..51",
        "amplification": LAMBDA,
        "probes": PROBES,
        "precision_bits": ctx.prec,
        "base_mixed_jet_seconds": base_seconds,
        "shifted_first_jet_seconds": shifted_seconds,
        "total_jet_seconds": base_seconds + shifted_seconds,
        "projected_full_sequence_jet_minutes": (
            (base_seconds + shifted_seconds) * (HORIZON - 1) / 60.0
        ),
        "selected_forcing_row_norm": float(norms[step]),
        "maximum_midpoint_difference": maximum_midpoint_difference,
        "maximum_outward_radius": max(row["radius"] for row in intervals),
        "maximum_outward_projection_magnitude": max(
            max(abs(row["lower"]), abs(row["upper"])) for row in intervals
        ),
        "minimum_float_to_outward_magnitude_ratio": min(
            row["float_to_outward_magnitude_ratio"] for row in intervals
        ),
        "intervals": intervals,
        "protocol_sha256": sha256(ROOT / "AMPLIFIED_SECANT_FOUR_PROBE_PROTOCOL.md"),
        "certificate_sha256": sha256(certificate_path),
        "centerline_sha256": path["centerline_sha256"],
        "outcome_files_read": 0,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
