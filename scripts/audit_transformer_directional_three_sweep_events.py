#!/usr/bin/env python3
"""Outcome-blind event audit for directional three-sweep closures."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from analytic_jet_release import logit_margin_radius
from audit_transformer_adaptive_sweep_cohort import (
    all_reduced_paths,
    first_persistent,
    raw_slacks,
    scaled,
    unscaled,
)
from audit_transformer_direct_image_green_panel import tensor_sha256
from transformer_block_envelope import ball_valid_envelope
from transformer_certificate_protocol import Candidate
from transformer_hvp_grokking import logits
from transformer_modal_forecast import optimizer_map
from transformer_optimizer_probe import make_scaled_optimizer_jvp_vjp
from transformer_v3_certificate import load_candidate, output_path, safe_json


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CLOSURE_PARENT = RESULTS / "transformer_directional_block_remainder_diagnostic.json"
EVENT_PARENT = RESULTS / "transformer_fully_recentered_three_sweep_audit.json"
PROTOCOL = ROOT / "DIRECTIONAL_THREE_SWEEP_EVENT_PROTOCOL.md"
OUTPUT = RESULTS / "transformer_directional_three_sweep_event_audit.json"
PERSISTENCE = 25
SWEEPS = 3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def identity(candidate: dict) -> tuple[int, float, int]:
    return (
        int(candidate["seed"]),
        float(candidate["threshold"]),
        int(candidate["anchor"]),
    )


def replay_corrected(candidate: Candidate, horizon: int):
    config, template, spec, data, parameter, velocity = load_candidate(candidate)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = data
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
        (
            torch.zeros_like(torch.stack(correction_rows)[:1]),
            torch.stack(correction_rows),
        ),
        dim=0,
    )
    corrected_scaled = scaled_center + correction
    corrected = unscaled(corrected_scaled, dimension, eta)
    return (
        config,
        template,
        spec,
        cert_pairs,
        cert_labels,
        dimension,
        corrected,
        corrected_scaled,
        pipeline,
    )


def main() -> None:
    closure_parent = safe_json(CLOSURE_PARENT)
    event_parent = safe_json(EVENT_PARENT)
    event_by_identity = {
        identity(row["candidate"]): row for row in event_parent["rows"]
    }
    closure_rows = [row for row in closure_parent["rows"] if row["closure_passed"]]
    if len(closure_rows) != 4:
        raise RuntimeError("directional closure set changed before event audit")
    rows = []
    for closure_row in closure_rows:
        candidate = Candidate(**closure_row["candidate"])
        old = event_by_identity[identity(closure_row["candidate"])]
        horizon = int(closure_row["horizon"])
        (
            config,
            template,
            spec,
            cert_pairs,
            cert_labels,
            dimension,
            corrected,
            corrected_scaled,
            pipeline,
        ) = replay_corrected(candidate, horizon)
        replay_hash = tensor_sha256(corrected_scaled)
        if replay_hash != closure_row["corrected_path_sha256"]:
            raise RuntimeError(f"closure corrected-path hash mismatch for {candidate}")
        if replay_hash != old["corrected_path_sha256"]:
            raise RuntimeError(f"event-parent corrected-path hash mismatch for {candidate}")

        domain = float(old["domain_radius_about_corrected_path"])
        blocks = [
            ball_valid_envelope(
                corrected[step, :dimension],
                spec,
                config,
                epsilon=domain,
                exact_values=True,
                sphere=True,
            )
            for step in range(1, horizon + 1)
        ]
        neural_domain = all(block["fixed_point_consistent"] for block in blocks)
        certificate = safe_json(output_path(candidate))
        required = int(certificate["required_correct"])
        raw = [
            raw_slacks(
                logits(corrected[step, :dimension], cert_pairs, template, spec),
                cert_labels,
                required,
            )
            for step in range(horizon + 1)
        ]
        radius = float(closure_row["closure"]["remainder_radius"])
        margins = [0.0] + [
            logit_margin_radius(first=float(block["first"]), state_radius=radius)
            for block in blocks
        ]
        guarantee = [float(pair[0]) - margin for pair, margin in zip(raw, margins)]
        exclusion = [float(pair[1]) - margin for pair, margin in zip(raw, margins)]
        lower = first_persistent([value <= 0.0 for value in exclusion])
        upper = first_persistent([value > 0.0 for value in guarantee])
        bracket = None
        logic_slack = None
        if lower is not None and upper is not None and lower <= upper:
            prior = [max(exclusion[start : start + PERSISTENCE]) for start in range(lower)]
            lower_slack = math.inf if not prior else min(prior)
            upper_slack = min(guarantee[upper : upper + PERSISTENCE])
            logic_slack = min(lower_slack, upper_slack)
            if logic_slack > 0.0:
                bracket = [lower, upper]
        issued = bracket is not None
        retained = issued and bracket == old["sealed_four_sweep_bracket"]
        row = {
            "candidate": closure_row["candidate"],
            "development_row": closure_row["development_row"],
            "horizon": horizon,
            "pipeline_diagnostics": pipeline,
            "corrected_path_sha256": replay_hash,
            "directional_remainder_radius": radius,
            "neural_stage_value_domain_passed": neural_domain,
            "issued": issued,
            "bracket": bracket,
            "logic_slack": logic_slack,
            "sealed_four_sweep_bracket": old["sealed_four_sweep_bracket"],
            "retains_sealed_four_sweep_bracket": retained,
            "maximum_output_margin_radius": max(margins),
            "outcome_files_read": 0,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    new_holdouts = [
        row
        for row in rows
        if not row["development_row"]
        and not event_by_identity[identity(row["candidate"])]["closure_passed"]
    ]
    gate = (
        len(new_holdouts) == 3
        and all(row["issued"] for row in new_holdouts)
        and all(row["retains_sealed_four_sweep_bracket"] for row in new_holdouts)
        and all(row["neural_stage_value_domain_passed"] for row in rows)
    )
    result = {
        "status": "directional three-sweep event audit complete",
        "evidence_boundary": (
            "Outcome-blind deterministic audit of the four directional closures; "
            "no new Green query."
        ),
        "closure_parent_sha256": sha256(CLOSURE_PARENT),
        "event_parent_sha256": sha256(EVENT_PARENT),
        "protocol_sha256": sha256(PROTOCOL),
        "script_sha256": sha256(Path(__file__)),
        "evaluated_closures": len(rows),
        "issued": sum(row["issued"] for row in rows),
        "retained_sealed_bracket": sum(
            row["retains_sealed_four_sweep_bracket"] for row in rows
        ),
        "new_nondevelopment_rows": len(new_holdouts),
        "new_nondevelopment_issued": sum(row["issued"] for row in new_holdouts),
        "new_nondevelopment_retained": sum(
            row["retains_sealed_four_sweep_bracket"] for row in new_holdouts
        ),
        "prespecified_practical_promotion_gate_passed": gate,
        "rows": rows,
        "outcome_files_read": 0,
    }
    OUTPUT.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
