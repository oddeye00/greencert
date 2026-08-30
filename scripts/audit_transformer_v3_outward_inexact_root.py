#!/usr/bin/env python3
"""Outward replay of the residual-corrected q=1 Transformer sensitivity.

This closes calibration and scalar-root arithmetic in 256-bit Arb conditional
on stored terminal norms and the hypothetical verified residual budget from
the inexact-operator tolerance audit.  It does not assert that the current
float64 neural kernels attain that budget.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from adaptive_witness_policy import WitnessQuery, acquire_witnesses
from one_shot_recenter_closure import conservative_one_shot_closure
from outward_inexact_anytime_gram import (
    folded_normal_calibration_lower,
    outward_inexact_gram_operator_upper_bound,
)
from transformer_block_envelope import objective_hessian_lipschitz
from transformer_certificate_protocol import Candidate
from transformer_v3_certificate import load_candidate, output_path, safe_json
from transformer_v3_protocol import PROBES


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOURCE = RESULTS / (
    "transformer_v3_combined_online_role_seed_366_gate_1_anchor_1120_"
    "matched-combined-v5.json"
)
TOLERANCE = RESULTS / "transformer_v3_inexact_operator_tolerance_postseal_audit.json"
OUTPUT = RESULTS / "transformer_v3_outward_inexact_root_postseal_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    source = safe_json(SOURCE)
    tolerance = safe_json(TOLERANCE)
    candidate = Candidate(**source["candidate"])
    certificate = safe_json(output_path(candidate))
    config, _, _, _, _, _ = load_candidate(candidate)
    rows = {int(row["step"]): row for row in source["output_rows"]}
    green = source["green_rows"][0]
    horizon = int(source["horizon"])
    event = int(source["frozen_predicted_persistent_event"])
    persistence = int(certificate["protocol"]["persistence"])
    geometry = source["geometry"]
    critical_relative = float(
        tolerance["tolerances"]["common"]["lower_passing_relative_gram_residual"]
    )
    # The binary64 threshold is intentionally the last passing point and has
    # essentially zero discriminant slack.  A proof-producing operational
    # contract must live strictly inside it.
    safety_factor = 0.99
    relative = safety_factor * critical_relative

    output_delta = float(source["family_budget"]["role_output_per_operator_delta"])
    green_delta = float(source["family_budget"]["green_per_operator_delta"])
    output_calibration = folded_normal_calibration_lower(
        delta=output_delta, probes=PROBES
    )
    green_calibration = folded_normal_calibration_lower(
        delta=green_delta, probes=PROBES
    )

    corrected_outputs = {
        step: outward_inexact_gram_operator_upper_bound(
            terminal_norm=float(row["Y"]),
            calibration_lower=output_calibration,
            residual_norms=(relative * float(row["Y"]),),
        )
        for step, row in rows.items()
    }
    corrected_green = outward_inexact_gram_operator_upper_bound(
        terminal_norm=float(green["Y"]),
        calibration_lower=green_calibration,
        residual_norms=(relative * float(green["Y"]),),
    )
    maximum_map_drift = 0.0
    for step in range(1, horizon):
        row = rows[step]
        first_ball = corrected_outputs[step] + float(row["block_second"]) * float(
            geometry["outer_domain_radius"]
        )
        drift = objective_hessian_lipschitz(
            first_ball,
            float(row["block_second"]),
            float(row["block_third"]),
        )
        maximum_map_drift = max(
            maximum_map_drift,
            math.sqrt(2.0) * config.learning_rate * drift,
        )
    closure = conservative_one_shot_closure(
        kappa=corrected_green,
        derivative_drift=maximum_map_drift,
        response_sequence_norm=float(geometry["signed_response_sequence_norm"]),
        response_max_state_norm=float(geometry["signed_response_max_state_norm"]),
        domain_radius=float(geometry["outer_domain_radius"]),
    )
    if not closure.closure_passed:
        raise AssertionError("outward scalar replay lost optimizer closure")
    radius = float(closure.total_pointwise_radius)
    raw_exclusions = {
        step: float(row["raw_exclusion_slack"]) for step, row in rows.items()
    }
    logic_slacks: list[float] = []

    def query(step: int) -> WitnessQuery:
        row = rows[step]
        margin = math.sqrt(2.0) * (
            corrected_outputs[step] * radius
            + 0.5 * float(row["block_second"]) * radius * radius
        )
        guarantee = float(row["raw_guarantee_slack"]) - margin
        exclusion = float(row["raw_exclusion_slack"]) - margin
        logic_slacks.append(
            guarantee if event <= step < event + persistence else exclusion
        )
        return WitnessQuery(step, guarantee > 0.0, exclusion > 0.0)

    policy = acquire_witnesses(
        event=event,
        persistence=persistence,
        horizon=horizon,
        raw_exclusion_slacks=raw_exclusions,
        query=query,
        exact_failures={0},
    )
    if not policy.issued:
        raise AssertionError(f"outward scalar replay abstained: {policy.reason}")

    float_replay = tolerance["tolerances"]["common"]["lower_replay"]
    ratios = [
        corrected_outputs[step]
        / (
            math.sqrt((float(row["Y"]) * (1.0 + relative)) / float(row["c_delta"]))
        )
        for step, row in rows.items()
    ]
    ratios.append(corrected_green / float(float_replay["green_upper"]))
    payload = {
        "status": "outward inexact Gram scalar replay passed",
        "scope": (
            "256-bit outward calibration and q=1 residual-root arithmetic, "
            "conditional on stored binary64 terminal norms and a hypothetical "
            "verified xi<=rY residual contract; not an estimate of kernel error."
        ),
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "tolerance_source": str(TOLERANCE.relative_to(ROOT)),
        "tolerance_source_sha256": sha256(TOLERANCE),
        "candidate": candidate.__dict__,
        "relative_residual_budget": relative,
        "binary64_critical_relative_residual": critical_relative,
        "strict_interior_safety_factor": safety_factor,
        "calibration_lower": {
            "output": output_calibration,
            "green": green_calibration,
        },
        "outward_green_upper": corrected_green,
        "maximum_outward_output_upper": max(corrected_outputs.values()),
        "maximum_outward_to_float_root_ratio": max(ratios),
        "closure": closure.as_dict(),
        "bracket": [event, event],
        "same_bracket": [event, event] == source["combined_bracket"],
        "minimum_queried_logic_slack": min(logic_slacks),
        "query_order": list(policy.query_order),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**payload, "output_sha256": sha256(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
