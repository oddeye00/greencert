#!/usr/bin/env python3
"""Independent replay of fresh confirmation certificate logic and seals."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from run_transformer_green_confirmation import (
    AGGREGATE,
    CERTIFICATE_SEAL,
    METHOD_SEAL,
    audit_path,
    sha256,
)
from transformer_certificate_protocol import Candidate
from transformer_green_confirmation_certificate import (
    CANDIDATE_SEAL,
    frozen_candidates,
    output_path,
    safe_json,
    verify_method_seal,
)
from transformer_green_confirmation_protocol import (
    FAMILY_FAILURE_PROBABILITY,
    PERSISTENCE,
    green_identity,
    output_identity,
    probe_config,
)


def first_persistent(values: np.ndarray, required: int) -> int | None:
    for start in range(max(len(values) - PERSISTENCE + 1, 0)):
        if np.all(values[start : start + PERSISTENCE] >= required):
            return int(start)
    return None


def verify_certificate(candidate: Candidate, horizon: int) -> dict:
    path = output_path(candidate)
    payload = safe_json(path)
    assert payload["candidate"] == candidate.__dict__
    assert int(payload["protocol"]["horizon"]) == horizon
    response = float(payload["signed_response_sequence_norm"])
    radius = float(payload["signed_radius"])
    drift = float(payload["maximum_optimizer_derivative_drift_upper"])
    minimum = float(payload["minimum_closure_lhs_using_kappa_ge_1"])
    assert math.isclose(radius, 2.0 * response, rel_tol=2e-15)
    assert math.isclose(minimum, 2.0 * drift * response, rel_tol=2e-15)

    geometry = payload["geometry"]
    assert len(geometry) == horizon
    for step, row in enumerate(geometry, start=1):
        assert int(row["step"]) == step
        assert tuple(row["output_probe"]["identity"]) == output_identity(candidate, step)
        assert bool(row["block_fixed_point_consistent"])

    green = payload["green_probe"]
    if green is None:
        assert payload["early_abstention_before_green_probe"]
        assert minimum > 1.0 or not payload["block_fixed_points_all_consistent"]
        assert payload["closure_lhs_2_kappa_M_Z"] is None
        closure = False
        expected_queries = horizon
    else:
        assert tuple(green["identity"]) == green_identity(candidate, horizon)
        kappa = float(green["green_operator_norm_upper_bound"])
        lhs = 2.0 * kappa * drift * response
        assert math.isclose(
            float(payload["closure_lhs_2_kappa_M_Z"]), lhs, rel_tol=2e-15
        )
        closure = lhs <= 1.0 and bool(payload["block_fixed_points_all_consistent"])
        expected_queries = horizon + 1
    assert bool(payload["closure_passed"]) == closure

    guaranteed = np.asarray(payload["guaranteed_correct"], dtype=np.int64)
    possible = np.asarray(payload["possibly_correct"], dtype=np.int64)
    required = int(payload["required_correct"])
    lower = first_persistent(possible, required)
    upper = first_persistent(guaranteed, required)
    raw = None if lower is None or upper is None or lower > upper else [lower, upper]
    expected_bracket = raw if closure else None
    assert payload["raw_margin_bracket"] == raw
    assert payload["certified_bracket"] == expected_bracket
    assert bool(payload["certificate_issued"]) == (expected_bracket is not None)
    if expected_bracket is not None:
        assert float(payload["certificate_output_logic_slack"]["minimum_logic_slack"]) > 0.0

    budget = payload["probability_budget"]
    assert int(budget["queried_operators"]) == expected_queries
    assert math.isclose(
        float(budget["queried_union_bound"]),
        expected_queries * probe_config().delta,
        rel_tol=2e-15,
    )
    assert float(budget["queried_union_bound"]) <= FAMILY_FAILURE_PROBABILITY
    assert bool(budget["all_queries_predeclared"])
    return {
        "candidate": candidate.__dict__,
        "sha256": sha256(path),
        "issued": payload["certificate_issued"],
        "bracket": expected_bracket,
    }


def verify_all(require_join: bool) -> dict:
    verify_method_seal()
    candidates, horizons, _ = frozen_candidates()
    certificate_seal = safe_json(CERTIFICATE_SEAL)
    sealed = {
        (
            int(row["candidate"]["seed"]),
            float(row["candidate"]["threshold"]),
            int(row["candidate"]["anchor"]),
        ): row
        for row in certificate_seal["certificate_files"]
    }
    rows = []
    for candidate in candidates:
        row = verify_certificate(candidate, horizons[candidate])
        key = (candidate.seed, candidate.threshold, candidate.anchor)
        assert sealed[key]["sha256"] == row["sha256"]
        if require_join:
            audit = safe_json(audit_path(candidate))
            assert audit["certificate_sha256"] == row["sha256"]
            if row["issued"]:
                assert audit["bracket_contains_actual"] is True
                assert audit["observed_sequence_tube_violation"] is False
                assert int(audit["observed_state_tube_violations"]) == 0
            row["actual"] = audit["actual_persistent_event"]
            row["covered"] = audit["bracket_contains_actual"]
        rows.append(row)
    if require_join:
        aggregate = safe_json(AGGREGATE)
        assert aggregate["candidate_seal_sha256"] == sha256(CANDIDATE_SEAL)
        assert aggregate["certificate_seal_sha256"] == sha256(CERTIFICATE_SEAL)
    return {
        "method_seal_sha256": sha256(METHOD_SEAL),
        "candidate_seal_sha256": sha256(CANDIDATE_SEAL),
        "certificate_seal_sha256": sha256(CERTIFICATE_SEAL),
        "candidates": len(candidates),
        "issued": sum(bool(row["issued"]) for row in rows),
        "rows": rows,
        "joined": require_join,
        "verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-join", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify_all(args.require_join), indent=2))


if __name__ == "__main__":
    main()
