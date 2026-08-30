#!/usr/bin/env python3
"""Independent mechanical verifier for saved signed-Green audit records."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from transformer_certificate_protocol import PERSISTENCE, Candidate
from transformer_green_protocol import green_identity, output_identity


def first_persistent(values: np.ndarray, required: int) -> int | None:
    for start in range(max(len(values) - PERSISTENCE + 1, 0)):
        if np.all(values[start : start + PERSISTENCE] >= required):
            return int(start)
    return None


def verify(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate = Candidate(**payload["candidate"])
    horizon = int(payload["protocol"]["horizon"])
    required = int(payload["required_correct"])
    kappa = float(payload["green_probe"]["green_operator_norm_upper_bound"])
    drift = float(payload["maximum_optimizer_derivative_drift_upper"])
    response = float(payload["signed_response_sequence_norm"])
    radius = float(payload["signed_radius"])
    closure = float(payload["closure_lhs_2_kappa_M_Z"])

    assert math.isclose(radius, 2.0 * response, rel_tol=2e-15, abs_tol=0.0)
    assert math.isclose(
        closure, 2.0 * kappa * drift * response, rel_tol=2e-15, abs_tol=0.0
    )
    assert bool(payload["closure_passed"]) == (
        closure <= 1.0 and bool(payload["block_fixed_points_all_consistent"])
    )
    assert tuple(payload["green_probe"]["identity"]) == green_identity(
        candidate, horizon
    )

    geometry = payload["geometry"]
    assert len(geometry) == horizon
    for step, row in enumerate(geometry, start=1):
        assert int(row["step"]) == step
        assert tuple(row["output_probe"]["identity"]) == output_identity(
            candidate, step
        )
        assert bool(row["block_fixed_point_consistent"])

    guaranteed = np.asarray(payload["guaranteed_correct"], dtype=np.int64)
    possible = np.asarray(payload["possibly_correct"], dtype=np.int64)
    assert len(guaranteed) == len(possible) == horizon + 1
    lower = first_persistent(possible, required)
    upper = first_persistent(guaranteed, required)
    recomputed = None if lower is None or upper is None or lower > upper else [lower, upper]
    expected = recomputed if payload["closure_passed"] else None
    assert payload["certified_bracket"] == expected
    assert bool(payload["certificate_issued"]) == (expected is not None)

    budget = payload["probability_budget"]
    assert int(budget["queried_operators"]) == horizon + 1
    assert int(budget["queried_operator_count"]) == horizon + 1
    probe_delta = float(payload["protocol"]["probe_config"]["delta"])
    assert math.isclose(
        float(budget["queried_union_bound"]),
        (horizon + 1) * probe_delta,
        rel_tol=2e-15,
        abs_tol=0.0,
    )
    assert float(budget["queried_union_bound"]) <= float(
        budget["maximum_family_union_bound"]
    )
    assert bool(budget["all_queries_predeclared"])

    if expected is not None:
        slack = payload["certificate_output_logic_slack"]
        assert float(slack["minimum_logic_slack"]) > 0.0
    if payload.get("outcome_joined"):
        actual = payload.get("actual_persistent_event")
        contains = (
            None
            if actual is None or expected is None
            else expected[0] <= actual <= expected[1]
        )
        assert payload.get("bracket_contains_actual") == contains
        if payload["closure_passed"]:
            assert not bool(payload["observed_sequence_tube_violation"])

    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
        "candidate": payload["candidate"],
        "horizon": horizon,
        "closure": closure,
        "bracket": expected,
        "actual": payload.get("actual_persistent_event"),
        "verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([verify(path) for path in args.paths], indent=2))


if __name__ == "__main__":
    main()
