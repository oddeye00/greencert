#!/usr/bin/env python3
"""Check that the sealed v3 loader consumes every shipped sparse anchor exactly."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from transformer_certificate_protocol import Candidate
from transformer_directional_anchor_bundle import load_anchor, verify
from transformer_v3_certificate import load_candidate, verify_method_seal


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "results" / "transformer_fully_recentered_three_sweep_audit.json"


def main() -> None:
    verify_method_seal()
    bundle = verify(ROOT)
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    identities = sorted(
        {
            (
                int(row["candidate"]["seed"]),
                float(row["candidate"]["threshold"]),
                int(row["candidate"]["anchor"]),
            )
            for row in parent["rows"]
        }
    )
    for seed, threshold, anchor in identities:
        candidate = Candidate(seed, threshold, anchor)
        *_, parameter, velocity = load_candidate(candidate)
        expected_parameter, expected_velocity = load_anchor(seed, anchor, ROOT)
        if not np.array_equal(parameter.numpy(), expected_parameter):
            raise AssertionError(f"sparse parameter mismatch: {candidate}")
        if not np.array_equal(velocity.numpy(), expected_velocity):
            raise AssertionError(f"sparse velocity mismatch: {candidate}")
    print(
        json.dumps(
            {
                "status": "sealed loader sparse-checkpoint test passed",
                "cases": len(identities),
                "anchors": int(bundle["anchors"]),
                "archives": int(bundle["archives"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
