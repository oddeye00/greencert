#!/usr/bin/env python3
"""Verify the immutable v1 and maintained v2 directional-theorem chain."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "DIRECTIONAL_BLOCK_REMAINDER_THEOREM.md"
V2 = ROOT / "DIRECTIONAL_BLOCK_REMAINDER_THEOREM_V2.md"
NOTE = ROOT / "DIRECTIONAL_BLOCK_REMAINDER_SOURCE_SUPERSESSION.md"
TEST = ROOT / "scripts" / "test_directional_block_symmetrization.py"
RESULT = ROOT / "results" / "transformer_directional_block_remainder_diagnostic.json"

EXPECTED = {
    V1: "F9C680DD7E6C47EFA8AE91753612464DF8622456A0F22179BF5174A6134B6AAF",
    V2: "6F59C5F579BC3CE882E456DDC6FBE083633774F2D3B153ED54974F9C92F48CBE",
    TEST: "11476070F0BF25F9FD582BF8B953C7F337B4B534DD3CAA6118B3C7626089FDD4",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    for path, expected in EXPECTED.items():
        observed = digest(path)
        if observed != expected:
            raise AssertionError(f"source hash changed: {path.name}: {observed}")

    result = json.loads(RESULT.read_text(encoding="utf-8"))
    if result["source_hashes"]["theorem"] != EXPECTED[V1]:
        raise AssertionError("frozen result no longer points to the immutable v1 theorem")

    note = NOTE.read_text(encoding="utf-8")
    for expected in EXPECTED.values():
        if expected not in note:
            raise AssertionError("supersession note omits a source hash")
    v2 = V2.read_text(encoding="utf-8")
    required = (
        "Lemma 1: symmetrization without loss",
        "all 24 summands are identical",
        "three-known, one-free contraction",
        "changes no bound, implementation, protocol, or result",
    )
    missing = [phrase for phrase in required if phrase not in v2]
    if missing:
        raise AssertionError(f"maintained theorem omits required statements: {missing}")

    print(
        json.dumps(
            {
                "status": "directional block theorem supersession verified",
                "immutable_v1_sha256": EXPECTED[V1],
                "maintained_v2_sha256": EXPECTED[V2],
                "symmetrization_test_sha256": EXPECTED[TEST],
                "frozen_result_points_to_v1": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
