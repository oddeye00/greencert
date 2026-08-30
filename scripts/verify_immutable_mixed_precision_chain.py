#!/usr/bin/env python3
"""Read-only verifier for immutable mixed-precision result/audit records."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGGREGATE = ROOT / "results" / "transformer_v3_mixed_precision_timing_aggregate_audit.json"
TOLERANCES = (
    ROOT / "results" / "transformer_v3_inexact_operator_tolerance_pre_timing_reframe.json",
    ROOT / "results" / "transformer_v3_inexact_operator_tolerance_pre_core_hash.json",
    ROOT / "results" / "transformer_v3_inexact_operator_tolerance_postseal_audit.json",
)
AGGREGATOR = ROOT / "scripts" / "audit_transformer_v3_mixed_precision_timing_aggregate.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def close(left: float, right: float) -> None:
    if not math.isclose(float(left), float(right), rel_tol=2.0e-14, abs_tol=1.0e-300):
        raise AssertionError(f"mismatch: {left} != {right}")


def main() -> None:
    aggregate = load(AGGREGATE)
    assert aggregate["aggregator_sha256"] == sha256(AGGREGATOR)
    tolerance_hashes = {sha256(path): path for path in TOLERANCES}
    assert len(tolerance_hashes) == 3
    critical_values = [
        float(load(path)["tolerances"]["common"]["lower_passing_relative_gram_residual"])
        for path in TOLERANCES
    ]
    for value in critical_values:
        close(value, 0.5999285448884594)
    pooled = []
    for row in aggregate["inputs"]:
        result_path = ROOT / row["result"]
        audit_path = ROOT / row["independent_audit"]
        assert sha256(result_path) == row["result_sha256"]
        assert sha256(audit_path) == row["independent_audit_sha256"]
        result = load(result_path)
        audit = load(audit_path)
        assert audit["result_sha256"] == row["result_sha256"]
        assert audit["tolerance_source_sha256"] in tolerance_hashes
        assert bool(result["same_bracket"]) and bool(audit["same_bracket"])
        speedups = [float(value) for value in row["paired_speedups"]]
        assert all(value > 1.0 for value in speedups)
        close(row["median_paired_speedup"], statistics.median(speedups))
        pooled.extend(speedups)
    assert len(aggregate["inputs"]) == aggregate["invocations"] == 4
    assert len(pooled) == aggregate["paired_trials"] == 20
    close(aggregate["pooled_median_paired_speedup"], statistics.median(pooled))
    close(aggregate["pooled_minimum_paired_speedup"], min(pooled))
    close(aggregate["pooled_maximum_paired_speedup"], max(pooled))
    assert aggregate["all_paired_speedups_above_one"]
    assert aggregate["same_bracket"]
    print(
        "PASS: immutable mixed-precision chain verifies 4 invocations, "
        "8 bound hashes, 3 tolerance generations, and 20 positive timing pairs"
    )


if __name__ == "__main__":
    main()
