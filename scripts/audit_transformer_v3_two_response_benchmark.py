#!/usr/bin/env python3
"""Independent arithmetic/hash verifier for the paired two-response benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOURCE = RESULTS / "transformer_v3_two_response_postseal_audit.json"
BENCHMARK = RESULTS / "transformer_v3_two_response_paired_benchmark.json"
OUTPUT = RESULTS / "transformer_v3_two_response_paired_benchmark_independent_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load(path: Path) -> dict:
    lowered = path.name.lower()
    if lowered.endswith(".outcomes.json") or lowered.endswith(".sealed.log"):
        raise RuntimeError(f"forbidden outcome read: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=2.0e-15, abs_tol=0.0)


def main() -> None:
    benchmark = load(BENCHMARK)
    source = load(SOURCE)
    if benchmark["status"] != "PAIRED TWO-RESPONSE INCREMENTAL BENCHMARK PASSED":
        raise AssertionError("benchmark status changed")
    if benchmark["candidate"] != {"seed": 366, "threshold": 0.7, "anchor": 1040}:
        raise AssertionError("benchmark candidate changed")
    if int(benchmark["horizon"]) != 52 or int(benchmark["repeats"]) != 4:
        raise AssertionError("benchmark design changed")
    if int(benchmark.get("outcome_files_read", -1)) != 0:
        raise AssertionError("benchmark reports an outcome read")
    if benchmark["two_response_source_sha256"] != sha256(SOURCE):
        raise AssertionError("benchmark source hash mismatch")
    rows = benchmark["records"]
    if len(rows) != 4:
        raise AssertionError("paired record count changed")
    directional = []
    powers = []
    ratios = []
    for index, row in enumerate(rows):
        expected_order = (
            ["directional", "power"] if index % 2 == 0 else ["power", "directional"]
        )
        if row["order"] != expected_order:
            raise AssertionError("alternating order changed")
        d = float(row["directional_seconds"])
        p = float(row["additional_gram_power_seconds"])
        ratio = float(row["paired_speedup"])
        if d <= 0.0 or p <= 0.0 or not close(ratio, p / d):
            raise AssertionError("invalid paired timing arithmetic")
        directional.append(d)
        powers.append(p)
        ratios.append(ratio)
    checks = (
        (benchmark["median_directional_seconds"], statistics.median(directional)),
        (benchmark["median_additional_gram_power_seconds"], statistics.median(powers)),
        (benchmark["median_paired_speedup"], statistics.median(ratios)),
        (benchmark["minimum_paired_speedup"], min(ratios)),
        (benchmark["maximum_paired_speedup"], max(ratios)),
    )
    if not all(close(left, right) for left, right in checks):
        raise AssertionError("aggregate timing arithmetic changed")
    if min(ratios) <= 2.7:
        raise AssertionError("paired speed advantage weakened")
    source_row = next(
        row
        for row in source["rows"]
        if row["candidate"] == benchmark["candidate"]
    )
    if not close(
        benchmark["quadratic_forcing_norm"],
        source_row["quadratic_surrogate_injection_norm"],
    ) or not close(
        benchmark["second_response_norm"],
        source_row["quadratic_surrogate_second_response_norm"],
    ):
        raise AssertionError("benchmark branch differs from cohort audit")
    payload = {
        "status": "INDEPENDENT PAIRED TWO-RESPONSE BENCHMARK AUDIT PASSED",
        "candidate": benchmark["candidate"],
        "horizon": benchmark["horizon"],
        "repeats": len(rows),
        "median_directional_seconds": statistics.median(directional),
        "median_additional_gram_power_seconds": statistics.median(powers),
        "median_paired_speedup": statistics.median(ratios),
        "minimum_paired_speedup": min(ratios),
        "maximum_paired_speedup": max(ratios),
        "outcome_files_read": 0,
        "benchmark_sha256": sha256(BENCHMARK),
        "source_sha256": sha256(SOURCE),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
