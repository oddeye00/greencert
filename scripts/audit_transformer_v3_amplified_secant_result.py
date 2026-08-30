#!/usr/bin/env python3
"""Independent arithmetic/hash audit of amplified-secant Transformer records."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FULL = RESULTS / "transformer_v3_amplified_secant_full_audit.json"
BENCHMARK = RESULTS / "transformer_v3_amplified_secant_paired_benchmark.json"
SOURCE = RESULTS / "transformer_v3_two_response_postseal_audit.json"
THEOREM = ROOT / "AMPLIFIED_SECANT_RESPONSE_THEOREM.md"
OUTPUT = RESULTS / "transformer_v3_amplified_secant_independent_audit.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def close(left: float, right: float, tolerance: float = 2.0e-12) -> None:
    if not math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=0.0):
        raise AssertionError((left, right))


def main() -> None:
    full = load(FULL)
    benchmark = load(BENCHMARK)
    source = load(SOURCE)
    if int(full.get("outcome_files_read", -1)) != 0:
        raise AssertionError("full audit read an outcome")
    if int(benchmark.get("outcome_files_read", -1)) != 0:
        raise AssertionError("benchmark read an outcome")
    if full["candidate"] != benchmark["candidate"]:
        raise AssertionError("candidate mismatch")
    if int(full["horizon"]) != int(benchmark["horizon"]):
        raise AssertionError("horizon mismatch")
    lam = float(benchmark["amplification"])
    if lam < 1.0 or not float(math.log2(lam)).is_integer():
        raise AssertionError("amplification is not a power of two")
    row = next(
        value for value in full["rows"] if float(value["amplification"]) == lam
    )
    if row["bracket"] != [28, 28] or not row["issued_without_arithmetic_budget"]:
        raise AssertionError("amplified secant did not reproduce the expected bracket")
    if not row["closure"]["closure_passed"]:
        raise AssertionError("amplified secant closure failed")
    kappa = float(row["closure"]["kappa"])
    expected_beta = float(row["secant_response_norm"]) + kappa * float(
        row["analytic_secant_discrepancy_upper"]
    )
    close(expected_beta, float(row["beta_without_arithmetic"]), 5.0e-15)
    cap = float(row["admissible_total_injection_error"])
    sigma = float(row["analytic_secant_discrepancy_upper"])
    close(cap - sigma, float(row["remaining_arithmetic_and_recurrence_headroom"]), 5.0e-15)
    close(cap / sigma, float(row["analytic_headroom_ratio"]), 5.0e-15)
    if float(row["remaining_arithmetic_and_recurrence_headroom"]) <= 0.0:
        raise AssertionError("no arithmetic headroom remains")
    if benchmark["full_audit_sha256"] != sha256(FULL):
        raise AssertionError("full-audit hash mismatch")
    if benchmark["two_response_source_sha256"] != sha256(SOURCE):
        raise AssertionError("two-response source hash mismatch")
    records = benchmark["records"]
    if len(records) != int(benchmark["repeats"]) or len(records) < 3:
        raise AssertionError("insufficient paired repeats")
    secant = [float(value["secant_seconds"]) for value in records]
    third = [float(value["third_product_seconds"]) for value in records]
    power = [float(value["additional_gram_power_seconds"]) for value in records]
    ratios_third = [
        float(value["third_product_seconds"]) / float(value["secant_seconds"])
        for value in records
    ]
    ratios_power = [
        float(value["additional_gram_power_seconds"])
        / float(value["secant_seconds"])
        for value in records
    ]
    close(statistics.median(secant), benchmark["median_secant_seconds"], 5.0e-15)
    close(statistics.median(third), benchmark["median_third_product_seconds"], 5.0e-15)
    close(statistics.median(power), benchmark["median_additional_gram_power_seconds"], 5.0e-15)
    close(
        statistics.median(ratios_third),
        benchmark["median_paired_third_over_secant_speedup"],
        5.0e-15,
    )
    close(
        statistics.median(ratios_power),
        benchmark["median_paired_power_over_secant_speedup"],
        5.0e-15,
    )
    close(min(ratios_power), benchmark["minimum_power_over_secant_speedup"], 5.0e-15)
    close(max(ratios_power), benchmark["maximum_power_over_secant_speedup"], 5.0e-15)
    checksums = benchmark["checksums"]
    close(checksums["secant"][0], row["secant_injection_norm"])
    close(checksums["secant"][1], row["secant_response_norm"])
    close(checksums["secant"][2], row["analytic_secant_discrepancy_upper"])
    source_row = next(
        value for value in source["rows"] if value.get("candidate") == full["candidate"]
    )
    close(checksums["third"][0], source_row["quadratic_surrogate_injection_norm"])
    close(checksums["third"][1], source_row["quadratic_surrogate_second_response_norm"])
    payload = {
        "status": "INDEPENDENT AMPLIFIED-SECANT AUDIT PASSED",
        "candidate": full["candidate"],
        "horizon": int(full["horizon"]),
        "amplification": lam,
        "bracket": row["bracket"],
        "analytic_headroom_ratio": float(row["analytic_headroom_ratio"]),
        "remaining_arithmetic_and_recurrence_headroom": float(
            row["remaining_arithmetic_and_recurrence_headroom"]
        ),
        "median_secant_seconds": statistics.median(secant),
        "median_third_product_seconds": statistics.median(third),
        "median_additional_gram_power_seconds": statistics.median(power),
        "median_paired_third_over_secant_speedup": statistics.median(ratios_third),
        "median_paired_power_over_secant_speedup": statistics.median(ratios_power),
        "minimum_power_over_secant_speedup": min(ratios_power),
        "maximum_power_over_secant_speedup": max(ratios_power),
        "full_audit_sha256": sha256(FULL),
        "benchmark_sha256": sha256(BENCHMARK),
        "two_response_source_sha256": sha256(SOURCE),
        "theorem_sha256": sha256(THEOREM),
        "outcome_files_read": 0,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**payload, "output_sha256": sha256(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
