#!/usr/bin/env python3
"""Aggregate four independently audited mixed-precision timing invocations."""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
INPUTS = (
    RESULTS
    / "transformer_v3_mixed_precision_residual_replication1_postseal_audit.json",
    RESULTS
    / "transformer_v3_mixed_precision_residual_replication2_postseal_audit.json",
    RESULTS
    / "transformer_v3_mixed_precision_residual_replication3_postseal_audit.json",
    RESULTS / "transformer_v3_mixed_precision_residual_postseal_audit.json",
)
AUDITS = (
    RESULTS
    / "transformer_v3_mixed_precision_residual_replication1_independent_audit.json",
    RESULTS
    / "transformer_v3_mixed_precision_residual_replication2_independent_audit.json",
    RESULTS
    / "transformer_v3_mixed_precision_residual_replication3_independent_audit.json",
    RESULTS / "transformer_v3_mixed_precision_residual_independent_audit.json",
)
TOLERANCE_RECORDS = (
    RESULTS / "transformer_v3_inexact_operator_tolerance_pre_timing_reframe.json",
    RESULTS / "transformer_v3_inexact_operator_tolerance_pre_core_hash.json",
    RESULTS / "transformer_v3_inexact_operator_tolerance_postseal_audit.json",
)
OUTPUT = RESULTS / "transformer_v3_mixed_precision_timing_aggregate_audit.json"


TIMING_KEYS = {
    "warmup_exact_operator_seconds",
    "warmup_float32_operator_seconds",
    "exact_repeat_seconds",
    "float32_repeat_seconds",
    "exact_operator_seconds",
    "float32_operator_seconds",
    "measured_kernel_speedup",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def without_timing(row: dict) -> dict:
    return {key: value for key, value in row.items() if key not in TIMING_KEYS}


def combined_speedups(record: dict) -> list[float]:
    timings = record["timings_seconds"]
    exact_output = timings["output_gram_binary64_trials"]
    approximate_output = timings["output_gram_float32_trials"]
    exact_green = timings["green_gram_binary64_trials"]
    approximate_green = timings["green_gram_float32_trials"]
    vectors = (exact_output, approximate_output, exact_green, approximate_green)
    if any(len(values) != 5 for values in vectors):
        raise AssertionError("each invocation must contain five paired trials")
    return [
        (exact_output[index] + exact_green[index])
        / (approximate_output[index] + approximate_green[index])
        for index in range(5)
    ]


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    records = [load(path) for path in INPUTS]
    audits = [load(path) for path in AUDITS]
    for path, record, audit_path, audit in zip(INPUTS, records, AUDITS, audits):
        if audit["result_sha256"] != sha256(path):
            raise AssertionError(f"independent audit does not bind {path.name}")
        if not audit["same_bracket"] or not audit["all_outward_scalar_supersolutions_rechecked"]:
            raise AssertionError(f"independent arithmetic audit failed for {path.name}")
    valid_tolerance_hashes = {sha256(path) for path in TOLERANCE_RECORDS}
    for record in records:
        if record["tolerance_source_sha256"] not in valid_tolerance_hashes:
            raise AssertionError("mixed result does not bind a retained tolerance record")

    reference = records[0]
    invariant_top = (
        "source_sha256",
        "candidate",
        "centerline_sha256",
        "precision_path",
        "measurement_inflation",
        "admissible_common_relative_residual_threshold",
        "maximum_measured_relative_residual",
        "admissible_to_measured_ratio",
        "all_measured_residuals_below_admissible_threshold",
        "closure",
        "bracket",
        "same_bracket",
        "minimum_queried_logic_slack",
        "claim_boundary",
    )
    for record in records[1:]:
        for key in invariant_top:
            if reference[key] != record[key]:
                raise AssertionError(
                    f"non-timing result changed between invocations: {key}"
                )
        if without_timing(reference["green"]) != without_timing(record["green"]):
            raise AssertionError("non-timing Green result changed between invocations")
        if len(reference["output_rows"]) != len(record["output_rows"]):
            raise AssertionError("output-row count changed")
        for left, right in zip(reference["output_rows"], record["output_rows"]):
            if without_timing(left) != without_timing(right):
                raise AssertionError(
                    f"non-timing output row changed at step {left['step']}"
                )

    invocation_trials = [combined_speedups(record) for record in records]
    for record, trials in zip(records, invocation_trials):
        stored = record["measured_operator_speedups"]
        if trials != stored["combined_q1_kernel_trials"]:
            raise AssertionError("stored paired speedups differ from direct recomputation")
        if statistics.median(trials) != stored["combined_q1_kernel"]:
            raise AssertionError("stored invocation median differs from recomputation")
    pooled = [value for trials in invocation_trials for value in trials]
    payload = {
        "status": "four-invocation mixed-precision timing aggregate passed",
        "scope": (
            "Twenty matched wall-time pairs across four separately launched, warmed, "
            "alternating-order invocations on one immutable q=1 candidate."
        ),
        "aggregator_sha256": sha256(Path(__file__)),
        "inputs": [
            {
                "result": str(path.relative_to(ROOT)),
                "result_sha256": sha256(path),
                "independent_audit": str(audit_path.relative_to(ROOT)),
                "independent_audit_sha256": sha256(audit_path),
                "paired_speedups": trials,
                "median_paired_speedup": statistics.median(trials),
            }
            for path, audit_path, trials in zip(INPUTS, AUDITS, invocation_trials)
        ],
        "invocations": 4,
        "paired_trials": len(pooled),
        "pooled_paired_speedups": pooled,
        "pooled_median_paired_speedup": statistics.median(pooled),
        "pooled_minimum_paired_speedup": min(pooled),
        "pooled_maximum_paired_speedup": max(pooled),
        "all_paired_speedups_above_one": all(value > 1.0 for value in pooled),
        "same_non_timing_certificate_record": True,
        "same_bracket": reference["bracket"],
        "maximum_measured_relative_residual": reference[
            "maximum_measured_relative_residual"
        ],
        "admissible_to_measured_ratio": reference["admissible_to_measured_ratio"],
        "claim_boundary": (
            "Wall-time variability is reported rather than filtered. This is a "
            "four-invocation systems benchmark, not a confidence interval or an "
            "outward exact-real neural-kernel proof."
        ),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**payload, "output": str(OUTPUT), "sha256": sha256(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
