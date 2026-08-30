#!/usr/bin/env python3
"""Independent arithmetic/integrity audit of the role-fused benchmark."""
from __future__ import annotations

import hashlib
import json
import math
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
GATE_INDEX = {0.7: 0, 0.8: 1, 0.9: 2}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def rows(mapping: dict) -> list[dict]:
    return list(mapping.values())


def close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=2.0e-13, abs_tol=1.0e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=369)
    parser.add_argument("--threshold", type=float, choices=tuple(GATE_INDEX), default=0.8)
    parser.add_argument("--anchor", type=int, default=4480)
    args = parser.parse_args()
    gate = GATE_INDEX[args.threshold]
    stem = f"transformer_v3_role_sparse_seed_{args.seed}_gate_{gate}_anchor_{args.anchor}"
    cache_path = RESULTS / f"{stem}_cache.json"
    benchmark_path = RESULTS / f"{stem}_audit.json"
    certificate_path = RESULTS / (
        f"transformer_v3_certificate_seed_{args.seed}_gate_{gate}_anchor_{args.anchor}.json"
    )
    output_path = RESULTS / f"{stem}_independent_audit.json"

    cache = load(cache_path)
    report = load(benchmark_path)
    certificate = load(certificate_path)

    expected_candidate = {
        "seed": args.seed,
        "threshold": args.threshold,
        "anchor": args.anchor,
    }
    if report["candidate"] != expected_candidate:
        raise AssertionError("benchmark candidate does not match requested candidate")

    if cache["certificate_sha256"] != sha256(certificate_path):
        raise AssertionError("cache does not bind the current sealed certificate")
    if report["certificate_sha256"] != sha256(certificate_path):
        raise AssertionError("benchmark does not bind the current sealed certificate")
    if report["centerline_sha256"] != certificate["centerline_sha256"]:
        raise AssertionError("centerline hash mismatch")
    if report["sealed_bracket"] != certificate["certified_bracket"]:
        raise AssertionError("sealed bracket mismatch")
    if report["role_sparse_bracket"] != certificate["certified_bracket"]:
        raise AssertionError("role-fused bracket mismatch")

    baseline = rows(cache["baseline"])
    separate = rows(cache["train"]) + rows(cache["cert"])
    fused = rows(cache["fused_train"]) + rows(cache["fused"])
    queried = {str(step) for step in cache["fused_policy"]["query_order"]}
    fused_extras = [
        row for step, row in cache["fused_cert_extra"].items() if step in queried
    ]
    fused += fused_extras

    horizon = int(report["horizon"])
    if len(baseline) != horizon:
        raise AssertionError("baseline output-time count changed")
    if len(cache["fused_train"]) + len(cache["fused"]) != horizon:
        raise AssertionError("fused path cardinality changed")
    if fused_extras:
        raise AssertionError("sealed benchmark unexpectedly used fallback queries")
    if len(cache["fused_policy"]["query_order"]) != int(
        report["query_counts"]["preplanned_event_times"]
    ):
        raise AssertionError("adaptive event-query count changed")
    if max(float(row["frozen_relative_error"]) for row in baseline) > 2.0e-12:
        raise AssertionError("baseline no longer reproduces the frozen traces")

    baseline_pairs = sum(int(row["pairs"]) for row in baseline)
    fused_pairs = sum(int(row["pairs"]) for row in fused)
    if report["pair_work"] != {
        "baseline": baseline_pairs,
        "fused_role_sparse": fused_pairs,
    }:
        raise AssertionError("reported pair work does not match cache arithmetic")

    baseline_operator = sum(float(row["operator_seconds"]) for row in baseline)
    fused_operator = sum(float(row["operator_seconds"]) for row in fused)
    baseline_wall = sum(float(row["wall_seconds"]) for row in baseline)
    fused_wall = sum(float(row["wall_seconds"]) for row in fused)
    timings = report["timings_seconds"]
    comparisons = {
        "baseline_operator": baseline_operator,
        "fused_role_sparse_operator": fused_operator,
        "baseline_output_wall": baseline_wall,
        "fused_role_sparse_output_wall": fused_wall,
        "operator_speedup": baseline_operator / fused_operator,
        "output_wall_speedup": baseline_wall / fused_wall,
    }
    for key, value in comparisons.items():
        if not close(value, timings[key]):
            raise AssertionError(f"timing mismatch for {key}: {value} != {timings[key]}")
    naive = report["naive_separate_role_result"]
    separate_operator = sum(float(row["operator_seconds"]) for row in separate)
    if not close(separate_operator, naive["operator_seconds"]):
        raise AssertionError("naive separate-role timing mismatch")
    if not (
        timings["operator_speedup"]
        > max(1.0, float(naive["operator_speedup"]))
    ):
        raise AssertionError("fusion does not strictly dominate both matched alternatives")

    payload = {
        "status": "independent role-fused benchmark audit passed",
        "candidate": expected_candidate,
        "horizon": horizon,
        "benchmark_sha256": sha256(benchmark_path),
        "cache_sha256": sha256(cache_path),
        "certificate_sha256": sha256(certificate_path),
        "same_bracket": True,
        "baseline_trace_maximum_relative_error": max(
            float(row["frozen_relative_error"]) for row in baseline
        ),
        "baseline_output_times": len(baseline),
        "fused_path_times": len(cache["fused_train"]) + len(cache["fused"]),
        "fallback_output_times": len(fused_extras),
        "baseline_pair_work": baseline_pairs,
        "fused_pair_work": fused_pairs,
        "baseline_operator_seconds": baseline_operator,
        "fused_operator_seconds": fused_operator,
        "baseline_output_wall_seconds": baseline_wall,
        "fused_output_wall_seconds": fused_wall,
        "naive_separate_operator_speedup": naive["operator_speedup"],
        "fused_operator_speedup": timings["operator_speedup"],
        "fused_output_wall_speedup": timings["output_wall_speedup"],
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"output": str(output_path), "sha256": sha256(output_path), **payload},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
