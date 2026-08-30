#!/usr/bin/env python3
"""Aggregate independently audited role-fused Transformer timing replays."""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "transformer_v3_role_sparse_panel_audit.json"
CANDIDATES = (
    (366, 1, 1120),
    (366, 0, 1040),
    (366, 2, 1360),
    (369, 1, 4480),
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    cases = []
    for seed, gate, anchor in CANDIDATES:
        stem = f"transformer_v3_role_sparse_seed_{seed}_gate_{gate}_anchor_{anchor}"
        independent_path = RESULTS / f"{stem}_independent_audit.json"
        benchmark_path = RESULTS / f"{stem}_audit.json"
        independent = load(independent_path)
        benchmark = load(benchmark_path)
        if independent["status"] != "independent role-fused benchmark audit passed":
            raise AssertionError(f"failed independent audit: {independent_path.name}")
        if independent["benchmark_sha256"] != sha256(benchmark_path):
            raise AssertionError(f"stale independent audit: {independent_path.name}")
        if not independent["same_bracket"] or independent["fallback_output_times"]:
            raise AssertionError(f"bracket/fallback failure: {independent_path.name}")
        if independent["baseline_trace_maximum_relative_error"] > 2.0e-12:
            raise AssertionError(f"trace mismatch: {independent_path.name}")

        timings = benchmark["timings_seconds"]
        naive = benchmark["naive_separate_role_result"]
        center = float(timings["centerline_shared"])
        baseline_wall = float(timings["baseline_output_wall"])
        fused_wall = float(timings["fused_role_sparse_output_wall"])
        if float(timings["operator_speedup"]) <= max(
            1.0, float(naive["operator_speedup"])
        ):
            raise AssertionError(f"fusion does not dominate: {benchmark_path.name}")
        cases.append(
            {
                "candidate": independent["candidate"],
                "horizon": independent["horizon"],
                "bracket": benchmark["sealed_bracket"],
                "independent_audit_sha256": sha256(independent_path),
                "benchmark_sha256": sha256(benchmark_path),
                "baseline_pair_work": independent["baseline_pair_work"],
                "fused_pair_work": independent["fused_pair_work"],
                "dense_event_queries": benchmark["query_counts"]["baseline_all_pair_times"],
                "preplanned_event_queries": benchmark["query_counts"]["preplanned_event_times"],
                "fallback_event_queries": benchmark["query_counts"]["fallback_certification_times"],
                "centerline_seconds": center,
                "baseline_operator_seconds": independent["baseline_operator_seconds"],
                "naive_separate_operator_seconds": naive["operator_seconds"],
                "fused_operator_seconds": independent["fused_operator_seconds"],
                "baseline_output_wall_seconds": baseline_wall,
                "fused_output_wall_seconds": fused_wall,
                "operator_speedup": independent["fused_operator_speedup"],
                "output_wall_speedup": independent["fused_output_wall_speedup"],
                "centerline_plus_output_speedup": (center + baseline_wall)
                / (center + fused_wall),
            }
        )

    cases.sort(key=lambda row: int(row["horizon"]))
    baseline_operator = sum(float(row["baseline_operator_seconds"]) for row in cases)
    fused_operator = sum(float(row["fused_operator_seconds"]) for row in cases)
    baseline_wall = sum(float(row["baseline_output_wall_seconds"]) for row in cases)
    fused_wall = sum(float(row["fused_output_wall_seconds"]) for row in cases)
    centerline = sum(float(row["centerline_seconds"]) for row in cases)
    baseline_pairs = sum(int(row["baseline_pair_work"]) for row in cases)
    fused_pairs = sum(int(row["fused_pair_work"]) for row in cases)
    dense_queries = sum(int(row["dense_event_queries"]) for row in cases)
    sparse_queries = sum(int(row["preplanned_event_queries"]) for row in cases)

    operator_speedups = [float(row["operator_speedup"]) for row in cases]
    wall_speedups = [float(row["output_wall_speedup"]) for row in cases]
    replay_speedups = [float(row["centerline_plus_output_speedup"]) for row in cases]
    payload = {
        "status": "independent four-horizon role-fused timing panel passed",
        "scope": (
            "Post-seal implementation audit on four immutable issued singleton "
            "certificates chosen to span H=26,52,94,142; not a prospective issuance study. "
            "Centerline and sealed Green response are shared; output-phase speedups "
            "exclude both, while the secondary replay metric includes centerline time."
        ),
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "same_brackets": len(cases),
            "fallback_event_queries": sum(
                int(row["fallback_event_queries"]) for row in cases
            ),
            "dense_event_queries": dense_queries,
            "preplanned_event_queries": sparse_queries,
            "event_query_reduction_factor": dense_queries / sparse_queries,
            "baseline_pair_work": baseline_pairs,
            "fused_pair_work": fused_pairs,
            "pair_work_reduction_fraction": 1.0 - fused_pairs / baseline_pairs,
            "aggregate_operator_speedup": baseline_operator / fused_operator,
            "median_operator_speedup": statistics.median(operator_speedups),
            "operator_speedup_range": [min(operator_speedups), max(operator_speedups)],
            "aggregate_output_wall_speedup": baseline_wall / fused_wall,
            "median_output_wall_speedup": statistics.median(wall_speedups),
            "output_wall_speedup_range": [min(wall_speedups), max(wall_speedups)],
            "aggregate_centerline_plus_output_speedup": (centerline + baseline_wall)
            / (centerline + fused_wall),
            "median_centerline_plus_output_speedup": statistics.median(replay_speedups),
            "centerline_plus_output_speedup_range": [
                min(replay_speedups),
                max(replay_speedups),
            ],
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "sha256": sha256(OUTPUT), **payload}, indent=2))


if __name__ == "__main__":
    main()
