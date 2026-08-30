#!/usr/bin/env python3
"""Audit the WDBC direct one-step outward continuation baseline.

The post-seal Arb verifier is a standard validated-trajectory comparator: from
an exact dyadic anchor it propagates a one-step scalar state tube around the
same four-sweep reference path.  The recurrence uses outward optimizer defects,
Jacobian norms, and a Hessian-drift remainder; it does not use the Green radius,
the randomized norm probe, or its Gaussian event.  This script makes that
baseline and its cost explicit without changing any sealed artifact.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import statistics
from pathlib import Path

import numpy as np

import outward_real_dataset_confirmation as outward


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CONFIRMATION = RESULTS / "real_dataset_confirmation"
CACHE = RESULTS / "real_dataset_outward_cache"
OUTPUT_JSON = RESULTS / "wdbc_direct_validated_baseline_audit.json"
OUTPUT_MD = RESULTS / "wdbc_direct_validated_baseline_audit.md"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _function_names(function) -> set[str]:
    tree = ast.parse(inspect.getsource(function))
    return {node.id.lower() for node in ast.walk(tree) if isinstance(node, ast.Name)}


def main() -> None:
    joined = read_json(RESULTS / "real_dataset_outward_joined.json")
    manifest = read_json(CONFIRMATION / "certificate_manifest.json")
    issued_records = [record for record in manifest["records"] if record["issued"]]
    certificates = [
        read_json(CONFIRMATION / "certificates" / record["path"])
        for record in issued_records
    ]

    cache_paths = sorted(CACHE.glob("seed_*_anchor_*.json"))
    caches = [read_json(path) for path in cache_paths]
    if not caches:
        raise RuntimeError("no WDBC outward cache records found")

    forbidden = {"green", "kappa", "probe", "random", "rng"}
    used_names = _function_names(outward.verified_tube)
    forbidden_used = sorted(forbidden & used_names)
    if forbidden_used:
        raise AssertionError(f"validated recurrence uses forbidden names: {forbidden_used}")

    transition_count = 0
    for path, payload in zip(cache_paths, caches):
        if "independently of the Green radius" not in payload["scope_note"]:
            raise AssertionError(f"missing independence declaration in {path.name}")
        radius_path = path.with_suffix(".npz")
        if not radius_path.is_file():
            raise FileNotFoundError(radius_path)
        radius = np.load(radius_path)["radius"]
        if len(radius) != int(payload["reached_horizon"]) + 1:
            raise AssertionError(f"radius length mismatch in {path.name}")
        if float(radius[0]) != 0.0:
            raise AssertionError(f"nonzero anchor radius in {path.name}")
        if not np.all(np.isfinite(radius)) or np.any(radius < 0.0):
            raise AssertionError(f"invalid radius in {path.name}")
        transition_count += int(payload["reached_horizon"])

    rows = joined["rows"]
    issued_rows = [row for row in rows if row["outward_issued"]]
    covered_rows = [row for row in issued_rows if row["outward_covered"]]
    identical_rows = [
        row for row in issued_rows
        if row["outward_bracket"] == row["green_float_bracket"]
    ]
    if len(issued_rows) != len(issued_records):
        raise AssertionError("issued event count does not match certificate manifest")

    direct_seconds = [float(payload["elapsed_seconds"]) for payload in caches]
    green_seconds = [float(payload["elapsed_seconds"]) for payload in certificates]
    direct_total = sum(direct_seconds)
    green_total = sum(green_seconds)

    summary = {
        "status": "PASS: POST-SEAL MATCHED DIRECT VALIDATED-CONTINUATION BASELINE",
        "baseline_definition": (
            "One-step outward state-tube recurrence from the exact dyadic anchor, "
            "centered on the same four-sweep path and using verified optimizer defect, "
            "Jacobian, Hessian-drift, and output-margin enclosures."
        ),
        "prospective_status": "post-seal descriptive comparator; sealed experiments unchanged",
        "matched_scope": (
            "The 56 Green-issued event coordinates only; the 15 Green abstentions were not "
            "run through this expensive comparator."
        ),
        "exact_input_scope": (
            "Exact-real continuation from stored binary checkpoint/reference values; not a "
            "verification of the PyTorch/BLAS training execution that produced the checkpoint."
        ),
        "independent_of_green_radius": True,
        "independent_of_randomized_green_event": True,
        "verified_tube_forbidden_names_found": forbidden_used,
        "unique_validated_tubes": len(caches),
        "validated_transitions": transition_count,
        "matched_events": len(issued_rows),
        "direct_outward_issued": len(issued_rows),
        "direct_outward_covered": len(covered_rows),
        "brackets_identical_to_green": len(identical_rows),
        "distinct_seeds": len({int(row["seed"]) for row in issued_rows}),
        "maximum_bracket_width": max(
            row["outward_bracket"][1] - row["outward_bracket"][0]
            for row in issued_rows
        ),
        "maximum_state_radius": max(float(payload["maximum_radius"]) for payload in caches),
        "minimum_output_logic_slack": min(
            float(payload["minimum_logic_slack"]) for payload in caches
        ),
        "direct_outward_total_seconds_excluding_shared_centerline": direct_total,
        "direct_outward_median_seconds_per_tube": statistics.median(direct_seconds),
        "direct_outward_mean_seconds_per_transition": direct_total / transition_count,
        "direct_outward_median_seconds_per_transition": statistics.median(
            seconds / int(payload["reached_horizon"])
            for seconds, payload in zip(direct_seconds, caches)
        ),
        "greencert_total_seconds_for_matched_event_records": green_total,
        "greencert_median_seconds_per_matched_event_record": statistics.median(green_seconds),
        "direct_to_greencert_aggregate_runtime_ratio": direct_total / green_total,
        "cost_comparison_is_conservative_for_direct_baseline": True,
        "cost_note": (
            "The direct timer excludes construction of the shared four-sweep centerline, while "
            "the GreenCert event-record timers include it; this biases the ratio in favor of the "
            "direct validated baseline."
        ),
        "cache_manifest_sha256": hashlib.sha256(
            "".join(f"{path.name}:{sha256(path)}\n" for path in cache_paths).encode("utf-8")
        ).hexdigest().upper(),
    }

    OUTPUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    OUTPUT_MD.write_text(
        "\n".join(
            [
                "# WDBC direct validated-continuation baseline",
                "",
                "Status: **PASS**. This is a post-seal matched comparator; no sealed artifact changed.",
                "",
                "The existing 192-bit Arb pass is a direct one-step validated trajectory method,",
                "not merely a floating-point replay of the Green inequality. Starting from zero",
                "radius at the exact dyadic checkpoint, it propagates verified optimizer defects,",
                "Jacobian norms, Hessian-drift remainders, and output margins. The recurrence",
                "contains no Green radius, randomized probe, or PRNG dependence.",
                "",
                f"- Matched issued events: **{summary['matched_events']}**.",
                f"- Direct outward brackets: **{summary['direct_outward_issued']}**, with "
                f"**{summary['direct_outward_covered']}/{summary['direct_outward_issued']}** containment.",
                f"- Brackets identical to GreenCert: **{summary['brackets_identical_to_green']}**.",
                f"- Unique tubes / transitions: **{summary['unique_validated_tubes']} / "
                f"{summary['validated_transitions']:,}**.",
                f"- Maximum state radius: **{summary['maximum_state_radius']:.6g}**.",
                f"- Minimum strict output slack: **{summary['minimum_output_logic_slack']:.6g}**.",
                f"- Direct outward aggregate time: **{direct_total / 3600:.2f} h**; median "
                f"**{summary['direct_outward_median_seconds_per_tube']:.2f} s/tube**.",
                f"- GreenCert matched event-record time: **{green_total / 60:.2f} min**.",
                f"- Aggregate direct/GreenCert runtime ratio: **{summary['direct_to_greencert_aggregate_runtime_ratio']:.2f}x**.",
                "",
                "The timing ratio is conservative for the direct comparator because its timer",
                "excludes construction of the shared four-sweep centerline, whereas the GreenCert",
                "event-record timers include centerline construction. The comparator was run only",
                "on the 56 Green-issued coordinates, so it establishes matched rigor and cost, not",
                "availability on the 15 Green abstentions.",
                "",
                "Consequently, the ideal-Gaussian failure budget remains part of the original",
                "Green issuance route but is not a condition of the 56 outward-retained WDBC",
                "brackets. Those are deterministic exact-real continuation statements from stored",
                "binary checkpoints, not end-to-end proofs of the preceding training program.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
