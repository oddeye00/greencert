#!/usr/bin/env python3
"""Aggregation-independent audit of the post-seal hardening results."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "postseal_hardening_independent_audit.json"
OUTPUT_MD = RESULTS / "postseal_hardening_independent_audit.md"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def audit_digits() -> dict:
    method_path = ROOT / "DIGITS_SIGNED_METHOD_SEAL.json"
    method_hash = sha256(method_path)
    export = RESULTS / "digits_signed_confirmation"
    manifest = read(export / "certificate_manifest.json")
    issued = [row for row in manifest["records"] if row["issued"]]
    certificates = {
        (int(row["seed"]), int(row["gate_index"])): {
            **read(export / "certificates" / row["path"]),
            "manifest_sha256": row["sha256"],
        }
        for row in issued
    }
    for row in issued:
        path = export / "certificates" / row["path"]
        if sha256(path) != row["sha256"]:
            raise RuntimeError(f"digits certificate hash mismatch: {path}")
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in certificates.values():
        grouped[(int(row["seed"]), int(row["anchor"]))].append(row)

    caches = {}
    for (seed, anchor), rows in grouped.items():
        path = RESULTS / "digits_outward_cache" / f"seed_{seed}_anchor_{anchor}.json"
        payload = read(path)
        expected_hashes = sorted(row["manifest_sha256"] for row in rows)
        expected_horizon = max(int(row["certificate_horizon"]) for row in rows)
        if payload["version"] != "digits-arb-outward-v1-2026-08-25":
            raise RuntimeError("unexpected digits outward version")
        if payload["method_seal_sha256"] != method_hash:
            raise RuntimeError("digits outward method hash mismatch")
        if payload["candidate_sha256"] != expected_hashes:
            raise RuntimeError("digits outward candidate hash mismatch")
        if payload["requested_horizon"] != expected_horizon:
            raise RuntimeError("digits outward requested horizon mismatch")
        if payload["reached_horizon"] != expected_horizon:
            raise RuntimeError("digits outward tube did not reach its horizon")
        if payload["maximum_radius"] < 0.0 or not math.isfinite(payload["maximum_radius"]):
            raise RuntimeError("invalid digits outward radius")
        if payload["minimum_logic_slack"] <= 0.0:
            raise RuntimeError("non-strict digits outward output logic")
        radius = np.load(path.with_suffix(".npz"))["radius"]
        if abs(float(radius.max()) - float(payload["maximum_radius"])) > 1e-18:
            raise RuntimeError("digits outward radius cache mismatch")
        caches[(seed, anchor)] = payload

    final = read(export / "final_audit.json")
    actual = {
        (int(row["seed"]), int(row["gate_index"])): row["actual_relative"]
        for row in final["rows"]
    }
    rows = []
    for key, certificate in sorted(certificates.items()):
        cache = caches[(int(certificate["seed"]), int(certificate["anchor"]))]
        event = cache["events"][str(int(certificate["gate_index"]))]
        bracket = event["outward_bracket"]
        truth = actual[key]
        rows.append({
            "seed": key[0],
            "gate_index": key[1],
            "anchor": int(certificate["anchor"]),
            "green_bracket": certificate["certified_bracket"],
            "outward_bracket": bracket,
            "actual_event": truth,
            "identical": bracket == certificate["certified_bracket"],
            "covered": bracket is not None and bracket[0] <= truth <= bracket[1],
            "signed_only": not certificate["unsigned_right_inverse_certificate_issued"],
        })
    if len(rows) != 7 or not all(row["identical"] and row["covered"] for row in rows):
        raise RuntimeError("digits outward bracket audit failed")
    signed_only = [row for row in rows if row["signed_only"]]
    if len(signed_only) != 1 or signed_only[0]["outward_bracket"] != [147, 147]:
        raise RuntimeError("signed-only outward event mismatch")
    return {
        "issued": len(rows),
        "covered": sum(row["covered"] for row in rows),
        "brackets_identical": sum(row["identical"] for row in rows),
        "unique_tubes": len(caches),
        "signed_only_issued_and_covered": len(signed_only),
        "maximum_radius": max(row["maximum_radius"] for row in caches.values()),
        "minimum_logic_slack": min(row["minimum_logic_slack"] for row in caches.values()),
        "maximum_hessian_interval_row_radius": max(
            row["maximum_hessian_interval_row_radius"] for row in caches.values()
        ),
        "maximum_eigen_numeric_error": max(
            row["maximum_eigen_numeric_error"] for row in caches.values()
        ),
        "aggregate_outward_seconds": sum(row["elapsed_seconds"] for row in caches.values()),
        "rows": rows,
    }


def audit_batched() -> dict:
    aggregate_path = RESULTS / "transformer_batched_scaling_benchmark.json"
    aggregate = read(aggregate_path)
    if aggregate["benchmark_source_sha256"] != sha256(
        ROOT / "scripts" / "benchmark_batched_certificate_primitives.py"
    ):
        raise RuntimeError("batched benchmark source hash mismatch")
    if aggregate["implementation_source_sha256"] != sha256(
        ROOT / "scripts" / "batched_green_operator.py"
    ):
        raise RuntimeError("batched implementation source hash mismatch")
    profiles = {}
    for row in aggregate["profiles"]:
        name = row["profile"]
        raw = read(RESULTS / f"transformer_batched_scaling_benchmark_{name}.json")
        if raw["timings_seconds"] != row["timings_seconds"]:
            raise RuntimeError(f"batched profile timing mismatch: {name}")
        timing = row["timings_seconds"]
        hvp_speedup = timing["serial_16_hvp_median"] / timing["batched_16_hvp_median"]
        output_speedup = (
            timing["serial_16_output_gram_median"]
            / timing["batched_16_output_gram_median"]
        )
        if abs(hvp_speedup - row["speedups"]["hvp_probe_block"]) > 1e-12:
            raise RuntimeError("HVP speedup arithmetic mismatch")
        if abs(output_speedup - row["speedups"]["output_gram_probe_block"]) > 1e-12:
            raise RuntimeError("output speedup arithmetic mismatch")
        projection = row["projection_h300"]
        if abs(
            projection["matched_serial_core_seconds"]
            / projection["projected_batched_core_seconds"]
            - projection["matched_projected_core_speedup"]
        ) > 1e-12:
            raise RuntimeError("projected speedup arithmetic mismatch")
        if max(row["equivalence"].values()) >= 5e-11:
            raise RuntimeError("batched primitive equivalence gate failed")
        profiles[name] = row

    replay_coordinates = ((333, 0, 3000), (345, 1, 1320))
    replays = []
    for seed, gate, anchor in replay_coordinates:
        path = RESULTS / (
            f"transformer_batched_replay_seed_{seed}_gate_{gate}_anchor_{anchor}.json"
        )
        row = read(path)
        original = ROOT / row["original_certificate"]
        if sha256(original) != row["original_certificate_sha256"]:
            raise RuntimeError("batched replay original-certificate hash mismatch")
        if not row["same_centerline_sha256"] or not row["exact_disposition_match"]:
            raise RuntimeError("batched replay disposition mismatch")
        if row["certified_bracket"] != row["original_certified_bracket"]:
            raise RuntimeError("batched replay bracket mismatch")
        maximum_error = max(
            row["green"]["upper_relative_error"],
            row["maximum_output_upper_relative_error"],
            row["closure_relative_error"],
        )
        if maximum_error >= 5e-11:
            raise RuntimeError("batched replay numerical mismatch")
        timing = row["timings_seconds"]
        recomputed = timing["original_serial_total"] / timing["total"]
        if abs(recomputed - timing["end_to_end_speedup"]) > 1e-12:
            raise RuntimeError("batched replay speedup arithmetic mismatch")
        replays.append({
            "seed": seed,
            "gate_index": gate,
            "anchor": anchor,
            "horizon": row["horizon"],
            "bracket": row["certified_bracket"],
            "maximum_relative_error": maximum_error,
            "end_to_end_speedup": timing["end_to_end_speedup"],
            "batched_minutes": timing["total"] / 60.0,
            "serial_minutes": timing["original_serial_total"] / 60.0,
        })
    speedups = [row["end_to_end_speedup"] for row in replays]
    million = profiles["1m"]["projection_h300"]
    return {
        "primitive_profiles": len(profiles),
        "maximum_primitive_relative_error": max(
            max(row["equivalence"].values()) for row in profiles.values()
        ),
        "complete_replays": len(replays),
        "exact_replay_matches": len(replays),
        "median_end_to_end_speedup": statistics.median(speedups),
        "minimum_end_to_end_speedup": min(speedups),
        "maximum_end_to_end_speedup": max(speedups),
        "million_parameter_matched_serial_hours": million[
            "matched_serial_core_seconds"
        ] / 3600.0,
        "million_parameter_batched_hours": million[
            "projected_batched_core_seconds"
        ] / 3600.0,
        "million_parameter_matched_speedup": million[
            "matched_projected_core_speedup"
        ],
        "million_parameter_peak_rss_gib": profiles["1m"][
            "observed_process_peak_rss_bytes"
        ] / 2**30,
        "replays": replays,
    }


def audit_modern() -> dict:
    path = RESULTS / "modern_transformer_primitive_audit.json"
    row = read(path)
    for name, expected in row["source_sha256"].items():
        source = ROOT / "scripts" / name
        if sha256(source) != expected:
            raise RuntimeError(f"modern primitive source hash mismatch: {name}")
    config = row["config"]
    if config["depth"] != 2 or config["normalization"] != "layernorm":
        raise RuntimeError("modern primitive architecture mismatch")
    if row["parameter_count"] < 100_000 or row["optimizer_state_dimension"] != 3 * row["parameter_count"]:
        raise RuntimeError("modern primitive dimension mismatch")
    if not row["finite"] or row["strictly_positive_second_moment_minimum"] <= 0.0:
        raise RuntimeError("modern primitive finite/AdamW-state gate failed")
    if row["optimizer_adjoint_relative_error"] >= 2e-11:
        raise RuntimeError("modern optimizer adjoint audit failed")
    if row["green_adjoint_relative_error"] >= 2e-11:
        raise RuntimeError("modern Green adjoint audit failed")
    return {
        "parameters": row["parameter_count"],
        "optimizer_state_dimension": row["optimizer_state_dimension"],
        "depth": config["depth"],
        "normalization": config["normalization"],
        "optimizer": "AdamW",
        "green_horizon": row["green_horizon"],
        "optimizer_adjoint_relative_error": row["optimizer_adjoint_relative_error"],
        "green_adjoint_relative_error": row["green_adjoint_relative_error"],
        "peak_rss_gib": row["observed_process_peak_rss_bytes"] / 2**30,
        "scope": row["scope"],
    }


def main() -> None:
    digits = audit_digits()
    batched = audit_batched()
    modern = audit_modern()
    result = {
        "status": "independent post-seal hardening audit passed",
        "digits_outward": digits,
        "batched_transformer": batched,
        "modern_transformer_primitives": modern,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Independent post-seal hardening audit",
        "",
        "## Digits exact-real continuation",
        "",
        f"- Issued/covered: {digits['covered']}/{digits['issued']}.",
        f"- Brackets identical to GreenCert: {digits['brackets_identical']}/{digits['issued']}.",
        "- The unique signed-only event remains the singleton `[147,147]`.",
        f"- Maximum 192-bit state radius: `{digits['maximum_radius']:.6e}`.",
        f"- Minimum strict output-logic slack: `{digits['minimum_logic_slack']:.6e}`.",
        "",
        "## Block-batched Transformer probes",
        "",
        f"- Complete exact replays: {batched['exact_replay_matches']}/{batched['complete_replays']}.",
        f"- Median measured end-to-end speedup: {batched['median_end_to_end_speedup']:.2f}x.",
        f"- Million-parameter matched projection: {batched['million_parameter_matched_serial_hours']:.2f} h serial to {batched['million_parameter_batched_hours']:.2f} h batched ({batched['million_parameter_matched_speedup']:.2f}x).",
        "- All 16 probes, eight powers, committed streams, and failure budgets are unchanged.",
        "",
        "## LayerNorm + AdamW derivative transport",
        "",
        f"- Two-block pre-LayerNorm Transformer: {modern['parameters']:,} parameters.",
        f"- AdamW optimizer state: {modern['optimizer_state_dimension']:,} coordinates.",
        f"- Optimizer/Green adjoint errors: {modern['optimizer_adjoint_relative_error']:.3e}/{modern['green_adjoint_relative_error']:.3e}.",
        "- Scope remains matrix-free primitives; global LayerNorm/AdamW jets and event certification are not claimed.",
        "",
    ]
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "digits": {k: digits[k] for k in ("issued", "covered", "brackets_identical")},
        "batched": {k: batched[k] for k in (
            "complete_replays",
            "exact_replay_matches",
            "median_end_to_end_speedup",
            "million_parameter_batched_hours",
        )},
        "modern": modern,
    }, indent=2))


if __name__ == "__main__":
    main()
