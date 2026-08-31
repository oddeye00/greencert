#!/usr/bin/env python3
"""Independent arithmetic verifier for the anchor-fixed structured audit."""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics

from direct_image_green_bound import direct_image_rows
from prefix_gram_enclosure import equal_family_stage_delta, prefix_gram_rows
from verify_structured_parameter_green_audit import (
    case_set,
    case_set_sha256,
    check_attempt,
    close,
    require,
    verify_dependency,
    verify_source_bridge,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "ANCHOR_FIXED_STRUCTURED_PARAMETER_GREEN_AUDIT_PROTOCOL.md"
RESULT = ROOT / "results" / "anchor_fixed_structured_parameter_green_transformer_audit.json"
BASELINE = ROOT / "results" / "structured_parameter_green_transformer_audit.json"
CACHE = ROOT / "results" / "anchor_fixed_structured_parameter_green_transformer_cache"
AUDIT_SOURCE = ROOT / "scripts" / "audit_anchor_fixed_structured_parameter_green_transformer.py"
OUTPUT = ROOT / "results" / "anchor_fixed_structured_parameter_green_independent_audit.json"
PREFIXES = (4, 8, 16)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def parse_protocol() -> tuple[str, str, dict[str, str]]:
    text = PROTOCOL.read_text(encoding="utf-8")
    nonce = re.search(r"MASTER_NONCE:\s*\n\s*`([0-9a-f]{64})`", text)
    case_hash = re.search(r"CASE_SET_SHA256:\s*\n\s*`([A-F0-9]{64})`", text)
    matches = re.findall(
        r"- DEPENDENCY:([^\r\n]+)\s*\r?\n\s*`([A-F0-9]{64})`", text
    )
    require(nonce is not None, "protocol nonce is missing")
    require(case_hash is not None, "protocol case-set hash is missing")
    require(len(matches) == 23, "protocol dependency count changed")
    dependencies = {name.strip(): digest for name, digest in matches}
    require(len(dependencies) == 23, "protocol has duplicate dependencies")
    return nonce.group(1), case_hash.group(1), dependencies


def no_outcome_reader_symbols() -> bool:
    tree = ast.parse(AUDIT_SOURCE.read_text(encoding="utf-8"), filename=str(AUDIT_SOURCE))
    forbidden = ("outcome_path", "audit_path", "reveal", "revealed", "future_trajectory")
    identifiers: list[str] = []
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.append(node.id.lower())
        elif isinstance(node, ast.Attribute):
            identifiers.append(node.attr.lower())
        elif isinstance(node, ast.Import):
            modules.extend(alias.name.lower() for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module.lower())
    return not any(
        token in identifier
        for identifier in identifiers + modules
        for token in forbidden
    )


def key(row: dict) -> tuple[int, float, int]:
    candidate = row["candidate"]
    return int(candidate["seed"]), float(candidate["threshold"]), int(candidate["anchor"])


def cache_path(row: dict) -> Path:
    seed, threshold, anchor = key(row)
    gate = int(round((threshold - 0.7) * 10.0))
    return CACHE / f"seed_{seed}_gate_{gate}_anchor_{anchor}_v3.json"


def main() -> None:
    nonce, protocol_case_hash, dependencies = parse_protocol()
    require(len(nonce) == 64, "nonce length changed")
    superseded_dependencies = 0
    for name, digest in dependencies.items():
        path, superseded = verify_dependency(name, digest)
        require(sha256(path) == digest, f"sealed dependency hash mismatch: {name}")
        superseded_dependencies += int(superseded)
    source_bridge = verify_source_bridge()

    manifest_rows = case_set()
    computed_case_hash = case_set_sha256(manifest_rows)
    require(computed_case_hash == protocol_case_hash, "protocol case-set hash mismatch")
    manifest = {tuple(row[:3]): row for row in manifest_rows}
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    baseline_payload = json.loads(BASELINE.read_text(encoding="utf-8"))
    baseline = {key(row): row for row in baseline_payload["rows"]}
    require(result["protocol_sha256"] == sha256(PROTOCOL), "result protocol hash mismatch")
    require(result["case_set_sha256"] == computed_case_hash, "result case-set hash mismatch")
    require(result["dependency_sha256"] == dependencies, "result dependency map mismatch")
    require(no_outcome_reader_symbols(), "audit source contains an outcome-reader symbol")

    rows = result["rows"]
    require(len(rows) == 15 and len({key(row) for row in rows}) == 15, "case rows changed")
    stage_delta = equal_family_stage_delta(
        family_failure=1.0e-6, operators=15, prefixes=PREFIXES
    )
    for row in rows:
        candidate = key(row)
        require(candidate in manifest and candidate in baseline, f"unknown case {candidate}")
        _, _, _, horizon, certificate_sha = manifest[candidate]
        require(int(row["horizon"]) == int(horizon), f"horizon changed: {candidate}")
        require(int(row["operator_input_blocks"]) == int(horizon) - 1, f"input blocks changed: {candidate}")
        require(bool(row["anchor_block_omitted"]), f"anchor block retained: {candidate}")
        require(row["certificate_sha256"] == certificate_sha, f"certificate hash changed: {candidate}")
        require(bool(row["issued"]), f"nonissued case: {candidate}")
        require(bool(row["bracket_preserved"]), f"bracket not preserved: {candidate}")
        require(row["bracket"] == row["inherited_bracket"], f"bracket mismatch: {candidate}")
        require(int(row["outcome_files_read"]) == 0, f"outcome flag changed: {candidate}")
        require(
            int(row["unrestricted_structured_green_sweeps"])
            == int(baseline[candidate]["logical_total_green_sweeps"]),
            f"baseline sweeps changed: {candidate}",
        )

        prefix = int(row["prefix"])
        require(prefix in PREFIXES, f"invalid prefix: {candidate}")
        require(len(row["initial_probe_norms"]) == prefix, f"initial probes changed: {candidate}")
        require(len(row["direct_image_norms"]) == prefix, f"image probes changed: {candidate}")
        require(len(row["gram_norms"]) in (0, prefix), f"Gram probes changed: {candidate}")
        stage_prefixes = tuple(int(stage["prefix"]) for stage in row["stages"])
        require(stage_prefixes == PREFIXES[: len(stage_prefixes)], f"stage order changed: {candidate}")
        for stage in row["stages"]:
            current = int(stage["prefix"])
            direct = direct_image_rows(
                image_norms=row["direct_image_norms"],
                initial_norms=row["initial_probe_norms"],
                prefixes=(current,),
                stage_delta=stage_delta,
            )[0]
            for field, expected in direct.items():
                require(close(stage["direct"][field], expected), f"direct {field} changed: {candidate}")
            check_attempt(stage["direct"])
            if stage.get("gram") is not None:
                gram = prefix_gram_rows(
                    final_norms=row["gram_norms"],
                    initial_norms=row["initial_probe_norms"],
                    prefixes=(current,),
                    power=1,
                    stage_delta=stage_delta,
                )[0]
                for field, expected in gram.items():
                    require(close(stage["gram"][field], expected), f"Gram {field} changed: {candidate}")
                check_attempt(stage["gram"])

        final = row["stages"][-1]
        if row["route"] == "direct_image":
            selected = final["direct"]
            forward, transpose = prefix, 0
        elif row["route"] == "gram_fallback":
            require(final.get("gram") is not None, f"missing Gram row: {candidate}")
            selected = final["gram"]
            forward, transpose = prefix, prefix
        else:
            raise RuntimeError(f"unknown route: {candidate}")
        require(bool(selected["issued"]), f"selected attempt does not issue: {candidate}")
        require(selected["bracket"] == row["bracket"], f"selected bracket changed: {candidate}")
        require(int(row["logical_forward_green_sweeps"]) == forward, f"forward sweeps changed: {candidate}")
        require(int(row["logical_transpose_green_sweeps"]) == transpose, f"transpose sweeps changed: {candidate}")
        total = forward + transpose
        require(int(row["logical_total_green_sweeps"]) == total, f"total sweeps changed: {candidate}")
        require(
            close(
                row["reduction_vs_unrestricted_structured"],
                float(row["unrestricted_structured_green_sweeps"]) / total,
            ),
            f"structured reduction changed: {candidate}",
        )
        cache = cache_path(row)
        require(cache.is_file(), f"cache missing: {cache.name}")
        require(json.loads(cache.read_text(encoding="utf-8")) == row, f"cache differs: {cache.name}")

    full_total = sum(int(row["full_state_staged_green_sweeps"]) for row in rows)
    unrestricted_total = sum(int(row["unrestricted_structured_green_sweeps"]) for row in rows)
    restricted_total = sum(int(row["logical_total_green_sweeps"]) for row in rows)
    pairwise = [float(row["reduction_vs_unrestricted_structured"]) for row in rows]
    reconstructed = {
        "cases": 15,
        "issued": sum(bool(row["issued"]) for row in rows),
        "brackets_preserved": sum(bool(row["bracket_preserved"]) for row in rows),
        "route_distribution": dict(Counter(row["route"] for row in rows)),
        "full_state_staged_green_sweeps": full_total,
        "unrestricted_structured_green_sweeps": unrestricted_total,
        "anchor_fixed_structured_green_sweeps": restricted_total,
        "reduction_vs_unrestricted_structured": unrestricted_total / restricted_total,
        "reduction_vs_full_state": full_total / restricted_total,
        "sweeps_saved_vs_unrestricted_structured": unrestricted_total - restricted_total,
        "median_pairwise_reduction_vs_unrestricted": statistics.median(pairwise),
        "minimum_pairwise_reduction_vs_unrestricted": min(pairwise),
        "maximum_pairwise_reduction_vs_unrestricted": max(pairwise),
        "combined_family_failure_upper": 2.0e-6,
        "outcome_files_read": sum(int(row["outcome_files_read"]) for row in rows),
        "promotion_gate_passed": (
            all(bool(row["bracket_preserved"]) for row in rows)
            and restricted_total < unrestricted_total
        ),
    }
    for field, expected in reconstructed.items():
        observed = result[field]
        if isinstance(expected, float):
            require(close(observed, expected), f"aggregate {field} changed")
        else:
            require(observed == expected, f"aggregate {field} changed")
    require(not reconstructed["promotion_gate_passed"], "negative promotion result changed")
    require(len(list(CACHE.glob("*_v3.json"))) == 15, "cache count changed")

    payload = {
        "status": "INDEPENDENT ANCHOR-FIXED STRUCTURED AUDIT PASSED",
        "scope": "mechanical replay; no neural or random-operator recomputation",
        **reconstructed,
        "sealed_dependency_hashes_verified": len(dependencies),
        "sealed_dependency_snapshots_used": superseded_dependencies,
        "source_supersession": source_bridge,
        "case_caches_verified": 15,
        "stored_direct_and_gram_bounds_recomputed": True,
        "stored_quadratic_closures_recomputed": True,
        "negative_promotion_result_reproduced": True,
        "audit_source_has_no_outcome_reader_symbol": True,
        "protocol_sha256": sha256(PROTOCOL),
        "result_sha256": sha256(RESULT),
        "verifier_sha256": sha256(Path(__file__)),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
