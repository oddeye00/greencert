#!/usr/bin/env python3
"""Independent mechanical verifier for the sealed structured-Green audit.

This script performs no neural-network or randomized operator computation.  It
checks the frozen dependency chain, parses the sealed case manifest without
executing the generating audit, recomputes all stored probabilistic-bound and
quadratic-closure arithmetic, checks every cache against the aggregate record,
and reconstructs the headline totals.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from direct_image_green_bound import direct_image_rows
from prefix_gram_enclosure import equal_family_stage_delta, prefix_gram_rows
from structured_parameter_green_sealed_v1 import structured_quadratic_root
from structured_parameter_green_source_bridge import (
    verify_dependency,
    verify_source_bridge,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "STRUCTURED_PARAMETER_GREEN_AUDIT_PROTOCOL_V2.md"
RESULT = ROOT / "results" / "structured_parameter_green_transformer_audit.json"
CACHE = ROOT / "results" / "structured_parameter_green_transformer_cache"
MANIFEST_SOURCE = ROOT / "scripts" / "audit_transformer_relinearized_prefix_panel.py"
AUDIT_SOURCE = ROOT / "scripts" / "audit_structured_parameter_green_transformer.py"
OUTPUT = ROOT / "results" / "structured_parameter_green_independent_audit.json"
PREFIXES = (4, 8, 16)
FAMILY_FAILURE = 1.0e-6


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def close(left: float, right: float, tolerance: float = 2.0e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=1.0e-30)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_protocol() -> tuple[str, str, dict[str, str]]:
    text = PROTOCOL.read_text(encoding="utf-8")
    nonce_match = re.search(r"MASTER_NONCE:\s*\n\s*`([0-9a-f]{64})`", text)
    case_match = re.search(r"CASE_SET_SHA256:\s*\n\s*`([A-F0-9]{64})`", text)
    dependency_matches = re.findall(
        r"- DEPENDENCY:([^\r\n]+)\s*\r?\n\s*`([A-F0-9]{64})`", text
    )
    require(nonce_match is not None, "protocol nonce is missing")
    require(case_match is not None, "protocol case-set hash is missing")
    require(len(dependency_matches) == 17, "protocol dependency count changed")
    dependencies = {name.strip(): digest for name, digest in dependency_matches}
    require(len(dependencies) == len(dependency_matches), "duplicate protocol dependency")
    return nonce_match.group(1), case_match.group(1), dependencies


def literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return ast.literal_eval(node.value)
    raise RuntimeError(f"literal assignment {name} not found in {path.name}")


def case_set() -> tuple[tuple, ...]:
    rows = tuple(literal_assignment(MANIFEST_SOURCE, "CASE_ROWS"))
    require(len(rows) == 15, "sealed case count changed")
    require(len({tuple(row[:3]) for row in rows}) == 15, "sealed cases are not unique")
    return rows


def case_set_sha256(rows: tuple[tuple, ...]) -> str:
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest().upper()


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


def candidate_key(row: dict) -> tuple[int, float, int]:
    candidate = row["candidate"]
    return int(candidate["seed"]), float(candidate["threshold"]), int(candidate["anchor"])


def cache_for(row: dict) -> Path:
    seed, threshold, anchor = candidate_key(row)
    gate = int(round((threshold - 0.7) * 10.0))
    return CACHE / f"seed_{seed}_gate_{gate}_anchor_{anchor}_v2.json"


def check_attempt(attempt: dict) -> None:
    kappa = float(attempt["structured_gain_upper"])
    forcing = float(attempt["parameter_forcing_upper"])
    response = float(attempt["parameter_response_upper"])
    lipschitz = float(attempt["objective_hessian_lipschitz_upper"])
    require(close(response, kappa * forcing), "parameter response arithmetic changed")
    discriminant = 1.0 - 2.0 * kappa * lipschitz * response
    require(close(discriminant, attempt["discriminant"]), "discriminant changed")
    radius = structured_quadratic_root(response, kappa, lipschitz)
    if radius is None:
        require(attempt["parameter_remainder_radius"] is None, "failed closure has a radius")
    else:
        require(
            close(radius, attempt["parameter_remainder_radius"]),
            "quadratic radius changed",
        )
    domain_passed = radius is not None and (
        float(attempt["correction_max_parameter_norm"]) + float(radius)
        <= float(attempt["domain_radius"])
    )
    require(domain_passed == bool(attempt["domain_passed"]), "domain decision changed")
    require(
        bool(attempt["issued"])
        == (domain_passed and attempt["bracket"] is not None),
        "issuance logic changed",
    )


def check_row(row: dict, manifest: dict[tuple[int, float, int], tuple]) -> None:
    key = candidate_key(row)
    require(key in manifest, f"row outside sealed manifest: {key}")
    _, _, _, horizon, certificate_sha = manifest[key]
    require(int(row["horizon"]) == int(horizon), f"horizon changed for {key}")
    require(row["certificate_sha256"] == certificate_sha, f"certificate hash changed for {key}")
    require(bool(row["issued"]), f"nonissued row in promoted panel: {key}")
    require(bool(row["bracket_preserved"]), f"bracket not preserved: {key}")
    require(row["bracket"] == row["inherited_bracket"], f"bracket mismatch: {key}")
    require(int(row["outcome_files_read"]) == 0, f"outcome-read flag changed: {key}")
    require(int(row["prefix"]) in PREFIXES, f"invalid stopping prefix: {key}")

    prefix = int(row["prefix"])
    require(len(row["initial_probe_norms"]) == prefix, f"initial probe count changed: {key}")
    require(len(row["direct_image_norms"]) == prefix, f"image probe count changed: {key}")
    require(len(row["gram_norms"]) in (0, prefix), f"Gram probe count changed: {key}")
    stage_delta = equal_family_stage_delta(
        family_failure=FAMILY_FAILURE, operators=15, prefixes=PREFIXES
    )
    stage_prefixes = tuple(int(stage["prefix"]) for stage in row["stages"])
    require(stage_prefixes == PREFIXES[: len(stage_prefixes)], f"stage order changed: {key}")

    for stage in row["stages"]:
        current = int(stage["prefix"])
        expected_direct = direct_image_rows(
            image_norms=row["direct_image_norms"],
            initial_norms=row["initial_probe_norms"],
            prefixes=(current,),
            stage_delta=stage_delta,
        )[0]
        for field, expected in expected_direct.items():
            require(close(stage["direct"][field], expected), f"direct {field} changed: {key}")
        check_attempt(stage["direct"])
        if stage.get("gram") is not None:
            expected_gram = prefix_gram_rows(
                final_norms=row["gram_norms"],
                initial_norms=row["initial_probe_norms"],
                prefixes=(current,),
                power=1,
                stage_delta=stage_delta,
            )[0]
            for field, expected in expected_gram.items():
                require(close(stage["gram"][field], expected), f"Gram {field} changed: {key}")
            check_attempt(stage["gram"])

    final = row["stages"][-1]
    if row["route"] == "direct_image":
        selected = final["direct"]
        expected_forward, expected_transpose = prefix, 0
    elif row["route"] == "gram_fallback":
        require(final.get("gram") is not None, f"Gram route lacks Gram row: {key}")
        selected = final["gram"]
        expected_forward, expected_transpose = prefix, prefix
    else:
        raise RuntimeError(f"unknown route for {key}")
    require(bool(selected["issued"]), f"selected stage does not issue: {key}")
    require(selected["bracket"] == row["bracket"], f"selected bracket changed: {key}")
    require(
        int(row["logical_forward_green_sweeps"]) == expected_forward,
        f"forward sweep count changed: {key}",
    )
    require(
        int(row["logical_transpose_green_sweeps"]) == expected_transpose,
        f"transpose sweep count changed: {key}",
    )
    total = expected_forward + expected_transpose
    require(int(row["logical_total_green_sweeps"]) == total, f"total sweeps changed: {key}")
    require(
        close(row["sweep_reduction"], float(row["full_state_staged_green_sweeps"]) / total),
        f"pairwise reduction changed: {key}",
    )

    cache = cache_for(row)
    require(cache.is_file(), f"cache missing: {cache.name}")
    require(json.loads(cache.read_text(encoding="utf-8")) == row, f"cache differs: {cache.name}")


def main() -> None:
    nonce, protocol_case_hash, dependencies = parse_protocol()
    require(len(nonce) == 64, "nonce length changed")
    superseded_dependencies = 0
    for name, expected in dependencies.items():
        path, superseded = verify_dependency(name, expected)
        require(sha256(path) == expected, f"sealed dependency hash mismatch: {name}")
        superseded_dependencies += int(superseded)
    source_bridge = verify_source_bridge()

    rows_literal = case_set()
    computed_case_hash = case_set_sha256(rows_literal)
    require(computed_case_hash == protocol_case_hash, "protocol case-set hash mismatch")
    manifest = {tuple(row[:3]): row for row in rows_literal}
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    require(result["protocol_sha256"] == sha256(PROTOCOL), "result protocol hash mismatch")
    require(result["case_set_sha256"] == computed_case_hash, "result case-set hash mismatch")
    require(result["dependency_sha256"] == dependencies, "result dependency map mismatch")
    require(no_outcome_reader_symbols(), "audit source contains an outcome-reader symbol")

    rows = result["rows"]
    require(len(rows) == 15, "aggregate row count changed")
    require(len({candidate_key(row) for row in rows}) == 15, "aggregate candidates repeat")
    for row in rows:
        check_row(row, manifest)

    full_total = sum(int(row["full_state_staged_green_sweeps"]) for row in rows)
    structured_total = sum(int(row["logical_total_green_sweeps"]) for row in rows)
    reductions = [float(row["sweep_reduction"]) for row in rows]
    routes = dict(Counter(row["route"] for row in rows))
    reconstructed = {
        "cases": len(rows),
        "issued": sum(bool(row["issued"]) for row in rows),
        "brackets_preserved": sum(bool(row["bracket_preserved"]) for row in rows),
        "route_distribution": routes,
        "full_state_staged_green_sweeps": full_total,
        "structured_parameter_green_sweeps": structured_total,
        "aggregate_sweep_reduction": full_total / structured_total,
        "median_pairwise_sweep_reduction": statistics.median(reductions),
        "minimum_pairwise_sweep_reduction": min(reductions),
        "maximum_pairwise_sweep_reduction": max(reductions),
        "combined_family_failure_upper": 2.0e-6,
        "outcome_files_read": sum(int(row["outcome_files_read"]) for row in rows),
    }
    for field, expected in reconstructed.items():
        observed = result[field]
        if isinstance(expected, float):
            require(close(observed, expected), f"aggregate {field} changed")
        else:
            require(observed == expected, f"aggregate {field} changed")

    cache_files = sorted(CACHE.glob("*_v2.json"))
    require(len(cache_files) == 15, "unexpected cache-file count")
    payload = {
        "status": "INDEPENDENT STRUCTURED PARAMETER GREEN AUDIT PASSED",
        "scope": "mechanical replay; no neural or random-operator recomputation",
        **reconstructed,
        "promotion_gate_passed": (
            reconstructed["issued"] == 15
            and reconstructed["brackets_preserved"] == 15
            and structured_total < full_total
        ),
        "sealed_dependency_hashes_verified": len(dependencies),
        "sealed_dependency_snapshots_used": superseded_dependencies,
        "source_supersession": source_bridge,
        "case_caches_verified": len(cache_files),
        "stored_direct_and_gram_bounds_recomputed": True,
        "stored_quadratic_closures_recomputed": True,
        "audit_source_has_no_outcome_reader_symbol": True,
        "protocol_sha256": sha256(PROTOCOL),
        "result_sha256": sha256(RESULT),
        "verifier_sha256": sha256(Path(__file__)),
    }
    require(payload["promotion_gate_passed"], "sealed promotion gate did not pass")
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
