#!/usr/bin/env python3
"""Verify the executable source/data closure of the v1.3 directional replay."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
OUTPUT = ROOT / "results" / "directional_replay_dependency_closure.json"
ENTRYPOINT_RESULTS = {
    "scripts/diagnose_transformer_directional_block_remainder.py": (
        "results/transformer_directional_block_remainder_diagnostic.json",
        ("source_hashes", "script"),
    ),
    "scripts/audit_transformer_directional_three_sweep_events.py": (
        "results/transformer_directional_three_sweep_event_audit.json",
        ("script_sha256",),
    ),
    "scripts/audit_transformer_mixed_directional_cohort.py": (
        "results/transformer_mixed_directional_cohort_audit.json",
        ("source_hashes", "script"),
    ),
}
DECLARED_BINDINGS = {
    "results/transformer_directional_block_remainder_diagnostic.json": {
        "results/transformer_fully_recentered_three_sweep_audit.json": ("source_hashes", "parent"),
        "DIRECTIONAL_BLOCK_REMAINDER_PROTOCOL.md": ("source_hashes", "protocol"),
        "DIRECTIONAL_BLOCK_REMAINDER_THEOREM.md": ("source_hashes", "theorem"),
        "scripts/transformer_directional_fourth_bound.py": ("source_hashes", "module"),
        "scripts/test_transformer_directional_fourth_bound.py": ("source_hashes", "test"),
        "scripts/diagnose_transformer_directional_block_remainder.py": ("source_hashes", "script"),
    },
    "results/transformer_mixed_directional_cohort_audit.json": {
        "results/transformer_directional_block_remainder_diagnostic.json": ("source_hashes", "parent"),
        "MIXED_DIRECTIONAL_JET_AUDIT_PROTOCOL.md": ("source_hashes", "protocol"),
        "scripts/transformer_mixed_directional_jet.py": ("source_hashes", "module"),
        "scripts/test_transformer_mixed_directional_jet.py": ("source_hashes", "test"),
        "scripts/audit_transformer_mixed_directional_cohort.py": ("source_hashes", "script"),
    },
    "results/transformer_directional_three_sweep_event_audit.json": {
        "results/transformer_directional_block_remainder_diagnostic.json": ("closure_parent_sha256",),
        "results/transformer_fully_recentered_three_sweep_audit.json": ("event_parent_sha256",),
        "DIRECTIONAL_THREE_SWEEP_EVENT_PROTOCOL.md": ("protocol_sha256",),
        "scripts/audit_transformer_directional_three_sweep_events.py": ("script_sha256",),
    },
}
CHECKPOINT_SEEDS = (360, 361, 366, 369, 370, 372, 373, 375, 378)
REQUIRED_DATA = (
    "ADAPTIVE_SWEEP_COHORT_PROTOCOL.md",
    "DIRECTIONAL_BLOCK_REMAINDER_PROTOCOL.md",
    "DIRECTIONAL_BLOCK_REMAINDER_THEOREM.md",
    "DIRECTIONAL_THREE_SWEEP_EVENT_PROTOCOL.md",
    "MIXED_DIRECTIONAL_JET_AUDIT_PROTOCOL.md",
    "results/transformer_fully_recentered_three_sweep_audit.json",
    "results/transformer_directional_anchor_states.npz",
    "results/transformer_directional_anchor_states_manifest.json",
    "results/transformer_directional_block_remainder_diagnostic.json",
    "results/transformer_directional_three_sweep_event_audit.json",
    "results/transformer_mixed_directional_cohort_audit.json",
) + tuple(
    f"results/transformer_hvp_prospective_seed_{seed}.checkpoints.npz"
    for seed in CHECKPOINT_SEEDS
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def nested(payload: dict, keys: tuple[str, ...]) -> object:
    value: object = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(".".join(keys))
        value = value[key]
    return value


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def source_closure(entrypoint: Path, modules: dict[str, Path]) -> dict:
    pending = [entrypoint]
    visited: set[Path] = set()
    external: set[str] = set()
    missing: set[str] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        for name in imported_roots(path):
            local = modules.get(name)
            if local is not None:
                if local not in visited:
                    pending.append(local)
            elif importlib.util.find_spec(name) is None:
                missing.add(name)
            else:
                external.add(name)
    return {
        "local_sources": {
            path.relative_to(ROOT).as_posix(): digest(path)
            for path in sorted(visited)
        },
        "external_modules": sorted(external),
        "missing_imports": sorted(missing),
    }


def build_record() -> dict:
    sys.path.insert(0, str(SCRIPTS))
    sys.path.insert(0, str(ROOT))
    modules = {path.stem: path for path in SCRIPTS.glob("*.py")}
    rows = {}
    union_sources: dict[str, str] = {}
    all_external: set[str] = set()
    declared_bindings = {}
    for result_relative, bindings in DECLARED_BINDINGS.items():
        result_path = ROOT / result_relative
        result = json.loads(result_path.read_text(encoding="utf-8"))
        checked = {}
        for target_relative, hash_keys in bindings.items():
            expected = str(nested(result, hash_keys)).upper()
            observed = digest(ROOT / target_relative)
            if observed != expected:
                raise AssertionError(
                    f"frozen source binding changed: {result_relative} -> {target_relative}"
                )
            checked[target_relative] = observed
        declared_bindings[result_relative] = checked
    for relative, (result_relative, hash_keys) in ENTRYPOINT_RESULTS.items():
        entrypoint = ROOT / relative
        result_path = ROOT / result_relative
        if not entrypoint.is_file() or not result_path.is_file():
            raise FileNotFoundError(relative if not entrypoint.is_file() else result_relative)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        expected_script_hash = str(nested(result, hash_keys)).upper()
        observed_script_hash = digest(entrypoint)
        source_relation = "exact byte match"
        if observed_script_hash != expected_script_hash:
            reconstructed = entrypoint.read_bytes() + b"\n"
            if digest_bytes(reconstructed) != expected_script_hash:
                raise AssertionError(
                    f"frozen result does not bind current entrypoint: {relative}"
                )
            source_relation = (
                "frozen bytes equal maintained source plus one terminal LF; "
                "Python AST is unchanged"
            )
        closure = source_closure(entrypoint, modules)
        if closure["missing_imports"]:
            raise AssertionError(
                f"missing imports for {relative}: {closure['missing_imports']}"
            )
        union_sources.update(closure["local_sources"])
        all_external.update(closure["external_modules"])
        rows[relative] = {
            "frozen_result": result_relative,
            "frozen_script_sha256": expected_script_hash,
            "maintained_script_sha256": observed_script_hash,
            "source_relation": source_relation,
            **closure,
        }

    missing_data = [name for name in REQUIRED_DATA if not (ROOT / name).is_file()]
    if missing_data:
        raise AssertionError(f"missing directional replay data: {missing_data}")
    required_data = {name: digest(ROOT / name) for name in REQUIRED_DATA}
    return {
        "status": "directional replay dependency closure passed",
        "scope": (
            "Static transitive Python import closure plus explicit frozen data "
            "dependencies; large Transformer checkpoints are regenerated separately."
        ),
        "entrypoints": rows,
        "declared_result_source_bindings": declared_bindings,
        "union_local_sources": dict(sorted(union_sources.items())),
        "external_modules": sorted(all_external),
        "required_data_files": required_data,
        "missing_imports": [],
        "large_checkpoint_archives_committed": False,
        "compact_exact_anchor_bundle_committed": True,
        "full_checkpoint_regeneration_required_for_directional_replay": False,
        "checkpoint_regeneration_seeds": list(CHECKPOINT_SEEDS),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    record = build_record()
    serialized = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.write:
        OUTPUT.write_text(serialized, encoding="utf-8", newline="\n")
    else:
        if not OUTPUT.is_file():
            raise FileNotFoundError(OUTPUT)
        existing = OUTPUT.read_text(encoding="utf-8")
        if existing != serialized:
            raise AssertionError(
                "directional replay dependency lock is stale; rerun with --write"
            )
    print(
        json.dumps(
            {
                "status": record["status"],
                "entrypoints": len(record["entrypoints"]),
                "local_sources": len(record["union_local_sources"]),
                "external_modules": len(record["external_modules"]),
                "required_data_files": len(record["required_data_files"]),
                "declared_result_bindings": sum(
                    len(row)
                    for row in record["declared_result_source_bindings"].values()
                ),
                "missing_imports": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
