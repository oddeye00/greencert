#!/usr/bin/env python3
"""Frozen 15-case composition audit for direct causal row closure."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import statistics
import time

from diagnose_transformer_causal_row_green import run
from transformer_certificate_protocol import Candidate
from transformer_v3_certificate import output_path, safe_json


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PARENT = RESULTS / "transformer_fully_recentered_three_sweep_audit.json"
BASELINE = RESULTS / "anchor_fixed_structured_parameter_green_transformer_audit.json"
PROTOCOL = ROOT / "CAUSAL_ROW_GREEN_COMPOSITION_PROTOCOL.md"
OUTPUT = RESULTS / "transformer_causal_row_green_composition_audit.json"
CACHE = RESULTS / "transformer_causal_row_green_composition_cache"
DEVELOPMENT = (360, 0.7, 3480)
EXPECTED_CASES = 15
SWEEPS = 4
PROBES = 4
FAMILY_FAILURE_UPPER = 1.0e-6
CLAIM_FILES = (
    "CAUSAL_ROW_GREEN_THEOREM.md",
    "CAUSAL_ROW_GREEN_COMPOSITION_PROTOCOL.md",
    "scripts/causal_row_green.py",
    "scripts/diagnose_transformer_causal_row_green.py",
    "scripts/transformer_mixed_directional_jet_v2.py",
    "scripts/test_causal_row_green.py",
    "scripts/test_causal_row_green_transformer_batch.py",
    "scripts/test_transformer_mixed_directional_jet_v2.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def candidate_from(row: dict) -> Candidate:
    value = row["candidate"]
    return Candidate(int(value["seed"]), float(value["threshold"]), int(value["anchor"]))


def key(candidate: Candidate) -> str:
    return f"seed_{candidate.seed}_gate_{candidate.gate_index}_anchor_{candidate.anchor}"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    temporary.replace(path)


def evaluate(task: dict) -> dict:
    candidate = Candidate(**task["candidate"])
    cache = CACHE / f"{key(candidate)}.json"
    if cache.exists():
        record = safe_json(cache)
        if record.get("cache_identity") == task["cache_identity"]:
            record["cache_hit"] = True
            return record
    record = run(
        candidate,
        SWEEPS,
        PROBES,
        "quadratic",
        family_delta=FAMILY_FAILURE_UPPER / EXPECTED_CASES,
    )
    record.update(
        {
            "development_row": (
                candidate.seed,
                candidate.threshold,
                candidate.anchor,
            )
            == DEVELOPMENT,
            "cache_identity": task["cache_identity"],
            "cache_hit": False,
        }
    )
    write_json(cache, record)
    return record


def summary(rows: list[dict]) -> dict:
    issued = [row for row in rows if row["issued"]]
    times = [float(row["timings_seconds"]["end_to_end"]) for row in rows]
    return {
        "cases": len(rows),
        "issued": len(issued),
        "retained_sealed_bracket": sum(row["retains_sealed_bracket"] for row in rows),
        "released_path_hash_matches": sum(
            row["released_corrected_path_match"] is True for row in rows
        ),
        "row_domain_passed": sum(row["row_domain_passed"] for row in rows),
        "old_global_closure_passed": sum(
            row["old_global_closure"]["closure_passed"] for row in rows
        ),
        "signed_global_closure_passed": sum(
            row["signed_global_closure"]["closure_passed"] for row in rows
        ),
        "row_only_issuance": sum(
            row["issued"] and not row["signed_global_closure"]["closure_passed"]
            for row in rows
        ),
        "median_end_to_end_seconds": statistics.median(times) if times else None,
        "total_end_to_end_seconds": sum(times),
        "issued_coordinates": [row["candidate"] for row in issued],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    started = time.perf_counter()

    parent = safe_json(PARENT)
    baseline = safe_json(BASELINE)
    candidates = [candidate_from(row) for row in parent["rows"]]
    if len(candidates) != EXPECTED_CASES or len(set(candidates)) != EXPECTED_CASES:
        raise RuntimeError("the declared 15-case cohort changed")
    if sum(
        (candidate.seed, candidate.threshold, candidate.anchor) == DEVELOPMENT
        for candidate in candidates
    ) != 1:
        raise RuntimeError("development coordinate does not identify one row")

    source_hashes = {name: sha256(ROOT / name) for name in CLAIM_FILES}
    source_hashes[str(PARENT.relative_to(ROOT)).replace("\\", "/")] = sha256(PARENT)
    source_hashes[str(BASELINE.relative_to(ROOT)).replace("\\", "/")] = sha256(BASELINE)
    script_hash = sha256(Path(__file__))
    tasks = []
    for candidate in candidates:
        identity = {
            "candidate": candidate.__dict__,
            "sweeps": SWEEPS,
            "probes": PROBES,
            "family_delta": FAMILY_FAILURE_UPPER / EXPECTED_CASES,
            "certificate_sha256": sha256(output_path(candidate)),
            "source_hashes": source_hashes,
            "audit_script_sha256": script_hash,
        }
        tasks.append(
            {
                "candidate": candidate.__dict__,
                "cache_identity": hashlib.sha256(
                    json.dumps(identity, sort_keys=True).encode("utf-8")
                ).hexdigest().upper(),
            }
        )

    rows = []
    if args.workers == 1:
        for task in tasks:
            rows.append(evaluate(task))
            print(f"completed {key(Candidate(**task['candidate']))} ({len(rows)}/15)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(evaluate, task): task for task in tasks}
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                print(f"completed {key(Candidate(**row['candidate']))} ({len(rows)}/15)", flush=True)

    rows.sort(
        key=lambda row: (
            row["candidate"]["seed"],
            row["candidate"]["threshold"],
            row["candidate"]["anchor"],
        )
    )
    seeds = [int(row["probe_seed"]) for row in rows]
    if len(set(seeds)) != EXPECTED_CASES:
        raise RuntimeError("probe streams are not unique")
    if any(int(row["outcome_files_read"]) != 0 for row in rows):
        raise RuntimeError("an outcome read was reported")

    holdout = [row for row in rows if not row["development_row"]]
    all_summary = summary(rows)
    holdout_summary = summary(holdout)
    logical_forward = PROBES * EXPECTED_CASES
    baseline_total = int(baseline["anchor_fixed_structured_green_sweeps"])
    conditions = {
        "fourteen_of_fourteen_holdouts_issue": holdout_summary["issued"] == 14,
        "fifteen_of_fifteen_brackets_retained": all_summary["retained_sealed_bracket"] == 15,
        "fifteen_of_fifteen_path_hashes_match": all_summary["released_path_hash_matches"] == 15,
        "all_issued_rows_inside_domain": all(
            not row["issued"] or row["row_domain_passed"] for row in rows
        ),
        "zero_outcome_reads": all(row["outcome_files_read"] == 0 for row in rows),
        "complete_denominator": len(rows) == EXPECTED_CASES,
        "exactly_sixty_forward_and_zero_transpose_applications": logical_forward == 60,
    }
    payload = {
        "status": "frozen causal row-Green released-composition audit complete",
        "evidence_boundary": (
            "Post-release systems composition on already sealed brackets; one "
            "development row and 14 held-out composition rows; no future outcome read."
        ),
        "source_hashes": source_hashes,
        "audit_script_sha256": script_hash,
        "family_failure_upper": FAMILY_FAILURE_UPPER,
        "candidate_failure_upper": FAMILY_FAILURE_UPPER / EXPECTED_CASES,
        "declared_operator_count": EXPECTED_CASES,
        "unique_probe_streams": len(set(seeds)),
        "all_cases": all_summary,
        "nondevelopment_cases": holdout_summary,
        "logical_forward_green_applications": logical_forward,
        "logical_transpose_green_applications": 0,
        "logical_total_green_applications": logical_forward,
        "baseline_logical_total_green_applications": baseline_total,
        "logical_green_reduction": baseline_total / logical_forward,
        "promotion_conditions": conditions,
        "promotion_passed": all(conditions.values()),
        "wall_seconds": time.perf_counter() - started,
        "cache_hits": sum(row["cache_hit"] for row in rows),
        "rows": rows,
    }
    write_json(args.output, payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "rows" and key != "source_hashes"}, indent=2))


if __name__ == "__main__":
    main()
