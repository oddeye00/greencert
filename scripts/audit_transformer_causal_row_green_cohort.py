#!/usr/bin/env python3
"""Frozen, outcome-blind cohort audit of causal row-Green closure."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from diagnose_transformer_causal_row_green import run
from transformer_certificate_protocol import Candidate
from transformer_v3_certificate import output_path, safe_json


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PARENT = RESULTS / "transformer_fully_recentered_three_sweep_audit.json"
PROTOCOL = ROOT / "CAUSAL_ROW_GREEN_COHORT_PROTOCOL.md"
OUTPUT = RESULTS / "transformer_causal_row_green_cohort_audit.json"
CACHE = RESULTS / "transformer_causal_row_green_cohort_cache"
DEVELOPMENT = (366, 0.8, 1120)
SWEEPS = 2
PROBES = 4
DEFECT_ROUTE = "quadratic"
FAMILY_FAILURE_UPPER = 1.0e-6
EXPECTED_CASES = 15
CLAIM_FILES = (
    "CAUSAL_ROW_GREEN_THEOREM.md",
    "CAUSAL_ROW_GREEN_COHORT_PROTOCOL.md",
    "scripts/causal_row_green.py",
    "scripts/diagnose_transformer_causal_row_green.py",
    "scripts/transformer_mixed_directional_jet_v2.py",
    "scripts/test_causal_row_green.py",
    "scripts/test_causal_row_green_transformer_batch.py",
    "scripts/test_transformer_mixed_directional_jet_v2.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def candidate_from_row(row: dict) -> Candidate:
    coordinates = row["candidate"]
    return Candidate(
        int(coordinates["seed"]),
        float(coordinates["threshold"]),
        int(coordinates["anchor"]),
    )


def candidate_key(candidate: Candidate) -> str:
    return (
        f"seed_{candidate.seed}_gate_{candidate.gate_index}_"
        f"anchor_{candidate.anchor}"
    )


def cache_path(candidate: Candidate) -> Path:
    return CACHE / f"{candidate_key(candidate)}.json"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    temporary.replace(path)


def evaluate_task(task: dict) -> dict:
    candidate = Candidate(**task["candidate"])
    path = cache_path(candidate)
    if path.exists():
        cached = safe_json(path)
        if cached.get("cache_identity") == task["cache_identity"]:
            cached["cache_hit"] = True
            return cached
    result = run(
        candidate,
        SWEEPS,
        PROBES,
        DEFECT_ROUTE,
        family_delta=FAMILY_FAILURE_UPPER / EXPECTED_CASES,
    )
    result.update(
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
    write_json(path, result)
    return result


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def summarize(rows: list[dict]) -> dict:
    issued = [row for row in rows if row["issued"]]
    ratios = [
        float(row["maximum_row_radius"]) / float(row["domain_radius"])
        for row in rows
    ]
    times = [float(row["timings_seconds"]["end_to_end"]) for row in rows]
    return {
        "cases": len(rows),
        "row_domain_passed": sum(bool(row["row_domain_passed"]) for row in rows),
        "old_global_closure_passed": sum(
            bool(row["old_global_closure"]["closure_passed"]) for row in rows
        ),
        "signed_global_closure_passed": sum(
            bool(row["signed_global_closure"]["closure_passed"]) for row in rows
        ),
        "issued": len(issued),
        "retained_sealed_bracket": sum(
            bool(row["retains_sealed_bracket"]) for row in rows
        ),
        "different_issued_brackets": sum(
            bool(row["issued"]) and not bool(row["retains_sealed_bracket"])
            for row in rows
        ),
        "row_certificates_rescued_from_old_global_failure": sum(
            bool(row["issued"])
            and not bool(row["old_global_closure"]["closure_passed"])
            for row in rows
        ),
        "row_certificates_rescued_from_signed_global_failure": sum(
            bool(row["issued"])
            and not bool(row["signed_global_closure"]["closure_passed"])
            for row in rows
        ),
        "median_row_radius_to_domain_ratio": median_or_none(ratios),
        "maximum_row_radius_to_domain_ratio": max(ratios, default=None),
        "median_end_to_end_seconds": median_or_none(times),
        "total_end_to_end_seconds": math.fsum(times),
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
    parent_rows = parent["rows"]
    if len(parent_rows) != EXPECTED_CASES:
        raise RuntimeError(
            f"frozen cohort size changed: {len(parent_rows)} != {EXPECTED_CASES}"
        )
    candidates = [candidate_from_row(row) for row in parent_rows]
    if len(set(candidates)) != EXPECTED_CASES:
        raise RuntimeError("frozen cohort contains duplicate candidates")
    if sum(
        (candidate.seed, candidate.threshold, candidate.anchor) == DEVELOPMENT
        for candidate in candidates
    ) != 1:
        raise RuntimeError("development-row declaration does not match the cohort")

    source_hashes = {name: sha256(ROOT / name) for name in CLAIM_FILES}
    source_hashes[str(PARENT.relative_to(ROOT)).replace("\\", "/")] = sha256(PARENT)
    script_hash = sha256(Path(__file__))
    tasks = []
    probe_seeds = set()
    for candidate in candidates:
        certificate_hash = sha256(output_path(candidate))
        identity_payload = {
            "candidate": candidate.__dict__,
            "sweeps": SWEEPS,
            "probes": PROBES,
            "defect_route": DEFECT_ROUTE,
            "family_delta": FAMILY_FAILURE_UPPER / EXPECTED_CASES,
            "certificate_sha256": certificate_hash,
            "source_hashes": source_hashes,
            "audit_script_sha256": script_hash,
        }
        cache_identity = hashlib.sha256(
            json.dumps(identity_payload, sort_keys=True).encode("utf-8")
        ).hexdigest().upper()
        tasks.append(
            {
                "candidate": candidate.__dict__,
                "cache_identity": cache_identity,
            }
        )

    rows = []
    if args.workers == 1:
        for task in tasks:
            rows.append(evaluate_task(task))
            print(
                f"completed {candidate_key(Candidate(**task['candidate']))} "
                f"({len(rows)}/{len(tasks)})",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            future_to_task = {
                pool.submit(evaluate_task, task): task for task in tasks
            }
            for future in as_completed(future_to_task):
                row = future.result()
                rows.append(row)
                print(
                    f"completed {candidate_key(Candidate(**row['candidate']))} "
                    f"({len(rows)}/{len(tasks)})",
                    flush=True,
                )

    rows.sort(
        key=lambda row: (
            int(row["candidate"]["seed"]),
            float(row["candidate"]["threshold"]),
            int(row["candidate"]["anchor"]),
        )
    )
    for row in rows:
        if int(row["outcome_files_read"]) != 0:
            raise RuntimeError("a cohort row reports an outcome read")
        seed = int(row["probe_seed"])
        if seed in probe_seeds:
            raise RuntimeError("two candidate operators share a probe stream")
        probe_seeds.add(seed)

    holdout = [row for row in rows if not row["development_row"]]
    all_summary = summarize(rows)
    holdout_summary = summarize(holdout)
    promotion_conditions = {
        "all_issued_rows_inside_domain": all(
            not row["issued"] or row["row_domain_passed"] for row in rows
        ),
        "no_issued_bracket_changed": all(
            not row["issued"] or row["retains_sealed_bracket"] for row in rows
        ),
        "at_least_three_holdout_certificates": holdout_summary["issued"] >= 3,
        "zero_outcome_reads": all(row["outcome_files_read"] == 0 for row in rows),
        "all_denominators_present": len(rows) == EXPECTED_CASES,
    }
    promotion_passed = all(promotion_conditions.values())
    payload = {
        "status": "frozen causal row-Green Transformer cohort audit complete",
        "evidence_boundary": (
            "The method was frozen on one disclosed development row before "
            "the remaining 14 rows were executed; no future outcome was read."
        ),
        "source_hashes": source_hashes,
        "audit_script_sha256": script_hash,
        "family_failure_upper": FAMILY_FAILURE_UPPER,
        "candidate_failure_upper": FAMILY_FAILURE_UPPER / EXPECTED_CASES,
        "declared_operator_count": EXPECTED_CASES,
        "unique_probe_streams": len(probe_seeds),
        "sweeps": SWEEPS,
        "probes_per_operator": PROBES,
        "defect_route": DEFECT_ROUTE,
        "all_cases": all_summary,
        "nondevelopment_cases": holdout_summary,
        "promotion_conditions": promotion_conditions,
        "promotion_passed": promotion_passed,
        "wall_seconds": time.perf_counter() - started,
        "cache_hits": sum(bool(row["cache_hit"]) for row in rows),
        "rows": rows,
    }
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "all_cases": all_summary,
                "nondevelopment_cases": holdout_summary,
                "promotion_conditions": promotion_conditions,
                "promotion_passed": promotion_passed,
                "wall_seconds": payload["wall_seconds"],
                "cache_hits": payload["cache_hits"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
