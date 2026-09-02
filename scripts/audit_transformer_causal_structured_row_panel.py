#!/usr/bin/env python3
"""Frozen 4/8-probe panel audit for structured causal row closure."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import torch

from combine_causal_row_probe_blocks import combine_probe_blocks
from diagnose_transformer_causal_row_green import run
from transformer_certificate_protocol import Candidate
from transformer_v3_certificate import output_path, safe_json


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BASELINE = RESULTS / "transformer_v3_relinearized_prefix_panel_audit.json"
PROTOCOL = ROOT / "CAUSAL_STRUCTURED_ROW_PANEL_PROTOCOL.md"
OUTPUT = RESULTS / "transformer_causal_structured_row_panel_audit.json"
CACHE = RESULTS / "transformer_causal_structured_row_panel_cache"
DEVELOPMENT = (373, 0.7, 1280)
EXPECTED_CASES = 15
SWEEPS = 4
BLOCK_PROBES = 4
MAXIMUM_PROBES = 8
FAMILY_FAILURE_UPPER = 1.0e-6
STAGE_DELTA = FAMILY_FAILURE_UPPER / (EXPECTED_CASES * 2)
BASELINE_SWEEPS = 144
CLAIM_FILES = (
    "CAUSAL_ROW_GREEN_THEOREM.md",
    "CAUSAL_STRUCTURED_ROW_PANEL_PROTOCOL.md",
    "scripts/causal_row_green.py",
    "scripts/diagnose_transformer_causal_row_green.py",
    "scripts/combine_causal_row_probe_blocks.py",
    "scripts/transformer_mixed_directional_jet_v2.py",
    "scripts/test_causal_row_green.py",
    "scripts/test_causal_structured_row_green.py",
    "scripts/test_combine_causal_row_probe_blocks.py",
    "scripts/test_causal_row_green_transformer_batch.py",
    "scripts/test_transformer_mixed_directional_jet_v2.py",
)
PREAUDIT_TESTS = (
    "scripts/test_causal_row_green.py",
    "scripts/test_causal_structured_row_green.py",
    "scripts/test_combine_causal_row_probe_blocks.py",
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


def frozen_git_commit() -> str:
    tracked = (*CLAIM_FILES, str(Path(__file__).relative_to(ROOT)).replace("\\", "/"))
    for name in tracked:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", name],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("tracked worktree is dirty; commit the frozen protocol first")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def run_preaudit_tests() -> list[dict]:
    records = []
    for name in PREAUDIT_TESTS:
        started = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, str(ROOT / name)],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        record = {
            "path": name,
            "exit_code": completed.returncode,
            "seconds": time.perf_counter() - started,
            "output": completed.stdout.strip(),
        }
        records.append(record)
        if completed.returncode != 0:
            raise RuntimeError(f"preaudit test failed: {name}\n{completed.stdout}")
    return records


def block_cache(candidate: Candidate, offset: int) -> Path:
    return CACHE / f"{key(candidate)}_offset_{offset}.json"


def evaluate_block(task: dict) -> dict:
    candidate = Candidate(**task["candidate"])
    offset = int(task["offset"])
    cache = block_cache(candidate, offset)
    if cache.exists():
        record = safe_json(cache)
        if record.get("cache_identity") == task["cache_identity"]:
            record["cache_hit"] = True
            return record
    record = run(
        candidate,
        SWEEPS,
        BLOCK_PROBES,
        "quadratic",
        family_delta=STAGE_DELTA,
        closure_channel="structured_parameter",
        probe_chunk_size=BLOCK_PROBES,
        probe_stream_size=MAXIMUM_PROBES,
        probe_offset=offset,
    )
    record.update(
        {
            "cache_identity": task["cache_identity"],
            "cache_hit": False,
            "development_row": (
                candidate.seed,
                candidate.threshold,
                candidate.anchor,
            )
            == DEVELOPMENT,
        }
    )
    write_json(cache, record)
    return record


def run_tasks(tasks: list[dict], workers: int) -> list[dict]:
    rows = []
    if workers == 1:
        for task in tasks:
            rows.append(evaluate_block(task))
            print(
                f"completed {key(Candidate(**task['candidate']))} "
                f"offset {task['offset']} ({len(rows)}/{len(tasks)})",
                flush=True,
            )
        return rows
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        max_tasks_per_child=1,
    ) as pool:
        futures = {pool.submit(evaluate_block, task): task for task in tasks}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"completed {key(Candidate(**row['candidate']))} "
                f"offset {row['probe_offset']} ({len(rows)}/{len(tasks)})",
                flush=True,
            )
    return rows


def make_task(
    candidate: Candidate,
    offset: int,
    source_hashes: dict[str, str],
    script_hash: str,
) -> dict:
    identity = {
        "candidate": candidate.__dict__,
        "offset": offset,
        "block_probes": BLOCK_PROBES,
        "maximum_probes": MAXIMUM_PROBES,
        "stage_delta": STAGE_DELTA,
        "certificate_sha256": sha256(output_path(candidate)),
        "source_hashes": source_hashes,
        "audit_script_sha256": script_hash,
    }
    return {
        "candidate": candidate.__dict__,
        "offset": offset,
        "cache_identity": hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest().upper(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    started = time.perf_counter()

    claim_commit = frozen_git_commit()
    preaudit_tests = run_preaudit_tests()

    baseline = safe_json(BASELINE)
    if int(baseline["new_total_theoretical_linearized_sweeps"]) != BASELINE_SWEEPS:
        raise RuntimeError("the released 144-sweep baseline changed")
    candidates = [candidate_from(row) for row in baseline["rows"]]
    if len(candidates) != EXPECTED_CASES or len(set(candidates)) != EXPECTED_CASES:
        raise RuntimeError("the declared panel changed")
    if sum(
        (candidate.seed, candidate.threshold, candidate.anchor) == DEVELOPMENT
        for candidate in candidates
    ) != 1:
        raise RuntimeError("development coordinate does not identify one row")

    source_hashes = {name: sha256(ROOT / name) for name in CLAIM_FILES}
    source_hashes[str(BASELINE.relative_to(ROOT)).replace("\\", "/")] = sha256(BASELINE)
    script_hash = sha256(Path(__file__))
    stage_one_tasks = [
        make_task(candidate, 0, source_hashes, script_hash) for candidate in candidates
    ]
    stage_one_rows = run_tasks(stage_one_tasks, args.workers)
    stage_one = {
        Candidate(**row["candidate"]): row for row in stage_one_rows
    }
    if len(stage_one) != EXPECTED_CASES:
        raise RuntimeError("stage one did not return every candidate")

    extended = [candidate for candidate in candidates if not stage_one[candidate]["issued"]]
    stage_two_tasks = [
        make_task(candidate, BLOCK_PROBES, source_hashes, script_hash)
        for candidate in extended
    ]
    stage_two_rows = run_tasks(stage_two_tasks, args.workers) if stage_two_tasks else []
    stage_two = {Candidate(**row["candidate"]): row for row in stage_two_rows}

    final_rows = []
    for candidate in candidates:
        first = stage_one[candidate]
        if candidate in stage_two:
            final = combine_probe_blocks(
                (first, stage_two[candidate]), stage_delta=STAGE_DELTA
            )
            final["prefixes_computed"] = MAXIMUM_PROBES
            final["block_cache_hits"] = [first["cache_hit"], stage_two[candidate]["cache_hit"]]
            final["measured_block_seconds"] = (
                float(first["timings_seconds"]["end_to_end"])
                + float(stage_two[candidate]["timings_seconds"]["end_to_end"])
            )
        else:
            final = dict(first)
            final["prefixes_computed"] = BLOCK_PROBES
            final["logical_forward_probe_applications"] = BLOCK_PROBES
            final["logical_signed_response_applications"] = 1
            final["logical_transpose_applications"] = 0
            final["logical_total_linearized_sweeps"] = BLOCK_PROBES + 1
            final["block_cache_hits"] = [first["cache_hit"]]
            final["measured_block_seconds"] = float(
                first["timings_seconds"]["end_to_end"]
            )
        final["development_row"] = (
            candidate.seed,
            candidate.threshold,
            candidate.anchor,
        ) == DEVELOPMENT
        final_rows.append(final)

    final_rows.sort(
        key=lambda row: (
            row["candidate"]["seed"],
            row["candidate"]["threshold"],
            row["candidate"]["anchor"],
        )
    )
    holdout = [row for row in final_rows if not row["development_row"]]
    total_sweeps = sum(int(row["logical_total_linearized_sweeps"]) for row in final_rows)
    prefixes = [int(row["prefixes_computed"]) for row in final_rows]
    conditions = {
        "fifteen_of_fifteen_issue": sum(row["issued"] for row in final_rows) == 15,
        "fourteen_of_fourteen_holdouts_issue": sum(row["issued"] for row in holdout) == 14,
        "all_brackets_retained": all(row["retains_sealed_bracket"] for row in final_rows),
        "all_released_paths_match": all(row["released_corrected_path_match"] for row in final_rows),
        "all_domains_pass": all(row["row_domain_passed"] for row in final_rows),
        "at_most_two_eight_probe_cases": sum(value == 8 for value in prefixes) <= 2,
        "at_most_eighty_three_sweeps": total_sweeps <= 83,
        "zero_transpose_sweeps": all(
            int(row["logical_transpose_applications"]) == 0 for row in final_rows
        ),
        "zero_outcome_reads": all(int(row["outcome_files_read"]) == 0 for row in final_rows),
        "complete_denominator": len(final_rows) == EXPECTED_CASES,
        "all_preaudit_tests_pass": all(
            int(row["exit_code"]) == 0 for row in preaudit_tests
        ),
    }
    measured = [float(row["measured_block_seconds"]) for row in final_rows]
    payload = {
        "status": "frozen structured causal-row Transformer panel audit complete",
        "evidence_boundary": (
            "Post-release systems/theorem audit with one disclosed development "
            "row and 14 structured-row holdouts; no future outcome read."
        ),
        "source_hashes": source_hashes,
        "audit_script_sha256": script_hash,
        "frozen_git_commit": claim_commit,
        "runtime_environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "preaudit_tests": preaudit_tests,
        "family_failure_upper": FAMILY_FAILURE_UPPER,
        "stage_delta": STAGE_DELTA,
        "cases": len(final_rows),
        "holdout_cases": len(holdout),
        "issued": sum(row["issued"] for row in final_rows),
        "holdout_issued": sum(row["issued"] for row in holdout),
        "brackets_retained": sum(row["retains_sealed_bracket"] for row in final_rows),
        "prefix_distribution": {
            "4": sum(value == 4 for value in prefixes),
            "8": sum(value == 8 for value in prefixes),
        },
        "logical_forward_probe_applications": sum(prefixes),
        "logical_signed_response_applications": EXPECTED_CASES,
        "logical_transpose_applications": 0,
        "logical_total_linearized_sweeps": total_sweeps,
        "baseline_total_linearized_sweeps": BASELINE_SWEEPS,
        "linearized_sweep_reduction": BASELINE_SWEEPS / total_sweeps,
        "median_measured_case_seconds": statistics.median(measured),
        "total_measured_case_seconds": sum(measured),
        "wall_seconds": time.perf_counter() - started,
        "promotion_conditions": conditions,
        "promotion_passed": all(conditions.values()),
        "outcome_files_read": 0,
        "rows": final_rows,
    }
    write_json(args.output, payload)
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key not in {"rows", "source_hashes"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
