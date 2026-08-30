#!/usr/bin/env python3
"""Independent 192-bit exact-real continuation for issued digits events.

This post-seal verifier does not reuse the signed Green radius or randomized
operator bound.  It rebuilds the three-sweep reference from the stored dyadic
checkpoint, encloses the exact-real gradient-descent map with Arb-backed
network/Hessian intervals, and derives persistent count brackets directly.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import flint
import numpy as np
import torch

from digits_parity_mlp import make_split, parameter_spec
from outward_interval_certificate import PRECISION_BITS
from outward_real_dataset_confirmation import (
    certified_count_paths,
    verified_tube,
)
from real_dataset_greencert import build_centerline, persistent_bracket
from real_dataset_mlp import RealMLPConfig


ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "results" / "digits_signed_confirmation"
CACHE = ROOT / "results" / "digits_outward_cache"
BLIND_SUMMARY = ROOT / "results" / "digits_outward_blind.json"
JOINED_SUMMARY = ROOT / "results" / "digits_outward_joined.json"
METHOD_SEAL = ROOT / "DIGITS_SIGNED_METHOD_SEAL.json"
VERSION = "digits-arb-outward-v1-2026-08-25"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def config_for(method: dict, seed: int) -> RealMLPConfig:
    payload = dict(method["config"])
    payload["seed"] = seed
    return RealMLPConfig(**payload)


def cache_path(seed: int, anchor: int) -> Path:
    return CACHE / f"seed_{seed}_anchor_{anchor}.json"


def issued_candidates() -> list[dict]:
    manifest = read_json(EXPORT / "certificate_manifest.json")
    rows = []
    for record in manifest["records"]:
        if not record["issued"]:
            continue
        certificate = read_json(EXPORT / "certificates" / record["path"])
        rows.append({**certificate, "certificate_sha256": record["sha256"]})
    return rows


@torch.no_grad()
def verify_anchor(
    seed: int,
    anchor: int,
    candidates: list[dict],
    *,
    use_cache: bool = True,
) -> dict:
    method = read_json(METHOD_SEAL)
    maximum_horizon = max(int(row["certificate_horizon"]) for row in candidates)
    candidate_hashes = sorted(row["certificate_sha256"] for row in candidates)
    output = cache_path(seed, anchor)
    if use_cache and output.exists():
        cached = read_json(output)
        expected = (
            VERSION,
            sha256(METHOD_SEAL),
            candidate_hashes,
            maximum_horizon,
        )
        observed = (
            cached.get("version"),
            cached.get("method_seal_sha256"),
            cached.get("candidate_sha256"),
            cached.get("requested_horizon"),
        )
        if observed == expected:
            return cached

    config = config_for(method, seed)
    torch.set_num_threads(config.threads)
    data = make_split(config)
    spec = parameter_spec(config)
    checkpoints = np.load(EXPORT / "checkpoints" / f"seed_{seed}.checkpoints.npz")
    parameter = torch.from_numpy(checkpoints[f"step_{anchor}"]).clone()
    paths, _ = build_centerline(
        parameter,
        data,
        spec,
        config,
        horizon=int(method["horizon"]),
        sweeps=int(method["sweeps"]),
    )
    reference = paths[-1][: maximum_horizon + 1].numpy()
    started = time.perf_counter()
    radius, reached, diagnostics = verified_tube(
        reference,
        data,
        spec,
        config,
        progress=f"digits seed {seed} anchor {anchor}",
    )
    guaranteed, possible, minimum_logic_slack = certified_count_paths(
        reference[: len(radius)], radius, data, spec
    )
    events = {}
    for candidate in candidates:
        bracket = persistent_bracket(
            guaranteed,
            possible,
            int(candidate["required_correct"]),
            int(method["persistence"]),
        )
        events[str(int(candidate["gate_index"]))] = {
            "threshold": float(candidate["threshold"]),
            "green_float_bracket": candidate["certified_bracket"],
            "outward_bracket": bracket,
        }
    payload = {
        "status": "post-seal digits outward exact-real map verification",
        "scope_note": (
            "The checkpoint and reference values are exact binary floats. The "
            "192-bit Arb tube encloses the exact-real full-batch optimizer map "
            "without the signed Green radius, randomized probe, or PRNG."
        ),
        "version": VERSION,
        "method_seal_sha256": sha256(METHOD_SEAL),
        "candidate_sha256": candidate_hashes,
        "python_flint_version": flint.__version__,
        "arb_precision_bits": PRECISION_BITS,
        "seed": seed,
        "anchor": anchor,
        "requested_horizon": maximum_horizon,
        "reached_horizon": reached,
        "events": events,
        "maximum_radius": float(np.max(radius)),
        "minimum_logic_slack": minimum_logic_slack,
        "maximum_hessian_interval_row_radius": max(
            (row["hessian_interval_row_radius"] for row in diagnostics), default=0.0
        ),
        "maximum_eigen_numeric_error": max(
            (row["eigen_numeric_error"] for row in diagnostics), default=0.0
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "diagnostics": diagnostics,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(output.with_suffix(".npz"), radius=radius)
    return payload


def _task(args):
    seed, anchor, rows, use_cache = args
    return seed, anchor, verify_anchor(seed, anchor, rows, use_cache=use_cache)


def verify_all(*, use_cache: bool = True, workers: int = 1) -> dict:
    issued = issued_candidates()
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in issued:
        grouped[(int(row["seed"]), int(row["anchor"]))].append(row)
    tasks = [
        (seed, anchor, rows, use_cache)
        for (seed, anchor), rows in sorted(grouped.items())
    ]
    caches = {}
    if workers <= 1:
        for index, task in enumerate(tasks, start=1):
            seed, anchor, _, _ = task
            print(
                f"digits outward anchor {index}/{len(tasks)}: seed {seed}, anchor {anchor}",
                flush=True,
            )
            _, _, payload = _task(task)
            caches[(seed, anchor)] = payload
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_task, task): task[:2] for task in tasks}
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                seed, anchor, payload = future.result()
                caches[(seed, anchor)] = payload
                completed += 1
                print(
                    f"digits outward complete {completed}/{len(tasks)}: seed {seed}, anchor {anchor}",
                    flush=True,
                )
    rows = []
    for candidate in issued:
        cache = caches[(int(candidate["seed"]), int(candidate["anchor"]))]
        event = cache["events"][str(int(candidate["gate_index"]))]
        rows.append({
            "seed": int(candidate["seed"]),
            "gate_index": int(candidate["gate_index"]),
            "threshold": float(candidate["threshold"]),
            "anchor": int(candidate["anchor"]),
            "green_float_bracket": candidate["certified_bracket"],
            "outward_bracket": event["outward_bracket"],
            "outward_issued": event["outward_bracket"] is not None,
            "signed_only": bool(candidate["certificate_issued"])
            and not bool(candidate["unsigned_right_inverse_certificate_issued"]),
        })
    blind = {
        "status": "post-seal outward verification before outcome join within this script",
        "version": VERSION,
        "issued_green_candidates": len(issued),
        "unique_seed_anchor_tubes": len(grouped),
        "outward_retained": sum(row["outward_issued"] for row in rows),
        "rows": rows,
    }
    BLIND_SUMMARY.write_text(json.dumps(blind, indent=2) + "\n", encoding="utf-8")
    return blind


def join_outcomes() -> dict:
    blind = read_json(BLIND_SUMMARY)
    final = read_json(EXPORT / "final_audit.json")
    actual = {
        (int(row["seed"]), int(row["gate_index"])): row["actual_relative"]
        for row in final["rows"]
    }
    rows = []
    for row in blind["rows"]:
        event = actual[(row["seed"], row["gate_index"])]
        bracket = row["outward_bracket"]
        rows.append({
            **row,
            "actual_event": event,
            "outward_covered": (
                None
                if bracket is None or event is None
                else int(bracket[0]) <= int(event) <= int(bracket[1])
            ),
        })
    issued = [row for row in rows if row["outward_issued"]]
    covered = [row for row in issued if row["outward_covered"]]
    signed_only = [row for row in issued if row["signed_only"]]
    summary = {
        "status": "post-seal digits outward verification joined to sealed outcomes",
        "green_issued": len(rows),
        "outward_issued": len(issued),
        "outward_covered": len(covered),
        "outward_brackets_identical_to_green": sum(
            row["outward_bracket"] == row["green_float_bracket"] for row in issued
        ),
        "signed_only_outward_issued": len(signed_only),
        "signed_only_outward_covered": sum(row["outward_covered"] for row in signed_only),
        "maximum_outward_bracket_width": max(
            (row["outward_bracket"][1] - row["outward_bracket"][0] for row in issued),
            default=None,
        ),
    }
    result = {"summary": summary, "rows": rows}
    JOINED_SUMMARY.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("one", "blind", "join", "all"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--anchor", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    if args.phase == "one":
        if args.seed is None or args.anchor is None:
            raise ValueError("one requires --seed and --anchor")
        rows = [
            row
            for row in issued_candidates()
            if int(row["seed"]) == args.seed and int(row["anchor"]) == args.anchor
        ]
        if not rows:
            raise ValueError("no issued candidate for seed/anchor")
        result = verify_anchor(
            args.seed, args.anchor, rows, use_cache=not args.no_cache
        )
        print(json.dumps({
            "seed": result["seed"],
            "anchor": result["anchor"],
            "requested_horizon": result["requested_horizon"],
            "reached_horizon": result["reached_horizon"],
            "events": result["events"],
            "maximum_radius": result["maximum_radius"],
            "minimum_logic_slack": result["minimum_logic_slack"],
            "elapsed_seconds": result["elapsed_seconds"],
            "cache": str(cache_path(result["seed"], result["anchor"])),
        }, indent=2))
    elif args.phase == "blind":
        verify_all(use_cache=not args.no_cache, workers=args.workers)
    elif args.phase == "join":
        join_outcomes()
    else:
        verify_all(use_cache=not args.no_cache, workers=args.workers)
        join_outcomes()


if __name__ == "__main__":
    main()
