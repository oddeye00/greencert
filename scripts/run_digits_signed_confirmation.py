#!/usr/bin/env python3
"""Sealed outcome-blind signed-Green confirmation on handwritten digits."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import secrets
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
import torch

from digits_parity_mlp import (
    initialize,
    make_selection_split,
    make_split,
    parameter_spec,
    raw_data_sha256,
)
from probe_jacobian_bound import ProbeConfig, ProbeRegistry
from real_dataset_greencert import certify_candidate, trigger_only_anchor
from real_dataset_mlp import RealMLPConfig, accuracy, gradient, objective, optimizer_map


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "DIGITS_SIGNED_CONFIRMATION_PROTOCOL.md"
METHOD_SEAL_PATH = ROOT / "DIGITS_SIGNED_METHOD_SEAL.json"
CANDIDATE_SEAL_PATH = ROOT / "DIGITS_SIGNED_CANDIDATE_SEAL.json"
CERTIFICATE_SEAL_PATH = ROOT / "DIGITS_SIGNED_CERTIFICATE_SEAL.json"
RUN_ROOT = ROOT / "results" / "digits_signed_confirmation"
SUMMARY_PATH = ROOT / "results" / "digits_signed_confirmation_summary.json"

FRESH_SEEDS = tuple(range(501, 513))
THRESHOLDS = (0.90, 0.925)
PERSISTENCE = 10
HORIZON = 400
SWEEPS = 3
MINIMUM_TRAIN_ACCURACY = 0.80
TRIGGER_BAND = 0.10
FAMILY_FAILURE_PROBABILITY = 1e-6
MAXIMUM_OPERATORS = len(FRESH_SEEDS) * len(THRESHOLDS)
PROBE = ProbeConfig(
    probes=8,
    power=4,
    delta=FAMILY_FAILURE_PROBABILITY / MAXIMUM_OPERATORS,
)
BASE_CONFIG = RealMLPConfig(
    width=8,
    learning_rate=0.03,
    weight_decay=1e-3,
    steps=600,
    checkpoint_every=5,
    seed=0,
    threads=1,
    dtype="float64",
)

SEALED_FILES = (
    "scripts/block_jet_bound.py",
    "scripts/digits_parity_mlp.py",
    "scripts/matrix_free_mlp.py",
    "scripts/probe_jacobian_bound.py",
    "scripts/real_dataset_greencert.py",
    "scripts/real_dataset_jet_bound.py",
    "scripts/real_dataset_mlp.py",
    "scripts/run_digits_signed_confirmation.py",
    "scripts/test_digits_parity_mlp.py",
    "scripts/test_digits_signed_confirmation_protocol.py",
    "scripts/transformer_green_operator.py",
    "scripts/transformer_modal_forecast.py",
    "DIGITS_SIGNED_CONFIRMATION_PROTOCOL.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def seed_config(seed: int) -> RealMLPConfig:
    return RealMLPConfig(**{**asdict(BASE_CONFIG), "seed": seed})


def method_hash() -> str:
    return sha256_file(METHOD_SEAL_PATH)


def verify_method_seal() -> dict:
    seal = read_json(METHOD_SEAL_PATH)
    if seal["raw_data_sha256"] != raw_data_sha256():
        raise RuntimeError("the bundled digits data changed after the method seal")
    for relative, expected in seal["code_sha256"].items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"sealed file changed: {relative}: {actual} != {expected}")
    return seal


def phase_seal() -> None:
    if any(path.exists() for path in (METHOD_SEAL_PATH, CANDIDATE_SEAL_PATH, CERTIFICATE_SEAL_PATH)):
        raise RuntimeError("a digits confirmation seal already exists")
    if RUN_ROOT.exists() and any(RUN_ROOT.rglob("*")):
        raise RuntimeError("fresh digits run root is not empty")
    missing = [relative for relative in SEALED_FILES if not (ROOT / relative).exists()]
    if missing:
        raise RuntimeError(f"sealed files are missing: {missing}")
    seal = {
        "status": "FROZEN BEFORE FRESH TRAINING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "development_disclosure": "Seeds 0--2 were inspected; fresh seeds 501--512 were untouched.",
        "external_timestamp": None,
        "external_timestamp_note": "Local SHA-256 seal; no priority claim relies on timestamping.",
        "raw_data_sha256": raw_data_sha256(),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "code_sha256": {relative: sha256_file(ROOT / relative) for relative in SEALED_FILES},
        "master_nonce": secrets.token_hex(32),
        "fresh_seeds": FRESH_SEEDS,
        "thresholds": THRESHOLDS,
        "persistence": PERSISTENCE,
        "horizon": HORIZON,
        "sweeps": SWEEPS,
        "minimum_train_accuracy": MINIMUM_TRAIN_ACCURACY,
        "trigger_band": TRIGGER_BAND,
        "family_failure_probability": FAMILY_FAILURE_PROBABILITY,
        "maximum_probabilistic_operators": MAXIMUM_OPERATORS,
        "probe": asdict(PROBE),
        "config": asdict(BASE_CONFIG),
        "software_environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
        },
    }
    write_json(METHOD_SEAL_PATH, seal)
    RUN_ROOT.mkdir(parents=True, exist_ok=False)
    shutil.copy2(METHOD_SEAL_PATH, RUN_ROOT / METHOD_SEAL_PATH.name)
    write_json(
        RUN_ROOT / "no_artifact_audit.json",
        {
            "status": "NO FRESH ARTIFACTS EXISTED AT METHOD SEAL",
            "run_root_was_empty": True,
            "method_seal_sha256": method_hash(),
            "fresh_seed_count": len(FRESH_SEEDS),
        },
    )
    print(json.dumps({"method_seal_sha256": method_hash(), "fresh_seeds": FRESH_SEEDS}, indent=2))


def _train_one(args: tuple[int, str]) -> dict:
    seed, root_text = args
    root = Path(root_text)
    config = seed_config(seed)
    torch.set_num_threads(config.threads)
    data = make_selection_split(config)
    spec = parameter_spec(config)
    parameter = initialize(config)
    train_accuracy: list[float] = []
    trigger_accuracy: list[float] = []
    train_loss: list[float] = []
    checkpoints: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    for step in range(config.steps + 1):
        train_accuracy.append(accuracy(parameter, data["train_x"], data["train_y"], spec))
        trigger_accuracy.append(accuracy(parameter, data["trigger_x"], data["trigger_y"], spec))
        train_loss.append(float(objective(parameter, data["train_x"], data["train_y"], spec, config)))
        if step % config.checkpoint_every == 0:
            checkpoints[f"step_{step}"] = parameter.detach().cpu().numpy().copy()
        if step < config.steps:
            parameter = parameter - config.learning_rate * gradient(
                parameter, data["train_x"], data["train_y"], spec, config
            )
    blind = {
        "status": "OUTCOME-BLIND TRAIN/TRIGGER RECORD",
        "seed": seed,
        "config": asdict(config),
        "parameter_count": spec.size,
        "dataset": data["metadata"],
        "train_accuracy": train_accuracy,
        "trigger_accuracy": trigger_accuracy,
        "train_loss": train_loss,
        "method_seal_sha256": method_hash(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    blind_dir = root / "blind"
    checkpoint_dir = root / "checkpoints"
    blind_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    blind_path = blind_dir / f"seed_{seed}.json"
    checkpoint_path = checkpoint_dir / f"seed_{seed}.checkpoints.npz"
    write_json(blind_path, blind)
    np.savez_compressed(checkpoint_path, **checkpoints)
    return {
        "seed": seed,
        "blind_sha256": sha256_file(blind_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "elapsed_seconds": blind["elapsed_seconds"],
    }


def phase_train(workers: int) -> None:
    verify_method_seal()
    if (RUN_ROOT / "training_manifest.json").exists():
        raise RuntimeError("fresh training already ran")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_train_one, [(seed, str(RUN_ROOT)) for seed in FRESH_SEEDS]))
    write_json(
        RUN_ROOT / "training_manifest.json",
        {
            "status": "FRESH TRAINING COMPLETE; CERTIFICATION TRAJECTORIES DO NOT EXIST",
            "method_seal_sha256": method_hash(),
            "runs": rows,
        },
    )
    print(json.dumps({"fresh_runs": len(rows), "total_seconds": sum(r["elapsed_seconds"] for r in rows)}, indent=2))


def phase_select() -> None:
    verify_method_seal()
    if CANDIDATE_SEAL_PATH.exists():
        raise RuntimeError("candidate seal already exists")
    rows = []
    candidates = []
    for seed in FRESH_SEEDS:
        blind = read_json(RUN_ROOT / "blind" / f"seed_{seed}.json")
        if any("certificate" in key.lower() for key in blind):
            raise RuntimeError("blind record contains a certification field")
        for gate_index, threshold in enumerate(THRESHOLDS):
            anchor = trigger_only_anchor(
                blind["train_accuracy"],
                blind["trigger_accuracy"],
                threshold=threshold,
                checkpoint_every=BASE_CONFIG.checkpoint_every,
                minimum_train_accuracy=MINIMUM_TRAIN_ACCURACY,
                trigger_band=TRIGGER_BAND,
            )
            row = {
                "seed": seed,
                "gate_index": gate_index,
                "threshold": threshold,
                "anchor": anchor,
                "disposition": "candidate frozen" if anchor is not None else "no trigger-only anchor",
            }
            rows.append(row)
            if anchor is not None:
                candidates.append(row)
    manifest_path = RUN_ROOT / "candidates_blind.json"
    write_json(
        manifest_path,
        {
            "status": "CANDIDATES FROZEN BEFORE FULL-SPLIT LOADER",
            "method_seal_sha256": method_hash(),
            "rows": rows,
            "candidates": candidates,
        },
    )
    seal = {
        "status": "FROZEN BEFORE FULL-SPLIT LOADER OR CERTIFICATION EVALUATION",
        "method_seal_sha256": method_hash(),
        "candidate_manifest_sha256": sha256_file(manifest_path),
        "candidate_count": len(candidates),
        "seed_threshold_cases": len(rows),
    }
    write_json(CANDIDATE_SEAL_PATH, seal)
    shutil.copy2(CANDIDATE_SEAL_PATH, RUN_ROOT / CANDIDATE_SEAL_PATH.name)
    print(json.dumps(seal, indent=2))


def candidate_identity(row: dict) -> tuple[int, ...]:
    return (417, int(row["seed"]), int(row["gate_index"]), int(row["anchor"]), SWEEPS, HORIZON)


def _certify_one(args: tuple[int, str]) -> dict:
    index, root_text = args
    root = Path(root_text)
    method = verify_method_seal()
    candidate_seal = read_json(CANDIDATE_SEAL_PATH)
    manifest_path = root / "candidates_blind.json"
    if sha256_file(manifest_path) != candidate_seal["candidate_manifest_sha256"]:
        raise RuntimeError("candidate manifest hash mismatch")
    manifest = read_json(manifest_path)
    candidate = manifest["candidates"][index]
    registry = ProbeRegistry(
        [candidate_identity(row) for row in manifest["candidates"]], method["master_nonce"]
    )
    seed = int(candidate["seed"])
    config = seed_config(seed)
    torch.set_num_threads(config.threads)
    data = make_split(config)
    spec = parameter_spec(config)
    checkpoints = np.load(root / "checkpoints" / f"seed_{seed}.checkpoints.npz")
    anchor = int(candidate["anchor"])
    parameter = torch.from_numpy(checkpoints[f"step_{anchor}"]).clone()
    result = certify_candidate(
        parameter,
        data,
        spec,
        config,
        seed=seed,
        gate_index=int(candidate["gate_index"]),
        threshold=float(candidate["threshold"]),
        anchor=anchor,
        horizon=HORIZON,
        persistence=PERSISTENCE,
        sweeps=SWEEPS,
        probe=PROBE,
        registry=registry,
        identity=candidate_identity(candidate),
    )
    result.update(
        {
            "status_tier": "FRESH OUTCOME-BLIND HIGH-CONFIDENCE FLOAT64",
            "method_seal_sha256": method_hash(),
            "candidate_seal_sha256": sha256_file(CANDIDATE_SEAL_PATH),
            "probability_budget": {
                "family_failure_probability": FAMILY_FAILURE_PROBABILITY,
                "maximum_operators": MAXIMUM_OPERATORS,
                "per_operator_failure_probability": PROBE.delta,
                "queried_operator": bool("green_probe" in result),
            },
        }
    )
    output_dir = root / "certificates"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"candidate_{index:03d}_seed_{seed}_gate_{candidate['gate_index']}.json"
    write_json(output, result)
    return {
        "index": index,
        "seed": seed,
        "gate_index": candidate["gate_index"],
        "path": output.name,
        "sha256": sha256_file(output),
        "issued": bool(result["certificate_issued"]),
        "unsigned_issued": bool(result.get("unsigned_right_inverse_certificate_issued")),
        "queried_operator": bool("green_probe" in result),
    }


def phase_certify(workers: int) -> None:
    verify_method_seal()
    if CERTIFICATE_SEAL_PATH.exists():
        raise RuntimeError("certificate seal already exists")
    if not CANDIDATE_SEAL_PATH.exists():
        raise RuntimeError("candidate seal is missing")
    manifest = read_json(RUN_ROOT / "candidates_blind.json")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_certify_one, [(i, str(RUN_ROOT)) for i in range(len(manifest["candidates"]))]))
    manifest_path = RUN_ROOT / "certificate_manifest.json"
    write_json(
        manifest_path,
        {
            "status": "ALL CERTIFICATE-OR-ABSTAIN RECORDS FROZEN; OUTCOMES UNOPENED",
            "method_seal_sha256": method_hash(),
            "candidate_seal_sha256": sha256_file(CANDIDATE_SEAL_PATH),
            "records": rows,
        },
    )
    seal = {
        "status": "FROZEN BEFORE OUTCOME JOIN",
        "method_seal_sha256": method_hash(),
        "candidate_seal_sha256": sha256_file(CANDIDATE_SEAL_PATH),
        "certificate_manifest_sha256": sha256_file(manifest_path),
        "certificate_count": len(rows),
        "issued_unopened": sum(row["issued"] for row in rows),
        "unsigned_issued_unopened": sum(row["unsigned_issued"] for row in rows),
        "signed_only_unopened": sum(row["issued"] and not row["unsigned_issued"] for row in rows),
        "queried_operators": sum(row["queried_operator"] for row in rows),
        "realized_union_failure_bound": sum(row["queried_operator"] for row in rows) * PROBE.delta,
    }
    write_json(CERTIFICATE_SEAL_PATH, seal)
    shutil.copy2(CERTIFICATE_SEAL_PATH, RUN_ROOT / CERTIFICATE_SEAL_PATH.name)
    print(json.dumps(seal, indent=2))


def first_persistent(values: list[float], threshold: float) -> int | None:
    run = 0
    for step, value in enumerate(values):
        run = run + 1 if value >= threshold else 0
        if run >= PERSISTENCE:
            return step - PERSISTENCE + 1
    return None


def _reconstruct_outcomes() -> tuple[dict[int, dict], dict[int, float]]:
    outcomes: dict[int, dict] = {}
    checkpoint_errors: dict[int, float] = {}
    outcome_dir = RUN_ROOT / "audit" / "outcomes"
    outcome_dir.mkdir(parents=True, exist_ok=False)
    for seed in FRESH_SEEDS:
        config = seed_config(seed)
        torch.set_num_threads(config.threads)
        data = make_split(config)
        spec = parameter_spec(config)
        parameter = initialize(config)
        checkpoints = np.load(RUN_ROOT / "checkpoints" / f"seed_{seed}.checkpoints.npz")
        trajectory: list[float] = []
        maximum_checkpoint_error = 0.0
        for step in range(config.steps + 1):
            trajectory.append(
                accuracy(parameter, data["certificate_x"], data["certificate_y"], spec)
            )
            if step % config.checkpoint_every == 0:
                saved = torch.from_numpy(checkpoints[f"step_{step}"])
                maximum_checkpoint_error = max(
                    maximum_checkpoint_error,
                    float(torch.linalg.vector_norm(parameter - saved)),
                )
            if step < config.steps:
                parameter = optimizer_map(
                    parameter, data["train_x"], data["train_y"], spec, config
                )
        record = {
            "status": "FIRST MATERIALIZED AFTER CERTIFICATE SEAL",
            "seed": seed,
            "events": {
                f"{threshold:.3f}": first_persistent(trajectory, threshold)
                for threshold in THRESHOLDS
            },
            "certificate_accuracy": trajectory,
            "maximum_checkpoint_reconstruction_error": maximum_checkpoint_error,
        }
        write_json(outcome_dir / f"seed_{seed}.outcomes.json", record)
        outcomes[seed] = record
        checkpoint_errors[seed] = maximum_checkpoint_error
    return outcomes, checkpoint_errors


def phase_join() -> None:
    verify_method_seal()
    certificate_seal = read_json(CERTIFICATE_SEAL_PATH)
    manifest_path = RUN_ROOT / "certificate_manifest.json"
    if sha256_file(manifest_path) != certificate_seal["certificate_manifest_sha256"]:
        raise RuntimeError("certificate manifest changed before outcome join")
    (RUN_ROOT / "audit").mkdir(parents=True, exist_ok=False)
    outcomes, checkpoint_errors = _reconstruct_outcomes()
    manifest = read_json(manifest_path)
    rows = []
    for record in manifest["records"]:
        path = RUN_ROOT / "certificates" / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"certificate changed before join: {path.name}")
        certificate = read_json(path)
        seed = int(certificate["seed"])
        threshold = float(certificate["threshold"])
        anchor = int(certificate["anchor"])
        absolute = outcomes[seed]["events"][f"{threshold:.3f}"]
        actual = None if absolute is None else int(absolute - anchor)
        bracket = certificate.get("certified_bracket")
        unsigned_bracket = certificate.get("unsigned_right_inverse_certified_bracket")
        issued = bool(certificate["certificate_issued"])
        unsigned_issued = bool(certificate.get("unsigned_right_inverse_certificate_issued"))
        covered = bool(issued and actual is not None and bracket[0] <= actual <= bracket[1])
        unsigned_covered = bool(
            unsigned_issued
            and actual is not None
            and unsigned_bracket[0] <= actual <= unsigned_bracket[1]
        )
        row = {
            "seed": seed,
            "gate_index": int(certificate["gate_index"]),
            "threshold": threshold,
            "anchor": anchor,
            "actual_absolute": absolute,
            "actual_relative": actual,
            "issued": issued,
            "bracket": bracket,
            "covered": covered,
            "unsigned_issued": unsigned_issued,
            "unsigned_bracket": unsigned_bracket,
            "unsigned_covered": unsigned_covered,
            "signed_only": bool(issued and not unsigned_issued),
            "predicted_event": certificate.get("predicted_event"),
            "closure_statistic": certificate.get("closure_statistic"),
            "unsigned_closure_statistic": certificate.get("unsigned_right_inverse_closure_statistic"),
            "directional_gain_ratio": certificate.get("directional_gain_ratio"),
            "minimum_output_slack": certificate.get("minimum_output_slack"),
            "elapsed_seconds": certificate.get("elapsed_seconds"),
        }
        rows.append(row)
        write_json(RUN_ROOT / "audit" / record["path"], row)

    issued_rows = [row for row in rows if row["issued"]]
    signed_only_rows = [row for row in rows if row["signed_only"]]
    unsigned_rows = [row for row in rows if row["unsigned_issued"]]
    leads = [row["actual_relative"] for row in issued_rows if row["actual_relative"] is not None]
    widths = [row["bracket"][1] - row["bracket"][0] for row in issued_rows]
    summary = {
        "status": "FRESH OUTCOME-BLIND CONFIRMATION JOINED AFTER CERTIFICATE SEAL",
        "method_seal_sha256": method_hash(),
        "candidate_seal_sha256": sha256_file(CANDIDATE_SEAL_PATH),
        "certificate_seal_sha256": sha256_file(CERTIFICATE_SEAL_PATH),
        "seed_threshold_cases": len(FRESH_SEEDS) * len(THRESHOLDS),
        "candidates": len(rows),
        "signed_issued": len(issued_rows),
        "signed_covered": sum(row["covered"] for row in issued_rows),
        "signed_distinct_seeds": len({row["seed"] for row in issued_rows}),
        "unsigned_issued": len(unsigned_rows),
        "unsigned_covered": sum(row["unsigned_covered"] for row in unsigned_rows),
        "signed_only": len(signed_only_rows),
        "signed_only_covered": sum(row["covered"] for row in signed_only_rows),
        "signed_only_distinct_seeds": len({row["seed"] for row in signed_only_rows}),
        "median_lead": None if not leads else float(np.median(leads)),
        "maximum_lead": None if not leads else int(max(leads)),
        "median_bracket_width": None if not widths else float(np.median(widths)),
        "maximum_checkpoint_reconstruction_error": max(checkpoint_errors.values()),
        "realized_union_failure_bound": certificate_seal["realized_union_failure_bound"],
        "rows": rows,
    }
    write_json(RUN_ROOT / "final_audit.json", summary)
    write_json(SUMMARY_PATH, summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("seal", "train", "select", "certify", "join"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.phase == "seal":
        phase_seal()
    elif args.phase == "train":
        phase_train(args.workers)
    elif args.phase == "select":
        phase_select()
    elif args.phase == "certify":
        phase_certify(args.workers)
    else:
        phase_join()


if __name__ == "__main__":
    main()
