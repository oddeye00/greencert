#!/usr/bin/env python3
"""Sealed, trigger-only GreenCert confirmation on a non-modular real dataset."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import platform
import secrets
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from probe_jacobian_bound import ProbeConfig, ProbeRegistry
from real_dataset_greencert import (
    build_centerline,
    certify_candidate,
    trigger_only_anchor,
)
from real_dataset_mlp import (
    ROOT,
    RealMLPConfig,
    accuracy,
    config_dict,
    gradient,
    initialize,
    make_split,
    make_selection_split,
    objective,
    optimizer_map,
    parameter_spec,
)


PROTOCOL_PATH = ROOT / "REAL_DATA_GREENCERT_CONFIRMATION_PROTOCOL.md"
METHOD_SEAL_PATH = ROOT / "REAL_DATA_GREENCERT_METHOD_SEAL.json"
CANDIDATE_SEAL_PATH = ROOT / "REAL_DATA_GREENCERT_CANDIDATE_SEAL.json"
CERTIFICATE_SEAL_PATH = ROOT / "REAL_DATA_GREENCERT_CERTIFICATE_SEAL.json"
DEFAULT_RUN_ROOT = Path(os.environ.get("LOCALAPPDATA", str(ROOT))) / "GreenCert" / "wdbc_confirmation_v1"
EXPORT_ROOT = ROOT / "results" / "real_dataset_confirmation"

FRESH_SEEDS = tuple(range(101, 125))
THRESHOLDS = (0.90, 0.925, 0.95)
PERSISTENCE = 10
HORIZON = 300
SWEEPS = 4
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
    learning_rate=0.005,
    weight_decay=1e-3,
    steps=1_000,
    checkpoint_every=5,
    threads=1,
    dtype="float64",
)

SEALED_FILES = (
    "data/wdbc_breast_cancer.csv",
    "scripts/block_jet_bound.py",
    "scripts/matrix_free_mlp.py",
    "scripts/probe_jacobian_bound.py",
    "scripts/real_dataset_greencert.py",
    "scripts/real_dataset_jet_bound.py",
    "scripts/real_dataset_mlp.py",
    "scripts/run_real_dataset_confirmation.py",
    "scripts/test_real_dataset_confirmation_protocol.py",
    "scripts/test_real_dataset_greencert_replay.py",
    "scripts/test_real_dataset_confirmation_e2e.py",
    "scripts/test_real_dataset_mlp.py",
    "scripts/test_real_dataset_jet_bound.py",
    "scripts/transformer_green_operator.py",
    "scripts/transformer_modal_forecast.py",
    "REAL_DATA_GREENCERT_CONFIRMATION_PROTOCOL.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def canonical_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def method_hash() -> str:
    if not METHOD_SEAL_PATH.exists():
        raise RuntimeError("method seal is missing")
    return sha256_file(METHOD_SEAL_PATH)


def verify_method_seal() -> dict:
    seal = read_json(METHOD_SEAL_PATH)
    for relative, expected in seal["code_sha256"].items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"sealed file changed: {relative}: {actual} != {expected}")
    if seal["protocol_sha256"] != sha256_file(PROTOCOL_PATH):
        raise RuntimeError("protocol hash changed")
    return seal


def phase_seal(run_root: Path) -> None:
    if METHOD_SEAL_PATH.exists():
        raise RuntimeError("method seal already exists; refusing to reseal")
    existing = list(run_root.rglob("*")) if run_root.exists() else []
    if existing:
        raise RuntimeError(f"fresh run root is not empty: {run_root}")
    if not PROTOCOL_PATH.exists():
        raise RuntimeError("protocol markdown must exist before sealing")
    code_hashes = {relative: sha256_file(ROOT / relative) for relative in SEALED_FILES}
    seal = {
        "status": "FROZEN BEFORE FRESH TRAINING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "external_timestamp": None,
        "external_timestamp_note": "Local SHA-256 seal only; no priority claim relies on timestamping.",
        "run_root_policy": "%LOCALAPPDATA%/GreenCert/wdbc_confirmation_v1 (non-synced mutable artifacts)",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "code_sha256": code_hashes,
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
        "config": config_dict(BASE_CONFIG),
        "software_environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
        },
    }
    write_json(METHOD_SEAL_PATH, seal)
    run_root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(METHOD_SEAL_PATH, run_root / METHOD_SEAL_PATH.name)
    audit = {
        "status": "NO FRESH ARTIFACTS EXISTED AT METHOD SEAL",
        "run_root_was_empty": True,
        "fresh_seed_count": len(FRESH_SEEDS),
        "method_seal_sha256": method_hash(),
    }
    write_json(run_root / "no_artifact_audit.json", audit)
    print(json.dumps({"method_seal_sha256": method_hash(), **audit}, indent=2))


def _events(values: list[float], threshold: float) -> int | None:
    run = 0
    for step, value in enumerate(values):
        run = run + 1 if value >= threshold else 0
        if run >= PERSISTENCE:
            return step - PERSISTENCE + 1
    return None


def _train_one(args: tuple[int, str]) -> dict:
    seed, root_text = args
    run_root = Path(root_text)
    config = RealMLPConfig(**{**asdict(BASE_CONFIG), "seed": seed})
    torch.set_num_threads(config.threads)
    data = make_selection_split(config)
    spec = parameter_spec(config)
    parameter = initialize(config)
    checkpoints: dict[str, np.ndarray] = {}
    train_accuracy: list[float] = []
    trigger_accuracy: list[float] = []
    losses: list[float] = []
    started = time.perf_counter()
    for step in range(config.steps + 1):
        train_accuracy.append(accuracy(parameter, data["train_x"], data["train_y"], spec))
        trigger_accuracy.append(accuracy(parameter, data["trigger_x"], data["trigger_y"], spec))
        losses.append(float(objective(parameter, data["train_x"], data["train_y"], spec, config)))
        if step % config.checkpoint_every == 0:
            checkpoints[f"step_{step}"] = parameter.detach().cpu().numpy().copy()
        if step < config.steps:
            parameter = parameter - config.learning_rate * gradient(
                parameter, data["train_x"], data["train_y"], spec, config
            )

    blind = {
        "status": "OUTCOME-BLIND TRAIN/TRIGGER RECORD",
        "seed": seed,
        "config": config_dict(config),
        "dataset": data["metadata"],
        "checkpoint_steps": list(range(0, config.steps + 1, config.checkpoint_every)),
        "train_accuracy": train_accuracy,
        "trigger_accuracy": trigger_accuracy,
        "train_loss": losses,
        "method_seal_sha256": method_hash(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    blind_dir = run_root / "blind"
    checkpoint_dir = run_root / "checkpoints"
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


def phase_train(run_root: Path, workers: int) -> None:
    verify_method_seal()
    existing_blind = (
        list((run_root / "blind").glob("seed_*.json"))
        if (run_root / "blind").exists()
        else []
    )
    if existing_blind:
        raise RuntimeError("training artifacts already exist")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_train_one, [(seed, str(run_root)) for seed in FRESH_SEEDS]))
    manifest = {
        "status": "FRESH TRAINING COMPLETE; CERTIFICATION TRAJECTORIES DO NOT YET EXIST",
        "method_seal_sha256": method_hash(),
        "runs": rows,
    }
    write_json(run_root / "training_manifest.json", manifest)
    print(json.dumps({"fresh_runs": len(rows), "blind_records": len(rows), "manifest_sha256": sha256_file(run_root / "training_manifest.json")}, indent=2))


def phase_select(run_root: Path) -> None:
    verify_method_seal()
    if CANDIDATE_SEAL_PATH.exists():
        raise RuntimeError("candidate seal already exists")
    rows = []
    candidates = []
    for seed in FRESH_SEEDS:
        blind_path = run_root / "blind" / f"seed_{seed}.json"
        blind = read_json(blind_path)
        if "certificate" in json.dumps(blind).lower():
            raise RuntimeError(f"blind record leaks certification data: {blind_path}")
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
    manifest = {
        "status": "CANDIDATES FROZEN BEFORE FULL-SPLIT LOADER OR CERTIFICATION EVALUATION",
        "method_seal_sha256": method_hash(),
        "rows": rows,
        "candidates": candidates,
    }
    manifest_path = run_root / "candidates_blind.json"
    write_json(manifest_path, manifest)
    seal = {
        "status": "FROZEN BEFORE FULL-SPLIT LOADER OR CERTIFICATION EVALUATION",
        "method_seal_sha256": method_hash(),
        "candidate_manifest_sha256": sha256_file(manifest_path),
        "candidate_count": len(candidates),
        "seed_threshold_cases": len(rows),
    }
    write_json(CANDIDATE_SEAL_PATH, seal)
    shutil.copy2(CANDIDATE_SEAL_PATH, run_root / CANDIDATE_SEAL_PATH.name)
    print(json.dumps(seal, indent=2))


def candidate_identity(row: dict) -> tuple[int, ...]:
    return (92, int(row["seed"]), int(row["gate_index"]), int(row["anchor"]), SWEEPS, HORIZON)


def _certify_one(args: tuple[int, str]) -> dict:
    index, root_text = args
    run_root = Path(root_text)
    seal = verify_method_seal()
    candidate_seal = read_json(CANDIDATE_SEAL_PATH)
    manifest_path = run_root / "candidates_blind.json"
    if sha256_file(manifest_path) != candidate_seal["candidate_manifest_sha256"]:
        raise RuntimeError("candidate manifest hash mismatch")
    manifest = read_json(manifest_path)
    candidate = manifest["candidates"][index]
    allowed = [candidate_identity(row) for row in manifest["candidates"]]
    registry = ProbeRegistry(allowed, seal["master_nonce"])
    seed = int(candidate["seed"])
    config = RealMLPConfig(**{**asdict(BASE_CONFIG), "seed": seed})
    torch.set_num_threads(config.threads)
    data = make_split(config)
    spec = parameter_spec(config)
    checkpoint_path = run_root / "checkpoints" / f"seed_{seed}.checkpoints.npz"
    checkpoints = np.load(checkpoint_path)
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
            "candidate_manifest_sha256": candidate_seal["candidate_manifest_sha256"],
            "probability_budget": {
                "family_failure_probability": FAMILY_FAILURE_PROBABILITY,
                "maximum_operators": MAXIMUM_OPERATORS,
                "per_operator_failure_probability": PROBE.delta,
                "queried_operator": bool("green_probe" in result),
            },
        }
    )
    output_dir = run_root / "certificates"
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
        "queried_operator": bool("green_probe" in result),
    }


def phase_certify(run_root: Path, workers: int) -> None:
    verify_method_seal()
    if CERTIFICATE_SEAL_PATH.exists():
        raise RuntimeError("certificate seal already exists")
    if not CANDIDATE_SEAL_PATH.exists():
        raise RuntimeError("candidate seal is missing")
    manifest = read_json(run_root / "candidates_blind.json")
    existing_certificates = (
        list((run_root / "certificates").glob("*.json"))
        if (run_root / "certificates").exists()
        else []
    )
    if existing_certificates:
        raise RuntimeError("partial certificate artifacts already exist; fresh study is incomplete")
    if (run_root / "audit").exists():
        raise RuntimeError("post-outcome audit directory exists before certificate seal")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_certify_one, [(i, str(run_root)) for i in range(len(manifest["candidates"]))]))
    certificate_manifest = {
        "status": "ALL CERTIFICATE-OR-ABSTAIN RECORDS FROZEN; OUTCOMES UNOPENED",
        "method_seal_sha256": method_hash(),
        "candidate_seal_sha256": sha256_file(CANDIDATE_SEAL_PATH),
        "records": rows,
    }
    manifest_path = run_root / "certificate_manifest.json"
    write_json(manifest_path, certificate_manifest)
    seal = {
        "status": "FROZEN BEFORE OUTCOME JOIN",
        "method_seal_sha256": method_hash(),
        "candidate_seal_sha256": sha256_file(CANDIDATE_SEAL_PATH),
        "certificate_manifest_sha256": sha256_file(manifest_path),
        "certificate_count": len(rows),
        "issued_unopened": sum(row["issued"] for row in rows),
        "queried_operators": sum(row["queried_operator"] for row in rows),
        "realized_union_failure_bound": sum(row["queried_operator"] for row in rows) * PROBE.delta,
    }
    write_json(CERTIFICATE_SEAL_PATH, seal)
    shutil.copy2(CERTIFICATE_SEAL_PATH, run_root / CERTIFICATE_SEAL_PATH.name)
    print(json.dumps(seal, indent=2))


def _reconstruct_outcomes(run_root: Path) -> tuple[dict[int, dict], dict[int, float]]:
    """First materialize exact certification trajectories after certificate seal."""
    outcomes: dict[int, dict] = {}
    checkpoint_errors: dict[int, float] = {}
    outcome_dir = run_root / "audit" / "outcomes"
    outcome_dir.mkdir(parents=True, exist_ok=False)
    for seed in FRESH_SEEDS:
        config = RealMLPConfig(**{**asdict(BASE_CONFIG), "seed": seed})
        torch.set_num_threads(config.threads)
        data = make_split(config)
        spec = parameter_spec(config)
        parameter = initialize(config)
        checkpoints = np.load(run_root / "checkpoints" / f"seed_{seed}.checkpoints.npz")
        certificate_accuracy: list[float] = []
        certificate_count: list[int] = []
        maximum_checkpoint_error = 0.0
        for step in range(config.steps + 1):
            value = accuracy(
                parameter, data["certificate_x"], data["certificate_y"], spec
            )
            certificate_accuracy.append(value)
            certificate_count.append(int(round(value * len(data["certificate_y"]))))
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
            "certificate_accuracy": certificate_accuracy,
            "certificate_count": certificate_count,
            "events": {
                f"{threshold:.3f}": _events(certificate_accuracy, threshold)
                for threshold in THRESHOLDS
            },
            "maximum_checkpoint_reconstruction_error": maximum_checkpoint_error,
        }
        path = outcome_dir / f"seed_{seed}.outcomes.json"
        write_json(path, record)
        outcomes[seed] = record
        checkpoint_errors[seed] = maximum_checkpoint_error
    return outcomes, checkpoint_errors


def phase_join(run_root: Path) -> None:
    verify_method_seal()
    certificate_seal = read_json(CERTIFICATE_SEAL_PATH)
    manifest_path = run_root / "certificate_manifest.json"
    if sha256_file(manifest_path) != certificate_seal["certificate_manifest_sha256"]:
        raise RuntimeError("certificate manifest changed before join")
    certificate_manifest = read_json(manifest_path)
    audit_dir = run_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=False)
    outcomes_by_seed, checkpoint_errors = _reconstruct_outcomes(run_root)
    rows = []
    for record in certificate_manifest["records"]:
        certificate_path = run_root / "certificates" / record["path"]
        if sha256_file(certificate_path) != record["sha256"]:
            raise RuntimeError(f"certificate changed before join: {record['path']}")
        certificate = read_json(certificate_path)
        seed = int(certificate["seed"])
        threshold = float(certificate["threshold"])
        anchor = int(certificate["anchor"])
        outcomes = outcomes_by_seed[seed]
        absolute = outcomes["events"][f"{threshold:.3f}"]
        actual = None if absolute is None else int(absolute - anchor)
        issued = bool(certificate["certificate_issued"])
        event_bracket = certificate.get("certified_bracket")
        covered = (
            None
            if not issued or actual is None
            else int(event_bracket[0]) <= actual <= int(event_bracket[1])
        )
        audit = {
            "seed": seed,
            "gate_index": certificate["gate_index"],
            "threshold": threshold,
            "anchor": anchor,
            "predicted_event": certificate.get("predicted_event"),
            "actual_event": actual,
            "raw_timing_error": (
                None
                if certificate.get("predicted_event") is None or actual is None
                else int(certificate["predicted_event"]) - actual
            ),
            "certificate_issued": issued,
            "certificate_status": certificate["status"],
            "certified_bracket": event_bracket,
            "bracket_contains_actual": covered,
            "closure_statistic": certificate.get("closure_statistic"),
            "closure_slack": certificate.get("closure_slack"),
            "minimum_output_slack": certificate.get("minimum_output_slack"),
            "directional_gain_ratio": certificate.get("directional_gain_ratio"),
            "unsigned_right_inverse_closure_statistic": certificate.get(
                "unsigned_right_inverse_closure_statistic"
            ),
            "unsigned_right_inverse_certificate_issued": certificate.get(
                "unsigned_right_inverse_certificate_issued"
            ),
            "unsigned_right_inverse_certified_bracket": certificate.get(
                "unsigned_right_inverse_certified_bracket"
            ),
            "certificate_sha256": record["sha256"],
        }
        if issued:
            config = RealMLPConfig(**{**asdict(BASE_CONFIG), "seed": seed})
            torch.set_num_threads(config.threads)
            data = make_split(config)
            spec = parameter_spec(config)
            checkpoints = np.load(run_root / "checkpoints" / f"seed_{seed}.checkpoints.npz")
            parameter = torch.from_numpy(checkpoints[f"step_{anchor}"]).clone()
            paths, _ = build_centerline(
                parameter, data, spec, config, horizon=HORIZON, sweeps=SWEEPS
            )
            if hashlib.sha256(paths[-1].numpy().tobytes(order="C")).hexdigest().upper() != certificate["centerline_sha256"]:
                raise RuntimeError("centerline reconstruction hash mismatch")
            h = int(certificate["certificate_horizon"])
            center = paths[-1][: h + 1]
            exact = [parameter]
            for _ in range(h):
                exact.append(
                    optimizer_map(exact[-1], data["train_x"], data["train_y"], spec, config)
                )
            exact_path = torch.stack(exact)
            sequence_error = float(torch.linalg.vector_norm(exact_path[1:] - center[1:]))
            state_error = float(torch.linalg.vector_norm(exact_path - center, dim=1).max())
            radius = float(certificate["minimal_admissible_radius"])
            audit.update(
                {
                    "observed_sequence_error": sequence_error,
                    "observed_sequence_error_to_radius": sequence_error / radius,
                    "maximum_observed_state_error": state_error,
                    "maximum_observed_state_error_to_radius": state_error / radius,
                    "observed_sequence_tube_violation": sequence_error > radius * (1 + 1e-8) + 1e-12,
                    "observed_state_tube_violation": state_error > radius * (1 + 1e-8) + 1e-12,
                }
            )
        audit_path = audit_dir / f"candidate_{record['index']:03d}_seed_{seed}_gate_{certificate['gate_index']}.json"
        write_json(audit_path, audit)
        rows.append(audit)

    issued_rows = [row for row in rows if row["certificate_issued"]]
    covered_rows = [row for row in issued_rows if row["bracket_contains_actual"]]
    leads = [row["actual_event"] for row in issued_rows if row["actual_event"] is not None]
    widths = [
        int(row["certified_bracket"][1]) - int(row["certified_bracket"][0])
        for row in issued_rows
    ]
    candidate_manifest = read_json(run_root / "candidates_blind.json")
    queried = certificate_seal["queried_operators"]
    closure_slacks = [
        row["closure_slack"] for row in rows if row["closure_slack"] is not None
    ]
    output_slacks = [
        row["minimum_output_slack"]
        for row in issued_rows
        if row["minimum_output_slack"] is not None
    ]
    summary = {
        "status": "FRESH REAL-DATA CONFIRMATION COMPLETE",
        "fresh_seeds": len(FRESH_SEEDS),
        "seed_threshold_cases": len(FRESH_SEEDS) * len(THRESHOLDS),
        "candidates": len(candidate_manifest["candidates"]),
        "candidate_rate": len(candidate_manifest["candidates"]) / (len(FRESH_SEEDS) * len(THRESHOLDS)),
        "issued": len(issued_rows),
        "overall_issuance_rate": len(issued_rows) / (len(FRESH_SEEDS) * len(THRESHOLDS)),
        "conditional_issuance_rate": len(issued_rows) / max(len(candidate_manifest["candidates"]), 1),
        "covered": len(covered_rows),
        "conditional_coverage": len(covered_rows) / max(len(issued_rows), 1),
        "distinct_issuing_seeds": len({row["seed"] for row in issued_rows}),
        "median_certified_lead": None if not leads else float(np.median(leads)),
        "maximum_certified_lead": None if not leads else int(max(leads)),
        "median_bracket_width": None if not widths else float(np.median(widths)),
        "maximum_bracket_width": None if not widths else int(max(widths)),
        "minimum_closure_slack": None if not closure_slacks else float(min(closure_slacks)),
        "minimum_issued_output_slack": None if not output_slacks else float(min(output_slacks)),
        "raw_exact_timing_matches": sum(row["raw_timing_error"] == 0 for row in rows if row["raw_timing_error"] is not None),
        "raw_timing_comparable": sum(row["raw_timing_error"] is not None for row in rows),
        "observed_sequence_tube_violations": sum(bool(row.get("observed_sequence_tube_violation")) for row in issued_rows),
        "observed_state_tube_violations": sum(bool(row.get("observed_state_tube_violation")) for row in issued_rows),
        "maximum_checkpoint_reconstruction_error": max(checkpoint_errors.values()),
        "unsigned_right_inverse_issued": sum(
            bool(row["unsigned_right_inverse_certificate_issued"]) for row in rows
        ),
        "certificate_dispositions": dict(
            sorted(Counter(row["certificate_status"] for row in rows).items())
        ),
        "candidate_dispositions": dict(
            sorted(Counter(row["disposition"] for row in candidate_manifest["rows"]).items())
        ),
        "queried_operators": queried,
        "realized_union_failure_bound": queried * PROBE.delta,
        "method_seal_sha256": method_hash(),
        "candidate_seal_sha256": sha256_file(CANDIDATE_SEAL_PATH),
        "certificate_seal_sha256": sha256_file(CERTIFICATE_SEAL_PATH),
    }
    aggregate = {"summary": summary, "rows": rows}
    aggregate_path = run_root / "final_audit.json"
    write_json(aggregate_path, aggregate)
    summary["final_audit_sha256"] = sha256_file(aggregate_path)
    write_json(run_root / "final_summary.json", summary)
    print(json.dumps(summary, indent=2))


def phase_export(run_root: Path) -> None:
    if not (run_root / "final_audit.json").exists():
        raise RuntimeError("final audit does not exist")
    if EXPORT_ROOT.exists():
        raise RuntimeError(f"export directory already exists: {EXPORT_ROOT}")
    shutil.copytree(run_root, EXPORT_ROOT)
    for name in (METHOD_SEAL_PATH, CANDIDATE_SEAL_PATH, CERTIFICATE_SEAL_PATH):
        shutil.copy2(name, EXPORT_ROOT / name.name)
    print(json.dumps({"export": str(EXPORT_ROOT), "files": len(list(EXPORT_ROOT.rglob('*')))}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("seal", "train", "select", "certify", "join", "export"))
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.phase == "seal":
        phase_seal(args.run_root)
    elif args.phase == "train":
        phase_train(args.run_root, args.workers)
    elif args.phase == "select":
        phase_select(args.run_root)
    elif args.phase == "certify":
        phase_certify(args.run_root, args.workers)
    elif args.phase == "join":
        phase_join(args.run_root)
    elif args.phase == "export":
        phase_export(args.run_root)


if __name__ == "__main__":
    main()
