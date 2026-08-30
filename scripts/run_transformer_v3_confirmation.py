#!/usr/bin/env python3
"""Freeze, execute, seal, and audit the Transformer v3 confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import secrets
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from transformer_certificate_protocol import Candidate
from transformer_four_sweep_development_audit import first_persistent, to_scaled
from transformer_green_development_audit import build_frozen_centerline
from transformer_green_operator import make_transformer_green_products
from transformer_hvp_grokking import (
    TransformerConfig,
    artifact_paths,
    flat_spec,
    logits,
    make_disjoint_split,
    make_template,
)
from transformer_modal_forecast import run as modal_run
from transformer_v3_certificate import (
    CANDIDATE_MANIFEST,
    CANDIDATE_SEAL,
    METHOD_SEAL,
    frozen_candidates,
    output_path as certificate_path,
    safe_json,
)
from transformer_v3_protocol import (
    CHECKPOINT_SPACING,
    HORIZON,
    MAX_DEFICIT,
    PERSISTENCE,
    SCAN_STEPS,
    SEEDS,
    SWEEPS,
    THRESHOLDS,
    maximum_operator_count,
    probe_config,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PROTOCOL = ROOT / "TRANSFORMER_V3_CONFIRMATION_PROTOCOL.md"
NO_ARTIFACT_AUDIT = RESULTS / "transformer_v3_no_artifact_audit.json"
CERTIFICATE_SEAL = ROOT / "TRANSFORMER_V3_CERTIFICATE_SEAL.json"
AGGREGATE = RESULTS / "transformer_v3_confirmation_audit.json"
WORKERS = 4

METHOD_FILES = (
    "TRANSFORMER_V3_CONFIRMATION_PROTOCOL.md",
    "ONE_SHOT_RECENTER_THEOREM.md",
    "scripts/run_transformer_v3_confirmation.py",
    "scripts/transformer_v3_protocol.py",
    "scripts/transformer_v3_certificate.py",
    "scripts/one_shot_recenter_closure.py",
    "scripts/batched_green_operator.py",
    "scripts/transformer_green_operator.py",
    "scripts/transformer_green_development_audit.py",
    "scripts/transformer_four_sweep_development_audit.py",
    "scripts/transformer_certificate_protocol.py",
    "scripts/transformer_block_envelope.py",
    "scripts/block_jet_bound.py",
    "scripts/probe_jacobian_bound.py",
    "scripts/transformer_optimizer_probe.py",
    "scripts/transformer_hvp_grokking.py",
    "scripts/transformer_modal_forecast.py",
    "scripts/matrix_free_mlp.py",
    "scripts/smooth_mlp_modular_grokking.py",
    "scripts/test_transformer_v3_protocol.py",
    "scripts/test_transformer_v3_preseal.py",
    "scripts/test_one_shot_signed_recenter.py",
    "scripts/test_progressive_gram_bound.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json_exclusive(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def expected_config(seed: int) -> TransformerConfig:
    return TransformerConfig(
        modulus=17,
        model_dim=32,
        hidden_dim=128,
        heads=4,
        depth=1,
        train_fraction=0.60,
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=0.01,
        steps=6_000,
        log_every=20,
        checkpoint_every=40,
        seed=seed,
        threads=4,
        dtype="float64",
        loss="cross_entropy",
        normalization="none",
    )


def training_command(seed: int) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "transformer_hvp_grokking.py"),
        "--seed", str(seed),
        "--modulus", "17",
        "--train-fraction", "0.60",
        "--steps", "6000",
        "--learning-rate", "0.01",
        "--weight-decay", "0.01",
        "--momentum", "0.9",
        "--model-dim", "32",
        "--hidden-dim", "128",
        "--log-every", "20",
        "--checkpoint-every", "40",
        "--loss", "cross_entropy",
        "--normalization", "none",
        "--prospective",
    ]


def _v3_log(seed: int) -> Path:
    return RESULTS / f"transformer_v3_training_seed_{seed}.sealed.log"


def _v3_certificate_log(candidate: Candidate) -> Path:
    return RESULTS / (
        f"transformer_v3_certificate_seed_{candidate.seed}_"
        f"gate_{candidate.gate_index}_anchor_{candidate.anchor}.process.log"
    )


def _v3_audit_path(candidate: Candidate) -> Path:
    return RESULTS / (
        f"transformer_v3_audit_seed_{candidate.seed}_"
        f"gate_{candidate.gate_index}_anchor_{candidate.anchor}.json"
    )


def fresh_targets() -> list[Path]:
    targets: list[Path] = []
    for seed in SEEDS:
        blind, checkpoint = artifact_paths(seed, development=False)
        targets.extend(
            [
                blind,
                blind.with_name(blind.stem + ".outcomes.json"),
                checkpoint,
                _v3_log(seed),
            ]
        )
        targets.extend(RESULTS.glob(f"transformer_modal_prospective_blind_seed_{seed}_*.json"))
        # The fresh-population claim is stronger than absence of the expected
        # v3 filenames.  Reject any earlier result whose filename identifies
        # one of the frozen seeds, including development or abandoned runs.
        targets.extend(RESULTS.rglob(f"*seed_{seed}*"))
        targets.extend(RESULTS.rglob(f"*seed-{seed}*"))
    targets.extend(
        [METHOD_SEAL, CANDIDATE_MANIFEST, CANDIDATE_SEAL, CERTIFICATE_SEAL, AGGREGATE]
    )
    targets.extend(RESULTS.glob("transformer_v3_certificate_seed_*.json"))
    targets.extend(RESULTS.glob("transformer_v3_audit_seed_*.json"))
    targets.extend(RESULTS.glob("transformer_v3_cache/*"))
    return targets


def freeze_method() -> dict:
    if METHOD_SEAL.exists() or NO_ARTIFACT_AUDIT.exists():
        raise FileExistsError("v3 method/no-artifact seal already exists")
    existing = sorted(
        {str(path.relative_to(ROOT)) for path in fresh_targets() if path.exists()}
    )
    audit = {
        "status": "PRE-TRAINING V3 NO-ARTIFACT AUDIT",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "fresh_seeds": list(SEEDS),
        "existing_target_artifacts": existing,
        "passed": not existing,
    }
    write_json_exclusive(NO_ARTIFACT_AUDIT, audit)
    if existing:
        raise RuntimeError(f"v3 fresh artifact audit failed: {existing}")
    missing = [name for name in METHOD_FILES if not (ROOT / name).exists()]
    if missing:
        raise FileNotFoundError(f"v3 method files missing: {missing}")
    payload = {
        "status": "FROZEN BEFORE V3 FRESH TRAINING",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha256(PROTOCOL),
        "no_artifact_audit": str(NO_ARTIFACT_AUDIT.relative_to(ROOT)),
        "no_artifact_audit_sha256": sha256(NO_ARTIFACT_AUDIT),
        "fresh_seeds": list(SEEDS),
        "thresholds": list(THRESHOLDS),
        "master_nonce": secrets.token_hex(32),
        "probe_config": asdict(probe_config()),
        "operator_accounting": maximum_operator_count(),
        "code_manifest": {name: sha256(ROOT / name) for name in METHOD_FILES},
        "single_entry_command": "python scripts/run_transformer_v3_confirmation.py --all",
        "training_commands": [training_command(seed)[1:] for seed in SEEDS],
        "primary_analysis": (
            "first q in 1..8 closing the conservative one-shot response-centered "
            "theorem and persistent output bracket"
        ),
        "matched_baseline": "fixed q=8, R=2Z signed-Green rule",
        "evidence_boundary": (
            "method frozen before seeds 355--378; candidate/certificate phases "
            "reject outcome files; certificate hashes freeze before exact rollout"
        ),
    }
    write_json_exclusive(METHOD_SEAL, payload)
    payload["sha256"] = sha256(METHOD_SEAL)
    return payload


def verify_method_seal() -> dict:
    seal = safe_json(METHOD_SEAL)
    if seal["protocol_sha256"] != sha256(PROTOCOL):
        raise RuntimeError("v3 protocol hash changed")
    if seal["no_artifact_audit_sha256"] != sha256(NO_ARTIFACT_AUDIT):
        raise RuntimeError("v3 no-artifact audit hash changed")
    for name, expected in seal["code_manifest"].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"v3 sealed file changed: {name}")
    return seal


def run_logged(command: list[str], log_path: Path) -> str:
    if log_path.exists():
        raise FileExistsError(f"refusing to overwrite process log: {log_path}")
    with log_path.open("wb") as stream:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"subprocess failed with code {completed.returncode}: {log_path}"
        )
    return str(log_path.relative_to(ROOT))


def train_one(seed: int) -> str:
    blind, checkpoint = artifact_paths(seed, development=False)
    outcome = blind.with_name(blind.stem + ".outcomes.json")
    log = _v3_log(seed)
    existing = [path.exists() for path in (blind, checkpoint, outcome, log)]
    if all(existing):
        payload = safe_json(blind)
        if payload["config"] != asdict(expected_config(seed)):
            raise RuntimeError(f"existing v3 seed {seed} has the wrong config")
        return str(blind.relative_to(ROOT))
    if any(existing):
        raise RuntimeError(f"partial v3 training artifacts for seed {seed}")
    run_logged(training_command(seed), log)
    if not all(path.exists() for path in (blind, checkpoint, outcome)):
        raise RuntimeError(f"v3 training seed {seed} omitted an artifact")
    return str(blind.relative_to(ROOT))


def train_all() -> list[str]:
    verify_method_seal()
    outputs = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(train_one, seed): seed for seed in SEEDS}
        for future in as_completed(futures):
            outputs.append(future.result())
    return sorted(outputs)


def first_eligible(trajectory: np.ndarray, threshold: float) -> int | None:
    trigger_gate = max(0.50, threshold - 0.20)
    indices = np.flatnonzero(
        (trajectory[:, 1] >= 0.99) & (trajectory[:, 2] >= trigger_gate)
    )
    return None if len(indices) == 0 else int(trajectory[indices[0], 0])


def round_up(step: int) -> int:
    return ((step + CHECKPOINT_SPACING - 1) // CHECKPOINT_SPACING) * CHECKPOINT_SPACING


def certificate_count(checkpoints, step, pairs, labels, template, spec) -> int:
    parameter = torch.from_numpy(checkpoints[f"step_{step}"]).to(torch.float64)
    return int((logits(parameter, pairs, template, spec).argmax(1) == labels).sum())


def scan_seed(seed: int) -> list[dict]:
    verify_method_seal()
    blind_path, checkpoint_path = artifact_paths(seed, development=False)
    payload = safe_json(blind_path)
    if payload["config"] != asdict(expected_config(seed)):
        raise RuntimeError(f"v3 seed {seed} config differs from protocol")
    if any("certificate" in name.lower() for name in payload["trajectory_columns"]):
        raise RuntimeError("v3 scanner received a certification outcome")
    config = TransformerConfig(**payload["config"])
    torch.set_num_threads(config.threads)
    trajectory = np.asarray(payload["trajectory"], dtype=np.float64)
    checkpoints = dict(np.load(checkpoint_path))
    template = make_template(config)
    spec = flat_spec(template)
    cert_pairs, cert_labels = make_disjoint_split(config)[4:]
    forecast_cache: dict[int, dict] = {}
    records = []

    for threshold in THRESHOLDS:
        required = int(math.ceil(threshold * len(cert_pairs)))
        eligible = first_eligible(trajectory, threshold)
        record = {
            "seed": seed,
            "threshold": threshold,
            "required": required,
            "eligibility_step": eligible,
            "anchor": None,
            "predicted_offset": None,
            "horizon": None,
            "disposition": None,
            "forecast_file": None,
            "forecast_sha256": None,
        }
        if eligible is None:
            record["disposition"] = "trigger never eligible"
            records.append(record)
            continue
        start = round_up(eligible)
        stop = min(start + SCAN_STEPS, config.steps)
        for anchor in range(start, stop + 1, CHECKPOINT_SPACING):
            current = certificate_count(
                checkpoints, anchor, cert_pairs, cert_labels, template, spec
            )
            deficit = required - current
            if deficit <= 0:
                record["disposition"] = "gate already present before candidate"
                break
            if deficit > MAX_DEFICIT:
                continue
            if anchor not in forecast_cache:
                forecast_cache[anchor] = modal_run(
                    seed,
                    anchor,
                    horizon=HORIZON,
                    sweeps=SWEEPS,
                    persistence=PERSISTENCE,
                    development=False,
                    evaluate_actual=False,
                )
            forecast = forecast_cache[anchor]
            event = forecast["events"][f"{threshold:.2f}"]["recentered"]
            if event is None or event <= 0:
                continue
            horizon = int(event) + PERSISTENCE - 1
            if horizon > HORIZON:
                raise RuntimeError("v3 modal event exceeds the frozen horizon")
            forecast_path = Path(forecast["output"])
            record.update(
                {
                    "anchor": anchor,
                    "predicted_offset": int(event),
                    "horizon": horizon,
                    "disposition": "candidate frozen",
                    "forecast_file": str(forecast_path.relative_to(ROOT)),
                    "forecast_sha256": sha256(forecast_path),
                }
            )
            break
        if record["disposition"] is None:
            record["disposition"] = "screened; no future modal candidate"
        records.append(record)
    return records


def blind_scan_all() -> dict:
    method = verify_method_seal()
    if CANDIDATE_MANIFEST.exists() or CANDIDATE_SEAL.exists():
        raise FileExistsError("v3 candidate manifest/seal already exists")
    records = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(scan_seed, seed): seed for seed in SEEDS}
        for future in as_completed(futures):
            records.extend(future.result())
    records.sort(key=lambda row: (int(row["seed"]), float(row["threshold"])))
    manifest = {
        "status": "FROZEN V3 CANDIDATES; OUTCOMES UNOPENED",
        "method_seal_sha256": sha256(METHOD_SEAL),
        "fresh_seeds": list(SEEDS),
        "thresholds": list(THRESHOLDS),
        "records": records,
    }
    write_json_exclusive(CANDIDATE_MANIFEST, manifest)
    selected = [row for row in records if row["disposition"] == "candidate frozen"]
    candidates = [
        {
            "seed": int(row["seed"]),
            "threshold": float(row["threshold"]),
            "anchor": int(row["anchor"]),
            "horizon": int(row["horizon"]),
        }
        for row in selected
    ]
    seal = {
        "status": "V3 CANDIDATES FROZEN BEFORE CERTIFICATION PROBES",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method_seal_sha256": sha256(METHOD_SEAL),
        "protocol_sha256": method["protocol_sha256"],
        "candidate_manifest": str(CANDIDATE_MANIFEST.relative_to(ROOT)),
        "candidate_manifest_sha256": sha256(CANDIDATE_MANIFEST),
        "seed_threshold_cases": len(records),
        "candidates": candidates,
        "distinct_candidate_seeds": len({row["seed"] for row in candidates}),
        "information_barrier": "outcome JSONs and training logs unopened",
    }
    write_json_exclusive(CANDIDATE_SEAL, seal)
    seal["sha256"] = sha256(CANDIDATE_SEAL)
    return seal


def certificate_command(candidate: Candidate) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "transformer_v3_certificate.py"),
        "--seed", str(candidate.seed),
        "--threshold", str(candidate.threshold),
        "--anchor", str(candidate.anchor),
    ]


def certify_all() -> dict:
    verify_method_seal()
    if CERTIFICATE_SEAL.exists():
        raise FileExistsError("v3 certificate seal already exists")
    candidates, _, _ = frozen_candidates()

    def one(candidate: Candidate) -> str:
        path = certificate_path(candidate)
        log = _v3_certificate_log(candidate)
        if path.exists() and log.exists():
            payload = safe_json(path)
            if payload["candidate"] != candidate.__dict__ or payload["outcome_joined"]:
                raise RuntimeError(f"invalid existing v3 certificate: {path}")
            return str(log.relative_to(ROOT))
        if path.exists() or log.exists():
            raise RuntimeError(f"partial v3 certificate artifacts: {candidate}")
        return run_logged(certificate_command(candidate), log)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(one, candidate): candidate for candidate in candidates}
        for future in as_completed(futures):
            future.result()

    files = []
    for candidate in candidates:
        path = certificate_path(candidate)
        payload = safe_json(path)
        if payload.get("outcome_joined") or "actual_persistent_event" in payload:
            raise RuntimeError("v3 certificate contains an opened outcome")
        files.append(
            {
                "candidate": candidate.__dict__,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "issued": bool(payload["certificate_issued"]),
                "baseline_issued": bool(
                    payload.get("matched_fixed_radius_baseline")
                    and payload["matched_fixed_radius_baseline"]["certificate_issued"]
                ),
                "earliest_power": payload.get("earliest_issuing_power"),
            }
        )
    seal = {
        "status": "V3 CERTIFICATES FROZEN BEFORE OUTCOME JOIN",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method_seal_sha256": sha256(METHOD_SEAL),
        "candidate_seal_sha256": sha256(CANDIDATE_SEAL),
        "certificate_files": files,
        "candidates": len(candidates),
        "v3_issued_unopened": sum(row["issued"] for row in files),
        "baseline_issued_unopened": sum(row["baseline_issued"] for row in files),
    }
    write_json_exclusive(CERTIFICATE_SEAL, seal)
    seal["sha256"] = sha256(CERTIFICATE_SEAL)
    return seal


def audit_one(candidate: Candidate) -> dict:
    verify_method_seal()
    certificate_seal = safe_json(CERTIFICATE_SEAL)
    expected = next(
        row
        for row in certificate_seal["certificate_files"]
        if row["candidate"] == candidate.__dict__
    )
    path = ROOT / expected["path"]
    if sha256(path) != expected["sha256"]:
        raise RuntimeError("v3 certificate hash changed before outcome join")
    certificate = safe_json(path)
    destination = _v3_audit_path(candidate)
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing["certificate_sha256"] != expected["sha256"]:
            raise RuntimeError("existing v3 audit points to another certificate")
        return existing

    blind_path, checkpoint_path = artifact_paths(candidate.seed, development=False)
    payload = safe_json(blind_path)
    config = TransformerConfig(**payload["config"])
    torch.set_num_threads(config.threads)
    template = make_template(config)
    spec = flat_spec(template)
    data = make_disjoint_split(config)
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = data
    checkpoints = np.load(checkpoint_path)
    parameter = torch.from_numpy(checkpoints[f"step_{candidate.anchor}"]).clone()
    velocity = torch.from_numpy(checkpoints[f"velocity_{candidate.anchor}"]).clone()
    centerline = build_frozen_centerline(
        config,
        template,
        spec,
        train_pairs,
        train_labels,
        parameter,
        velocity,
    )
    if centerline["centerline_sha256"] != certificate["centerline_sha256"]:
        raise RuntimeError("v3 audit centerline hash mismatch")
    horizon = int(certificate["protocol"]["horizon"])
    exact = [torch.cat((parameter, velocity))]
    for _ in range(horizon):
        exact.append(centerline["map_step"](exact[-1]))
    exact = torch.stack(exact)
    dimension = parameter.numel()
    exact_counts = np.asarray(
        [
            int(
                (
                    logits(state[:dimension], cert_pairs, template, spec).argmax(1)
                    == cert_labels
                ).sum()
            )
            for state in exact
        ],
        dtype=np.int64,
    )
    actual_event = first_persistent(exact_counts, int(certificate["required_correct"]))

    scaled_exact = to_scaled(exact, dimension, config.learning_rate)
    scaled_center = centerline["scaled_center"][: horizon + 1]
    state_error = torch.linalg.vector_norm(scaled_exact - scaled_center, dim=1)
    residual = torch.stack(
        [
            to_scaled(
                centerline["map_step"](centerline["center"][step]),
                dimension,
                config.learning_rate,
            )
            - scaled_center[step + 1]
            for step in range(horizon)
        ]
    )
    apply_green, _ = make_transformer_green_products(
        centerline["center"][:horizon, :dimension],
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )
    signed_response = apply_green(residual.reshape(-1)).reshape(horizon, -1)
    response_centered_error = (
        scaled_exact[1:] - scaled_center[1:] - signed_response
    )
    response_centered_sequence_norm = float(
        torch.linalg.vector_norm(response_centered_error)
    )
    maximum_state_error = float(state_error.max())

    bracket = certificate.get("certified_bracket")
    issued = bool(certificate["certificate_issued"])
    contains = bool(
        issued
        and actual_event is not None
        and bracket[0] <= actual_event <= bracket[1]
    )
    baseline = certificate.get("matched_fixed_radius_baseline")
    baseline_bracket = None if baseline is None else baseline["certified_bracket"]
    baseline_issued = bool(baseline and baseline["certificate_issued"])
    baseline_contains = bool(
        baseline_issued
        and actual_event is not None
        and baseline_bracket[0] <= actual_event <= baseline_bracket[1]
    )
    total_radius = certificate.get("certified_total_pointwise_radius")
    remainder_radius = certificate.get("certified_remainder_sequence_radius")
    result = {
        "status": "POST-CERTIFICATE-SEAL V3 OUTCOME AUDIT",
        "candidate": candidate.__dict__,
        "certificate_path": str(path.relative_to(ROOT)),
        "certificate_sha256": expected["sha256"],
        "certificate_seal_sha256": sha256(CERTIFICATE_SEAL),
        "predicted_persistent_event": certificate["predicted_persistent_event"],
        "actual_persistent_event": actual_event,
        "raw_timing_error": (
            None
            if actual_event is None
            else int(certificate["predicted_persistent_event"] - actual_event)
        ),
        "certificate_issued": issued,
        "earliest_issuing_power": certificate.get("earliest_issuing_power"),
        "certified_bracket": bracket,
        "bracket_contains_actual": contains if issued else None,
        "baseline_certificate_issued": baseline_issued,
        "baseline_certified_bracket": baseline_bracket,
        "baseline_bracket_contains_actual": (
            baseline_contains if baseline_issued else None
        ),
        "maximum_observed_state_error": maximum_state_error,
        "certified_total_pointwise_radius": total_radius,
        "observed_state_tube_violation": (
            None if not issued else maximum_state_error > float(total_radius)
        ),
        "observed_response_centered_sequence_error": response_centered_sequence_norm,
        "certified_remainder_sequence_radius": remainder_radius,
        "observed_response_centered_sequence_tube_violation": (
            None
            if not issued
            else response_centered_sequence_norm > float(remainder_radius)
        ),
        "exact_count": exact_counts.tolist(),
    }
    write_json_exclusive(destination, result)
    return result


def join_and_summarize() -> dict:
    verify_method_seal()
    candidates, _, _ = frozen_candidates()
    audits = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(audit_one, candidate): candidate for candidate in candidates}
        for future in as_completed(futures):
            audits.append(future.result())
    audits.sort(
        key=lambda row: (
            int(row["candidate"]["seed"]),
            float(row["candidate"]["threshold"]),
        )
    )
    manifest = safe_json(CANDIDATE_MANIFEST)
    dispositions = Counter(row["disposition"] for row in manifest["records"])
    issued = [row for row in audits if row["certificate_issued"]]
    baseline = [row for row in audits if row["baseline_certificate_issued"]]
    comparable = [row for row in audits if row["actual_persistent_event"] is not None]
    inaccurate = [row for row in comparable if row["raw_timing_error"] != 0]
    issued_with_observed_events = [
        row for row in issued if row["actual_persistent_event"] is not None
    ]
    leads = [row["actual_persistent_event"] for row in issued_with_observed_events]
    widths = [
        row["certified_bracket"][1] - row["certified_bracket"][0]
        for row in issued
        if row["certified_bracket"] is not None
    ]
    q_counts = Counter(str(row["earliest_issuing_power"]) for row in issued)
    certificates = [safe_json(certificate_path(candidate)) for candidate in candidates]
    queried = sum(
        int(row["probability_budget"]["queried_operators"]) for row in certificates
    )
    summary = {
        "status": "FRESH PROSPECTIVE TRANSFORMER V3 CONFIRMATION COMPLETE",
        "method_seal_sha256": sha256(METHOD_SEAL),
        "candidate_seal_sha256": sha256(CANDIDATE_SEAL),
        "certificate_seal_sha256": sha256(CERTIFICATE_SEAL),
        "fresh_seeds": len(SEEDS),
        "seed_threshold_cases": len(SEEDS) * len(THRESHOLDS),
        "candidate_dispositions": dict(sorted(dispositions.items())),
        "candidates": len(audits),
        "distinct_candidate_seeds": len(
            {row["candidate"]["seed"] for row in audits}
        ),
        "v3_issued": len(issued),
        "v3_covered": sum(row["bracket_contains_actual"] is True for row in issued),
        "v3_issued_without_observed_event": len(issued) - len(issued_with_observed_events),
        "v3_conditional_coverage": (
            None
            if not issued
            else sum(row["bracket_contains_actual"] is True for row in issued)
            / len(issued)
        ),
        "v3_distinct_issuing_seeds": len(
            {row["candidate"]["seed"] for row in issued}
        ),
        "baseline_issued": len(baseline),
        "baseline_covered": sum(
            row["baseline_bracket_contains_actual"] is True for row in baseline
        ),
        "v3_additional_issued_over_baseline": sum(
            row["certificate_issued"] and not row["baseline_certificate_issued"]
            for row in audits
        ),
        "earliest_power_distribution": dict(sorted(q_counts.items())),
        "median_certified_lead": None if not leads else float(np.median(leads)),
        "maximum_certified_lead": None if not leads else int(max(leads)),
        "median_bracket_width": None if not widths else float(np.median(widths)),
        "maximum_bracket_width": None if not widths else int(max(widths)),
        "finite_prediction_outcome_pairs": len(comparable),
        "exact_finite_predictions": sum(row["raw_timing_error"] == 0 for row in comparable),
        "inaccurate_finite_predictions": len(inaccurate),
        "inaccurate_issued": sum(row["certificate_issued"] for row in inaccurate),
        "inaccurate_abstained": sum(not row["certificate_issued"] for row in inaccurate),
        "observed_state_tube_violations": sum(
            row["observed_state_tube_violation"] is True for row in issued
        ),
        "observed_response_centered_sequence_tube_violations": sum(
            row["observed_response_centered_sequence_tube_violation"] is True
            for row in issued
        ),
        "queried_random_operators": queried,
        "realized_union_failure_bound": queried * probe_config().delta,
        "rows": audits,
    }
    if AGGREGATE.exists():
        raise FileExistsError(f"v3 aggregate already exists: {AGGREGATE}")
    write_json_exclusive(AGGREGATE, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--certify", action="store_true")
    parser.add_argument("--join", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not any((args.freeze, args.train, args.select, args.certify, args.join, args.all)):
        parser.error("choose a phase or --all")
    outputs = {}
    if args.freeze or args.all:
        outputs["freeze"] = freeze_method()
    if args.train or args.all:
        outputs["train"] = train_all()
    if args.select or args.all:
        outputs["select"] = blind_scan_all()
    if args.certify or args.all:
        outputs["certify"] = certify_all()
    if args.join or args.all:
        outputs["join"] = join_and_summarize()
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
