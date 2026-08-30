#!/usr/bin/env python3
"""Freeze, execute, and audit the fresh signed-Green Transformer study."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from transformer_certificate_protocol import Candidate
from transformer_four_sweep_development_audit import first_persistent, to_scaled
from transformer_green_confirmation_certificate import (
    CANDIDATE_MANIFEST,
    CANDIDATE_SEAL,
    build_frozen_centerline,
    frozen_candidates,
    output_path as certificate_path,
    safe_json,
)
from transformer_green_confirmation_protocol import (
    CHECKPOINT_SPACING,
    HORIZON,
    MASTER_NONCE,
    MAX_DEFICIT,
    PERSISTENCE,
    SCAN_STEPS,
    SEEDS,
    SWEEPS,
    THRESHOLDS,
    maximum_operator_count,
    probe_config,
)
from transformer_hvp_grokking import (
    TransformerConfig,
    artifact_paths,
    flat_spec,
    logits,
    make_disjoint_split,
    make_template,
)
from transformer_modal_forecast import optimizer_map, run as modal_run

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PROTOCOL = ROOT / "TRANSFORMER_GREEN_CONFIRMATION_PROTOCOL.md"
METHOD_SEAL = ROOT / "TRANSFORMER_GREEN_CONFIRMATION_METHOD_SEAL.json"
CERTIFICATE_SEAL = ROOT / "TRANSFORMER_GREEN_CONFIRMATION_CERTIFICATE_SEAL.json"
NO_ARTIFACT_AUDIT = RESULTS / "transformer_green_confirmation_no_artifact_audit.json"
AGGREGATE = RESULTS / "transformer_green_confirmation_audit.json"
WORKERS = 4

METHOD_FILES = (
    "TRANSFORMER_GREEN_CONFIRMATION_PROTOCOL.md",
    "GREEN_OPERATOR_SHADOWING_THEOREM.md",
    "VARIATIONAL_SHADOWING_THEOREM.md",
    "PROJECTED_HVP_SHADOWING_THEOREM.md",
    "scripts/run_transformer_green_confirmation.py",
    "scripts/transformer_green_confirmation_protocol.py",
    "scripts/transformer_green_confirmation_certificate.py",
    "scripts/transformer_green_operator.py",
    "scripts/transformer_green_protocol.py",
    "scripts/transformer_green_development_audit.py",
    "scripts/transformer_four_sweep_development_audit.py",
    "scripts/transformer_certificate_protocol.py",
    "scripts/transformer_optimizer_probe.py",
    "scripts/transformer_block_envelope.py",
    "scripts/block_jet_bound.py",
    "scripts/probe_jacobian_bound.py",
    "scripts/transformer_hvp_grokking.py",
    "scripts/transformer_modal_forecast.py",
    "scripts/matrix_free_mlp.py",
    "scripts/smooth_mlp_modular_grokking.py",
    "scripts/test_transformer_green_confirmation_protocol.py",
    "scripts/test_transformer_green_operator.py",
    "scripts/test_probe_jacobian_bound.py",
    "scripts/test_block_jet_bound.py",
    "scripts/verify_transformer_green_result.py",
    "scripts/verify_transformer_green_confirmation.py",
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


def fresh_targets() -> list[Path]:
    rows: list[Path] = []
    for seed in SEEDS:
        blind, checkpoint = artifact_paths(seed, development=False)
        rows.extend(
            [
                blind,
                blind.with_name(blind.stem + ".outcomes.json"),
                checkpoint,
                RESULTS / f"transformer_green_confirmation_seed_{seed}.sealed.log",
            ]
        )
    rows.extend([CANDIDATE_MANIFEST, CANDIDATE_SEAL, CERTIFICATE_SEAL, AGGREGATE])
    rows.extend(RESULTS.glob("transformer_green_confirmation_certificate_seed_*.json"))
    rows.extend(RESULTS.glob("transformer_green_confirmation_audit_seed_*.json"))
    rows.extend(RESULTS.glob("transformer_green_confirmation_cache/*"))
    return rows


def freeze_method() -> dict:
    if METHOD_SEAL.exists() or NO_ARTIFACT_AUDIT.exists():
        raise FileExistsError("method/no-artifact seal already exists")
    existing = sorted({str(path.relative_to(ROOT)) for path in fresh_targets() if path.exists()})
    audit = {
        "status": "pre-training no-artifact audit",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "fresh_seeds": list(SEEDS),
        "existing_target_artifacts": existing,
        "passed": not existing,
    }
    write_json_exclusive(NO_ARTIFACT_AUDIT, audit)
    if existing:
        raise RuntimeError(f"fresh artifact audit failed: {existing}")
    missing = [name for name in METHOD_FILES if not (ROOT / name).exists()]
    if missing:
        raise FileNotFoundError(f"claim-relevant files missing: {missing}")
    manifest = {name: sha256(ROOT / name) for name in METHOD_FILES}
    payload = {
        "status": "FROZEN BEFORE FRESH TRAINING",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha256(PROTOCOL),
        "no_artifact_audit": str(NO_ARTIFACT_AUDIT.relative_to(ROOT)),
        "no_artifact_audit_sha256": sha256(NO_ARTIFACT_AUDIT),
        "fresh_seeds": list(SEEDS),
        "thresholds": list(THRESHOLDS),
        "master_nonce": MASTER_NONCE,
        "probe_config": asdict(probe_config()),
        "operator_accounting": maximum_operator_count(),
        "code_manifest": manifest,
        "single_entry_command": "python scripts/run_transformer_green_confirmation.py --all",
        "training_commands": [training_command(seed)[1:] for seed in SEEDS],
        "information_barrier": (
            "candidate and certificate processes reject outcome/log reads; "
            "certificate hashes freeze before exact future rollout"
        ),
    }
    write_json_exclusive(METHOD_SEAL, payload)
    payload["sha256"] = sha256(METHOD_SEAL)
    return payload


def verify_method_seal() -> dict:
    seal = safe_json(METHOD_SEAL)
    if seal["protocol_sha256"] != sha256(PROTOCOL):
        raise RuntimeError("frozen confirmation protocol hash mismatch")
    if seal["no_artifact_audit_sha256"] != sha256(NO_ARTIFACT_AUDIT):
        raise RuntimeError("no-artifact audit hash mismatch")
    for name, expected in seal["code_manifest"].items():
        observed = sha256(ROOT / name)
        if observed != expected:
            raise RuntimeError(f"claim-relevant code changed after freeze: {name}")
    return seal


def run_logged(command: list[str], log_path: Path) -> str:
    if log_path.exists():
        raise FileExistsError(f"refusing to overwrite process log: {log_path}")
    with log_path.open("wb") as handle:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"subprocess failed with code {completed.returncode}; sealed log: {log_path}"
        )
    return str(log_path.relative_to(ROOT))


def train_one(seed: int) -> str:
    blind, checkpoint = artifact_paths(seed, development=False)
    outcome = blind.with_name(blind.stem + ".outcomes.json")
    log = RESULTS / f"transformer_green_confirmation_seed_{seed}.sealed.log"
    for path in (blind, checkpoint, outcome, log):
        if path.exists():
            raise FileExistsError(f"fresh training target already exists: {path}")
    run_logged(training_command(seed), log)
    if not all(path.exists() for path in (blind, checkpoint, outcome)):
        raise RuntimeError(f"training seed {seed} did not create all split artifacts")
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
    rows = np.flatnonzero(
        (trajectory[:, 1] >= 0.99) & (trajectory[:, 2] >= trigger_gate)
    )
    return None if len(rows) == 0 else int(trajectory[rows[0], 0])


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
        raise RuntimeError(f"seed {seed} config differs from the frozen protocol")
    if any("certificate" in name.lower() for name in payload["trajectory_columns"]):
        raise RuntimeError("blind scanner received a certification outcome column")
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
                raise RuntimeError("modal event persistence suffix exceeds the frozen horizon")
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
    seal = verify_method_seal()
    if CANDIDATE_MANIFEST.exists() or CANDIDATE_SEAL.exists():
        raise FileExistsError("candidate manifest/seal already exists")
    records = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(scan_seed, seed): seed for seed in SEEDS}
        for future in as_completed(futures):
            records.extend(future.result())
    records.sort(key=lambda row: (int(row["seed"]), float(row["threshold"])))
    manifest = {
        "status": "FROZEN fresh Transformer candidates; outcomes unopened",
        "method_seal_sha256": sha256(METHOD_SEAL),
        "fresh_seeds": list(SEEDS),
        "thresholds": list(THRESHOLDS),
        "records": records,
    }
    write_json_exclusive(CANDIDATE_MANIFEST, manifest)
    candidate_rows = [row for row in records if row["disposition"] == "candidate frozen"]
    candidates = [
        {
            "seed": int(row["seed"]),
            "threshold": float(row["threshold"]),
            "anchor": int(row["anchor"]),
            "horizon": int(row["horizon"]),
        }
        for row in candidate_rows
    ]
    candidate_seal = {
        "status": "FROZEN BEFORE ANY CONFIRMATION PROBE",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method_seal_sha256": sha256(METHOD_SEAL),
        "protocol_sha256": seal["protocol_sha256"],
        "candidate_manifest": str(CANDIDATE_MANIFEST.relative_to(ROOT)),
        "candidate_manifest_sha256": sha256(CANDIDATE_MANIFEST),
        "seed_threshold_cases": len(records),
        "candidates": candidates,
        "distinct_candidate_seeds": len({row["seed"] for row in candidates}),
        "information_barrier": "outcome JSONs and sealed logs unopened",
    }
    write_json_exclusive(CANDIDATE_SEAL, candidate_seal)
    candidate_seal["sha256"] = sha256(CANDIDATE_SEAL)
    return candidate_seal


def certificate_command(candidate: Candidate) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "transformer_green_confirmation_certificate.py"),
        "--seed", str(candidate.seed),
        "--threshold", str(candidate.threshold),
        "--anchor", str(candidate.anchor),
    ]


def certify_all() -> dict:
    verify_method_seal()
    if CERTIFICATE_SEAL.exists():
        raise FileExistsError("certificate seal already exists")
    candidates, _, _ = frozen_candidates()
    for candidate in candidates:
        if certificate_path(candidate).exists():
            raise FileExistsError(f"fresh certificate already exists: {candidate}")

    def one(candidate: Candidate) -> str:
        log = RESULTS / (
            f"transformer_green_confirmation_certificate_seed_{candidate.seed}_"
            f"gate_{candidate.gate_index}_anchor_{candidate.anchor}.process.log"
        )
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
            raise RuntimeError("fresh certificate contains a pre-seal outcome")
        files.append(
            {
                "candidate": candidate.__dict__,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "issued": bool(payload["certificate_issued"]),
            }
        )
    certificate_seal = {
        "status": "FROZEN BEFORE OUTCOME JOIN",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method_seal_sha256": sha256(METHOD_SEAL),
        "candidate_seal_sha256": sha256(CANDIDATE_SEAL),
        "certificate_files": files,
        "candidates": len(candidates),
        "issued_unopened": sum(int(row["issued"]) for row in files),
    }
    write_json_exclusive(CERTIFICATE_SEAL, certificate_seal)
    certificate_seal["sha256"] = sha256(CERTIFICATE_SEAL)
    return certificate_seal


def audit_path(candidate: Candidate) -> Path:
    return RESULTS / (
        f"transformer_green_confirmation_audit_seed_{candidate.seed}_"
        f"gate_{candidate.gate_index}_anchor_{candidate.anchor}.json"
    )


def audit_one(candidate: Candidate) -> dict:
    verify_method_seal()
    certificate_seal = safe_json(CERTIFICATE_SEAL)
    expected = next(
        row for row in certificate_seal["certificate_files"]
        if row["candidate"] == candidate.__dict__
    )
    cert_path = ROOT / expected["path"]
    if sha256(cert_path) != expected["sha256"]:
        raise RuntimeError("certificate hash changed before outcome join")
    certificate = safe_json(cert_path)
    destination = audit_path(candidate)
    if destination.exists():
        raise FileExistsError(f"fresh audit already exists: {destination}")

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
    path = build_frozen_centerline(
        config,
        template,
        spec,
        train_pairs,
        train_labels,
        parameter,
        velocity,
    )
    if path["centerline_sha256"] != certificate["centerline_sha256"]:
        raise RuntimeError("post-seal centerline differs from certificate centerline")
    horizon = int(certificate["protocol"]["horizon"])
    exact = [torch.cat((parameter, velocity))]
    for _ in range(horizon):
        exact.append(path["map_step"](exact[-1]))
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
    actual = first_persistent(exact_counts, int(certificate["required_correct"]))
    bracket = certificate["certified_bracket"]
    contains = (
        None
        if actual is None or bracket is None
        else int(bracket[0]) <= actual <= int(bracket[1])
    )
    scaled_exact = to_scaled(exact, dimension, config.learning_rate)
    error = torch.linalg.vector_norm(
        scaled_exact - path["scaled_center"][: horizon + 1], dim=1
    )
    sequence_error = float(torch.linalg.vector_norm(error[1:]))
    radius = float(certificate["signed_radius"])
    tolerance = radius * (1.0 + 1e-9) + 1e-30
    audit = {
        "status": "post-certificate-seal outcome audit",
        "candidate": candidate.__dict__,
        "certificate_path": str(cert_path.relative_to(ROOT)),
        "certificate_sha256": expected["sha256"],
        "certificate_seal_sha256": sha256(CERTIFICATE_SEAL),
        "predicted_persistent_event": certificate["predicted_persistent_event"],
        "actual_persistent_event": actual,
        "raw_timing_error": (
            None if actual is None else int(certificate["predicted_persistent_event"]) - actual
        ),
        "certificate_issued": bool(certificate["certificate_issued"]),
        "certified_bracket": bracket,
        "bracket_contains_actual": contains,
        "actual_sequence_error": sequence_error,
        "actual_sequence_error_to_radius_ratio": (
            None if radius == 0.0 else sequence_error / radius
        ),
        "actual_max_state_error": float(error.max()),
        "observed_sequence_tube_violation": (
            bool(sequence_error > tolerance)
            if certificate["closure_passed"] else None
        ),
        "observed_state_tube_violations": (
            int(torch.sum(error[1:] > tolerance))
            if certificate["closure_passed"] else None
        ),
        "exact_count": exact_counts.tolist(),
    }
    write_json_exclusive(destination, audit)
    audit["output"] = str(destination)
    audit["sha256"] = sha256(destination)
    return audit


def audit_command(candidate: Candidate) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_transformer_green_confirmation.py"),
        "--audit-one",
        "--seed", str(candidate.seed),
        "--threshold", str(candidate.threshold),
        "--anchor", str(candidate.anchor),
    ]


def join_all() -> dict:
    verify_method_seal()
    certificate_seal = safe_json(CERTIFICATE_SEAL)
    candidates, _, _ = frozen_candidates()
    for row in certificate_seal["certificate_files"]:
        if sha256(ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError("certificate changed before outcome join")

    def one(candidate: Candidate) -> str:
        log = RESULTS / (
            f"transformer_green_confirmation_audit_seed_{candidate.seed}_"
            f"gate_{candidate.gate_index}_anchor_{candidate.anchor}.process.log"
        )
        return run_logged(audit_command(candidate), log)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(one, candidate): candidate for candidate in candidates}
        for future in as_completed(futures):
            future.result()

    manifest = safe_json(CANDIDATE_MANIFEST)
    cert_rows = []
    audit_rows = []
    for candidate in candidates:
        cert_rows.append(safe_json(certificate_path(candidate)))
        audit_rows.append(safe_json(audit_path(candidate)))
    issued_indices = [i for i, row in enumerate(cert_rows) if row["certificate_issued"]]
    issued_audits = [audit_rows[i] for i in issued_indices]
    leads = [int(row["actual_persistent_event"]) for row in issued_audits]
    widths = [
        int(cert_rows[i]["certified_bracket"][1])
        - int(cert_rows[i]["certified_bracket"][0])
        for i in issued_indices
    ]
    dispositions: dict[str, int] = {}
    for row in manifest["records"]:
        dispositions[row["disposition"]] = dispositions.get(row["disposition"], 0) + 1
    queried = sum(int(row["probability_budget"]["queried_operators"]) for row in cert_rows)
    aggregate = {
        "status": "complete frozen fresh signed-Green Transformer confirmation",
        "method_seal_sha256": sha256(METHOD_SEAL),
        "candidate_seal_sha256": sha256(CANDIDATE_SEAL),
        "certificate_seal_sha256": sha256(CERTIFICATE_SEAL),
        "summary": {
            "fresh_seeds": len(SEEDS),
            "seed_threshold_cases": len(manifest["records"]),
            "dispositions": dispositions,
            "candidates": len(candidates),
            "distinct_candidate_seeds": len({candidate.seed for candidate in candidates}),
            "issued": len(issued_indices),
            "covered": sum(bool(row["bracket_contains_actual"]) for row in issued_audits),
            "distinct_issuing_seeds": len(
                {candidates[i].seed for i in issued_indices}
            ),
            "abstention_rate_among_candidates": (
                None if not candidates else 1.0 - len(issued_indices) / len(candidates)
            ),
            "median_bracket_width": None if not widths else float(np.median(widths)),
            "median_certified_lead": None if not leads else float(np.median(leads)),
            "maximum_certified_lead": None if not leads else max(leads),
            "raw_exact_timing_matches": sum(
                row["raw_timing_error"] == 0 for row in audit_rows
            ),
            "observed_issued_sequence_tube_violations": sum(
                bool(row["observed_sequence_tube_violation"]) for row in issued_audits
            ),
            "observed_issued_state_tube_violations": sum(
                int(row["observed_state_tube_violations"] or 0) for row in issued_audits
            ),
        },
        "probability_budget": {
            "queried_operators": queried,
            "queried_union_bound": queried * probe_config().delta,
            "maximum_family_union_bound": 1.0e-6,
            "maximum_operator_accounting": maximum_operator_count(),
        },
        "rows": [
            {
                "candidate": candidate.__dict__,
                "predicted_event": cert["predicted_persistent_event"],
                "actual_event": audit["actual_persistent_event"],
                "raw_timing_error": audit["raw_timing_error"],
                "early_abstention": cert["early_abstention_before_green_probe"],
                "green_upper": (
                    None if cert["green_probe"] is None
                    else cert["green_probe"]["green_operator_norm_upper_bound"]
                ),
                "closure_lhs": cert["closure_lhs_2_kappa_M_Z"],
                "closure_slack": cert["closure_slack"],
                "signed_radius": cert["signed_radius"],
                "certificate_issued": cert["certificate_issued"],
                "certified_bracket": cert["certified_bracket"],
                "output_logic_slack": cert["certificate_output_logic_slack"],
                "bracket_contains_actual": audit["bracket_contains_actual"],
                "actual_sequence_error": audit["actual_sequence_error"],
                "actual_sequence_error_to_radius_ratio": audit[
                    "actual_sequence_error_to_radius_ratio"
                ],
                "actual_max_state_error": audit["actual_max_state_error"],
                "queried_operators": cert["probability_budget"]["queried_operators"],
                "certificate_sha256": sha256(certificate_path(candidate)),
                "audit_sha256": sha256(audit_path(candidate)),
            }
            for candidate, cert, audit in zip(candidates, cert_rows, audit_rows)
        ],
    }
    write_json_exclusive(AGGREGATE, aggregate)
    aggregate["output"] = str(AGGREGATE)
    aggregate["sha256"] = sha256(AGGREGATE)
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    phases = parser.add_mutually_exclusive_group(required=True)
    phases.add_argument("--freeze", action="store_true")
    phases.add_argument("--train", action="store_true")
    phases.add_argument("--blind-scan", action="store_true")
    phases.add_argument("--certify", action="store_true")
    phases.add_argument("--join", action="store_true")
    phases.add_argument("--audit-one", action="store_true")
    phases.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--anchor", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.freeze:
        result = freeze_method()
    elif args.train:
        result = {"training_outputs": train_all()}
    elif args.blind_scan:
        result = blind_scan_all()
    elif args.certify:
        result = certify_all()
    elif args.join:
        result = join_all()
    elif args.audit_one:
        if args.seed is None or args.threshold is None or args.anchor is None:
            raise ValueError("--audit-one requires seed, threshold, and anchor")
        result = audit_one(Candidate(args.seed, args.threshold, args.anchor))
    else:
        if not METHOD_SEAL.exists():
            freeze_method()
        train_all()
        blind_scan_all()
        certify_all()
        result = join_all()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
