#!/usr/bin/env python3
"""Conservative recovery for the frozen Transformer signed-Green audit.

The original mathematical implementation remains hash-sealed and untouched.
This program only retries two transient cache-I/O failures with that exact
executable and maps one deterministic reference-path construction failure to
abstention before any future optimizer trajectory is rolled out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from transformer_certificate_protocol import Candidate
from transformer_four_sweep_development_audit import first_persistent
from transformer_green_confirmation_certificate import (
    CANDIDATE_MANIFEST,
    CANDIDATE_SEAL,
    METHOD_SEAL,
    frozen_candidates,
    load_candidate,
    output_path as certificate_path,
    safe_json,
    sha256,
    verify_method_seal,
)
from transformer_green_confirmation_protocol import (
    FAMILY_FAILURE_PROBABILITY,
    MASTER_NONCE,
    PERSISTENCE,
    SWEEPS,
    maximum_operator_count,
    probe_config,
)
from transformer_hvp_grokking import logits
from transformer_modal_forecast import optimizer_map


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
AMENDMENT_DOC = ROOT / "TRANSFORMER_GREEN_CONFIRMATION_EXECUTION_AMENDMENT_V1_1.md"
AMENDMENT_SEAL = ROOT / "TRANSFORMER_GREEN_CONFIRMATION_EXECUTION_AMENDMENT_SEAL_V1_1.json"
CERTIFICATE_SEAL = ROOT / "TRANSFORMER_GREEN_CONFIRMATION_CERTIFICATE_SEAL.json"
AGGREGATE = RESULTS / "transformer_green_confirmation_audit.json"
ARCHIVE = RESULTS / "transformer_green_confirmation_execution_failure_archive_v1_1"

CONSTRUCTION_ABSTENTION = Candidate(335, 0.70, 2440)
IO_RETRIES = (
    Candidate(333, 0.80, 3080),
    Candidate(350, 0.70, 1440),
)
MAX_IO_ATTEMPTS = 3


def write_json_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def amendment_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def failure_log(candidate: Candidate) -> Path:
    return RESULTS / (
        f"transformer_green_confirmation_certificate_seed_{candidate.seed}_"
        f"gate_{candidate.gate_index}_anchor_{candidate.anchor}.process.log"
    )


def audit_path(candidate: Candidate) -> Path:
    return RESULTS / (
        f"transformer_green_confirmation_audit_seed_{candidate.seed}_"
        f"gate_{candidate.gate_index}_anchor_{candidate.anchor}.json"
    )


def verify_amendment_seal() -> dict:
    seal = json.loads(AMENDMENT_SEAL.read_text(encoding="utf-8"))
    if seal["original_method_seal_sha256"] != sha256(METHOD_SEAL):
        raise RuntimeError("execution amendment points to a different method seal")
    if seal["candidate_seal_sha256"] != sha256(CANDIDATE_SEAL):
        raise RuntimeError("execution amendment points to a different candidate seal")
    if seal["candidate_manifest_sha256"] != sha256(CANDIDATE_MANIFEST):
        raise RuntimeError("execution amendment points to a different candidate manifest")
    for relative, expected in seal["amendment_manifest"].items():
        if amendment_sha256(ROOT / relative) != expected:
            raise RuntimeError(f"execution-amendment file changed after freeze: {relative}")
    return seal


def check_failure_evidence() -> dict:
    evidence = {}
    for candidate in IO_RETRIES:
        path = failure_log(candidate)
        text = path.read_text(encoding="utf-8")
        if "PermissionError" not in text or "cache" not in text:
            raise RuntimeError(f"I/O retry lacks the frozen PermissionError evidence: {candidate}")
        evidence[str(candidate)] = amendment_sha256(path)
    construction_log = failure_log(CONSTRUCTION_ABSTENTION)
    construction_text = construction_log.read_text(encoding="utf-8")
    if "recentring sweep 2 truncated" not in construction_text:
        raise RuntimeError("construction abstention lacks the frozen truncation evidence")
    evidence[str(CONSTRUCTION_ABSTENTION)] = amendment_sha256(construction_log)
    return evidence


def archive_failure_logs() -> None:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for candidate in (*IO_RETRIES, CONSTRUCTION_ABSTENTION):
        source = failure_log(candidate)
        destination = ARCHIVE / source.name
        if destination.exists():
            if amendment_sha256(destination) != amendment_sha256(source):
                raise RuntimeError(f"failure-log archive collision: {destination}")
            continue
        shutil.copy2(source, destination)


def validate_outcome_blind_certificate(path: Path, candidate: Candidate) -> dict:
    payload = safe_json(path)
    if payload.get("candidate") != candidate.__dict__:
        raise RuntimeError(f"certificate candidate mismatch: {path}")
    if payload.get("outcome_joined") or "actual_persistent_event" in payload:
        raise RuntimeError(f"pre-seal certificate contains an outcome: {path}")
    return payload


def original_certificate_command(candidate: Candidate) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "transformer_green_confirmation_certificate.py"),
        "--seed",
        str(candidate.seed),
        "--threshold",
        str(candidate.threshold),
        "--anchor",
        str(candidate.anchor),
    ]


def retry_original_certificate(candidate: Candidate) -> None:
    destination = certificate_path(candidate)
    if destination.exists():
        validate_outcome_blind_certificate(destination, candidate)
        return
    for attempt in range(1, MAX_IO_ATTEMPTS + 1):
        log = RESULTS / (
            f"transformer_green_confirmation_certificate_seed_{candidate.seed}_"
            f"gate_{candidate.gate_index}_anchor_{candidate.anchor}.recovery_v1_1_"
            f"attempt_{attempt}.log"
        )
        with log.open("x", encoding="utf-8") as handle:
            completed = subprocess.run(
                original_certificate_command(candidate),
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode == 0:
            validate_outcome_blind_certificate(destination, candidate)
            return
        text = log.read_text(encoding="utf-8")
        if "PermissionError" not in text:
            raise RuntimeError(
                f"exact sealed retry failed non-transiently for {candidate}; see {log}"
            )
        if destination.exists():
            raise RuntimeError("failed retry unexpectedly produced a certificate")
        time.sleep(2.0 * attempt)
    raise RuntimeError(f"cache I/O retry budget exhausted for {candidate}")


def write_construction_abstention(candidate: Candidate, horizon: int, seal: dict) -> None:
    destination = certificate_path(candidate)
    if destination.exists():
        validate_outcome_blind_certificate(destination, candidate)
        return
    required = int(math.ceil(candidate.threshold * 58))
    predicted = horizon - PERSISTENCE + 1
    payload = {
        "status": "FROZEN FRESH conservative construction abstention; outcomes unopened",
        "candidate": candidate.__dict__,
        "method_seal_sha256": sha256(METHOD_SEAL),
        "candidate_seal_sha256": sha256(CANDIDATE_SEAL),
        "candidate_manifest_sha256": seal["candidate_manifest_sha256"],
        "execution_amendment_seal_sha256": amendment_sha256(AMENDMENT_SEAL),
        "protocol": {
            "sweeps": SWEEPS,
            "horizon": horizon,
            "persistence": PERSISTENCE,
            "radius_rule": "R = 2 ||K_H s||_sequence",
            "probe_config": probe_config().__dict__,
            "family_failure_probability": FAMILY_FAILURE_PROBABILITY,
            "maximum_operator_accounting": maximum_operator_count(),
            "master_nonce": MASTER_NONCE,
        },
        "required_correct": required,
        "predicted_persistent_event": predicted,
        "construction_abstention_reason": "frozen recentering sweep 2 truncated",
        "centerline_sha256": None,
        "signed_radius": None,
        "green_probe": None,
        "minimum_closure_lhs_using_kappa_ge_1": None,
        "early_abstention_before_green_probe": True,
        "closure_lhs_2_kappa_M_Z": None,
        "closure_slack": None,
        "closure_passed": False,
        "raw_margin_bracket": None,
        "certified_bracket": None,
        "certificate_issued": False,
        "certificate_output_logic_slack": None,
        "probability_budget": {
            "queried_operators": 0,
            "queried_union_bound": 0.0,
            "maximum_family_union_bound": FAMILY_FAILURE_PROBABILITY,
        },
        "outcome_joined": False,
    }
    write_json_exclusive(destination, payload)


def prepare() -> dict:
    verify_method_seal()
    amendment = verify_amendment_seal()
    if CERTIFICATE_SEAL.exists():
        raise FileExistsError("certificate seal already exists")
    evidence = check_failure_evidence()
    archive_failure_logs()
    candidates, horizons, candidate_seal = frozen_candidates()
    if CONSTRUCTION_ABSTENTION not in horizons:
        raise RuntimeError("construction abstention is outside the sealed candidates")
    for candidate in IO_RETRIES:
        if candidate not in horizons:
            raise RuntimeError(f"I/O retry is outside the sealed candidates: {candidate}")

    # Retry only the two environmental failures, with the original executable.
    for candidate in IO_RETRIES:
        retry_original_certificate(candidate)
    write_construction_abstention(
        CONSTRUCTION_ABSTENTION,
        horizons[CONSTRUCTION_ABSTENTION],
        candidate_seal,
    )

    files = []
    for candidate in candidates:
        path = certificate_path(candidate)
        payload = validate_outcome_blind_certificate(path, candidate)
        files.append(
            {
                "candidate": candidate.__dict__,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "issued": bool(payload["certificate_issued"]),
                "construction_abstention": candidate == CONSTRUCTION_ABSTENTION,
            }
        )
    certificate_seal = {
        "status": "FROZEN BEFORE OUTCOME JOIN; execution amendment v1.1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method_seal_sha256": sha256(METHOD_SEAL),
        "candidate_seal_sha256": sha256(CANDIDATE_SEAL),
        "candidate_manifest_sha256": sha256(CANDIDATE_MANIFEST),
        "execution_amendment_seal_sha256": amendment_sha256(AMENDMENT_SEAL),
        "pre_amendment_failure_log_sha256": evidence,
        "certificate_files": files,
        "candidates": len(candidates),
        "issued_unopened": sum(int(row["issued"]) for row in files),
        "construction_abstentions": 1,
        "exact_original_executable_retries": len(IO_RETRIES),
    }
    write_json_exclusive(CERTIFICATE_SEAL, certificate_seal)
    certificate_seal["sha256"] = amendment_sha256(CERTIFICATE_SEAL)
    return certificate_seal


def original_audit_command(candidate: Candidate) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_transformer_green_confirmation.py"),
        "--audit-one",
        "--seed",
        str(candidate.seed),
        "--threshold",
        str(candidate.threshold),
        "--anchor",
        str(candidate.anchor),
    ]


def run_original_audit(candidate: Candidate) -> None:
    destination = audit_path(candidate)
    if destination.exists():
        return
    log = RESULTS / (
        f"transformer_green_confirmation_audit_seed_{candidate.seed}_"
        f"gate_{candidate.gate_index}_anchor_{candidate.anchor}.process.log"
    )
    with log.open("x", encoding="utf-8") as handle:
        completed = subprocess.run(
            original_audit_command(candidate),
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"post-seal audit failed for {candidate}; see {log}")


def audit_construction_abstention(candidate: Candidate, horizon: int) -> dict:
    destination = audit_path(candidate)
    if destination.exists():
        return json.loads(destination.read_text(encoding="utf-8"))
    config, template, spec, data, parameter, velocity = load_candidate(candidate)
    torch.set_num_threads(config.threads)
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = data
    state = torch.cat((parameter, velocity))
    counts = []
    for step in range(horizon + 1):
        counts.append(
            int(
                (
                    logits(state[: parameter.numel()], cert_pairs, template, spec).argmax(1)
                    == cert_labels
                ).sum()
            )
        )
        if step < horizon:
            state = optimizer_map(
                state, train_pairs, train_labels, template, spec, config
            )
    required = int(math.ceil(candidate.threshold * len(cert_pairs)))
    actual = first_persistent(np.asarray(counts, dtype=np.int64), required)
    predicted = horizon - PERSISTENCE + 1
    audit = {
        "status": "post-seal exact rollout of construction abstention",
        "candidate": candidate.__dict__,
        "certificate_path": str(certificate_path(candidate).relative_to(ROOT)),
        "certificate_sha256": sha256(certificate_path(candidate)),
        "certificate_seal_sha256": amendment_sha256(CERTIFICATE_SEAL),
        "predicted_persistent_event": predicted,
        "actual_persistent_event": actual,
        "raw_timing_error": None if actual is None else predicted - actual,
        "certificate_issued": False,
        "certified_bracket": None,
        "bracket_contains_actual": None,
        "actual_sequence_error": None,
        "actual_sequence_error_to_radius_ratio": None,
        "actual_max_state_error": None,
        "observed_sequence_tube_violation": None,
        "observed_state_tube_violations": None,
        "construction_abstention_reason": "frozen recentering sweep 2 truncated",
        "exact_count": counts,
    }
    write_json_exclusive(destination, audit)
    return audit


def join() -> dict:
    verify_method_seal()
    verify_amendment_seal()
    if not CERTIFICATE_SEAL.exists():
        raise FileNotFoundError("certificate seal does not exist")
    if AGGREGATE.exists():
        raise FileExistsError("confirmation aggregate already exists")
    certificate_seal = json.loads(CERTIFICATE_SEAL.read_text(encoding="utf-8"))
    candidates, horizons, _ = frozen_candidates()
    expected_rows = {tuple(row["candidate"].values()): row for row in certificate_seal["certificate_files"]}
    for candidate in candidates:
        row = next(
            item
            for item in certificate_seal["certificate_files"]
            if item["candidate"] == candidate.__dict__
        )
        if sha256(ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError("certificate changed before outcome rollout")

    normal = [candidate for candidate in candidates if candidate != CONSTRUCTION_ABSTENTION]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_original_audit, candidate): candidate for candidate in normal}
        for future in as_completed(futures):
            future.result()
    audit_construction_abstention(
        CONSTRUCTION_ABSTENTION,
        horizons[CONSTRUCTION_ABSTENTION],
    )

    manifest = safe_json(CANDIDATE_MANIFEST)
    cert_rows = [safe_json(certificate_path(candidate)) for candidate in candidates]
    audit_rows = [
        json.loads(audit_path(candidate).read_text(encoding="utf-8"))
        for candidate in candidates
    ]
    issued_indices = [i for i, row in enumerate(cert_rows) if row["certificate_issued"]]
    issued_audits = [audit_rows[i] for i in issued_indices]
    leads = [
        int(row["actual_persistent_event"])
        for row in issued_audits
        if row["actual_persistent_event"] is not None
    ]
    widths = [
        int(cert_rows[i]["certified_bracket"][1])
        - int(cert_rows[i]["certified_bracket"][0])
        for i in issued_indices
    ]
    dispositions: dict[str, int] = {}
    for row in manifest["records"]:
        dispositions[row["disposition"]] = dispositions.get(row["disposition"], 0) + 1
    queried = sum(
        int(row.get("probability_budget", {}).get("queried_operators", 0))
        for row in cert_rows
    )
    aggregate = {
        "status": "complete frozen fresh signed-Green Transformer confirmation; execution amendment v1.1",
        "method_seal_sha256": sha256(METHOD_SEAL),
        "candidate_seal_sha256": sha256(CANDIDATE_SEAL),
        "certificate_seal_sha256": amendment_sha256(CERTIFICATE_SEAL),
        "execution_amendment_seal_sha256": amendment_sha256(AMENDMENT_SEAL),
        "summary": {
            "fresh_seeds": 24,
            "seed_threshold_cases": len(manifest["records"]),
            "dispositions": dispositions,
            "candidates": len(candidates),
            "distinct_candidate_seeds": len({candidate.seed for candidate in candidates}),
            "construction_abstentions": 1,
            "issued": len(issued_indices),
            "covered": sum(bool(row["bracket_contains_actual"]) for row in issued_audits),
            "distinct_issuing_seeds": len({candidates[i].seed for i in issued_indices}),
            "abstention_rate_among_candidates": 1.0 - len(issued_indices) / len(candidates),
            "median_bracket_width": None if not widths else float(np.median(widths)),
            "median_certified_lead": None if not leads else float(np.median(leads)),
            "maximum_certified_lead": None if not leads else max(leads),
            "raw_exact_timing_matches": sum(row["raw_timing_error"] == 0 for row in audit_rows),
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
            "maximum_family_union_bound": FAMILY_FAILURE_PROBABILITY,
            "maximum_operator_accounting": maximum_operator_count(),
        },
        "rows": [
            {
                "candidate": candidate.__dict__,
                "predicted_event": cert["predicted_persistent_event"],
                "actual_event": audit["actual_persistent_event"],
                "raw_timing_error": audit["raw_timing_error"],
                "construction_abstention": candidate == CONSTRUCTION_ABSTENTION,
                "early_abstention": cert.get("early_abstention_before_green_probe"),
                "green_upper": (
                    None
                    if cert.get("green_probe") is None
                    else cert["green_probe"]["green_operator_norm_upper_bound"]
                ),
                "closure_lhs": cert.get("closure_lhs_2_kappa_M_Z"),
                "closure_slack": cert.get("closure_slack"),
                "signed_radius": cert.get("signed_radius"),
                "certificate_issued": cert["certificate_issued"],
                "certified_bracket": cert["certified_bracket"],
                "output_logic_slack": cert.get("certificate_output_logic_slack"),
                "bracket_contains_actual": audit["bracket_contains_actual"],
                "actual_sequence_error": audit["actual_sequence_error"],
                "actual_sequence_error_to_radius_ratio": audit["actual_sequence_error_to_radius_ratio"],
                "actual_max_state_error": audit["actual_max_state_error"],
                "queried_operators": cert.get("probability_budget", {}).get("queried_operators", 0),
                "certificate_sha256": sha256(certificate_path(candidate)),
                "audit_sha256": amendment_sha256(audit_path(candidate)),
            }
            for candidate, cert, audit in zip(candidates, cert_rows, audit_rows)
        ],
    }
    write_json_exclusive(AGGREGATE, aggregate)
    aggregate["output"] = str(AGGREGATE)
    aggregate["sha256"] = amendment_sha256(AGGREGATE)
    return aggregate


def verify(require_join: bool) -> dict:
    verify_method_seal()
    verify_amendment_seal()
    candidates, _, _ = frozen_candidates()
    seal = json.loads(CERTIFICATE_SEAL.read_text(encoding="utf-8"))
    if len(seal["certificate_files"]) != len(candidates):
        raise RuntimeError("certificate seal is incomplete")
    for candidate in candidates:
        row = next(
            item for item in seal["certificate_files"]
            if item["candidate"] == candidate.__dict__
        )
        path = ROOT / row["path"]
        if sha256(path) != row["sha256"]:
            raise RuntimeError(f"certificate hash mismatch: {candidate}")
        validate_outcome_blind_certificate(path, candidate)
    result = {
        "certificate_files": len(seal["certificate_files"]),
        "issued_unopened": int(seal["issued_unopened"]),
        "construction_abstentions": int(seal["construction_abstentions"]),
        "certificate_seal_sha256": amendment_sha256(CERTIFICATE_SEAL),
    }
    if require_join:
        aggregate = json.loads(AGGREGATE.read_text(encoding="utf-8"))
        if aggregate["certificate_seal_sha256"] != amendment_sha256(CERTIFICATE_SEAL):
            raise RuntimeError("aggregate points to a different certificate seal")
        for row in aggregate["rows"]:
            candidate = Candidate(**row["candidate"])
            if amendment_sha256(audit_path(candidate)) != row["audit_sha256"]:
                raise RuntimeError(f"audit hash mismatch: {candidate}")
        result["summary"] = aggregate["summary"]
        result["aggregate_sha256"] = amendment_sha256(AGGREGATE)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--join", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--require-join", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chosen = sum((args.prepare, args.join, args.verify))
    if chosen != 1:
        raise SystemExit("choose exactly one of --prepare, --join, or --verify")
    if args.prepare:
        result = prepare()
    elif args.join:
        result = join()
    else:
        result = verify(args.require_join)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
