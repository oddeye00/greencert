#!/usr/bin/env python3
"""Serialize and audit the one frozen Transformer-v3 execution abstention.

This file is deliberately outside the original method manifest.  It does not
repair or rerun the failed candidate.  Before any outcome join, it converts a
deterministic frozen-centerline construction failure into an explicit
zero-query abstention and hash-seals that disposition.  After the ordinary
certificate seal, it creates the exceptional post-seal event audit and then
delegates all remaining joins to the frozen runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from run_transformer_v3_confirmation import (
    AGGREGATE,
    CERTIFICATE_SEAL,
    ROOT,
    _v3_audit_path,
    _v3_certificate_log,
    first_persistent,
    join_and_summarize,
)
from transformer_certificate_protocol import Candidate
from transformer_hvp_grokking import (
    TransformerConfig,
    artifact_paths,
    flat_spec,
    logits,
    make_disjoint_split,
    make_template,
)
from transformer_modal_forecast import optimizer_map
from transformer_v3_certificate import (
    CANDIDATE_MANIFEST,
    CANDIDATE_SEAL,
    METHOD_SEAL,
    frozen_candidates,
    output_path,
    safe_json,
)
from transformer_v3_protocol import (
    FAMILY_FAILURE_PROBABILITY,
    PERSISTENCE,
    SWEEPS,
    maximum_operator_count,
    probe_config,
)


NOTE = ROOT / "TRANSFORMER_V3_EXECUTION_AMENDMENT.md"
AMENDMENT_SEAL = ROOT / "TRANSFORMER_V3_EXECUTION_AMENDMENT_SEAL.json"
JOIN_SEAL = ROOT / "TRANSFORMER_V3_EXECUTION_AMENDMENT_JOIN_SEAL.json"
SCRIPT = ROOT / "scripts" / "transformer_v3_execution_amendment.py"
EXCEPTION = Candidate(365, 0.70, 1480)
EXPECTED_FAILURE = "RuntimeError: recentring sweep 2 truncated"
AMENDMENT_FILES = (
    "TRANSFORMER_V3_EXECUTION_AMENDMENT.md",
    "scripts/transformer_v3_execution_amendment.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json_exclusive(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen amendment artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def exceptional_manifest_row() -> dict:
    manifest = safe_json(CANDIDATE_MANIFEST)
    matches = [
        row
        for row in manifest["records"]
        if row["disposition"] == "candidate frozen"
        and int(row["seed"]) == EXCEPTION.seed
        and math.isclose(float(row["threshold"]), EXCEPTION.threshold)
        and int(row["anchor"]) == EXCEPTION.anchor
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one exceptional frozen row, got {len(matches)}")
    return matches[0]


def preflight() -> dict:
    method = safe_json(METHOD_SEAL)
    for name, expected in method["code_manifest"].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"sealed method file changed before amendment: {name}")
    candidates, horizons, candidate_seal = frozen_candidates()
    if EXCEPTION not in horizons:
        raise RuntimeError("exceptional candidate is absent from the frozen set")
    row = exceptional_manifest_row()
    expected_horizon = int(row["predicted_offset"]) + PERSISTENCE - 1
    if horizons[EXCEPTION] != expected_horizon or int(row["horizon"]) != expected_horizon:
        raise RuntimeError("exceptional horizon differs across frozen records")

    failed_log = _v3_certificate_log(EXCEPTION)
    if not failed_log.exists():
        raise FileNotFoundError(f"missing failed process log: {failed_log}")
    log_text = failed_log.read_text(encoding="utf-8", errors="replace")
    if EXPECTED_FAILURE not in log_text:
        raise RuntimeError("failed process log does not contain the frozen failure")

    forecast_path = ROOT / row["forecast_file"]
    if sha256(forecast_path) != row["forecast_sha256"]:
        raise RuntimeError("blind modal forecast hash changed")
    forecast = json.loads(forecast_path.read_text(encoding="utf-8"))
    if forecast["events"]["0.70"]["recentered"] != int(row["predicted_offset"]):
        raise RuntimeError("blind modal event differs from the candidate manifest")
    sweeps = forecast["sweeps"]
    if len(sweeps) != SWEEPS or int(sweeps[1]["reached_horizon"]) >= int(forecast["horizon"]):
        raise RuntimeError("blind record does not exhibit the sealed truncation")

    missing = [candidate for candidate in candidates if not output_path(candidate).exists()]
    if missing != [EXCEPTION]:
        raise RuntimeError(f"expected exactly one missing certificate, got {missing}")
    if CERTIFICATE_SEAL.exists():
        raise RuntimeError("certificate seal already exists; amendment must precede it")

    return {
        "candidate": EXCEPTION.__dict__,
        "candidate_count": len(candidates),
        "other_completed_certificates": len(candidates) - len(missing),
        "horizon": expected_horizon,
        "predicted_event": int(row["predicted_offset"]),
        "required_correct": int(row["required"]),
        "forecast_path": str(forecast_path.relative_to(ROOT)),
        "forecast_sha256": sha256(forecast_path),
        "failed_log": str(failed_log.relative_to(ROOT)),
        "failed_log_sha256": sha256(failed_log),
        "sweep_diagnostics": sweeps,
        "method_seal_sha256": sha256(METHOD_SEAL),
        "candidate_seal_sha256": sha256(CANDIDATE_SEAL),
        "candidate_manifest_sha256": candidate_seal["candidate_manifest_sha256"],
    }


def serialize_abstention() -> dict:
    if AMENDMENT_SEAL.exists() or output_path(EXCEPTION).exists():
        raise FileExistsError("execution amendment or abstention record already exists")
    evidence = preflight()
    probe = probe_config()
    certificate = {
        "status": "FROZEN V3 OUTCOME-BLIND EXECUTION ABSTENTION",
        "candidate": EXCEPTION.__dict__,
        "method_seal_sha256": evidence["method_seal_sha256"],
        "candidate_seal_sha256": evidence["candidate_seal_sha256"],
        "candidate_manifest_sha256": evidence["candidate_manifest_sha256"],
        "protocol": {
            "sweeps": SWEEPS,
            "horizon": evidence["horizon"],
            "persistence": PERSISTENCE,
            "disposition": "deterministic reference-path construction failure",
        },
        "centerline_sha256": None,
        "sweep_diagnostics": evidence["sweep_diagnostics"],
        "required_correct": evidence["required_correct"],
        "predicted_persistent_event": evidence["predicted_event"],
        "defect_sequence_norm": None,
        "signed_response_sequence_norm": None,
        "signed_response_max_state_norm": None,
        "outer_domain_radius": None,
        "directional_green_norm_lower_bound": None,
        "optimistic_one_shot_closure": None,
        "early_abstention_before_green_probe": True,
        "block_fixed_points_all_consistent": None,
        "green_trace": None,
        "power_rows": [],
        "earliest_issuing_power": None,
        "certified_total_pointwise_radius": None,
        "certified_remainder_sequence_radius": None,
        "certified_bracket": None,
        "certificate_issued": False,
        "certificate_logic_slack": None,
        "matched_fixed_radius_baseline": None,
        "output_rows": [],
        "probability_budget": {
            "queried_operators": 0,
            "queried_union_bound": 0.0,
            "maximum_family_union_bound": FAMILY_FAILURE_PROBABILITY,
            "no_union_over_power_levels": True,
            "allowed_operator_count": maximum_operator_count()[
                "maximum_probabilistic_operators"
            ],
            "collision_free_stream_count": None,
            "queried_operator_count": 0,
            "all_queries_predeclared": True,
            "probe_delta_if_queried": probe.delta,
        },
        "timings_seconds": {
            "centerline": None,
            "output_phase_this_process": 0.0,
            "green_phase_this_process": 0.0,
            "total_this_process": None,
        },
        "execution_failure": {
            "reason": EXPECTED_FAILURE,
            "failed_log": evidence["failed_log"],
            "failed_log_sha256": evidence["failed_log_sha256"],
            "blind_forecast": evidence["forecast_path"],
            "blind_forecast_sha256": evidence["forecast_sha256"],
            "handling": "candidate retained; zero-query abstention; no retry",
        },
        "outcome_joined": False,
    }
    destination = output_path(EXCEPTION)
    write_json_exclusive(destination, certificate)
    seal = {
        "status": "SEALED OUTCOME-BLIND V3 EXECUTION AMENDMENT",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method_seal_sha256": evidence["method_seal_sha256"],
        "candidate_seal_sha256": evidence["candidate_seal_sha256"],
        "candidate_manifest_sha256": evidence["candidate_manifest_sha256"],
        "candidate": EXCEPTION.__dict__,
        "candidate_count_unchanged": evidence["candidate_count"],
        "future_outcome_opened": False,
        "randomized_operators_queried": 0,
        "code_manifest": {name: sha256(ROOT / name) for name in AMENDMENT_FILES},
        "failed_log": evidence["failed_log"],
        "failed_log_sha256": evidence["failed_log_sha256"],
        "blind_forecast": evidence["forecast_path"],
        "blind_forecast_sha256": evidence["forecast_sha256"],
        "abstention_certificate": str(destination.relative_to(ROOT)),
        "abstention_certificate_sha256": sha256(destination),
        "disposition": "deterministic construction failure -> abstain; no retry",
    }
    write_json_exclusive(AMENDMENT_SEAL, seal)
    seal["sha256"] = sha256(AMENDMENT_SEAL)
    return seal


def verify_amendment() -> dict:
    seal = json.loads(AMENDMENT_SEAL.read_text(encoding="utf-8"))
    for name, expected in seal["code_manifest"].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"execution amendment file changed: {name}")
    certificate = ROOT / seal["abstention_certificate"]
    if sha256(certificate) != seal["abstention_certificate_sha256"]:
        raise RuntimeError("execution-abstention certificate hash changed")
    if sha256(METHOD_SEAL) != seal["method_seal_sha256"]:
        raise RuntimeError("method seal changed after execution amendment")
    if sha256(CANDIDATE_SEAL) != seal["candidate_seal_sha256"]:
        raise RuntimeError("candidate seal changed after execution amendment")
    return seal


def exceptional_outcome_audit() -> dict:
    amendment = verify_amendment()
    if not CERTIFICATE_SEAL.exists():
        raise FileNotFoundError("ordinary v3 certificate seal is required before join")
    certificate_seal = safe_json(CERTIFICATE_SEAL)
    expected = next(
        row
        for row in certificate_seal["certificate_files"]
        if row["candidate"] == EXCEPTION.__dict__
    )
    path = ROOT / expected["path"]
    if expected["sha256"] != amendment["abstention_certificate_sha256"]:
        raise RuntimeError("ordinary certificate seal did not retain the amendment hash")
    certificate = safe_json(path)
    if certificate["certificate_issued"] or certificate["probability_budget"][
        "queried_operators"
    ]:
        raise RuntimeError("exceptional candidate is no longer a zero-query abstention")

    destination = _v3_audit_path(EXCEPTION)
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing["certificate_sha256"] != expected["sha256"]:
            raise RuntimeError("existing exceptional audit points to another certificate")
        return existing

    blind_path, checkpoint_path = artifact_paths(EXCEPTION.seed, development=False)
    payload = safe_json(blind_path)
    config = TransformerConfig(**payload["config"])
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    template = make_template(config)
    spec = flat_spec(template)
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = make_disjoint_split(config)
    checkpoints = np.load(checkpoint_path)
    parameter = torch.from_numpy(checkpoints[f"step_{EXCEPTION.anchor}"]).clone()
    velocity = torch.from_numpy(checkpoints[f"velocity_{EXCEPTION.anchor}"]).clone()

    def map_step(state: torch.Tensor) -> torch.Tensor:
        return optimizer_map(
            state, train_pairs, train_labels, template, spec, config
        )

    horizon = int(certificate["protocol"]["horizon"])
    exact = [torch.cat((parameter, velocity))]
    for _ in range(horizon):
        exact.append(map_step(exact[-1]))
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
    result = {
        "status": "POST-CERTIFICATE-SEAL V3 EXECUTION-ABSTENTION OUTCOME AUDIT",
        "candidate": EXCEPTION.__dict__,
        "certificate_path": str(path.relative_to(ROOT)),
        "certificate_sha256": expected["sha256"],
        "certificate_seal_sha256": sha256(CERTIFICATE_SEAL),
        "execution_amendment_seal_sha256": sha256(AMENDMENT_SEAL),
        "predicted_persistent_event": certificate["predicted_persistent_event"],
        "actual_persistent_event": actual_event,
        "raw_timing_error": (
            None
            if actual_event is None
            else int(certificate["predicted_persistent_event"] - actual_event)
        ),
        "certificate_issued": False,
        "earliest_issuing_power": None,
        "certified_bracket": None,
        "bracket_contains_actual": None,
        "baseline_certificate_issued": False,
        "baseline_certified_bracket": None,
        "baseline_bracket_contains_actual": None,
        "maximum_observed_state_error": None,
        "certified_total_pointwise_radius": None,
        "observed_state_tube_violation": None,
        "observed_response_centered_sequence_error": None,
        "certified_remainder_sequence_radius": None,
        "observed_response_centered_sequence_tube_violation": None,
        "reference_path_available": False,
        "reference_path_failure": EXPECTED_FAILURE,
        "exact_count": exact_counts.tolist(),
    }
    write_json_exclusive(destination, result)
    return result


def join_with_amendment() -> dict:
    if JOIN_SEAL.exists():
        raise FileExistsError(f"join seal already exists: {JOIN_SEAL}")
    exceptional = exceptional_outcome_audit()
    summary = join_and_summarize()
    join_seal = {
        "status": "TRANSFORMER V3 EXECUTION AMENDMENT JOIN COMPLETE",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "execution_amendment_seal_sha256": sha256(AMENDMENT_SEAL),
        "certificate_seal_sha256": sha256(CERTIFICATE_SEAL),
        "exceptional_audit": str(_v3_audit_path(EXCEPTION).relative_to(ROOT)),
        "exceptional_audit_sha256": sha256(_v3_audit_path(EXCEPTION)),
        "aggregate": str(AGGREGATE.relative_to(ROOT)),
        "aggregate_sha256": sha256(AGGREGATE),
        "candidate_count": summary["candidates"],
        "exceptional_actual_event": exceptional["actual_persistent_event"],
        "exceptional_raw_timing_error": exceptional["raw_timing_error"],
    }
    write_json_exclusive(JOIN_SEAL, join_seal)
    return {"summary": summary, "join_seal": join_seal}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--serialize-abstention", action="store_true")
    parser.add_argument("--join", action="store_true")
    args = parser.parse_args()
    if sum((args.verify, args.serialize_abstention, args.join)) != 1:
        parser.error("choose exactly one phase")
    if args.verify:
        result = preflight()
    elif args.serialize_abstention:
        result = serialize_abstention()
    else:
        result = join_with_amendment()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
