#!/usr/bin/env python3
"""Post-seal all-cohort audit of deterministic analytic-jet release.

The audit reuses the already committed corrected Green bounds, but replaces
every randomized output-Jacobian bound by the ball-valid first/second/third
neural jets already stored in the sealed certificate record.  It reads no
revealed future trajectory.  Cases not closed by the deterministic release
retain their existing output-probe fallback.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from analytic_jet_release import analytic_jet_release, logit_margin_radius
from audit_transformer_relinearized_prefix_panel import CASE_ROWS
from transformer_hvp_grokking import logits
from transformer_v3_certificate import (
    _gate_raw_slacks,
    _logic_slack,
    _persistent_bracket,
    load_candidate,
    output_path,
    safe_json,
)
from transformer_certificate_protocol import Candidate


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DIRECT_PANEL = RESULTS / "transformer_direct_image_green_panel_audit.json"
PREFIX_PANEL = RESULTS / "transformer_v3_relinearized_prefix_panel_audit.json"
OUTPUT = RESULTS / "transformer_analytic_jet_release_postseal_audit.json"
THEOREM = ROOT / "ANALYTIC_JET_RELEASE_THEOREM.md"

EXPECTED_DIRECT_SHA256 = (
    "931CBF5750510C49DEB92F16F77E8CCA355C7969A18BCF4EFA1A0701335ED705"
)
EXPECTED_PREFIX_SHA256 = (
    "08E501B51FEAC3D96FFE02BE0B5D84E0E682C2E73CB906C083ED0FEF7E75E12B"
)
EXPECTED_ANALYTIC_ISSUANCE = 8


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def candidate_key(candidate: dict) -> tuple[int, float, int]:
    return (
        int(candidate["seed"]),
        float(candidate["threshold"]),
        int(candidate["anchor"]),
    )


def selected_stage(row: dict) -> dict:
    stage = row["stage_rows"][-1]
    selected = stage["direct"] if row["route"] == "direct_image" else stage["gram"]
    if selected is None or not bool(selected["issued"]):
        raise RuntimeError("direct/Gram source row has no issuing terminal stage")
    return selected


def audit_case(row: dict, prefix: dict) -> dict:
    key = candidate_key(row["candidate"])
    seed, threshold, anchor = key
    candidate = Candidate(seed, threshold, anchor)
    certificate_path = output_path(candidate)
    certificate = safe_json(certificate_path)
    expected_certificate_sha = next(
        item[4]
        for item in CASE_ROWS
        if (int(item[0]), float(item[1]), int(item[2])) == key
    )
    if sha256(certificate_path) != expected_certificate_sha:
        raise RuntimeError(f"certificate hash mismatch for {candidate}")
    if int(row["horizon"]) != int(prefix["horizon"]):
        raise RuntimeError(f"panel horizon mismatch for {candidate}")

    selected = selected_stage(row)
    output_rows = certificate["output_rows"]
    horizon = int(row["horizon"])
    if len(output_rows) != horizon:
        raise RuntimeError(f"output-row count mismatch for {candidate}")
    if not all(bool(item["block_fixed_point_consistent"]) for item in output_rows):
        raise RuntimeError(f"invalid analytic jet fixed point for {candidate}")

    release = analytic_jet_release(
        kappa=float(selected["operator_norm_upper_bound"]),
        corrected_defect_response_bound=float(selected["forcing_response_upper"]),
        correction_max_state_norm=float(prefix["correction_max_state_norm"]),
        domain_radius=float(prefix["domain_radius"]),
        learning_rate=0.01,
        transition_jets=[
            (
                float(item["block_first"]),
                float(item["block_second"]),
                float(item["block_third"]),
            )
            for item in output_rows[:-1]
        ],
        output_first_bounds=[float(item["block_first"]) for item in output_rows],
    )

    bracket = None
    logic_slack = None
    margin_radii: list[float] = []
    raw_guarantees: list[float] = []
    raw_exclusions: list[float] = []
    if release.closure.closure_passed:
        config, template, spec, data, parameter, _ = load_candidate(candidate)
        _, _, _, _, cert_pairs, cert_labels = data
        raw_zero = _gate_raw_slacks(
            logits(parameter, cert_pairs, template, spec),
            cert_labels,
            int(certificate["required_correct"]),
        )
        raw_guarantees = [float(raw_zero[0])]
        raw_exclusions = [float(raw_zero[1])]
        margin_radii = [0.0]
        total_radius = float(release.state_radius_about_original_reference)
        for item in output_rows:
            margin = logit_margin_radius(
                first=float(item["block_first"]), state_radius=total_radius
            )
            margin_radii.append(margin)
            raw_guarantees.append(float(item["raw_guarantee_slack"]))
            raw_exclusions.append(float(item["raw_exclusion_slack"]))
        guarantee_slacks = [
            raw - margin for raw, margin in zip(raw_guarantees, margin_radii)
        ]
        exclusion_slacks = [
            raw - margin for raw, margin in zip(raw_exclusions, margin_radii)
        ]
        bracket = _persistent_bracket(guarantee_slacks, exclusion_slacks)
        logic_slack = _logic_slack(bracket, guarantee_slacks, exclusion_slacks)

    issued = bracket is not None
    if issued and bracket != row["bracket"]:
        raise RuntimeError(f"analytic release changed bracket for {candidate}")
    probes = int(certificate["protocol"]["probe_config"]["probes"])
    old_output_power = int(selected["output_power"])
    old_drift = float(selected["closure"]["derivative_drift"])
    return {
        "candidate": row["candidate"],
        "horizon": horizon,
        "green_route": row["route"],
        "analytic_jet_issued": issued,
        "analytic_jet_bracket": bracket,
        "existing_bracket": row["bracket"],
        "fallback_required": not issued,
        "analytic_logic_slack": logic_slack,
        "maximum_analytic_margin_radius": (
            None if not margin_radii else max(margin_radii)
        ),
        "maximum_optimizer_jacobian_drift": (
            release.maximum_optimizer_jacobian_drift
        ),
        "prior_probe_based_derivative_drift": old_drift,
        "analytic_to_probe_drift_ratio": (
            release.maximum_optimizer_jacobian_drift / old_drift
        ),
        "closure": release.closure.as_dict(),
        "randomized_output_operators_eliminated": horizon if issued else 0,
        "randomized_output_gram_applications_eliminated": (
            horizon * probes * old_output_power if issued else 0
        ),
        "output_failure_probability_eliminated": 1.0e-6 if issued else 0.0,
        "remaining_combined_failure_upper": 1.0e-6 if issued else 2.0e-6,
        "future_outcome_files_read": 0,
    }


def main() -> None:
    if sha256(DIRECT_PANEL) != EXPECTED_DIRECT_SHA256:
        raise RuntimeError("direct-image panel hash changed")
    if sha256(PREFIX_PANEL) != EXPECTED_PREFIX_SHA256:
        raise RuntimeError("corrected-prefix panel hash changed")
    direct = safe_json(DIRECT_PANEL)
    prefix_payload = safe_json(PREFIX_PANEL)
    prefix_index = {candidate_key(row["candidate"]): row for row in prefix_payload["rows"]}
    rows = [
        audit_case(row, prefix_index[candidate_key(row["candidate"])])
        for row in direct["rows"]
    ]
    rows.sort(
        key=lambda item: (
            int(item["candidate"]["seed"]),
            float(item["candidate"]["threshold"]),
        )
    )
    issued = [row for row in rows if row["analytic_jet_issued"]]
    if len(issued) != EXPECTED_ANALYTIC_ISSUANCE:
        raise RuntimeError(
            f"expected {EXPECTED_ANALYTIC_ISSUANCE} analytic releases, got {len(issued)}"
        )
    if not all(row["analytic_jet_bracket"] == row["existing_bracket"] for row in issued):
        raise RuntimeError("an analytic release changed an issued bracket")
    payload = {
        "status": "POST-SEAL DETERMINISTIC ANALYTIC-JET RELEASE AUDIT PASSED",
        "scope": (
            "Complete 15-case pre-existing Green-operator cohort; deterministic "
            "output release reads no future trajectory and changes no frozen count."
        ),
        "cases": len(rows),
        "analytic_jet_issued": len(issued),
        "probe_fallback_required": len(rows) - len(issued),
        "staged_total_issued": len(rows),
        "same_brackets": all(
            (not row["analytic_jet_issued"])
            or row["analytic_jet_bracket"] == row["existing_bracket"]
            for row in rows
        ),
        "randomized_output_operators_eliminated": sum(
            row["randomized_output_operators_eliminated"] for row in rows
        ),
        "randomized_output_gram_applications_eliminated": sum(
            row["randomized_output_gram_applications_eliminated"] for row in rows
        ),
        "minimum_analytic_logic_slack": min(
            float(row["analytic_logic_slack"]) for row in issued
        ),
        "maximum_analytic_margin_radius": max(
            float(row["maximum_analytic_margin_radius"]) for row in issued
        ),
        "green_only_failure_upper_for_analytic_releases": 1.0e-6,
        "fallback_combined_failure_upper": 2.0e-6,
        "future_outcome_files_read": 0,
        "source_sha256": sha256(Path(__file__)),
        "theorem_sha256": sha256(THEOREM),
        "direct_panel_sha256": EXPECTED_DIRECT_SHA256,
        "prefix_panel_sha256": EXPECTED_PREFIX_SHA256,
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
