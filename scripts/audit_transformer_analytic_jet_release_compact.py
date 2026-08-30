#!/usr/bin/env python3
"""Checkpoint-free consistency audit for the analytic neural-jet release."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOURCE = RESULTS / "transformer_analytic_jet_release_postseal_audit.json"
INDEPENDENT = RESULTS / "transformer_analytic_jet_release_independent_audit.json"
DIRECT = RESULTS / "transformer_direct_image_green_panel_audit.json"
PREFIX = RESULTS / "transformer_v3_relinearized_prefix_panel_audit.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def candidate_key(row: dict) -> tuple[int, float, int]:
    candidate = row["candidate"]
    return int(candidate["seed"]), float(candidate["threshold"]), int(candidate["anchor"])


def close(left: float, right: float, *, rel: float = 2.0e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=rel, abs_tol=1.0e-300)


def main() -> None:
    source = read(SOURCE)
    independent = read(INDEPENDENT)
    direct = read(DIRECT)
    prefix = read(PREFIX)

    assert source["direct_panel_sha256"] == sha256(DIRECT)
    assert source["prefix_panel_sha256"] == sha256(PREFIX)
    assert independent["source_sha256"] == sha256(SOURCE)
    assert source["future_outcome_files_read"] == independent["future_outcome_files_read"] == 0

    source_rows = {candidate_key(row): row for row in source["rows"]}
    independent_rows = {candidate_key(row): row for row in independent["rows"]}
    direct_rows = {candidate_key(row): row for row in direct["rows"]}
    prefix_rows = {candidate_key(row): row for row in prefix["rows"]}
    assert len(source_rows) == 15
    assert source_rows.keys() == independent_rows.keys() == direct_rows.keys() == prefix_rows.keys()

    issued = 0
    eliminated_operators = 0
    eliminated_grams = 0
    for identity, row in source_rows.items():
        audit = independent_rows[identity]
        closure = row["closure"]
        kappa = float(closure["kappa"])
        drift = float(closure["derivative_drift"])
        forcing = float(closure["corrected_defect_response_bound"])
        discriminant = 1.0 - 2.0 * kappa * drift * forcing
        assert close(discriminant, closure["discriminant"])
        if discriminant >= 0.0:
            radius = 2.0 * forcing / (1.0 + math.sqrt(discriminant))
            assert close(radius, closure["remainder_radius"])
            assert close(
                float(closure["correction_max_state_norm"]) + radius,
                closure["total_radius_about_original_reference"],
            )
        else:
            assert closure["remainder_radius"] is None

        released = bool(row["analytic_jet_issued"])
        assert released == bool(audit["issued"])
        assert row["analytic_jet_bracket"] == audit["bracket"]
        assert released == (not bool(row["fallback_required"]))
        assert close(row["maximum_optimizer_jacobian_drift"], audit["drift"])
        assert row["existing_bracket"] == direct_rows[identity]["bracket"]
        if released:
            issued += 1
            assert row["analytic_jet_bracket"] == row["existing_bracket"]
            assert close(row["analytic_logic_slack"], audit["logic_slack"])
            assert close(row["maximum_analytic_margin_radius"], audit["maximum_margin_radius"])
        eliminated_operators += int(row["randomized_output_operators_eliminated"])
        eliminated_grams += int(row["randomized_output_gram_applications_eliminated"])

    assert issued == source["analytic_jet_issued"] == independent["analytic_jet_issued"] == 8
    assert source["probe_fallback_required"] == 7
    assert source["staged_total_issued"] == 15
    assert eliminated_operators == source["randomized_output_operators_eliminated"] == 1432
    assert eliminated_grams == source["randomized_output_gram_applications_eliminated"] == 22912

    print(
        json.dumps(
            {
                "status": "COMPACT ANALYTIC-JET ARTIFACT AUDIT PASSED",
                "cases": len(source_rows),
                "analytic_releases": issued,
                "probe_fallbacks": source["probe_fallback_required"],
                "randomized_output_operators_eliminated": eliminated_operators,
                "output_gram_applications_eliminated": eliminated_grams,
                "checkpoint_free": True,
                "full_derivative_replay": "scripts/audit_transformer_analytic_jet_release_result.py after regenerating checkpoints",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
