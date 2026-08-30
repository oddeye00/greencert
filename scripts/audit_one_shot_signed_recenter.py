#!/usr/bin/env python3
"""Post-seal audit of the conservative one-shot recentered closure.

This audit changes no prospective record.  It applies a theorem derived after
the Transformer outcomes were opened to the already stored signed response,
Green bound, derivative envelope, margin bracket, and outcome audit.  Any gain
reported here is retrospective until confirmed under a newly frozen protocol.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from one_shot_recenter_closure import conservative_one_shot_closure


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "one_shot_signed_recenter_postseal_audit.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _audit_path(certificate: dict) -> Path:
    candidate = certificate["candidate"]
    gate = int(round(float(candidate["threshold"]) * 10.0)) - 7
    return RESULTS / (
        f"transformer_green_confirmation_audit_seed_{candidate['seed']}_"
        f"gate_{gate}_anchor_{candidate['anchor']}.json"
    )


def _contains(bracket: list[int] | None, event: int | None) -> bool | None:
    if bracket is None or event is None:
        return None
    return int(bracket[0]) <= int(event) <= int(bracket[1])


def build_audit() -> dict:
    rows = []
    certificate_paths = sorted(
        RESULTS.glob("transformer_green_confirmation_certificate_seed_*.json")
    )
    for certificate_path in certificate_paths:
        certificate = _load(certificate_path)
        audit_path = _audit_path(certificate)
        if not audit_path.exists():
            raise FileNotFoundError(f"missing outcome audit for {certificate_path.name}")
        outcome = _load(audit_path)
        candidate = certificate["candidate"]
        row = {
            "candidate": candidate,
            "certificate_path": certificate_path.relative_to(ROOT).as_posix(),
            "certificate_sha256": _sha256(certificate_path),
            "outcome_audit_path": audit_path.relative_to(ROOT).as_posix(),
            "outcome_audit_sha256": _sha256(audit_path),
            "predicted_persistent_event": certificate.get(
                "predicted_persistent_event"
            ),
            "actual_persistent_event": outcome.get("actual_persistent_event"),
            "old_certificate_issued": bool(certificate.get("certificate_issued")),
            "old_certified_bracket": certificate.get("certified_bracket"),
            "raw_margin_bracket": certificate.get("raw_margin_bracket"),
            "green_queried": certificate.get("green_probe") is not None,
        }

        # Because z = K s, ||K|| >= ||z||/||s||.  Together with ||K|| >= 1,
        # this gives a deterministic, response-aware lower bound that is free
        # once the signed response has been formed.
        sequence_norm = certificate.get("signed_response_sequence_norm")
        max_state_norm = certificate.get("signed_response_max_state_norm")
        derivative_drift = certificate.get(
            "maximum_optimizer_derivative_drift_upper"
        )
        defect_norm = certificate.get("defect_sequence_norm")
        if None not in (sequence_norm, max_state_norm, derivative_drift, defect_norm):
            directional_kappa_lower = max(
                1.0,
                float(sequence_norm) / max(float(defect_norm), 1.0e-300),
            )
            optimistic = conservative_one_shot_closure(
                kappa=directional_kappa_lower,
                derivative_drift=float(derivative_drift),
                response_sequence_norm=float(sequence_norm),
                response_max_state_norm=float(max_state_norm),
                domain_radius=float(certificate["signed_radius"]),
            )
            row.update(
                {
                    "directional_green_norm_lower_bound": directional_kappa_lower,
                    "optimistic_one_shot_closure": optimistic.as_dict(),
                    "safe_early_abstention_without_green": (
                        not optimistic.closure_passed
                    ),
                }
            )
        else:
            row.update(
                {
                    "directional_green_norm_lower_bound": None,
                    "optimistic_one_shot_closure": None,
                    "safe_early_abstention_without_green": False,
                }
            )

        required = (
            sequence_norm,
            max_state_norm,
            derivative_drift,
        )
        if certificate.get("green_probe") is None or any(x is None for x in required):
            row.update(
                {
                    "one_shot_evaluable": False,
                    "one_shot_closure": None,
                    "new_certificate_issued": False,
                    "new_certified_bracket": None,
                    "converted_from_old_abstention": False,
                    "new_bracket_contains_actual": None,
                    "observed_max_state_inside_new_total_radius": None,
                }
            )
            rows.append(row)
            continue

        kappa = float(certificate["green_probe"]["green_operator_norm_upper_bound"])
        derivative_drift = float(derivative_drift)
        sequence_norm = float(sequence_norm)
        max_state_norm = float(max_state_norm)
        old_radius = float(certificate["signed_radius"])
        closure = conservative_one_shot_closure(
            kappa=kappa,
            derivative_drift=derivative_drift,
            response_sequence_norm=sequence_norm,
            response_max_state_norm=max_state_norm,
            domain_radius=old_radius,
        )
        raw_bracket = certificate.get("raw_margin_bracket")
        fixed_points = bool(certificate.get("block_fixed_points_all_consistent"))
        issued = closure.closure_passed and fixed_points and raw_bracket is not None
        bracket = raw_bracket if issued else None
        actual = outcome.get("actual_persistent_event")
        observed_max_state_error = outcome.get("actual_max_state_error")
        observed_inside = None
        if observed_max_state_error is not None and closure.total_pointwise_radius is not None:
            observed_inside = (
                float(observed_max_state_error) <= closure.total_pointwise_radius
            )

        closure_dict = closure.as_dict()
        closure_dict.update(
            {
                "old_radius": old_radius,
                "new_to_old_radius_ratio": (
                    closure.total_pointwise_radius / old_radius
                    if closure.total_pointwise_radius is not None and old_radius > 0.0
                    else None
                ),
                "fixed_point_envelopes_consistent": fixed_points,
                "reuses_old_ball_valid_drift_envelope": closure.domain_passed,
                "reuses_old_margin_bracket_by_radius_monotonicity": (
                    closure.domain_passed and raw_bracket is not None
                ),
            }
        )
        row.update(
            {
                "one_shot_evaluable": True,
                "one_shot_closure": closure_dict,
                "new_certificate_issued": issued,
                "new_certified_bracket": bracket,
                "converted_from_old_abstention": (
                    issued and not row["old_certificate_issued"]
                ),
                "new_bracket_contains_actual": _contains(bracket, actual),
                "observed_max_state_inside_new_total_radius": observed_inside,
            }
        )
        rows.append(row)

    evaluable = [row for row in rows if row["one_shot_evaluable"]]
    issued = [row for row in rows if row["new_certificate_issued"]]
    converted = [row for row in rows if row["converted_from_old_abstention"]]
    covered = [row for row in issued if row["new_bracket_contains_actual"]]
    old_issued = [row for row in rows if row["old_certificate_issued"]]
    ratios = [
        row["one_shot_closure"]["new_to_old_radius_ratio"]
        for row in evaluable
        if row["one_shot_closure"]["closure_passed"]
    ]
    algebraic_slacks = [
        row["one_shot_closure"]["discriminant"] for row in evaluable
        if row["one_shot_closure"]["closure_passed"]
    ]
    actual_tube = [
        row["observed_max_state_inside_new_total_radius"] for row in issued
        if row["observed_max_state_inside_new_total_radius"] is not None
    ]
    safe_early = [row for row in rows if row["safe_early_abstention_without_green"]]

    return {
        "status": "POST-SEAL THEOREM AUDIT; not prospective confirmation",
        "evidence_boundary": (
            "The one-shot closure was derived after Transformer outcomes were "
            "available. It reuses sealed quantities without changing them; issuance "
            "and coverage gains require a new frozen batch for confirmatory status."
        ),
        "method_or_stored_quantity_changed": False,
        "certificate_records": len(rows),
        "green_evaluable_records": len(evaluable),
        "old_issued": len(old_issued),
        "new_issued": len(issued),
        "new_covered": len(covered),
        "converted_old_abstentions": len(converted),
        "converted_candidates": [row["candidate"] for row in converted],
        "distinct_new_issuing_seeds": len(
            {int(row["candidate"]["seed"]) for row in issued}
        ),
        "minimum_passing_discriminant": min(algebraic_slacks) if algebraic_slacks else None,
        "maximum_new_to_old_radius_ratio": max(ratios) if ratios else None,
        "observed_state_tube_checks": len(actual_tube),
        "observed_state_tube_violations": sum(not value for value in actual_tube),
        "response_lower_bound_early_abstentions": len(safe_early),
        "response_lower_bound_early_abstention_candidates": [
            row["candidate"] for row in safe_early
        ],
        "early_abstentions_preserved_without_random_green_query": all(
            not row["green_queried"] for row in safe_early
        ),
        "rows": rows,
    }


def main() -> None:
    audit = build_audit()
    OUTPUT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(
        "PASS: conservative one-shot recentering changes "
        f"{audit['old_issued']} -> {audit['new_issued']} issued, with "
        f"{audit['new_covered']}/{audit['new_issued']} retrospective coverage and "
        f"{audit['observed_state_tube_violations']} observed state-tube violations."
    )


if __name__ == "__main__":
    main()
