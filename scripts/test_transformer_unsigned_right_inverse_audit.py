#!/usr/bin/env python3
"""Arithmetic and aggregation checks for the post-seal unsigned baseline."""
from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "results" / "transformer_unsigned_right_inverse_audit.json"


def close(left: float, right: float, *, rel: float = 2e-14) -> None:
    assert math.isclose(float(left), float(right), rel_tol=rel, abs_tol=1e-300), (
        left,
        right,
    )


def main() -> None:
    payload = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    rows = payload["rows"]
    comparable = [row for row in rows if row["green_operator_available"]]
    assert len(rows) == 23
    assert len(comparable) == 18

    for row in comparable:
        expected_unsigned = (
            float(row["green_operator_norm_upper_bound"])
            * float(row["defect_sequence_norm"])
        )
        close(row["unsigned_response_upper"], expected_unsigned)
        close(
            row["unsigned_to_signed_response_ratio"],
            expected_unsigned / float(row["signed_response_sequence_norm"]),
        )
        expected_favourable = (
            2.0
            * float(row["green_operator_norm_upper_bound"])
            * float(row["signed_radius_derivative_drift_upper"])
            * expected_unsigned
        )
        close(
            row["favourable_unsigned_closure_using_signed_radius_drift"],
            expected_favourable,
        )
        assert row["unsigned_to_signed_response_ratio"] > 2.0
        if row.get("radii_polynomial_impossible"):
            assert expected_favourable > 1.0
            assert row["unsigned_certificate_issued"] is False

    full = [row for row in comparable if row.get("full_evaluation")]
    assert len(full) == 1
    survivor = full[0]
    expected_closure = (
        2.0
        * float(survivor["green_operator_norm_upper_bound"])
        * float(survivor["outer_derivative_drift_upper"])
        * float(survivor["unsigned_response_upper"])
    )
    close(survivor["outer_closure_statistic"], expected_closure)
    expected_radius = 2.0 * float(survivor["unsigned_response_upper"]) / (
        1.0 + math.sqrt(1.0 - expected_closure)
    )
    close(survivor["minimal_admissible_radius"], expected_radius)
    assert survivor["outer_fixed_points_all_consistent"] is True
    assert survivor["output_fixed_points_all_consistent"] is True
    assert survivor["unsigned_certificate_issued"] is True
    assert survivor["unsigned_certified_bracket"] == [30, 30]
    assert survivor["actual_event"] == 30
    assert survivor["unsigned_bracket_contains_actual"] is True

    signed_issued = [row for row in comparable if row["signed_certificate_issued"]]
    unsigned_issued = [row for row in comparable if row["unsigned_certificate_issued"]]
    lost = [
        row
        for row in signed_issued
        if not row["unsigned_certificate_issued"]
    ]
    summary = payload["summary"]
    assert len(signed_issued) == summary["signed_issued_in_matched_cases"] == 9
    assert len(unsigned_issued) == summary["strong_unsigned_issued_in_matched_cases"] == 1
    assert len(lost) == summary["signed_certificates_lost_by_unsigned_norm_replacement"] == 8
    assert summary["unsigned_covered"] == 1
    assert summary["cases_eliminated_without_reconstruction"] == 17
    print(
        "PASS: matched unsigned baseline retains 1/9 signed certificates; "
        "17/18 cases fail the favourable closure screen."
    )


if __name__ == "__main__":
    main()
