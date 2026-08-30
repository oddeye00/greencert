#!/usr/bin/env python3
"""Independent arithmetic/hash audit of the output-recentering replay."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOURCE = RESULTS / "transformer_v3_output_recentering_postseal_audit.json"
OUTPUT = RESULTS / "transformer_v3_output_recentering_independent_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def close(left: float, right: float) -> None:
    if not math.isclose(float(left), float(right), rel_tol=2.0e-13, abs_tol=1.0e-300):
        raise AssertionError(f"mismatch: {left} != {right}")


def main() -> None:
    source = load(SOURCE)
    rows = source["rows"]
    assert len(rows) == 19
    assert source["outcome_files_read"] == 0
    assert source["randomized_queries_added"] == 0
    script = ROOT / "scripts" / "audit_transformer_v3_output_recentering.py"
    assert sha256(script) == source["audit_script_sha256"]

    evaluable = [row for row in rows if row["evaluable"]]
    old_issued = [row for row in rows if row["old_certificate_issued"]]
    recentered = [row for row in evaluable if row["recentered_issued"]]
    hybrid = [row for row in evaluable if row["hybrid_issued"]]
    assert (len(evaluable), len(old_issued), len(recentered), len(hybrid)) == (15, 11, 11, 11)

    ratios = []
    for row in rows:
        certificate_path = ROOT / row["certificate_path"]
        assert sha256(certificate_path) == row["certificate_sha256"]
        certificate = load(certificate_path)
        assert certificate["candidate"] == row["candidate"]
        assert bool(certificate["certificate_issued"]) == row["old_certificate_issued"]
        if not row["evaluable"]:
            continue
        assert row["outcome_files_read"] == 0
        assert row["randomized_queries_added"] == 0
        for power in row["power_audits"]:
            if not power["closure_passed"]:
                continue
            old_radius = float(power["maximum_origin_margin_radius"])
            new_radius = float(power["maximum_recentered_margin_radius"])
            assert new_radius <= old_radius + 2.0e-14 * max(1.0, old_radius)
            expected_ratio = 0.0 if old_radius == 0.0 else new_radius / old_radius
            close(power["maximum_margin_radius_ratio"], expected_ratio)
            if power["power"] == row["hybrid_earliest_power"]:
                ratios.append(expected_ratio)
        if row["old_certificate_issued"]:
            assert row["hybrid_issued"]
            assert row["hybrid_earliest_power"] == row["old_earliest_power"]
            assert row["hybrid_bracket"] == row["old_bracket"]

    assert len(ratios) == 11
    close(source["median_maximum_margin_radius_ratio_at_hybrid_issue"], statistics.median(ratios))
    close(source["maximum_margin_radius_ratio_at_hybrid_issue"], max(ratios))
    assert source["hybrid_converted_old_abstentions"] == 0
    assert source["earlier_power_cases"] == 0
    payload = {
        "status": "INDEPENDENT OUTPUT-RECENTERING AUDIT PASSED",
        "source": SOURCE.relative_to(ROOT).as_posix(),
        "source_sha256": sha256(SOURCE),
        "generator_script_sha256": sha256(script),
        "records": len(rows),
        "evaluable": len(evaluable),
        "old_issued_retained": len(old_issued),
        "hybrid_issued": len(hybrid),
        "minimum_maximum_margin_radius_ratio": min(ratios),
        "median_maximum_margin_radius_ratio": statistics.median(ratios),
        "maximum_maximum_margin_radius_ratio": max(ratios),
        "same_power_and_bracket": True,
        "outcome_files_read": 0,
        "randomized_queries_added": 0,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        "PASS: 11/11 sealed certificates retained; maximum output-radius ratio "
        f"range {min(ratios):.3f}--{max(ratios):.3f}, median {statistics.median(ratios):.3f}."
    )


if __name__ == "__main__":
    main()
