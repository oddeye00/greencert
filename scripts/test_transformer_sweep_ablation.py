#!/usr/bin/env python3
"""Consistency checks for the 0--4 sweep Transformer ablation."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "results" / "transformer_sweep_ablation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    payload = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    records = payload["records"]
    assert len(records) == 3
    for record in records:
        assert len(record["stages"]) == 5
        source_path = ROOT / record["source_audit"]
        source = json.loads(source_path.read_text(encoding="utf-8"))
        assert sha256(source_path) == record["source_audit_sha256"]
        assert record["stages"][-1]["centerline_sha256"] == source["centerline_sha256"]
        assert record["stages"][-1]["predicted_persistent_event"] == source[
            "predicted_persistent_event"
        ]
        assert record["actual_persistent_event"] == source["actual_persistent_event"]
        for sweep, stage in enumerate(record["stages"]):
            assert stage["sweep"] == sweep
            assert stage["cumulative_hvp_calls"] == (sweep + 1) * 300
            assert math.isclose(
                stage["maximum_raw_defect_norm"],
                source["delta_raw_0_through_4"][sweep],
                rel_tol=0.0,
                abs_tol=0.0,
            )
            assert math.isclose(
                stage["maximum_scaled_defect_norm"],
                source["delta_scaled_0_through_4"][sweep],
                rel_tol=0.0,
                abs_tol=0.0,
            )
            predicted = stage["predicted_persistent_event"]
            expected_error = predicted - record["actual_persistent_event"]
            assert stage["signed_timing_error"] == expected_error
            assert stage["absolute_timing_error"] == abs(expected_error)

    summary = payload["summary_by_sweep"]
    assert len(summary) == 5
    for sweep, row in enumerate(summary):
        stages = [record["stages"][sweep] for record in records]
        errors = [stage["absolute_timing_error"] for stage in stages]
        defects = [stage["maximum_scaled_defect_norm"] for stage in stages]
        assert row["sweep"] == sweep
        assert row["exact_event_clocks"] == sum(error == 0 for error in errors)
        assert row["median_absolute_timing_error"] == statistics.median(errors)
        assert row["maximum_absolute_timing_error"] == max(errors)
        assert row["median_maximum_scaled_defect_norm"] == statistics.median(defects)
        assert row["maximum_scaled_defect_norm"] == max(defects)

    assert summary[0]["exact_event_clocks"] == 0
    assert all(row["exact_event_clocks"] == 3 for row in summary[1:])
    contraction = (
        summary[0]["median_maximum_scaled_defect_norm"]
        / summary[4]["median_maximum_scaled_defect_norm"]
    )
    assert contraction > 1.0e5
    print(
        f"PASS: one sweep repairs 3/3 event clocks; four sweeps reduce the "
        f"median maximum scaled defect by {contraction:.3g}x."
    )


if __name__ == "__main__":
    main()
