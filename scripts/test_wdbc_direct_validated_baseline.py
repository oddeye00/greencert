#!/usr/bin/env python3
"""Regression test for the WDBC direct validated-continuation audit."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_wdbc_direct_validated_baseline.py"
RESULT = ROOT / "results" / "wdbc_direct_validated_baseline_audit.json"


def main() -> None:
    subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, check=True, capture_output=True)
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    assert payload["independent_of_green_radius"]
    assert payload["independent_of_randomized_green_event"]
    assert payload["verified_tube_forbidden_names_found"] == []
    assert payload["unique_validated_tubes"] == 40
    assert payload["validated_transitions"] == 5089
    assert payload["matched_events"] == 56
    assert payload["direct_outward_issued"] == 56
    assert payload["direct_outward_covered"] == 56
    assert payload["brackets_identical_to_green"] == 56
    assert payload["maximum_bracket_width"] == 0
    assert payload["direct_to_greencert_aggregate_runtime_ratio"] > 10.0
    print(
        "PASS: direct outward WDBC baseline is Green-independent, covers 56/56, "
        f"and costs {payload['direct_to_greencert_aggregate_runtime_ratio']:.2f}x matched GreenCert time."
    )


if __name__ == "__main__":
    main()
