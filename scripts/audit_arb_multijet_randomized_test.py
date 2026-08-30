#!/usr/bin/env python3
"""Run and seal the randomized frozen-source Arb multijet test output."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "scripts" / "test_arb_transformer_multijet_randomized.py"
FROZEN = ROOT / "scripts" / "arb_transformer_multijet.py"
OUTPUT = ROOT / "results" / "transformer_arb_multijet_randomized_test_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    completed = subprocess.run(
        [sys.executable, str(TEST)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("randomized test emitted no result")
    result = ast.literal_eval(lines[-1])
    if result.get("status") != "randomized Arb Transformer multijet tests passed":
        raise AssertionError("randomized test status changed")
    if int(result["cases"]) != 12 or int(result["probe_directions_per_case"]) != 3:
        raise AssertionError("randomized test design changed")
    if float(result["maximum_jet_midpoint_error"]) > 7.0e-16:
        raise AssertionError("randomized jet discrepancy increased")
    payload = {
        **result,
        "status": "SEALED RANDOMIZED ARB MULTIJET TEST PASSED",
        "test_sha256": sha256(TEST),
        "frozen_multijet_sha256": sha256(FROZEN),
        "stderr": completed.stderr,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
