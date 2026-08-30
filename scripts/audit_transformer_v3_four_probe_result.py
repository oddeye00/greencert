#!/usr/bin/env python3
"""Independent audit of the frozen four-probe response-free result."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from probe_jacobian_bound import c_delta


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "transformer_v3_four_probe_audit.json"
PROTOCOL = ROOT / "AMPLIFIED_SECANT_FOUR_PROBE_PROTOCOL.md"
OUTPUT = ROOT / "results" / "transformer_v3_four_probe_independent_audit.json"
NONCE = "5a37e5ccaf6834c438fde251d52ec1de313329314377d70cc1cb25e62fc52f2a"
DOMAIN = "greencert-response-free-secant-four-probe-v1|"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=5.0e-13, abs_tol=0.0)


def main() -> None:
    row = json.loads(SOURCE.read_text(encoding="utf-8"))
    if row["status"] != "OUTCOME-BLIND FOUR-PROBE AUDIT COMPLETED":
        raise AssertionError("status changed")
    if row["candidate"] != {"seed": 366, "threshold": 0.7, "anchor": 1040}:
        raise AssertionError("candidate changed")
    if (int(row["horizon"]), int(row["power"]), int(row["probes"])) != (52, 1, 4):
        raise AssertionError("frozen design changed")
    if float(row["delta"]) != 1.0e-6 or float(row["amplification"]) != 4096.0:
        raise AssertionError("frozen constants changed")
    seed = int.from_bytes(
        hashlib.sha256((DOMAIN + NONCE).encode("ascii")).digest()[:8], "little"
    ) % (2**63 - 1)
    if int(row["probe_seed"]) != seed:
        raise AssertionError("seed derivation mismatch")
    if row["protocol_sha256"] != sha256(PROTOCOL):
        raise AssertionError("protocol hash mismatch")
    projections = [float(value) for value in row["projections_float64"]]
    if len(projections) != 4:
        raise AssertionError("projection count changed")
    maximum = max(abs(value) for value in projections)
    calibration = c_delta(1.0e-6, 4)
    upper = maximum / calibration
    if not close(maximum, row["projection_absolute_max_float64"]):
        raise AssertionError("projection maximum mismatch")
    if not close(calibration, row["calibration"]):
        raise AssertionError("calibration mismatch")
    if not close(upper, row["secant_norm_upper_point_projection"]):
        raise AssertionError("projection bound mismatch")
    observed = float(row["secant_injection_norm_float64"])
    ratio = upper / observed
    if not close(ratio, row["probe_bound_to_observed_norm_ratio"]):
        raise AssertionError("bound inflation mismatch")
    kappa = float(row["closure"]["kappa"])
    sigma = float(row["analytic_secant_discrepancy_upper"])
    beta = kappa * (sigma + upper)
    if not close(beta, row["response_free_beta_upper_point_projection"]):
        raise AssertionError("beta mismatch")
    headroom = float(row["forcing_cap"]) / (sigma + upper)
    if not close(headroom, row["forcing_headroom_ratio"]):
        raise AssertionError("headroom mismatch")
    if row["bracket"] != [28, 28] or not row["closure"]["closure_passed"]:
        raise AssertionError("closure changed")
    if int(row["outcome_files_read"]) != 0:
        raise AssertionError("outcome boundary violated")
    payload = {
        "status": "INDEPENDENT FOUR-PROBE AUDIT PASSED",
        "source_sha256": sha256(SOURCE),
        "protocol_sha256": sha256(PROTOCOL),
        "candidate": row["candidate"],
        "probes": 4,
        "calibration": calibration,
        "bound_to_observed_ratio": ratio,
        "forcing_headroom_ratio": headroom,
        "bracket": row["bracket"],
        "outcome_files_read": 0,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
