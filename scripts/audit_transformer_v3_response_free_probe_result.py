#!/usr/bin/env python3
"""Independent arithmetic/hash audit of the response-free probe result."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from probe_jacobian_bound import c_delta


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOURCE = RESULTS / "transformer_v3_response_free_probe_audit.json"
PROTOCOL = ROOT / "AMPLIFIED_SECANT_PROBE_PROTOCOL.md"
OUTPUT = RESULTS / "transformer_v3_response_free_probe_independent_audit.json"
NONCE = "2df178250abfd4272951e2493a1c1b93ddc7d29e73a4359246eee8998f8a0778"
DOMAIN = "greencert-response-free-secant-v1|"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def close(left: float, right: float, tolerance: float = 5.0e-13) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=0.0)


def main() -> None:
    row = json.loads(SOURCE.read_text(encoding="utf-8"))
    if row["status"] != "OUTCOME-BLIND RESPONSE-FREE PROBE AUDIT COMPLETED":
        raise AssertionError("status changed")
    if row["candidate"] != {"seed": 366, "threshold": 0.7, "anchor": 1040}:
        raise AssertionError("candidate changed")
    if int(row["horizon"]) != 52 or int(row["power"]) != 1:
        raise AssertionError("window changed")
    if int(row["probes"]) != 16 or float(row["delta"]) != 1.0e-6:
        raise AssertionError("probe block changed")
    expected_seed = int.from_bytes(
        hashlib.sha256((DOMAIN + NONCE).encode("ascii")).digest()[:8], "little"
    ) % (2**63 - 1)
    if int(row["probe_seed"]) != expected_seed:
        raise AssertionError("probe seed derivation changed")
    if row["protocol_sha256"] != sha256(PROTOCOL):
        raise AssertionError("protocol hash mismatch")
    projections = [float(value) for value in row["projections_float64"]]
    if len(projections) != 16:
        raise AssertionError("projection count changed")
    maximum = max(abs(value) for value in projections)
    calibration = c_delta(float(row["delta"]), int(row["probes"]))
    norm_upper = maximum / calibration
    if not close(maximum, row["projection_absolute_max_float64"]):
        raise AssertionError("projection maximum mismatch")
    if not close(calibration, row["calibration"]):
        raise AssertionError("calibration mismatch")
    if not close(norm_upper, row["secant_norm_upper_point_projection"]):
        raise AssertionError("norm bound mismatch")
    observed = float(row["secant_injection_norm_float64"])
    if not observed <= norm_upper:
        raise AssertionError("observed norm exceeds the point-projection bound")
    kappa = float(row["closure"]["kappa"])
    sigma = float(row["analytic_secant_discrepancy_upper"])
    beta = kappa * (sigma + norm_upper)
    if not close(beta, row["response_free_beta_upper_point_projection"]):
        raise AssertionError("response-free beta mismatch")
    headroom = float(row["forcing_cap"]) / (sigma + norm_upper)
    if not close(headroom, row["forcing_headroom_ratio"]):
        raise AssertionError("headroom mismatch")
    if row["bracket"] != [28, 28] or not row["closure"]["closure_passed"]:
        raise AssertionError("closure or bracket changed")
    if int(row["outcome_files_read"]) != 0:
        raise AssertionError("outcome boundary violated")
    payload = {
        "status": "INDEPENDENT RESPONSE-FREE PROBE AUDIT PASSED",
        "source_sha256": sha256(SOURCE),
        "protocol_sha256": sha256(PROTOCOL),
        "candidate": row["candidate"],
        "probes": len(projections),
        "calibration": calibration,
        "projection_maximum": maximum,
        "secant_norm_upper": norm_upper,
        "observed_norm": observed,
        "bound_to_observed_ratio": norm_upper / observed,
        "forcing_headroom_ratio": headroom,
        "bracket": row["bracket"],
        "outcome_files_read": 0,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
