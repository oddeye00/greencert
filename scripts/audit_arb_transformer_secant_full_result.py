#!/usr/bin/env python3
"""Independent interval/hash audit of the full Arb secant result."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from flint import arb, ctx

from one_shot_recenter_closure import exact_one_shot_closure


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "transformer_v3_arb_secant_full_v2_audit.json"
PROTOCOL = ROOT / "AMPLIFIED_SECANT_OUTWARD_EXECUTION_PROTOCOL_V2.md"
FOUR_PROTOCOL = ROOT / "AMPLIFIED_SECANT_FOUR_PROBE_PROTOCOL.md"
OUTPUT = ROOT / "results" / "transformer_v3_arb_secant_full_v2_independent_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def close(left: float, right: float, tolerance: float = 5.0e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=0.0)


def main() -> None:
    ctx.prec = 192
    row = json.loads(SOURCE.read_text(encoding="utf-8"))
    if row["status"] != "FULL-SEQUENCE OUTWARD ARB SECANT PROBE AUDIT V2 PASSED":
        raise AssertionError("status changed")
    if row["candidate"] != {"seed": 366, "threshold": 0.7, "anchor": 1040}:
        raise AssertionError("candidate changed")
    if (
        int(row["horizon"]),
        int(row["checkpoints"]),
        int(row["probes"]),
        int(row["scalar_intervals"]),
        int(row["precision_bits"]),
    ) != (52, 51, 4, 204, 192):
        raise AssertionError("frozen execution shape changed")
    if row["protocol_sha256"] != sha256(PROTOCOL):
        raise AssertionError("outward protocol hash mismatch")
    if row["four_probe_protocol_sha256"] != sha256(FOUR_PROTOCOL):
        raise AssertionError("probe protocol hash mismatch")
    for relative, expected in row["source_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise AssertionError(f"source hash changed: {relative}")
    checkpoint_rows = row["checkpoint_rows"]
    if [int(item["step"]) for item in checkpoint_rows] != list(range(1, 52)):
        raise AssertionError("checkpoint order changed")
    sums = [arb(0) for _ in range(4)]
    parsed_intervals = 0
    for item in checkpoint_rows:
        if len(item["intervals"]) != 4:
            raise AssertionError("checkpoint probe count changed")
        for index, text in enumerate(item["intervals"]):
            sums[index] += arb(text)
            parsed_intervals += 1
    if parsed_intervals != 204:
        raise AssertionError("interval count mismatch")
    stored_sums = [arb(text) for text in row["summed_projection_intervals"]]
    for recomputed, stored in zip(sums, stored_sums):
        if not stored.contains(recomputed):
            raise AssertionError("stored sum does not enclose recomputed sum")
    magnitudes = [abs(value) for value in sums]
    projection_upper = max(
        (value.upper() for value in magnitudes), key=lambda value: float(value)
    )
    delta = arb(1) / 10**6
    calibration = arb(2).sqrt() * (delta.root(4)).erfinv()
    calibration_lower = calibration.lower()
    norm_bound = projection_upper / calibration_lower
    norm_upper = math.nextafter(float(norm_bound.upper()), math.inf)
    if not arb(row["calibration_interval"]).contains(calibration):
        raise AssertionError("stored calibration does not enclose recomputation")
    if not close(norm_upper, row["secant_forcing_norm_upper"]):
        raise AssertionError("forcing norm bound mismatch")
    if norm_upper >= 2.0e-29:
        raise AssertionError("forcing norm bound unexpectedly large")
    sigma = float(row["analytic_secant_discrepancy_upper"])
    closure_row = row["closure"]
    kappa = float(closure_row["kappa"])
    beta = math.nextafter(kappa * (sigma + norm_upper), math.inf)
    if not close(beta, row["response_free_beta_upper"]):
        raise AssertionError("response-free beta mismatch")
    closure = exact_one_shot_closure(
        kappa=kappa,
        derivative_drift=float(closure_row["derivative_drift"]),
        response_sequence_norm=float(closure_row["response_sequence_norm"]),
        response_max_state_norm=float(closure_row["response_max_state_norm"]),
        corrected_defect_response_bound=beta,
        domain_radius=float(closure_row["domain_radius"]),
    )
    if not closure.closure_passed:
        raise AssertionError("recomputed closure fails")
    if not close(closure.total_pointwise_radius, closure_row["total_pointwise_radius"]):
        raise AssertionError("closure radius mismatch")
    headroom = float(row["forcing_cap"]) / (sigma + norm_upper)
    if not close(headroom, row["forcing_headroom_ratio"]):
        raise AssertionError("headroom mismatch")
    if row["bracket"] != [28, 28]:
        raise AssertionError("bracket changed")
    if int(row["outcome_files_read"]) != 0:
        raise AssertionError("outcome boundary violated")
    payload = {
        "status": "INDEPENDENT FULL-SEQUENCE ARB SECANT AUDIT V2 PASSED",
        "source_sha256": sha256(SOURCE),
        "protocol_sha256": sha256(PROTOCOL),
        "candidate": row["candidate"],
        "intervals_recomputed": parsed_intervals,
        "calibration_interval": calibration.str(90, radius=True, more=True),
        "secant_forcing_norm_upper": norm_upper,
        "forcing_headroom_ratio": headroom,
        "closure_radius": closure.total_pointwise_radius,
        "bracket": row["bracket"],
        "jet_wall_seconds": row["jet_wall_seconds"],
        "total_wall_seconds": row["total_wall_seconds"],
        "outcome_files_read": 0,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
