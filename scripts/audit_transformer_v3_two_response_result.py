#!/usr/bin/env python3
"""Independent structural verifier for the post-seal two-response audits."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOURCE = RESULTS / "transformer_v3_two_response_postseal_audit.json"
POLICY = RESULTS / "transformer_v3_two_response_policy_audit.json"
LOCAL = RESULTS / "transformer_v3_two_response_local_fourth_audit.json"
OUTPUT = RESULTS / "transformer_v3_two_response_independent_audit.json"
PROBES = 16
MAXIMUM_POWER = 8


def load(path: Path) -> dict:
    lowered = path.name.lower()
    if lowered.endswith(".outcomes.json") or lowered.endswith(".sealed.log"):
        raise RuntimeError(f"forbidden outcome read: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def key(row: dict) -> tuple[int, float, int]:
    candidate = row["candidate"]
    return (
        int(candidate["seed"]),
        float(candidate["threshold"]),
        int(candidate["anchor"]),
    )


def main() -> None:
    source = load(SOURCE)
    policy = load(POLICY)
    local = load(LOCAL)
    evaluable = [row for row in source["rows"] if row.get("evaluable")]
    if len(source["rows"]) != 19 or len(evaluable) != 15:
        raise AssertionError("unexpected source denominators")
    if any(row.get("outcome_files_read") != 0 for row in source["rows"]):
        raise AssertionError("source audit reports an outcome read")
    if not all(row["surrogate_issued"] for row in evaluable):
        raise AssertionError("not every Green-evaluable row issues")
    if sum(row["old_certificate_issued"] for row in evaluable) != 11:
        raise AssertionError("old issuance count changed")
    converted = [
        row
        for row in evaluable
        if row["surrogate_issued"] and not row["old_certificate_issued"]
    ]
    earlier = [
        row
        for row in evaluable
        if row["old_earliest_power"] is not None
        and row["surrogate_earliest_power"] < row["old_earliest_power"]
    ]
    if len(converted) != 4 or len(earlier) != 5:
        raise AssertionError("conversion/earlier-power counts changed")
    for row in evaluable:
        for power in row["power_audits"]:
            if power["quadratic_surrogate_injection_norm"] > (
                power["taylor_injection_bound"] * (1.0 + 1.0e-12) + 1.0e-300
            ):
                raise AssertionError("quadratic injection exceeds Taylor bound")
            if power["observed_response_to_kappa_injection_ratio"] > 1.0 + 1.0e-12:
                raise AssertionError("second response exceeds sealed Green bound")

    source_by_key = {key(row): row for row in evaluable}
    policy_rows = policy["rows"]
    if len(policy_rows) != 15 or set(map(key, policy_rows)) != set(source_by_key):
        raise AssertionError("policy/source candidate mismatch")
    invoked = [row for row in policy_rows if row["adaptive_second_response_invoked"]]
    if len(invoked) != 9:
        raise AssertionError("adaptive invocation count changed")
    saved_levels = 0
    gross = 0
    response_calls = 0
    third_products = 0
    for row in policy_rows:
        source_row = source_by_key[key(row)]
        baseline = (
            int(row["old_earliest_power"])
            if row["old_earliest_power"] is not None
            else MAXIMUM_POWER
        )
        expected_saved = (
            baseline - int(row["surrogate_earliest_power"])
            if row["adaptive_second_response_invoked"]
            else 0
        )
        if expected_saved != int(row["progressive_power_levels_saved"]):
            raise AssertionError("saved power accounting mismatch")
        horizon = int(row["horizon"])
        expected_gross = expected_saved * 2 * PROBES * horizon
        expected_response = horizon if row["adaptive_second_response_invoked"] else 0
        if int(row["gross_objective_hvp_calls_saved"]) != expected_gross:
            raise AssertionError("gross HVP accounting mismatch")
        if int(row["second_response_hvp_calls"]) != expected_response:
            raise AssertionError("response HVP accounting mismatch")
        if (
            float(row["quadratic_surrogate_response_norm"])
            != float(source_row["quadratic_surrogate_second_response_norm"])
        ):
            raise AssertionError("policy response norm differs from source")
        saved_levels += expected_saved
        gross += expected_gross
        response_calls += expected_response
        third_products += (
            max(0, horizon - 1) if row["adaptive_second_response_invoked"] else 0
        )
    if (saved_levels, gross, response_calls, third_products) != (
        22,
        169056,
        1806,
        1797,
    ):
        raise AssertionError("aggregate operator accounting changed")

    local_rows = local["rows"]
    if len(local_rows) != 9 or set(map(key, local_rows)) != set(map(key, invoked)):
        raise AssertionError("local fourth-order cohort mismatch")
    if not all(row["local_fourth_order_taylor_gate_passed"] for row in local_rows):
        raise AssertionError("a local fourth-order gate failed")
    ratios = [float(row["headroom_to_taylor_error_ratio"]) for row in local_rows]
    if min(ratios) <= 55.0:
        raise AssertionError("local fourth-order headroom unexpectedly weakened")
    for row in local_rows:
        recomputed = (
            float(row["admissible_sigma_q_plus_tau_q"])
            - float(row["directional_quadratic_taylor_error_upper"])
        )
        if not math.isclose(
            recomputed,
            float(row["remaining_arithmetic_and_recurrence_headroom"]),
            rel_tol=2.0e-15,
            abs_tol=0.0,
        ):
            raise AssertionError("remaining fourth-order headroom mismatch")
        if row.get("outcome_files_read") != 0:
            raise AssertionError("local audit reports an outcome read")

    payload = {
        "status": "INDEPENDENT TWO-RESPONSE STRUCTURAL AUDIT PASSED",
        "certificate_records": len(source["rows"]),
        "green_evaluable_records": len(evaluable),
        "surrogate_issued": sum(row["surrogate_issued"] for row in evaluable),
        "old_issued": sum(row["old_certificate_issued"] for row in evaluable),
        "converted_old_abstentions": len(converted),
        "earlier_power_cases": len(earlier),
        "adaptive_response_invocations": len(invoked),
        "unchanged_cases_without_added_response": len(policy_rows) - len(invoked),
        "progressive_power_levels_saved": saved_levels,
        "gross_hvp_calls_saved": gross,
        "second_response_hvp_calls": response_calls,
        "net_hvp_calls_saved_excluding_third_products": gross - response_calls,
        "directional_third_products": third_products,
        "local_fourth_order_passes": len(local_rows),
        "minimum_local_fourth_headroom_ratio": min(ratios),
        "median_local_fourth_headroom_ratio": statistics.median(ratios),
        "outcome_files_read": 0,
        "source_sha256": sha256(SOURCE),
        "policy_sha256": sha256(POLICY),
        "local_fourth_sha256": sha256(LOCAL),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
