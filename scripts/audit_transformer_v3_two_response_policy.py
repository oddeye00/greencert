#!/usr/bin/env python3
"""Outcome-blind policy and fourth-order headroom audit for two responses."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

from transformer_certificate_protocol import Candidate
from transformer_v3_certificate import load_candidate, safe_json
from transformer_v3_protocol import MAXIMUM_POWER, PROBES


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOURCE = RESULTS / "transformer_v3_two_response_postseal_audit.json"
OUTPUT = RESULTS / "transformer_v3_two_response_policy_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    source = safe_json(SOURCE)
    rows = []
    for record in source["rows"]:
        if not record.get("evaluable"):
            continue
        candidate = Candidate(**record["candidate"])
        config, _, _, _, _, _ = load_candidate(candidate)
        response_norm = float(record["response_sequence_norm"])
        response_max = float(record["response_max_state_norm"])
        denominator = (
            math.sqrt(2.0)
            * float(config.learning_rate)
            * response_max**2
            * response_norm
            / 6.0
        )
        surrogate_power = int(record["surrogate_earliest_power"])
        old_power = record["old_earliest_power"]
        baseline_terminal_power = (
            int(old_power) if old_power is not None else MAXIMUM_POWER
        )
        beneficial = surrogate_power < baseline_terminal_power
        selected = next(
            row
            for row in record["power_audits"]
            if int(row["power"]) == surrogate_power
        )
        linear = float(
            selected["surrogate_closure"]["linearized_remainder_coefficient"]
        )
        if linear >= 1.0:
            raise AssertionError("surrogate issued before the necessary linear gate")
        previous = [
            row
            for row in record["power_audits"]
            if int(row["power"]) < surrogate_power
        ]
        if any(
            float(row["surrogate_closure"]["linearized_remainder_coefficient"]) < 1.0
            and bool(row["surrogate_closure"]["closure_passed"])
            for row in previous
        ):
            raise AssertionError("surrogate stopping power is not earliest")

        saved_levels = (
            baseline_terminal_power - surrogate_power if beneficial else 0
        )
        gross_saved_hvps = saved_levels * 2 * PROBES * int(record["horizon"])
        second_response_hvps = int(record["horizon"]) if beneficial else 0
        net_saved_hvps = gross_saved_hvps - second_response_hvps
        headroom = float(selected["admissible_sigma_q_plus_tau_q"])
        fourth_cap = math.inf if denominator == 0.0 else headroom / denominator
        rows.append(
            {
                "candidate": record["candidate"],
                "horizon": int(record["horizon"]),
                "old_earliest_power": old_power,
                "surrogate_earliest_power": surrogate_power,
                "adaptive_second_response_invoked": beneficial,
                "converted_old_abstention": (
                    old_power is None and bool(record["surrogate_issued"])
                ),
                "progressive_power_levels_saved": saved_levels,
                "gross_objective_hvp_calls_saved": gross_saved_hvps,
                "second_response_hvp_calls": second_response_hvps,
                "net_objective_hvp_calls_saved_excluding_third_products": (
                    net_saved_hvps
                ),
                "directional_third_products_if_invoked": (
                    max(0, int(record["horizon"]) - 1) if beneficial else 0
                ),
                "linear_gate_at_surrogate_power": linear,
                "quadratic_surrogate_response_norm": float(
                    record["quadratic_surrogate_second_response_norm"]
                ),
                "old_response_bound_at_surrogate_power": float(
                    selected["old_corrected_defect_response_bound"]
                ),
                "surrogate_to_old_response_bound_ratio": float(
                    selected["surrogate_to_old_response_bound_ratio"]
                ),
                "admissible_sigma_q_plus_tau_q": headroom,
                "uniform_objective_fourth_bound_cap_from_p2Z": fourth_cap,
            }
        )

    invoked = [row for row in rows if row["adaptive_second_response_invoked"]]
    caps = [
        row["uniform_objective_fourth_bound_cap_from_p2Z"] for row in invoked
    ]
    payload = {
        "status": "OUTCOME-BLIND ADAPTIVE TWO-RESPONSE POLICY AUDIT PASSED",
        "evidence_boundary": (
            "Post-seal method-development audit. No future outcomes are read and "
            "prospective v3 counts remain unchanged. The fourth-order caps are "
            "required uniform objective bounds, not measured derivatives."
        ),
        "policy": (
            "At each progressive power: stop if zero-extra closure issues; refine "
            "if b*p>=1; otherwise compute the cancellation-safe quadratic injection "
            "and one deterministic second Green response, then reuse it at later powers."
        ),
        "evaluable_records": len(rows),
        "adaptive_second_response_invocations": len(invoked),
        "unchanged_cases_with_no_added_response": len(rows) - len(invoked),
        "converted_old_abstentions": sum(
            row["converted_old_abstention"] for row in rows
        ),
        "earlier_existing_certificates": sum(
            row["old_earliest_power"] is not None
            and row["adaptive_second_response_invoked"]
            for row in rows
        ),
        "total_progressive_power_levels_saved": sum(
            row["progressive_power_levels_saved"] for row in rows
        ),
        "total_gross_objective_hvp_calls_saved": sum(
            row["gross_objective_hvp_calls_saved"] for row in rows
        ),
        "total_second_response_hvp_calls": sum(
            row["second_response_hvp_calls"] for row in rows
        ),
        "total_net_objective_hvp_calls_saved_excluding_third_products": sum(
            row["net_objective_hvp_calls_saved_excluding_third_products"]
            for row in rows
        ),
        "total_directional_third_products": sum(
            row["directional_third_products_if_invoked"] for row in rows
        ),
        "median_forcing_ratio_when_invoked": statistics.median(
            row["surrogate_to_old_response_bound_ratio"] for row in invoked
        ),
        "maximum_forcing_ratio_when_invoked": max(
            row["surrogate_to_old_response_bound_ratio"] for row in invoked
        ),
        "minimum_uniform_objective_fourth_bound_cap": min(caps),
        "median_uniform_objective_fourth_bound_cap": statistics.median(caps),
        "source": SOURCE.relative_to(ROOT).as_posix(),
        "source_sha256": sha256(SOURCE),
        "outcome_files_read": 0,
        "randomized_queries_added": 0,
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
