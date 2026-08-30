#!/usr/bin/env python3
"""Post-seal replay of chi-Frobenius and hybrid q=1 probe bounds."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

from adaptive_witness_policy import WitnessQuery, acquire_witnesses
from chi_block_gram import chi_block_operator_upper_bound, chi_lower_calibration
from one_shot_recenter_closure import conservative_one_shot_closure
from probe_jacobian_bound import ProbeConfig
from transformer_block_envelope import objective_hessian_lipschitz
from transformer_certificate_protocol import Candidate
from transformer_v3_certificate import load_candidate, output_path, safe_json
from transformer_v3_protocol import MAXIMUM_POWER, PROBES


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOURCE = RESULTS / (
    "transformer_v3_combined_online_role_seed_366_gate_1_anchor_1120_"
    "matched-combined-v5.json"
)
OUTPUT = RESULTS / "transformer_v3_chi_block_postseal_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    source = safe_json(SOURCE)
    candidate = Candidate(**source["candidate"])
    certificate = safe_json(output_path(candidate))
    config, _, _, _, _, _ = load_candidate(candidate)
    rows = {int(row["step"]): row for row in source["output_rows"]}
    green = source["green_rows"][0]
    horizon = int(source["horizon"])
    event = int(source["frozen_predicted_persistent_event"])
    persistence = int(certificate["protocol"]["persistence"])
    geometry = source["geometry"]
    output_delta = float(source["family_budget"]["role_output_per_operator_delta"])
    green_delta = float(source["family_budget"]["green_per_operator_delta"])
    raw_exclusions = {
        step: float(row["raw_exclusion_slack"]) for step, row in rows.items()
    }

    def max_bound(row: dict, delta: float) -> float:
        calibration = ProbeConfig(PROBES, MAXIMUM_POWER, delta).c_delta()
        return math.sqrt(float(row["Y"]) / calibration)

    def chi_bound(row: dict, delta: float) -> float:
        calibration = chi_lower_calibration(probes=PROBES, delta=delta)
        return chi_block_operator_upper_bound(
            terminal_frobenius_norm=float(row["block_frobenius_terminal_norm"]),
            calibration=calibration,
            power=1,
        )

    methods = {
        "sealed_max": {
            "outputs": {step: max_bound(row, output_delta) for step, row in rows.items()},
            "green": max_bound(green, green_delta),
            "failure_budget": "delta per operator",
        },
        "chi_frobenius": {
            "outputs": {step: chi_bound(row, output_delta) for step, row in rows.items()},
            "green": chi_bound(green, green_delta),
            "failure_budget": "delta per operator",
        },
        "bonferroni_hybrid": {
            "outputs": {
                step: min(max_bound(row, output_delta / 2.0), chi_bound(row, output_delta / 2.0))
                for step, row in rows.items()
            },
            "green": min(max_bound(green, green_delta / 2.0), chi_bound(green, green_delta / 2.0)),
            "failure_budget": "delta/2 for max and delta/2 for chi per operator",
        },
    }

    def replay(method: dict) -> dict:
        output_uppers = method["outputs"]
        maximum_map_drift = 0.0
        for step in range(1, horizon):
            row = rows[step]
            first_ball = float(output_uppers[step]) + float(row["block_second"]) * float(
                geometry["outer_domain_radius"]
            )
            objective_drift = objective_hessian_lipschitz(
                first_ball,
                float(row["block_second"]),
                float(row["block_third"]),
            )
            maximum_map_drift = max(
                maximum_map_drift,
                math.sqrt(2.0) * config.learning_rate * objective_drift,
            )
        closure = conservative_one_shot_closure(
            kappa=float(method["green"]),
            derivative_drift=maximum_map_drift,
            response_sequence_norm=float(geometry["signed_response_sequence_norm"]),
            response_max_state_norm=float(geometry["signed_response_max_state_norm"]),
            domain_radius=float(geometry["outer_domain_radius"]),
        )
        if not closure.closure_passed:
            return {
                "issued": False,
                "reason": "optimizer closure failed",
                "closure": closure.as_dict(),
            }
        radius = float(closure.total_pointwise_radius)
        slacks: list[float] = []

        def query(step: int) -> WitnessQuery:
            row = rows[step]
            margin = math.sqrt(2.0) * (
                float(output_uppers[step]) * radius
                + 0.5 * float(row["block_second"]) * radius * radius
            )
            guarantee = float(row["raw_guarantee_slack"]) - margin
            exclusion = float(row["raw_exclusion_slack"]) - margin
            slacks.append(
                guarantee if event <= step < event + persistence else exclusion
            )
            return WitnessQuery(step, guarantee > 0.0, exclusion > 0.0)

        policy = acquire_witnesses(
            event=event,
            persistence=persistence,
            horizon=horizon,
            raw_exclusion_slacks=raw_exclusions,
            query=query,
            exact_failures={0},
        )
        return {
            "issued": bool(policy.issued),
            "reason": policy.reason,
            "bracket": [event, event] if policy.issued else None,
            "green_upper": float(method["green"]),
            "maximum_output_upper": max(float(value) for value in output_uppers.values()),
            "maximum_map_drift": maximum_map_drift,
            "closure": closure.as_dict(),
            "minimum_queried_logic_slack": min(slacks),
            "query_order": list(policy.query_order),
        }

    replays = {name: replay(method) for name, method in methods.items()}
    if not replays["sealed_max"]["issued"]:
        raise AssertionError("sealed max replay failed")
    ratios = [
        methods["chi_frobenius"]["outputs"][step]
        / methods["sealed_max"]["outputs"][step]
        for step in rows
    ]
    payload = {
        "status": "post-seal chi-block probe audit passed",
        "scope": (
            "Alternative exact-arithmetic Gaussian calibration on one immutable "
            "q=1 case; not a new prospective certificate."
        ),
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "candidate": candidate.__dict__,
        "calibrations": {
            "output_max": ProbeConfig(PROBES, MAXIMUM_POWER, output_delta).c_delta(),
            "output_chi": chi_lower_calibration(probes=PROBES, delta=output_delta),
            "green_max": ProbeConfig(PROBES, MAXIMUM_POWER, green_delta).c_delta(),
            "green_chi": chi_lower_calibration(probes=PROBES, delta=green_delta),
        },
        "replays": replays,
        "chi_to_max_output_upper_ratio": {
            "minimum": min(ratios),
            "median": statistics.median(ratios),
            "maximum": max(ratios),
        },
        "chi_to_max_green_upper_ratio": (
            methods["chi_frobenius"]["green"] / methods["sealed_max"]["green"]
        ),
        "interpretation": (
            "Chi-Frobenius uses the complete committed probe block at no extra "
            "Gram application cost.  It is retained only if the replay improves "
            "the downstream certificate geometry."
        ),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**payload, "output_sha256": sha256(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
