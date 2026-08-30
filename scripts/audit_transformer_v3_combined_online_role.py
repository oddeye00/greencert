#!/usr/bin/env python3
"""Independent arithmetic and protocol audit of the combined v3 benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from adaptive_witness_policy import WitnessQuery, acquire_witnesses
from benchmark_transformer_v3_role_sparse import CERT_ROLE, FUSED_ROLE, TRAIN_ROLE
from one_shot_recenter_closure import conservative_one_shot_closure
from probe_jacobian_bound import ProbeConfig
from transformer_block_envelope import objective_hessian_lipschitz
from transformer_certificate_protocol import Candidate
from transformer_v3_certificate import load_candidate, output_path, safe_json
from transformer_v3_protocol import (
    FAMILY_FAILURE_PROBABILITY,
    HORIZON,
    MAXIMUM_POWER,
    PROBES,
    maximum_operator_count,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BENCHMARK_SCRIPT = ROOT / "scripts" / "benchmark_transformer_v3_combined_online_role.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def close(left: float, right: float, tolerance: float = 2.0e-12) -> bool:
    return abs(left - right) <= tolerance * max(abs(right), 1.0e-300)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=(
            "results/transformer_v3_combined_online_role_seed_366_"
            "gate_1_anchor_1120_matched-combined-v5.json"
        ),
    )
    args = parser.parse_args()
    source = (ROOT / args.input).resolve()
    result = safe_json(source)
    if result["benchmark_script_sha256"] != sha256(BENCHMARK_SCRIPT):
        raise AssertionError("combined benchmark script hash mismatch")
    candidate = Candidate(**result["candidate"])
    certificate_path = output_path(candidate)
    certificate = safe_json(certificate_path)
    if result["certificate_sha256"] != sha256(certificate_path):
        raise AssertionError("sealed certificate hash mismatch")
    if result["centerline_sha256"] != certificate["centerline_sha256"]:
        raise AssertionError("centerline identity mismatch")
    event = int(certificate["predicted_persistent_event"])
    horizon = int(result["horizon"])
    persistence = int(certificate["protocol"]["persistence"])
    if result["frozen_predicted_persistent_event"] != event:
        raise AssertionError("benchmark did not key witnesses to the frozen prediction")
    if result["combined_bracket"] != [event, event]:
        raise AssertionError("combined bracket differs from frozen predicted event")

    maximum_candidates = int(maximum_operator_count()["maximum_candidates"])
    green_delta = 0.5 * FAMILY_FAILURE_PROBABILITY / maximum_candidates
    role_outputs = HORIZON + (HORIZON + 1)
    output_delta = (
        0.5 * FAMILY_FAILURE_PROBABILITY / (maximum_candidates * role_outputs)
    )
    green_config = ProbeConfig(PROBES, MAXIMUM_POWER, green_delta)
    output_config = ProbeConfig(PROBES, MAXIMUM_POWER, output_delta)
    budgets = result["family_budget"]
    if not close(float(budgets["green_per_operator_delta"]), green_delta):
        raise AssertionError("Green failure budget mismatch")
    if not close(float(budgets["role_output_per_operator_delta"]), output_delta):
        raise AssertionError("output failure budget mismatch")

    config, _, _, data, parameter, _ = load_candidate(candidate)
    train_pairs, _, _, _, cert_pairs, _ = data
    train_count = int(len(train_pairs))
    cert_count = int(len(cert_pairs))
    all_count = int(config.modulus * config.modulus)
    dimension = int(parameter.numel())
    del dimension  # Documents that the candidate is loaded without rebuilding a trajectory.
    frozen_rows = {int(row["step"]): row for row in certificate["output_rows"]}
    raw_exclusions = {
        step: float(row["raw_exclusion_slack"])
        for step, row in frozen_rows.items()
    }

    def optimistic(step: int) -> WitnessQuery:
        return WitnessQuery(
            step,
            event <= step < event + persistence,
            raw_exclusions.get(step, -math.inf) > 0.0,
        )

    preplan = acquire_witnesses(
        event=event,
        persistence=persistence,
        horizon=horizon,
        raw_exclusion_slacks=raw_exclusions,
        query=optimistic,
        exact_failures={0},
    )
    if not preplan.issued:
        raise AssertionError("independent deterministic witness preplan failed")
    planned = set(preplan.query_order)

    rows = {int(row["step"]): row for row in result["output_rows"]}
    if set(rows) != set(range(1, horizon + 1)):
        raise AssertionError("primary output times are incomplete")
    maximum_map_drift = 0.0
    primary_pair_work = 0
    primary_operator_seconds = 0.0
    primary_wall_seconds = 0.0
    for step, row in rows.items():
        if not close(float(row["c_delta"]), output_config.c_delta()):
            raise AssertionError(f"output calibration mismatch at step {step}")
        expected_upper = (
            0.0
            if float(row["Y"]) <= 0.0
            else (float(row["Y"]) / output_config.c_delta()) ** 0.5
        )
        if not close(float(row["operator_norm_upper_bound"]), expected_upper):
            raise AssertionError(f"output root mismatch at step {step}")
        frozen = frozen_rows[step]
        for key in ("block_second", "block_third", "raw_guarantee_slack", "raw_exclusion_slack"):
            if not close(float(row[key]), float(frozen[key])):
                raise AssertionError(f"stored geometry mismatch at step {step}: {key}")
        if step < horizon:
            expected_role = FUSED_ROLE if step in planned else TRAIN_ROLE
            expected_pairs = train_count + cert_count if step in planned else train_count
            if not bool(row["supports_training"]):
                raise AssertionError(f"training role missing at step {step}")
            first_ball = float(row["operator_norm_upper_bound"]) + float(
                row["block_second"]
            ) * float(result["geometry"]["outer_domain_radius"])
            drift = objective_hessian_lipschitz(
                first_ball,
                float(row["block_second"]),
                float(row["block_third"]),
            )
            maximum_map_drift = max(
                maximum_map_drift,
                math.sqrt(2.0) * config.learning_rate * drift,
            )
        else:
            expected_role = CERT_ROLE
            expected_pairs = cert_count
        if int(row["role"]) != expected_role or int(row["pairs"]) != expected_pairs:
            raise AssertionError(f"role/pair schedule mismatch at step {step}")
        if bool(row["supports_certificate"]) != (step in planned):
            raise AssertionError(f"certificate-role flag mismatch at step {step}")
        primary_pair_work += int(row["pairs"])
        primary_operator_seconds += float(row["operator_seconds"])
        primary_wall_seconds += float(row["wall_seconds"])

    fallback = {int(row["step"]): row for row in result["fallback_rows"]}
    if fallback:
        raise AssertionError("frozen H=26 benchmark unexpectedly required fallback operators")
    if not close(
        maximum_map_drift,
        float(result["maximum_optimizer_derivative_drift_upper"]),
    ):
        raise AssertionError("optimizer derivative drift replay mismatch")

    green_rows = result["green_rows"]
    if len(green_rows) != int(result["decision"]["q_green"]):
        raise AssertionError("Green stopping power mismatch")
    for q, row in enumerate(green_rows, start=1):
        if int(row["power"]) != q or not close(
            float(row["c_delta"]), green_config.c_delta()
        ):
            raise AssertionError(f"Green trace metadata mismatch at q={q}")
        expected_upper = (
            0.0
            if float(row["Y"]) <= 0.0
            else (float(row["Y"]) / green_config.c_delta()) ** (1.0 / (2.0 * q))
        )
        if not close(float(row["operator_norm_upper_bound"]), expected_upper):
            raise AssertionError(f"Green root mismatch at q={q}")
    kappa = float(green_rows[-1]["operator_norm_upper_bound"])
    geometry = result["geometry"]
    closure = conservative_one_shot_closure(
        kappa=kappa,
        derivative_drift=maximum_map_drift,
        response_sequence_norm=float(geometry["signed_response_sequence_norm"]),
        response_max_state_norm=float(geometry["signed_response_max_state_norm"]),
        domain_radius=float(geometry["outer_domain_radius"]),
    )
    if not closure.closure_passed:
        raise AssertionError("independent closure replay failed")
    if not close(
        float(closure.total_pointwise_radius),
        float(result["decision"]["total_pointwise_radius"]),
    ):
        raise AssertionError("closure radius mismatch")

    def query(step: int) -> WitnessQuery:
        row = rows[step]
        radius = float(closure.total_pointwise_radius)
        margin = math.sqrt(2.0) * (
            float(row["operator_norm_upper_bound"]) * radius
            + 0.5 * float(row["block_second"]) * radius * radius
        )
        return WitnessQuery(
            step,
            float(row["raw_guarantee_slack"]) - margin > 0.0,
            float(row["raw_exclusion_slack"]) - margin > 0.0,
        )

    policy = acquire_witnesses(
        event=event,
        persistence=persistence,
        horizon=horizon,
        raw_exclusion_slacks=raw_exclusions,
        query=query,
        exact_failures={0},
    )
    if not policy.issued:
        raise AssertionError(f"independent witness replay abstained: {policy.reason}")
    if list(policy.query_order) != result["decision"]["query_order"]:
        raise AssertionError("witness query order mismatch")

    expected_dense_pairs = horizon * all_count
    if result["pair_work"]["dense_q1"] != expected_dense_pairs:
        raise AssertionError("dense pair-work accounting mismatch")
    if result["pair_work"]["combined_role_fused_q1"] != primary_pair_work:
        raise AssertionError("role-fused pair-work accounting mismatch")
    if not close(
        float(result["operator_seconds"]["combined_output"]),
        primary_operator_seconds,
    ):
        raise AssertionError("combined output operator timing sum mismatch")
    if not close(
        float(result["timings_seconds"]["combined_role_fused_operator_wall"]),
        primary_wall_seconds,
    ):
        raise AssertionError("combined output wall timing sum mismatch")

    audit = {
        "status": "independent combined online-role benchmark audit passed",
        "source": str(source.relative_to(ROOT)),
        "source_sha256": sha256(source),
        "benchmark_script_sha256": sha256(BENCHMARK_SCRIPT),
        "candidate": candidate.__dict__,
        "same_bracket": result["combined_bracket"] == certificate["certified_bracket"],
        "green_stopping_power": len(green_rows),
        "fallback_operators": len(fallback),
        "dense_pair_work": expected_dense_pairs,
        "combined_pair_work": primary_pair_work,
        "pair_work_reduction_fraction": 1.0 - primary_pair_work / expected_dense_pairs,
        "same_process_output_operator_speedup": result["operator_seconds"][
            "same_process_output_speedup"
        ],
        "same_process_output_wall_speedup": result["timings_seconds"][
            "same_process_output_operator_wall_speedup"
        ],
        "matched_end_to_end_replay_speedup": result["timings_seconds"][
            "matched_end_to_end_replay_speedup"
        ],
        "combined_monolithic_end_to_end_seconds": result["timings_seconds"][
            "combined_monolithic_end_to_end"
        ],
        "future_outcome_role": (
            "No joined outcome is used in witness selection, stopping, or arithmetic; "
            "the replay keys to predicted_persistent_event."
        ),
    }
    destination = source.with_name(source.stem + "_independent_audit.json")
    destination.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**audit, "output": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
