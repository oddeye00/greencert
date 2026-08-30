#!/usr/bin/env python3
"""Monolithic online-Green plus role-fused output benchmark.

This post-seal replay composes two theorem-preserving implementation changes on
one immutable issued Transformer candidate:

1. stop the Green power cascade as soon as the certificate issues; and
2. transport only the output roles required by optimizer closure and the
   predictable first-passage witness policy, fusing roles at shared times.

No joined future outcome enters witness selection, stopping, or certificate
arithmetic.  The prospective certificate population and all headline counts
remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from adaptive_witness_policy import WitnessQuery, acquire_witnesses
from batched_green_operator import (
    make_batched_output_gram_operator,
    make_batched_transformer_green_products,
)
from benchmark_transformer_v3_role_sparse import (
    CERT_ROLE,
    FUSED_ROLE,
    TRAIN_ROLE,
    role_identity,
)
from one_shot_recenter_closure import conservative_one_shot_closure
from online_progressive_gram import OnlineGramState
from probe_jacobian_bound import ProbeConfig, namespaced_probe_seed
from transformer_block_envelope import (
    ball_valid_envelope,
    objective_hessian_lipschitz,
)
from transformer_certificate_protocol import Candidate
from transformer_four_sweep_development_audit import to_scaled
from transformer_green_development_audit import build_frozen_centerline
from transformer_green_operator import make_transformer_green_products
from transformer_hvp_grokking import logits
from transformer_v3_certificate import (
    METHOD_SEAL,
    _gate_raw_slacks,
    frozen_candidates,
    load_candidate,
    output_path,
    safe_json,
)
from transformer_v3_protocol import (
    FAMILY_FAILURE_PROBABILITY,
    HORIZON,
    MAXIMUM_POWER,
    PROBES,
    green_identity,
    maximum_operator_count,
    output_identity,
    probe_config,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_CANDIDATE = Candidate(366, 0.80, 1120)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def relative_error(left: float, right: float) -> float:
    return abs(left - right) / max(abs(right), torch.finfo(torch.float64).tiny)


def run_q1(
    *,
    theta: torch.Tensor,
    pairs: torch.Tensor,
    template,
    spec,
    config: ProbeConfig,
    seed: int,
) -> tuple[dict, OnlineGramState, float]:
    started = time.perf_counter()
    apply = make_batched_output_gram_operator(theta, pairs, template, spec)
    state = OnlineGramState.initialize(
        dimension=theta.numel(),
        dtype=theta.dtype,
        device=theta.device,
        config=config,
        seed=seed,
    )
    row = state.step(apply)
    row["block_frobenius_terminal_norm"] = float(
        torch.linalg.vector_norm(state.vectors)
    )
    return row, state, time.perf_counter() - started


def benchmark(candidate: Candidate, *, run_label: str) -> dict:
    method = safe_json(METHOD_SEAL)
    candidates, horizons, _ = frozen_candidates()
    if candidate not in horizons:
        raise ValueError(f"candidate is outside the sealed v3 set: {candidate}")
    horizon = int(horizons[candidate])
    certificate_path = output_path(candidate)
    certificate = safe_json(certificate_path)
    if not bool(certificate.get("certificate_issued")):
        raise ValueError("combined benchmark requires a sealed issued certificate")
    # The witness schedule is keyed to the frozen centerline prediction, not to
    # the subsequently joined outcome or to a certificate endpoint.
    event = int(certificate["predicted_persistent_event"])
    if certificate["certified_bracket"] != [event, event]:
        raise ValueError(
            "combined benchmark expects the sealed singleton bracket to equal "
            "the frozen centerline event"
        )

    config, template, spec, data, parameter, velocity = load_candidate(candidate)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = data
    dimension = int(parameter.numel())

    maximum_candidates = int(maximum_operator_count()["maximum_candidates"])
    green_delta = 0.5 * FAMILY_FAILURE_PROBABILITY / maximum_candidates
    role_outputs_per_candidate = HORIZON + (HORIZON + 1)
    output_delta = (
        0.5
        * FAMILY_FAILURE_PROBABILITY
        / (maximum_candidates * role_outputs_per_candidate)
    )
    role_config = ProbeConfig(PROBES, MAXIMUM_POWER, output_delta)
    green_config = ProbeConfig(PROBES, MAXIMUM_POWER, green_delta)
    master_nonce = str(method["master_nonce"])

    started = time.perf_counter()
    center_started = time.perf_counter()
    path = build_frozen_centerline(
        config,
        template,
        spec,
        train_pairs,
        train_labels,
        parameter,
        velocity,
    )
    center = path["center"]
    scaled_center = path["scaled_center"]
    residual = torch.stack(
        [
            to_scaled(path["map_step"](center[step]), dimension, config.learning_rate)
            - scaled_center[step + 1]
            for step in range(horizon)
        ]
    )
    scalar_green, _ = make_transformer_green_products(
        center[:horizon, :dimension],
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )
    signed_response = scalar_green(residual.reshape(-1)).reshape(horizon, -1)
    response_norm = float(torch.linalg.vector_norm(signed_response))
    response_max = float(torch.linalg.vector_norm(signed_response, dim=1).max())
    domain_radius = 2.0 * response_norm
    center_seconds = time.perf_counter() - center_started
    if path["centerline_sha256"] != certificate["centerline_sha256"]:
        raise RuntimeError("combined benchmark centerline hash mismatch")
    for key, observed in (
        ("signed_response_sequence_norm", response_norm),
        ("signed_response_max_state_norm", response_max),
        ("outer_domain_radius", domain_radius),
    ):
        if relative_error(observed, float(certificate[key])) > 2.0e-12:
            raise RuntimeError(f"combined benchmark geometry differs at {key}")

    frozen_rows = {int(row["step"]): row for row in certificate["output_rows"]}
    required = int(certificate["required_correct"])
    raw_zero = _gate_raw_slacks(
        logits(center[0, :dimension], cert_pairs, template, spec),
        cert_labels,
        required,
    )
    raw_exclusions = {
        step: float(row["raw_exclusion_slack"])
        for step, row in frozen_rows.items()
    }

    def optimistic_query(step: int) -> WitnessQuery:
        return WitnessQuery(
            step,
            event <= step < event + int(certificate["protocol"]["persistence"]),
            raw_exclusions.get(step, -math.inf) > 0.0,
        )

    preplan = acquire_witnesses(
        event=event,
        persistence=int(certificate["protocol"]["persistence"]),
        horizon=horizon,
        raw_exclusion_slacks=raw_exclusions,
        query=optimistic_query,
        exact_failures={0},
    )
    if not preplan.issued:
        raise RuntimeError(f"deterministic witness preplan failed: {preplan.reason}")
    planned_times = set(preplan.query_order)
    train_cert_pairs = torch.cat((train_pairs, cert_pairs), dim=0)

    output_started = time.perf_counter()
    output_entries: dict[int, dict] = {}
    for step in range(1, horizon + 1):
        if step < horizon:
            planned = step in planned_times
            pairs = train_cert_pairs if planned else train_pairs
            role = FUSED_ROLE if planned else TRAIN_ROLE
            supports_training = True
            supports_certificate = planned
        else:
            planned = step in planned_times
            pairs = cert_pairs
            role = CERT_ROLE
            supports_training = False
            supports_certificate = planned
        theta = center[step, :dimension]
        block = ball_valid_envelope(
            theta,
            spec,
            config,
            epsilon=domain_radius,
            exact_values=True,
            sphere=True,
        )
        row, state, wall = run_q1(
            theta=theta,
            pairs=pairs,
            template=template,
            spec=spec,
            config=role_config,
            seed=namespaced_probe_seed(
                master_nonce, role_identity(candidate, step, role)
            ),
        )
        sealed = frozen_rows[step]
        for key in ("second", "third"):
            if relative_error(float(block[key]), float(sealed[f"block_{key}"])) > 2.0e-12:
                raise RuntimeError(f"block envelope mismatch at step {step}, {key}")
        output_entries[step] = {
            "step": step,
            "pairs": int(len(pairs)),
            "role": int(role),
            "supports_training": supports_training,
            "supports_certificate": supports_certificate,
            "operator_norm_upper_bound": float(row["operator_norm_upper_bound"]),
            "Y": float(row["Y"]),
            "block_frobenius_terminal_norm": float(
                row["block_frobenius_terminal_norm"]
            ),
            "c_delta": float(row["c_delta"]),
            "operator_seconds": float(state.cumulative_operator_seconds),
            "wall_seconds": float(wall),
            "block_second": float(block["second"]),
            "block_third": float(block["third"]),
            "block_fixed_point_consistent": bool(block["fixed_point_consistent"]),
            "raw_guarantee_slack": float(sealed["raw_guarantee_slack"]),
            "raw_exclusion_slack": float(sealed["raw_exclusion_slack"]),
        }
    output_seconds = time.perf_counter() - output_started

    maximum_map_drift = 0.0
    for step in range(1, horizon):
        row = output_entries[step]
        if not row["supports_training"]:
            raise RuntimeError(f"missing training role at step {step}")
        first_ball = (
            float(row["operator_norm_upper_bound"])
            + float(row["block_second"]) * domain_radius
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

    batch_green, batch_green_t = make_batched_transformer_green_products(
        center[:horizon, :dimension],
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )

    def green_gram(rows: torch.Tensor) -> torch.Tensor:
        return batch_green_t(batch_green(rows))

    green_state = OnlineGramState.initialize(
        dimension=horizon * 2 * dimension,
        dtype=parameter.dtype,
        device=parameter.device,
        config=green_config,
        seed=namespaced_probe_seed(
            master_nonce, green_identity(candidate, horizon)
        ),
    )
    fallback_entries: dict[int, dict] = {}
    policy_trace: list[dict] = []
    green_rows: list[dict] = []
    green_started = time.perf_counter()
    decision = None

    def certificate_query(step: int, radius: float) -> WitnessQuery:
        primary = output_entries[step]

        def verdict(row: dict) -> tuple[bool, bool]:
            margin = math.sqrt(2.0) * (
                float(row["operator_norm_upper_bound"]) * radius
                + 0.5 * float(row["block_second"]) * radius * radius
            )
            return (
                float(row["raw_guarantee_slack"]) - margin > 0.0,
                float(row["raw_exclusion_slack"]) - margin > 0.0,
            )

        guarantee, exclusion = (
            verdict(primary)
            if bool(primary["supports_certificate"])
            else (False, False)
        )
        need = (event <= step < event + int(certificate["protocol"]["persistence"]) and not guarantee) or (
            not (event <= step < event + int(certificate["protocol"]["persistence"]))
            and not exclusion
        )
        if need:
            if step not in fallback_entries:
                theta = center[step, :dimension]
                row, state, wall = run_q1(
                    theta=theta,
                    pairs=cert_pairs,
                    template=template,
                    spec=spec,
                    config=role_config,
                    seed=namespaced_probe_seed(
                        master_nonce, role_identity(candidate, step, CERT_ROLE)
                    ),
                )
                fallback_entries[step] = {
                    **primary,
                    "pairs": int(len(cert_pairs)),
                    "role": int(CERT_ROLE),
                    "supports_training": False,
                    "supports_certificate": True,
                    "operator_norm_upper_bound": float(row["operator_norm_upper_bound"]),
                    "Y": float(row["Y"]),
                    "block_frobenius_terminal_norm": float(
                        row["block_frobenius_terminal_norm"]
                    ),
                    "c_delta": float(row["c_delta"]),
                    "operator_seconds": float(state.cumulative_operator_seconds),
                    "wall_seconds": float(wall),
                }
            extra_guarantee, extra_exclusion = verdict(fallback_entries[step])
            guarantee = guarantee or extra_guarantee
            exclusion = exclusion or extra_exclusion
        return WitnessQuery(step, guarantee, exclusion)

    for q_green in range(1, MAXIMUM_POWER + 1):
        green_row = green_state.step(green_gram)
        green_row["block_frobenius_terminal_norm"] = float(
            torch.linalg.vector_norm(green_state.vectors)
        )
        green_rows.append(green_row)
        kappa = float(green_row["operator_norm_upper_bound"])
        closure = conservative_one_shot_closure(
            kappa=kappa,
            derivative_drift=maximum_map_drift,
            response_sequence_norm=response_norm,
            response_max_state_norm=response_max,
            domain_radius=domain_radius,
        )
        trial = {
            "q_output": 1,
            "q_green": q_green,
            "closure_passed": bool(closure.closure_passed),
            "certificate_issued": False,
            "certified_bracket": None,
            "total_pointwise_radius": float(closure.total_pointwise_radius),
            "closure": closure.as_dict(),
        }
        if closure.closure_passed and all(
            bool(row["block_fixed_point_consistent"])
            for row in output_entries.values()
        ):
            radius = float(closure.total_pointwise_radius)
            policy = acquire_witnesses(
                event=event,
                persistence=int(certificate["protocol"]["persistence"]),
                horizon=horizon,
                raw_exclusion_slacks=raw_exclusions,
                query=lambda step: certificate_query(step, radius),
                exact_failures={0},
            )
            if policy.issued:
                trial.update(
                    {
                        "certificate_issued": True,
                        "certified_bracket": [event, event],
                        "query_order": list(policy.query_order),
                        "success_times": list(policy.success_times),
                        "failure_witnesses": list(policy.failure_witnesses),
                    }
                )
                decision = trial
        policy_trace.append(trial)
        if decision is not None:
            break
    green_seconds = time.perf_counter() - green_started
    combined_end_to_end = time.perf_counter() - started
    if decision is None:
        raise RuntimeError("combined online-role path failed to issue")
    if decision["certified_bracket"] != certificate["certified_bracket"]:
        raise RuntimeError("combined online-role path changed the sealed bracket")

    # Same-process dense q=1 output control.  It runs after the monolithic path,
    # so it does not contaminate the combined end-to-end stopwatch.
    dense_started = time.perf_counter()
    dense_operator_seconds = 0.0
    dense_pair_work = 0
    dense_maximum_trace_deviation = 0.0
    all_pairs = torch.cartesian_prod(
        torch.arange(config.modulus), torch.arange(config.modulus)
    ).long()
    dense_config = probe_config()
    for step in range(1, horizon + 1):
        row, state, _ = run_q1(
            theta=center[step, :dimension],
            pairs=all_pairs,
            template=template,
            spec=spec,
            config=dense_config,
            seed=namespaced_probe_seed(
                master_nonce, output_identity(candidate, step)
            ),
        )
        frozen = certificate["output_rows"][step - 1]["trace"]["rows"][0]
        dense_maximum_trace_deviation = max(
            dense_maximum_trace_deviation,
            relative_error(
                float(row["operator_norm_upper_bound"]),
                float(frozen["operator_norm_upper_bound"]),
            ),
        )
        dense_operator_seconds += float(state.cumulative_operator_seconds)
        dense_pair_work += int(len(all_pairs))
    dense_wall_seconds = time.perf_counter() - dense_started
    if dense_maximum_trace_deviation > 2.0e-12:
        raise RuntimeError(
            f"same-process dense q1 trace mismatch: {dense_maximum_trace_deviation}"
        )

    primary_rows = list(output_entries.values())
    fallback_rows = list(fallback_entries.values())
    fused_operator_seconds = sum(
        float(row["operator_seconds"]) for row in primary_rows + fallback_rows
    )
    fused_operator_wall = sum(
        float(row["wall_seconds"]) for row in primary_rows + fallback_rows
    )
    common_envelope_wall = max(0.0, output_seconds - fused_operator_wall)
    fused_pair_work = sum(int(row["pairs"]) for row in primary_rows + fallback_rows)
    green_operator_seconds = float(green_state.cumulative_operator_seconds)
    matched_combined_replay = (
        center_seconds + common_envelope_wall + fused_operator_wall + green_seconds
    )
    matched_dense_replay = (
        center_seconds + common_envelope_wall + dense_wall_seconds + green_seconds
    )
    result = {
        "status": "POST-SEAL MONOLITHIC ONLINE-GREEN + ROLE-FUSED BENCHMARK",
        "scope": (
            "Immutable replay; no joined future outcome enters selection, "
            "stopping, or arithmetic; prospective certificate counts unchanged."
        ),
        "benchmark_script_sha256": sha256(Path(__file__).resolve()),
        "candidate": candidate.__dict__,
        "certificate_sha256": sha256(certificate_path),
        "centerline_sha256": path["centerline_sha256"],
        "horizon": horizon,
        "frozen_predicted_persistent_event": event,
        "sealed_bracket": certificate["certified_bracket"],
        "combined_bracket": decision["certified_bracket"],
        "same_bracket": decision["certified_bracket"] == certificate["certified_bracket"],
        "run_label": run_label,
        "family_budget": {
            "family_failure_probability": FAMILY_FAILURE_PROBABILITY,
            "green_fraction": 0.5,
            "output_fraction": 0.5,
            "green_per_operator_delta": green_delta,
            "role_output_per_operator_delta": output_delta,
            "maximum_candidates": maximum_candidates,
            "maximum_role_outputs_per_candidate": role_outputs_per_candidate,
        },
        "decision": decision,
        "policy_trace": policy_trace,
        "query_counts": {
            "preplanned_output_times": len(planned_times),
            "primary_output_operators": len(primary_rows),
            "fallback_certification_operators": len(fallback_rows),
            "green_power": int(green_state.power),
            "output_power": 1,
        },
        "geometry": {
            "signed_response_sequence_norm": response_norm,
            "signed_response_max_state_norm": response_max,
            "outer_domain_radius": domain_radius,
        },
        "pair_work": {
            "dense_q1": dense_pair_work,
            "combined_role_fused_q1": fused_pair_work,
            "reduction_fraction": 1.0 - fused_pair_work / dense_pair_work,
        },
        "operator_seconds": {
            "combined_output": fused_operator_seconds,
            "combined_green": green_operator_seconds,
            "combined_total": fused_operator_seconds + green_operator_seconds,
            "same_process_dense_output_control": dense_operator_seconds,
            "same_process_output_speedup": dense_operator_seconds / fused_operator_seconds,
        },
        "timings_seconds": {
            "centerline_and_signed_response": center_seconds,
            "combined_role_fused_output_q1_and_envelopes": output_seconds,
            "combined_online_green_until_issue": green_seconds,
            "combined_monolithic_end_to_end": combined_end_to_end,
            "combined_role_fused_operator_wall": fused_operator_wall,
            "common_output_envelope_wall": common_envelope_wall,
            "same_process_dense_q1_output_control": dense_wall_seconds,
            "same_process_output_operator_wall_speedup": (
                dense_wall_seconds / fused_operator_wall
            ),
            "matched_combined_replay": matched_combined_replay,
            "matched_dense_replay": matched_dense_replay,
            "matched_end_to_end_replay_speedup": (
                matched_dense_replay / matched_combined_replay
            ),
        },
        "dense_control_maximum_relative_trace_deviation": dense_maximum_trace_deviation,
        "output_rows": primary_rows,
        "fallback_rows": fallback_rows,
        "green_rows": green_rows,
        "maximum_optimizer_derivative_drift_upper": maximum_map_drift,
        "interpretation": (
            "This is the first measured composition of online Green stopping and "
            "role-fused output transport in one certificate execution."
        ),
    }
    destination = RESULTS / (
        f"transformer_v3_combined_online_role_seed_{candidate.seed}_"
        f"gate_{candidate.gate_index}_anchor_{candidate.anchor}_{run_label}.json"
    )
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite combined benchmark: {destination}")
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["output"] = str(destination)
    result["sha256"] = sha256(destination)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_CANDIDATE.seed)
    parser.add_argument("--threshold", type=float, default=DEFAULT_CANDIDATE.threshold)
    parser.add_argument("--anchor", type=int, default=DEFAULT_CANDIDATE.anchor)
    parser.add_argument("--run-label", default="matched-combined")
    args = parser.parse_args()
    result = benchmark(
        Candidate(args.seed, args.threshold, args.anchor),
        run_label=args.run_label,
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("policy_trace", "output_rows", "fallback_rows")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
