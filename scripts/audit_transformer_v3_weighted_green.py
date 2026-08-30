#!/usr/bin/env python3
"""Post-seal audit of a diagonally time-scaled Green certificate.

The time weights are chosen *before* a fresh Green probe block from only the
signed response, local drift envelopes, and a scalar directional-gain proxy.
The ensuing randomized operator query validates the actual weighted causal
operator; the proxy is used only to choose its deterministic metric.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import differential_evolution

from adaptive_witness_policy import WitnessQuery, acquire_witnesses
from batched_green_operator import make_batched_transformer_green_products
from heterogeneous_recenter_closure import heterogeneous_one_shot_closure
from online_progressive_gram import OnlineGramState
from probe_jacobian_bound import ProbeConfig
from transformer_block_envelope import objective_hessian_lipschitz
from transformer_certificate_protocol import Candidate
from transformer_four_sweep_development_audit import to_scaled
from transformer_green_development_audit import build_frozen_centerline
from transformer_green_operator import make_transformer_green_products
from transformer_v3_certificate import METHOD_SEAL, load_candidate, output_path, safe_json
from transformer_v3_protocol import MAXIMUM_POWER, PROBES
from weighted_green_operator import make_weighted_batched_green_products
from weighted_recenter_closure import weighted_one_shot_closure


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOURCE = RESULTS / (
    "transformer_v3_combined_online_role_seed_366_gate_1_anchor_1120_"
    "matched-combined-v5.json"
)
OUTPUT = RESULTS / "transformer_v3_weighted_green_postseal_audit.json"
WEIGHT_CAP = 4.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def fresh_seed(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def scalar_green_matrix(gains: np.ndarray) -> np.ndarray:
    horizon = gains.size
    matrix = np.zeros((horizon, horizon), dtype=np.float64)
    for column in range(horizon):
        state = 0.0
        for step in range(horizon):
            state = gains[step] * state + (1.0 if step == column else 0.0)
            matrix[step, column] = state
    return matrix


def metric_family(
    parameters: np.ndarray,
    *,
    drift_state: np.ndarray,
    response: np.ndarray,
) -> np.ndarray:
    horizon = response.size
    time_axis = np.linspace(-1.0, 1.0, horizon)
    safe_drift = np.maximum(drift_state, np.max(drift_state) * 1.0e-12)
    safe_response = np.maximum(response, np.max(response) * 5.0e-2)
    log_drift = np.log(safe_drift) - np.mean(np.log(safe_drift))
    log_response = np.log(safe_response) - np.mean(np.log(safe_response))
    log_weight = (
        parameters[0] * time_axis
        + parameters[1] * log_drift
        + parameters[2] * log_response
    )
    log_weight = np.clip(log_weight, -math.log(WEIGHT_CAP), math.log(WEIGHT_CAP))
    log_weight -= np.mean(log_weight)
    return np.exp(log_weight)


def choose_weights(
    *,
    gains: np.ndarray,
    drift: np.ndarray,
    response: np.ndarray,
    domain_radius: float,
) -> tuple[np.ndarray, dict]:
    horizon = response.size
    drift_state = np.concatenate((drift, drift[-1:])) if drift.size else np.ones(horizon)
    proxy_green = scalar_green_matrix(gains)
    injection = np.ones(horizon)

    def evaluate(parameters: np.ndarray) -> float:
        state = metric_family(
            parameters, drift_state=drift_state, response=response
        )
        kappa = float(np.linalg.norm(np.diag(state) @ proxy_green, ord=2))
        closure = weighted_one_shot_closure(
            green_operator_bound=kappa,
            drift_bounds=drift,
            response_state_norms=response,
            state_weights=state,
            injection_weights=injection,
            domain_radii=np.full(horizon, domain_radius),
        )
        if not closure.closure_passed:
            deficit = max(0.0, closure.linearized_remainder_coefficient - 1.0)
            return 1.0e6 + 1.0e3 * deficit + float(np.var(np.log(state)))
        smoothness = float(np.mean(np.diff(np.log(state)) ** 2)) if horizon > 1 else 0.0
        return math.log(max(closure.maximum_pointwise_total_radius, 1.0e-300)) + 0.002 * smoothness

    optimization = differential_evolution(
        evaluate,
        bounds=((-2.5, 2.5), (-2.0, 2.0), (-2.0, 2.0)),
        seed=20260827,
        polish=True,
        workers=1,
        updating="immediate",
        tol=1.0e-9,
        maxiter=100,
    )
    weights = metric_family(
        np.asarray(optimization.x), drift_state=drift_state, response=response
    )
    return weights, {
        "parameters": np.asarray(optimization.x).astype(float).tolist(),
        "objective": float(optimization.fun),
        "success": bool(optimization.success),
        "message": str(optimization.message),
        "minimum_weight": float(np.min(weights)),
        "maximum_weight": float(np.max(weights)),
        "condition_ratio": float(np.max(weights) / np.min(weights)),
        "selection_inputs": "signed response, drift bounds, scalar directional gains only",
    }


def main() -> None:
    source = safe_json(SOURCE)
    candidate = Candidate(**source["candidate"])
    certificate_path = output_path(candidate)
    certificate = safe_json(certificate_path)
    config, template, spec, data, parameter, velocity = load_candidate(candidate)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    train_pairs, train_labels, _, _, _, _ = data
    dimension = int(parameter.numel())
    horizon = int(source["horizon"])
    method = safe_json(METHOD_SEAL)

    started = time.perf_counter()
    path = build_frozen_centerline(
        config,
        template,
        spec,
        train_pairs,
        train_labels,
        parameter,
        velocity,
    )
    if path["centerline_sha256"] != source["centerline_sha256"]:
        raise AssertionError("centerline identity mismatch")
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
    signed = scalar_green(residual.reshape(-1)).reshape(horizon, -1)
    response = torch.linalg.vector_norm(signed, dim=1).detach().cpu().numpy()
    domain_radius = float(source["geometry"]["outer_domain_radius"])
    if not math.isclose(
        float(np.linalg.norm(response)),
        float(source["geometry"]["signed_response_sequence_norm"]),
        rel_tol=2.0e-12,
    ):
        raise AssertionError("signed response mismatch")

    rows = {int(row["step"]): row for row in source["output_rows"]}
    drift = []
    for step in range(1, horizon):
        row = rows[step]
        first_ball = float(row["operator_norm_upper_bound"]) + float(
            row["block_second"]
        ) * domain_radius
        drift.append(
            math.sqrt(2.0)
            * config.learning_rate
            * objective_hessian_lipschitz(
                first_ball,
                float(row["block_second"]),
                float(row["block_third"]),
            )
        )
    drift_array = np.asarray(drift, dtype=np.float64)

    gains = np.zeros(horizon, dtype=np.float64)
    for step in range(1, horizon):
        previous = signed[step - 1]
        action = signed[step] - residual[step]
        denominator = float(torch.linalg.vector_norm(previous))
        gains[step] = (
            0.0
            if denominator == 0.0
            else float(torch.linalg.vector_norm(action)) / denominator
        )
    weights, selection = choose_weights(
        gains=gains,
        drift=drift_array,
        response=response,
        domain_radius=domain_radius,
    )
    injection_weights = np.ones(horizon, dtype=np.float64)

    original_kappa = float(source["green_rows"][0]["operator_norm_upper_bound"])
    unweighted = heterogeneous_one_shot_closure(
        kappa=original_kappa,
        drift_bounds=drift_array,
        response_input_state_norms=response[:-1],
        response_sequence_norm=float(np.linalg.norm(response)),
        response_max_state_norm=float(np.max(response)),
        domain_radius=domain_radius,
    )

    batch_green, batch_green_t = make_batched_transformer_green_products(
        center[:horizon, :dimension],
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )
    weighted_green, weighted_green_t = make_weighted_batched_green_products(
        batch_green,
        batch_green_t,
        state_weights=weights,
        injection_weights=injection_weights,
        state_dimension=2 * dimension,
        dtype=parameter.dtype,
        device=parameter.device,
    )

    def weighted_gram(vectors: torch.Tensor) -> torch.Tensor:
        return weighted_green_t(weighted_green(vectors))

    delta = float(source["family_budget"]["green_per_operator_delta"])
    probe = ProbeConfig(PROBES, MAXIMUM_POWER, delta)
    seed = fresh_seed(
        "weighted-green-v1|"
        + str(method["master_nonce"])
        + f"|{candidate.seed}|{candidate.gate_index}|{candidate.anchor}|{horizon}"
    )
    state = OnlineGramState.initialize(
        dimension=horizon * 2 * dimension,
        dtype=parameter.dtype,
        device=parameter.device,
        config=probe,
        seed=seed,
    )
    traces = []
    decision = None
    persistence = int(certificate["protocol"]["persistence"])
    event = int(source["frozen_predicted_persistent_event"])
    raw_exclusions = {
        step: float(row["raw_exclusion_slack"]) for step, row in rows.items()
    }

    for power in range(1, MAXIMUM_POWER + 1):
        row = state.step(weighted_gram)
        kappa = float(row["operator_norm_upper_bound"])
        closure = weighted_one_shot_closure(
            green_operator_bound=kappa,
            drift_bounds=drift_array,
            response_state_norms=response,
            state_weights=weights,
            injection_weights=injection_weights,
            domain_radii=np.full(horizon, domain_radius),
        )
        trial = {
            "power": power,
            "Y": float(row["Y"]),
            "kappa": kappa,
            "closure": closure.as_dict(),
            "issued": False,
            "bracket": None,
        }
        if closure.closure_passed:
            radii = np.asarray(closure.pointwise_total_radii)

            def query(step: int) -> WitnessQuery:
                output = rows[step]
                radius = float(radii[step - 1])
                margin = math.sqrt(2.0) * (
                    float(output["operator_norm_upper_bound"]) * radius
                    + 0.5 * float(output["block_second"]) * radius * radius
                )
                return WitnessQuery(
                    step,
                    float(output["raw_guarantee_slack"]) - margin > 0.0,
                    float(output["raw_exclusion_slack"]) - margin > 0.0,
                )

            policy = acquire_witnesses(
                event=event,
                persistence=persistence,
                horizon=horizon,
                raw_exclusion_slacks=raw_exclusions,
                query=query,
                exact_failures={0},
            )
            trial.update(
                {
                    "issued": bool(policy.issued),
                    "bracket": [event, event] if policy.issued else None,
                    "query_order": list(policy.query_order),
                }
            )
            if policy.issued and decision is None:
                decision = trial
        traces.append(trial)
        if decision is not None:
            break

    payload = {
        "status": "post-seal diagonally weighted Green audit passed",
        "scope": (
            "Fresh randomized Green block on one immutable case; metric selected "
            "without that block or future outcome. Method-development evidence, "
            "not a change to prospective counts."
        ),
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "candidate": candidate.__dict__,
        "fresh_green_seed": seed,
        "green_delta": delta,
        "selection": selection,
        "state_weights": weights.astype(float).tolist(),
        "injection_weights": injection_weights.astype(float).tolist(),
        "directional_gain_proxy": gains.astype(float).tolist(),
        "unweighted_q1_time_resolved": unweighted.as_dict(),
        "weighted_traces": traces,
        "earliest_issuing_power": None if decision is None else decision["power"],
        "same_bracket": decision is not None and decision["bracket"] == source["combined_bracket"],
        "weighted_q1_maximum_radius_ratio_to_unweighted_q1": (
            None
            if not traces or not traces[0]["closure"]["closure_passed"]
            else traces[0]["closure"]["maximum_pointwise_total_radius"]
            / unweighted.total_pointwise_radius
        ),
        "weighted_q1_kappa_ratio_to_unweighted_q1": (
            traces[0]["kappa"] / original_kappa
        ),
        "logical_green_gram_applications": int(state.power * PROBES),
        "green_operator_seconds": float(state.cumulative_operator_seconds),
        "total_wall_seconds": time.perf_counter() - started,
    }
    if decision is None:
        payload["status"] = "post-seal diagonally weighted Green audit abstained"
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**payload, "output_sha256": sha256(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
