#!/usr/bin/env python3
"""Burned-candidate audit of the signed finite-window Green certificate.

The candidate coordinates and modal event offsets were sealed before outcomes
were joined.  This script keeps the four-sweep centreline unchanged, computes
the known signed response ``z = K_H s``, fixes ``R = 2 ||z||``, and then tests
the Green shadowing closure.  It never selects a candidate, horizon, radius, or
method constant from a probabilistic probe or from a training outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from matrix_free_mlp import signed_variational_recenter
from probe_jacobian_bound import jacobian_norm_bound
from transformer_block_envelope import ball_valid_envelope, objective_hessian_lipschitz
from transformer_certificate_protocol import HORIZON, PERSISTENCE, SWEEPS, Candidate
from transformer_four_sweep_development_audit import (
    CANDIDATES,
    count_envelope,
    first_persistent,
    persistent_bracket,
    to_scaled,
    verify_burned_candidate_seal,
)
from transformer_green_operator import green_norm_bound, make_transformer_green_products
from transformer_green_protocol import (
    FAMILY_FAILURE_PROBABILITY,
    candidate_universe,
    green_identity,
    make_registry,
    maximum_operator_count,
    output_identity,
    probe_config,
)
from transformer_hvp_grokking import (
    TransformerConfig,
    artifact_paths,
    flat_spec,
    logits,
    make_disjoint_split,
    make_template,
)
from transformer_modal_forecast import affine_reference, optimizer_jvp, optimizer_map

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MASTER_NONCE = "independent-signed-green-development-v1"
NUMERIC_RADIUS_CAP = 1.0e3

# Frozen sealed modal offsets plus the 25-state persistence suffix.  These are
# fixed before any Green/output probe is instantiated.
HORIZONS = {
    Candidate(321, 0.70, 1440): 211 + PERSISTENCE - 1,
    Candidate(322, 0.70, 2400): 34 + PERSISTENCE - 1,
    Candidate(322, 0.80, 2640): 87 + PERSISTENCE - 1,
}


def cache_path(candidate: Candidate) -> Path:
    return RESULTS / "transformer_green_development_cache" / (
        f"seed_{candidate.seed}_gate_{candidate.gate_index}_anchor_{candidate.anchor}.json"
    )


def output_path(candidate: Candidate) -> Path:
    return RESULTS / (
        f"transformer_green_development_seed_{candidate.seed}_"
        f"gate_{candidate.gate_index}_anchor_{candidate.anchor}.json"
    )


def load_candidate(candidate: Candidate):
    result_path, checkpoint_path = artifact_paths(candidate.seed, development=False)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    config = TransformerConfig(**payload["config"])
    if config.checkpoint_every != 40 or candidate.anchor % config.checkpoint_every:
        raise RuntimeError("candidate anchor is not on the frozen checkpoint grid")
    template = make_template(config)
    spec = flat_spec(template)
    data = make_disjoint_split(config)
    checkpoints = np.load(checkpoint_path)
    parameter = torch.from_numpy(checkpoints[f"step_{candidate.anchor}"]).clone()
    velocity = torch.from_numpy(checkpoints[f"velocity_{candidate.anchor}"]).clone()
    return config, template, spec, data, parameter, velocity


def build_frozen_centerline(
    config,
    template,
    spec,
    train_pairs,
    train_labels,
    parameter,
    velocity,
) -> dict:
    """Construct exactly four sweeps without rolling out the true trajectory."""
    anchor = torch.cat((parameter, velocity))

    def map_step(state):
        return optimizer_map(state, train_pairs, train_labels, template, spec, config)

    def jvp(center, direction):
        return optimizer_jvp(
            center, direction, train_pairs, train_labels, template, spec, config
        )

    raw = affine_reference(
        anchor, map_step, lambda direction: jvp(anchor, direction), horizon=HORIZON
    )
    centers = [raw]
    diagnostics = []
    for sweep in range(SWEEPS):
        corrected, diagnostic = signed_variational_recenter(
            centers[-1], map_step, jvp, numeric_cap=1.0e6
        )
        if diagnostic["reached_horizon"] != HORIZON:
            raise RuntimeError(f"recentring sweep {sweep + 1} truncated")
        diagnostics.append({"sweep": sweep + 1, **diagnostic})
        centers.append(corrected)
    if len(diagnostics) != 4:
        raise RuntimeError("the frozen method requires exactly four sweeps")

    dimension = parameter.numel()
    center = centers[-1]
    scaled_center = to_scaled(center, dimension, config.learning_rate)
    return {
        "map_step": map_step,
        "center": center,
        "scaled_center": scaled_center,
        "diagnostics": diagnostics,
        "centerline_sha256": hashlib.sha256(
            scaled_center.numpy().tobytes(order="C")
        ).hexdigest().upper(),
    }


def gate_slacks(
    center_logits: torch.Tensor,
    labels: torch.Tensor,
    margin_radius: float,
    required: int,
) -> tuple[float, float]:
    """Return strict slacks for ``guaranteed >= r`` and ``possible < r``.

    Positive first output means at least ``required`` examples remain correct.
    Positive second output means enough examples are definitely incorrect that
    fewer than ``required`` examples can be correct.
    """
    true = center_logits.gather(1, labels[:, None])
    margins = true - center_logits
    rows = torch.arange(len(labels))
    lower = margins - margin_radius
    upper = margins + margin_radius
    lower[rows, labels] = torch.inf
    upper[rows, labels] = torch.inf
    per_example_lower = torch.min(lower, dim=1).values
    per_example_upper = torch.min(upper, dim=1).values
    guaranteed_cut = torch.sort(per_example_lower, descending=True).values[required - 1]
    definitely_incorrect_needed = len(labels) - required + 1
    incorrect_cut = torch.sort(per_example_upper).values[
        definitely_incorrect_needed - 1
    ]
    return float(guaranteed_cut), -float(incorrect_cut)


def persistent_certificate_slack(
    bracket: list[int] | None,
    guaranteed_slack: list[float],
    exclusion_slack: list[float],
) -> dict | None:
    if bracket is None:
        return None
    lower, upper = bracket
    upper_slack = min(guaranteed_slack[upper : upper + PERSISTENCE])
    prior_block_slacks = [
        max(exclusion_slack[start : start + PERSISTENCE])
        for start in range(lower)
    ]
    lower_slack = math.inf if not prior_block_slacks else min(prior_block_slacks)
    return {
        "lower_endpoint_exclusion_slack": lower_slack,
        "upper_endpoint_guarantee_slack": upper_slack,
        "minimum_logic_slack": min(lower_slack, upper_slack),
    }


def load_cache(candidate: Candidate, horizon: int, centerline_sha256: str) -> dict:
    path = cache_path(candidate)
    if not path.exists():
        return {"output_rows": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = (MASTER_NONCE, horizon, centerline_sha256)
    observed = (
        payload.get("master_nonce"),
        int(payload.get("horizon", -1)),
        payload.get("centerline_sha256"),
    )
    if observed != expected:
        raise RuntimeError(f"stale Green cache metadata: {observed} != {expected}")
    return payload


def save_cache(
    candidate: Candidate,
    horizon: int,
    centerline_sha256: str,
    green_probe: dict | None,
    output_rows: list[dict],
) -> None:
    path = cache_path(candidate)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "burned signed-Green development cache",
        "candidate": candidate.__dict__,
        "horizon": horizon,
        "master_nonce": MASTER_NONCE,
        "centerline_sha256": centerline_sha256,
        "probe_config": probe_config().__dict__,
        "green_probe": green_probe,
        "output_rows": output_rows,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_candidate(candidate: Candidate, *, join_outcome: bool) -> dict:
    if candidate not in HORIZONS:
        raise ValueError(f"candidate is outside the sealed burned set: {candidate}")
    started = time.perf_counter()
    seal = verify_burned_candidate_seal()
    horizon = HORIZONS[candidate]
    config, template, spec, data, parameter, velocity = load_candidate(candidate)
    torch.set_num_threads(config.threads)
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = data
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
    dimension = parameter.numel()

    # Verify the pre-probe horizon directly from the frozen four-sweep path.
    center_counts = np.asarray(
        [
            int(
                (
                    logits(state[:dimension], cert_pairs, template, spec).argmax(1)
                    == cert_labels
                ).sum()
            )
            for state in center
        ],
        dtype=np.int64,
    )
    required = int(math.ceil(candidate.threshold * len(cert_pairs)))
    predicted_event = first_persistent(center_counts, required)
    if predicted_event is None or predicted_event + PERSISTENCE - 1 != horizon:
        raise RuntimeError(
            f"sealed horizon mismatch: predicted {predicted_event}, frozen H={horizon}"
        )

    residual = torch.stack(
        [
            to_scaled(path["map_step"](center[step]), dimension, config.learning_rate)
            - scaled_center[step + 1]
            for step in range(horizon)
        ]
    )
    defect_norm = float(torch.linalg.vector_norm(residual))

    # Candidate, centreline, horizon, signed defect, and operator universe are
    # all fixed before the first probabilistic operator is claimed.
    registry = make_registry(CANDIDATES, HORIZONS, MASTER_NONCE)
    probe = probe_config()
    cache = load_cache(candidate, horizon, path["centerline_sha256"])
    green_probe = cache.get("green_probe")
    output_rows_by_step = {
        int(row["step"]): row for row in cache.get("output_rows", [])
    }
    if green_probe is not None:
        registry.claim(tuple(green_probe["identity"]))
    for row in output_rows_by_step.values():
        registry.claim(tuple(row["output_probe"]["identity"]))

    green_apply, _ = make_transformer_green_products(
        center[:horizon, :dimension],
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )
    signed_response = green_apply(residual.reshape(-1)).reshape(horizon, -1)
    signed_response_norm = float(torch.linalg.vector_norm(signed_response))
    signed_response_max_state = float(
        torch.linalg.vector_norm(signed_response, dim=1).max()
    )
    radius = 2.0 * signed_response_norm
    if not math.isfinite(radius) or radius > NUMERIC_RADIUS_CAP:
        raise RuntimeError(f"fixed Green radius is numerically unusable: {radius}")

    if green_probe is None:
        green_probe = green_norm_bound(
            center[:horizon, :dimension],
            train_pairs,
            train_labels,
            template,
            spec,
            config,
            probe,
            green_identity(candidate, horizon),
            registry,
        )
        save_cache(
            candidate,
            horizon,
            path["centerline_sha256"],
            green_probe,
            [output_rows_by_step[key] for key in sorted(output_rows_by_step)],
        )

    all_pairs = torch.cartesian_prod(
        torch.arange(config.modulus), torch.arange(config.modulus)
    ).long()
    guaranteed = [int(center_counts[0])]
    possible = [int(center_counts[0])]
    logits_zero = logits(center[0, :dimension], cert_pairs, template, spec)
    guarantee_slacks = []
    exclusion_slacks = []
    zero_guarantee, zero_exclusion = gate_slacks(
        logits_zero, cert_labels, 0.0, required
    )
    guarantee_slacks.append(zero_guarantee)
    exclusion_slacks.append(zero_exclusion)
    geometry_rows = []
    maximum_map_drift = 0.0
    fixed_point_consistent = True

    for step in range(1, horizon + 1):
        theta = center[step, :dimension]
        cached_row = output_rows_by_step.get(step)
        if cached_row is None:
            output_probe = jacobian_norm_bound(
                theta,
                all_pairs,
                template,
                spec,
                probe,
                output_identity(candidate, step),
                registry=registry,
            )
            cached_row = {"step": step, "output_probe": output_probe}
            output_rows_by_step[step] = cached_row
            save_cache(
                candidate,
                horizon,
                path["centerline_sha256"],
                green_probe,
                [output_rows_by_step[key] for key in sorted(output_rows_by_step)],
            )

        block = ball_valid_envelope(
            theta,
            spec,
            config,
            epsilon=radius,
            exact_values=True,
            sphere=True,
        )
        fixed_point_consistent &= bool(block["fixed_point_consistent"])
        output_upper = cached_row["output_probe"]["jacobian_norm_upper_bound"]
        first_ball = output_upper + block["second"] * radius
        objective_lipschitz = objective_hessian_lipschitz(
            first_ball, block["second"], block["third"]
        )
        map_drift = math.sqrt(2.0) * config.learning_rate * objective_lipschitz
        if step < horizon:
            maximum_map_drift = max(maximum_map_drift, map_drift)

        margin_radius = math.sqrt(2.0) * (
            output_upper * radius + 0.5 * block["second"] * radius * radius
        )
        center_logits = logits(theta, cert_pairs, template, spec)
        lower_count, upper_count = count_envelope(
            center_logits, cert_labels, margin_radius
        )
        guarantee_slack, exclusion_slack = gate_slacks(
            center_logits, cert_labels, margin_radius, required
        )
        guaranteed.append(lower_count)
        possible.append(upper_count)
        guarantee_slacks.append(guarantee_slack)
        exclusion_slacks.append(exclusion_slack)
        geometry_rows.append(
            {
                "step": step,
                "output_probe": cached_row["output_probe"],
                "block_first": block["first"],
                "block_second": block["second"],
                "block_third": block["third"],
                "block_fixed_point_consistent": block["fixed_point_consistent"],
                "block_fixed_point_iterations": block["fixed_point_iterations_used"],
                "first_ball": first_ball,
                "objective_hessian_lipschitz_upper": objective_lipschitz,
                "optimizer_derivative_drift_upper": map_drift,
                "margin_radius": margin_radius,
                "guaranteed_correct": lower_count,
                "possibly_correct": upper_count,
                "guaranteed_gate_slack": guarantee_slack,
                "possible_below_gate_slack": exclusion_slack,
            }
        )

    kappa = float(green_probe["green_operator_norm_upper_bound"])
    closure_lhs = 2.0 * kappa * maximum_map_drift * signed_response_norm
    closure_passed = fixed_point_consistent and closure_lhs <= 1.0
    raw_bracket = persistent_bracket(
        np.asarray(guaranteed, dtype=np.int64),
        np.asarray(possible, dtype=np.int64),
        required,
    )
    bracket = raw_bracket if closure_passed else None
    certificate_slack = persistent_certificate_slack(
        bracket, guarantee_slacks, exclusion_slacks
    )
    registry_summary = registry.summary()
    queried = registry_summary["queried_operator_count"]
    result = {
        "status": "burned-candidate signed-Green development certificate",
        "candidate": candidate.__dict__,
        "sealed_inputs": seal,
        "protocol": {
            "sweeps": SWEEPS,
            "horizon": horizon,
            "persistence": PERSISTENCE,
            "radius_rule": "R = 2 ||K_H s||_sequence",
            "probe_config": probe.__dict__,
            "family_failure_probability": FAMILY_FAILURE_PROBABILITY,
            "maximum_operator_accounting": maximum_operator_count(),
            "actual_candidate_operator_universe": len(
                candidate_universe(CANDIDATES, HORIZONS)
            ),
            "master_nonce": MASTER_NONCE,
        },
        "centerline_sha256": path["centerline_sha256"],
        "sweep_diagnostics": path["diagnostics"],
        "required_correct": required,
        "predicted_persistent_event": predicted_event,
        "defect_sequence_norm": defect_norm,
        "signed_response_sequence_norm": signed_response_norm,
        "signed_response_max_state_norm": signed_response_max_state,
        "norm_only_linear_radius": kappa * defect_norm,
        "signed_radius": radius,
        "green_probe": green_probe,
        "maximum_optimizer_derivative_drift_upper": maximum_map_drift,
        "closure_lhs_2_kappa_M_Z": closure_lhs,
        "closure_slack": 1.0 - closure_lhs,
        "closure_passed": closure_passed,
        "block_fixed_points_all_consistent": fixed_point_consistent,
        "raw_margin_bracket": raw_bracket,
        "certified_bracket": bracket,
        "certificate_issued": bracket is not None,
        "certificate_output_logic_slack": certificate_slack,
        "guaranteed_correct": guaranteed,
        "possibly_correct": possible,
        "geometry": geometry_rows,
        "probability_budget": {
            "queried_operators": queried,
            "queried_union_bound": queried * probe.delta,
            "maximum_family_union_bound": FAMILY_FAILURE_PROBABILITY,
            **registry_summary,
        },
        "elapsed_seconds_before_outcome_join": time.perf_counter() - started,
        "outcome_joined": False,
    }
    output_path(candidate).write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    if join_outcome:
        exact = [torch.cat((parameter, velocity))]
        for _ in range(horizon):
            exact.append(path["map_step"](exact[-1]))
        exact = torch.stack(exact)
        exact_counts = np.asarray(
            [
                int(
                    (
                        logits(state[:dimension], cert_pairs, template, spec).argmax(1)
                        == cert_labels
                    ).sum()
                )
                for state in exact
            ],
            dtype=np.int64,
        )
        actual_event = first_persistent(exact_counts, required)
        state_error = torch.linalg.vector_norm(
            to_scaled(exact, dimension, config.learning_rate) - scaled_center[: horizon + 1],
            dim=1,
        )
        actual_sequence_error = float(torch.linalg.vector_norm(state_error[1:]))
        actual_max_error = float(state_error.max())
        result.update(
            {
                "outcome_joined": True,
                "actual_persistent_event": actual_event,
                "bracket_contains_actual": (
                    None
                    if bracket is None or actual_event is None
                    else bracket[0] <= actual_event <= bracket[1]
                ),
                "actual_sequence_error": actual_sequence_error,
                "actual_max_state_error": actual_max_error,
                "observed_sequence_tube_violation": actual_sequence_error
                > radius * (1.0 + 1e-9) + 1e-12,
                "observed_state_tube_violations": int(
                    torch.sum(state_error[1:] > radius * (1.0 + 1e-9) + 1e-12)
                ),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        output_path(candidate).write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=322)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--anchor", type=int, default=2400)
    parser.add_argument("--join-outcome", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate = Candidate(args.seed, args.threshold, args.anchor)
    result = run_candidate(candidate, join_outcome=args.join_outcome)
    summary = {
        key: result.get(key)
        for key in (
            "candidate",
            "predicted_persistent_event",
            "actual_persistent_event",
            "signed_response_sequence_norm",
            "signed_radius",
            "green_probe",
            "maximum_optimizer_derivative_drift_upper",
            "closure_lhs_2_kappa_M_Z",
            "closure_passed",
            "certified_bracket",
            "bracket_contains_actual",
            "elapsed_seconds",
        )
    }
    if isinstance(summary.get("green_probe"), dict):
        probe = summary["green_probe"]
        summary["green_probe"] = {
            "upper": probe["green_operator_norm_upper_bound"],
            "lower": probe["green_operator_norm_lower_estimate"],
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
