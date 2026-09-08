#!/usr/bin/env python3
"""Adversarial four-sweep certificate audit on burned Transformer candidates.

This script never trains a seed and never changes a method constant.  It first
verifies the previously sealed candidate file, constructs exactly four signed
recentring sweeps without any probe input, and only then instantiates the
predeclared probabilistic-operator registry.  Outcomes are used solely for the
post-certificate audit fields.
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
from transformer_block_envelope import (
    ball_valid_envelope,
    objective_hessian_lipschitz,
)
from transformer_certificate_protocol import (
    FAMILY_FAILURE_PROBABILITY,
    HORIZON,
    OPTIMIZER_JACOBIAN,
    OUTPUT_JACOBIAN,
    PERSISTENCE,
    SWEEPS,
    Candidate,
    candidate_universe,
    make_registry,
    maximum_operator_count,
    operator_identity,
    per_operator_failure_probability,
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
from transformer_modal_forecast import (
    affine_reference,
    optimizer_jvp,
    optimizer_map,
)
from transformer_optimizer_probe import scaled_optimizer_norm_bound

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MASTER_NONCE = "independent-development-audit-v1"
NUMERIC_RADIUS_CAP = 1.0e6
CANDIDATES = (
    Candidate(321, 0.70, 1440),
    Candidate(322, 0.70, 2400),
    Candidate(322, 0.80, 2640),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def verify_burned_candidate_seal() -> dict:
    seal_path = ROOT / "TRANSFORMER_HVP_CANDIDATE_SEAL_V2.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    candidate_path = ROOT / seal["candidate_manifest"]
    actual_hash = sha256(candidate_path)
    if actual_hash != seal["candidate_manifest_sha256"]:
        raise RuntimeError(
            f"burned candidate manifest hash mismatch: {actual_hash} != "
            f"{seal['candidate_manifest_sha256']}"
        )
    expected = [
        {
            "seed": row.seed,
            "threshold": row.threshold,
            "anchor": row.anchor,
        }
        for row in CANDIDATES
    ]
    sealed = [
        {
            "seed": int(row["seed"]),
            "threshold": float(row["threshold"]),
            "anchor": int(row["anchor"]),
        }
        for row in seal["predictions"]
    ]
    if expected != sealed:
        raise RuntimeError(f"candidate coordinates differ from the seal: {sealed}")
    return {
        "seal": str(seal_path.relative_to(ROOT)),
        "seal_sha256": sha256(seal_path),
        "candidate_manifest": str(candidate_path.relative_to(ROOT)),
        "candidate_manifest_sha256": actual_hash,
    }


def first_persistent(values: np.ndarray, required: int) -> int | None:
    starts = len(values) - PERSISTENCE + 1
    for start in range(max(starts, 0)):
        if np.all(values[start : start + PERSISTENCE] >= required):
            return int(start)
    return None


def persistent_bracket(
    guaranteed: np.ndarray, possible: np.ndarray, required: int
) -> list[int] | None:
    possible_event = first_persistent(possible, required)
    guaranteed_event = first_persistent(guaranteed, required)
    if possible_event is None or guaranteed_event is None:
        return None
    if possible_event > guaranteed_event:
        return None
    return [possible_event, guaranteed_event]


def count_envelope(
    center_logits: torch.Tensor, labels: torch.Tensor, margin_radius: float
) -> tuple[int, int]:
    true = center_logits.gather(1, labels[:, None])
    margins = true - center_logits
    lower = margins - margin_radius
    upper = margins + margin_radius
    rows = torch.arange(len(labels))
    lower[rows, labels] = torch.inf
    upper[rows, labels] = torch.inf
    guaranteed = int(torch.all(lower > 0.0, dim=1).sum())
    definitely_incorrect = int(torch.any(upper < 0.0, dim=1).sum())
    return guaranteed, len(labels) - definitely_incorrect


def to_scaled(path: torch.Tensor, dimension: int, eta: float) -> torch.Tensor:
    return torch.cat((path[..., :dimension], eta * path[..., dimension:]), dim=-1)


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
    return payload, config, template, spec, data, parameter, velocity


def build_four_sweep_path(
    config,
    template,
    spec,
    train_pairs,
    train_labels,
    parameter,
    velocity,
) -> dict:
    anchor = torch.cat((parameter, velocity))

    def map_step(state):
        return optimizer_map(
            state, train_pairs, train_labels, template, spec, config
        )

    def jvp(center, direction):
        return optimizer_jvp(
            center,
            direction,
            train_pairs,
            train_labels,
            template,
            spec,
            config,
        )

    raw = affine_reference(
        anchor, map_step, lambda direction: jvp(anchor, direction), horizon=HORIZON
    )
    centers = [raw]
    diagnostics = []
    for sweep in range(SWEEPS):
        corrected, diagnostic = signed_variational_recenter(
            centers[-1], map_step, jvp, numeric_cap=NUMERIC_RADIUS_CAP
        )
        if diagnostic["reached_horizon"] != HORIZON:
            raise RuntimeError(f"recentring sweep {sweep + 1} truncated")
        diagnostics.append({"sweep": sweep + 1, **diagnostic})
        centers.append(corrected)
    if len(diagnostics) != 4:
        raise RuntimeError("protocol requires exactly four recentering sweeps")

    dimension = parameter.numel()
    scaled_centers = [
        to_scaled(row, dimension, config.learning_rate) for row in centers
    ]
    delta_raw = []
    delta_scaled = []
    for raw_center, scaled_center in zip(centers, scaled_centers):
        raw_residual = torch.stack(
            [
                map_step(raw_center[step]) - raw_center[step + 1]
                for step in range(HORIZON)
            ]
        )
        scaled_residual = to_scaled(
            raw_residual, dimension, config.learning_rate
        )
        delta_raw.append(float(torch.linalg.vector_norm(raw_residual, dim=1).max()))
        delta_scaled.append(
            float(torch.linalg.vector_norm(scaled_residual, dim=1).max())
        )

    exact = [anchor]
    for _ in range(HORIZON):
        exact.append(map_step(exact[-1]))
    exact = torch.stack(exact)
    final = centers[-1]
    scaled_final = scaled_centers[-1]
    scaled_exact = to_scaled(exact, dimension, config.learning_rate)
    residual = torch.stack(
        [
            to_scaled(map_step(final[step]), dimension, config.learning_rate)
            - scaled_final[step + 1]
            for step in range(HORIZON)
        ]
    )
    state_error = torch.linalg.vector_norm(scaled_exact - scaled_final, dim=1)
    return {
        "map_step": map_step,
        "center": final,
        "scaled_center": scaled_final,
        "exact": exact,
        "residual_norm": torch.linalg.vector_norm(residual, dim=1).numpy(),
        "state_error": state_error.numpy(),
        "delta_raw": delta_raw,
        "delta_scaled": delta_scaled,
        "sweep_diagnostics": diagnostics,
        "centerline_sha256": hashlib.sha256(
            scaled_final.numpy().tobytes(order="C")
        ).hexdigest().upper(),
    }


def cache_path(candidate: Candidate) -> Path:
    return (
        RESULTS
        / "transformer_four_sweep_development_cache"
        / f"seed_{candidate.seed}_gate_{candidate.gate_index}_anchor_{candidate.anchor}.json"
    )


def output_path(candidate: Candidate) -> Path:
    return RESULTS / (
        f"transformer_four_sweep_development_seed_{candidate.seed}_"
        f"gate_{candidate.gate_index}_anchor_{candidate.anchor}.json"
    )


def load_cache(candidate: Candidate) -> dict:
    path = cache_path(candidate)
    if not path.exists():
        return {"rows": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("master_nonce") != MASTER_NONCE:
        raise RuntimeError("development probe cache uses a different master nonce")
    return payload


def save_cache(candidate: Candidate, rows: list[dict]) -> None:
    path = cache_path(candidate)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "burned development probe cache",
        "candidate": candidate.__dict__,
        "master_nonce": MASTER_NONCE,
        "probe_config": probe_config().__dict__,
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_candidate(candidate: Candidate) -> dict:
    started = time.perf_counter()
    seal = verify_burned_candidate_seal()
    payload, config, template, spec, data, parameter, velocity = load_candidate(candidate)
    torch.set_num_threads(config.threads)
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = data
    path = build_four_sweep_path(
        config,
        template,
        spec,
        train_pairs,
        train_labels,
        parameter,
        velocity,
    )
    center = path["center"]
    dimension = parameter.numel()

    # Candidate and centreline are fixed before the probabilistic registry is
    # instantiated.  Outcomes play no role in any path or bound below.
    registry = make_registry(CANDIDATES, MASTER_NONCE)
    config_probe = probe_config()
    cache = load_cache(candidate)
    cached = {int(row["step"]): row for row in cache.get("rows", [])}
    for row in cached.values():
        registry.claim(tuple(row["output_probe"]["identity"]))
        if row.get("optimizer_probe") is not None:
            registry.claim(tuple(row["optimizer_probe"]["identity"]))

    guaranteed = []
    possible = []
    epsilon = [0.0]
    geometry_rows = []
    stop_reason = None
    all_pairs = torch.cartesian_prod(
        torch.arange(config.modulus), torch.arange(config.modulus)
    ).long()

    # State zero is exact and needs no probabilistic derivative query.
    theta0 = center[0, :dimension]
    logits0 = logits(theta0, cert_pairs, template, spec)
    g0, p0 = count_envelope(logits0, cert_labels, 0.0)
    guaranteed.append(g0)
    possible.append(p0)
    epsilon.append(float(path["residual_norm"][0]))

    for step in range(1, HORIZON + 1):
        radius = epsilon[step]
        if not math.isfinite(radius) or radius > NUMERIC_RADIUS_CAP:
            stop_reason = "state radius exceeded the fixed numerical audit cap"
            break
        cached_row = cached.get(step)
        theta = center[step, :dimension]
        if cached_row is None:
            output_probe = jacobian_norm_bound(
                theta,
                all_pairs,
                template,
                spec,
                config_probe,
                operator_identity(candidate, step, OUTPUT_JACOBIAN),
                registry=registry,
            )
            optimizer_probe = None
            if step < HORIZON:
                optimizer_probe = scaled_optimizer_norm_bound(
                    theta,
                    train_pairs,
                    train_labels,
                    template,
                    spec,
                    config,
                    config_probe,
                    operator_identity(candidate, step, OPTIMIZER_JACOBIAN),
                    registry,
                )
            cached_row = {
                "step": step,
                "epsilon": radius,
                "output_probe": output_probe,
                "optimizer_probe": optimizer_probe,
            }
        elif abs(float(cached_row["epsilon"]) - radius) > 1e-12 * max(radius, 1.0):
            raise RuntimeError(f"cached tube radius mismatch at step {step}")

        # Deterministic envelopes are deliberately recomputed when resuming so
        # a verification-only tightening cannot silently reuse stale geometry.
        block_full = ball_valid_envelope(
            theta,
            spec,
            config,
            epsilon=radius,
            exact_values=True,
            sphere=True,
        )
        cached_row["block"] = {
            "first": block_full["first"],
            "second": block_full["second"],
            "third": block_full["third"],
            "fixed_point_consistent": block_full["fixed_point_consistent"],
            "fixed_point_iterations_used": block_full["fixed_point_iterations_used"],
            "maximum_value_inflation": max(block_full["inflation"].values()),
        }
        cached[step] = cached_row
        save_cache(candidate, [cached[key] for key in sorted(cached)])

        block = cached_row["block"]
        if not block["fixed_point_consistent"]:
            stop_reason = "ball-valid block value chain did not reach a fixed point"
            break
        output_upper = cached_row["output_probe"]["jacobian_norm_upper_bound"]
        first_ball = output_upper + block["second"] * radius
        margin_radius = math.sqrt(2.0) * (
            output_upper * radius + 0.5 * block["second"] * radius * radius
        )
        center_logits = logits(theta, cert_pairs, template, spec)
        lower_count, upper_count = count_envelope(
            center_logits, cert_labels, margin_radius
        )
        guaranteed.append(lower_count)
        possible.append(upper_count)

        l_cert = objective_hessian_lipschitz(
            first_ball, block["second"], block["third"]
        )
        row = {
            **cached_row,
            "first_ball": first_ball,
            "objective_hessian_lipschitz_upper": l_cert,
            "margin_radius": margin_radius,
            "guaranteed_correct": lower_count,
            "possibly_correct": upper_count,
        }
        geometry_rows.append(row)

        if step < HORIZON:
            beta = cached_row["optimizer_probe"][
                "optimizer_jacobian_norm_upper_bound"
            ]
            map_lipschitz = math.sqrt(2.0) * config.learning_rate * l_cert
            next_radius = (
                beta * radius
                + float(path["residual_norm"][step])
                + 0.5 * map_lipschitz * radius * radius
            )
            epsilon.append(next_radius)

    reached = len(guaranteed) - 1
    guaranteed_array = np.asarray(guaranteed, dtype=np.int64)
    possible_array = np.asarray(possible, dtype=np.int64)
    required = int(math.ceil(candidate.threshold * len(cert_pairs)))
    bracket = persistent_bracket(guaranteed_array, possible_array, required)

    exact_count = []
    center_count = []
    for exact_state, center_state in zip(path["exact"], center):
        exact_count.append(
            int(
                (
                    logits(exact_state[:dimension], cert_pairs, template, spec).argmax(1)
                    == cert_labels
                ).sum()
            )
        )
        center_count.append(
            int(
                (
                    logits(center_state[:dimension], cert_pairs, template, spec).argmax(1)
                    == cert_labels
                ).sum()
            )
        )
    actual_event = first_persistent(np.asarray(exact_count), required)
    predicted_event = first_persistent(np.asarray(center_count), required)

    error = path["state_error"][: reached + 1]
    tube = np.asarray(epsilon[: reached + 1])
    positive = tube > 0.0
    ratio = np.divide(error[positive], tube[positive]) if np.any(positive) else np.asarray([])
    violations = int(np.sum(error > tube * (1 + 1e-9) + 1e-12))
    max_actual_error = float(np.max(path["state_error"]))
    allowed_l = (
        math.inf
        if max_actual_error == 0.0
        else 1.0 / (HORIZON * config.learning_rate * max_actual_error)
    )
    l_values = [row["objective_hessian_lipschitz_upper"] for row in geometry_rows]
    output_ratios = [
        row["output_probe"]["operator_norm_upper_bound"]
        / max(row["output_probe"]["operator_norm_lower_estimate"], 1e-300)
        for row in geometry_rows
    ]
    optimizer_rows = [row for row in geometry_rows if row["optimizer_probe"] is not None]
    optimizer_ratios = [
        row["optimizer_probe"]["operator_norm_upper_bound"]
        / max(row["optimizer_probe"]["operator_norm_lower_estimate"], 1e-300)
        for row in optimizer_rows
    ]
    queried = registry.summary()["queried_operator_count"]
    result = {
        "status": "burned-candidate independent four-sweep development audit",
        "candidate": candidate.__dict__,
        "sealed_inputs": seal,
        "protocol": {
            "sweeps": SWEEPS,
            "horizon": HORIZON,
            "persistence": PERSISTENCE,
            "probe_config": config_probe.__dict__,
            "family_failure_probability": FAMILY_FAILURE_PROBABILITY,
            "per_operator_failure_probability": per_operator_failure_probability(),
            "maximum_operator_accounting": maximum_operator_count(),
            "actual_candidate_operator_universe": len(candidate_universe(CANDIDATES)),
            "master_nonce": MASTER_NONCE,
        },
        "centerline_sha256": path["centerline_sha256"],
        "delta_raw_0_through_4": path["delta_raw"],
        "delta_scaled_0_through_4": path["delta_scaled"],
        "sweep_diagnostics": path["sweep_diagnostics"],
        "required_correct": required,
        "predicted_persistent_event": predicted_event,
        "actual_persistent_event": actual_event,
        "actual_lead": actual_event,
        "state_horizon": reached,
        "stop_reason": stop_reason,
        "maximum_certified_radius": float(np.max(tube)),
        "maximum_actual_scaled_state_error_full_window": max_actual_error,
        "maximum_actual_error_to_bound_ratio": (
            None if len(ratio) == 0 else float(np.max(ratio))
        ),
        "observed_state_tube_violations": violations,
        "certified_bracket": bracket,
        "certificate_issued": bracket is not None,
        "bracket_contains_actual": (
            None
            if bracket is None or actual_event is None
            else bracket[0] <= actual_event <= bracket[1]
        ),
        "raw_timing_error": (
            None
            if predicted_event is None or actual_event is None
            else predicted_event - actual_event
        ),
        "maximum_probe_jacobian_upper": (
            None
            if not geometry_rows
            else max(row["output_probe"]["operator_norm_upper_bound"] for row in geometry_rows)
        ),
        "maximum_probe_jacobian_lower_estimate": (
            None
            if not geometry_rows
            else max(row["output_probe"]["operator_norm_lower_estimate"] for row in geometry_rows)
        ),
        "maximum_output_probe_upper_lower_ratio": (
            None if not output_ratios else max(output_ratios)
        ),
        "maximum_optimizer_jacobian_upper": (
            None
            if not optimizer_rows
            else max(
                row["optimizer_probe"]["optimizer_jacobian_norm_upper_bound"]
                for row in optimizer_rows
            )
        ),
        "maximum_optimizer_probe_upper_lower_ratio": (
            None if not optimizer_ratios else max(optimizer_ratios)
        ),
        "maximum_certified_objective_hessian_lipschitz": (
            None if not l_values else max(l_values)
        ),
        "heuristic_allowed_l_for_full_window": allowed_l,
        "heuristic_l_headroom": (
            None if not l_values else allowed_l / max(l_values)
        ),
        "probability_budget": {
            "queried_operators": queried,
            "queried_union_bound": queried * config_probe.delta,
            "maximum_family_union_bound": (
                maximum_operator_count()["maximum_probabilistic_operators"]
                * config_probe.delta
            ),
            **registry.summary(),
        },
        "operator_calls": {
            "output_gram": sum(
                row["output_probe"]["gram_applications"] for row in geometry_rows
            ),
            "output_jvp": sum(
                row["output_probe"]["jvp_calls"] for row in geometry_rows
            ),
            "output_vjp": sum(
                row["output_probe"]["vjp_calls"] for row in geometry_rows
            ),
            "optimizer_gram": sum(
                row["optimizer_probe"]["gram_applications"]
                for row in optimizer_rows
            ),
            "optimizer_jvp": sum(
                row["optimizer_probe"]["optimizer_jvp_calls"]
                for row in optimizer_rows
            ),
            "optimizer_vjp": sum(
                row["optimizer_probe"]["optimizer_vjp_calls"]
                for row in optimizer_rows
            ),
            "objective_hvp_inside_optimizer_gram": sum(
                row["optimizer_probe"]["objective_hvp_calls"]
                for row in optimizer_rows
            ),
            "recentring_hvp": SWEEPS * HORIZON,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "epsilon": epsilon[: reached + 1],
        "guaranteed_correct": guaranteed,
        "possibly_correct": possible,
        "actual_count": exact_count,
        "center_count": center_count,
        "geometry": geometry_rows,
    }
    out = output_path(candidate)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(out),
                "candidate": result["candidate"],
                "state_horizon": result["state_horizon"],
                "bracket": result["certified_bracket"],
                "actual": actual_event,
                "queries": queried,
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )
    return result


def aggregate() -> dict:
    rows = []
    for candidate in CANDIDATES:
        path = output_path(candidate)
        if not path.exists():
            raise FileNotFoundError(f"missing development audit: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                key: payload[key]
                for key in (
                    "candidate",
                    "delta_raw_0_through_4",
                    "delta_scaled_0_through_4",
                    "actual_lead",
                    "predicted_persistent_event",
                    "raw_timing_error",
                    "state_horizon",
                    "maximum_certified_radius",
                    "maximum_actual_scaled_state_error_full_window",
                    "maximum_actual_error_to_bound_ratio",
                    "observed_state_tube_violations",
                    "certified_bracket",
                    "certificate_issued",
                    "bracket_contains_actual",
                    "maximum_probe_jacobian_upper",
                    "maximum_probe_jacobian_lower_estimate",
                    "maximum_output_probe_upper_lower_ratio",
                    "maximum_optimizer_jacobian_upper",
                    "maximum_optimizer_probe_upper_lower_ratio",
                    "maximum_certified_objective_hessian_lipschitz",
                    "heuristic_allowed_l_for_full_window",
                    "heuristic_l_headroom",
                    "probability_budget",
                    "operator_calls",
                    "elapsed_seconds",
                    "stop_reason",
                )
            }
        )
    output = {
        "status": "complete independent four-sweep burned-candidate audit",
        "summary": {
            "candidates": len(rows),
            "issued": sum(row["certificate_issued"] for row in rows),
            "covered": sum(row["bracket_contains_actual"] is True for row in rows),
            "state_tube_violations": sum(
                row["observed_state_tube_violations"] for row in rows
            ),
        },
        "rows": rows,
    }
    path = RESULTS / "transformer_four_sweep_development_audit.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(path), **output["summary"]}, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-index", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if sum((args.candidate_index is not None, args.all, args.aggregate)) != 1:
        parser.error("choose exactly one of --candidate-index, --all, or --aggregate")
    if args.aggregate:
        aggregate()
    elif args.all:
        for candidate in CANDIDATES:
            run_candidate(candidate)
        aggregate()
    else:
        if not 0 <= args.candidate_index < len(CANDIDATES):
            parser.error("candidate index is outside the burned candidate list")
        run_candidate(CANDIDATES[args.candidate_index])


if __name__ == "__main__":
    main()
