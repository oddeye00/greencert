#!/usr/bin/env python3
"""Full-space HVP modal forecasts for the smooth modular Transformer.

This development tool measures whether the signed recentering mechanism
transfers to an optimizer-state Transformer trajectory before any rigorous
uncertainty envelope is attempted.  It never forms a dense Hessian.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from matrix_free_mlp import signed_variational_recenter
from transformer_hvp_grokking_v15 import (
    TransformerConfig,
    accuracy,
    artifact_paths,
    flat_spec,
    gradient,
    gradient_and_objective_hvp,
    gradient_hvp_and_third_contraction,
    logits,
    make_disjoint_split,
    make_template,
    objective_hvp,
    replayable_gradient_and_hvp,
)


ROOT = Path(__file__).resolve().parents[1]


def split_state(state: Tensor) -> tuple[Tensor, Tensor]:
    if state.numel() % 2:
        raise ValueError("optimizer state dimension must be even")
    return state.chunk(2)


def optimizer_map(
    state: Tensor,
    train_pairs: Tensor,
    train_labels: Tensor,
    template,
    spec,
    config: TransformerConfig,
) -> Tensor:
    parameter, velocity = split_state(state)
    next_velocity = config.momentum * velocity + gradient(
        parameter, train_pairs, train_labels, template, spec, config
    )
    next_parameter = parameter - config.learning_rate * next_velocity
    return torch.cat((next_parameter, next_velocity))


def optimizer_jvp(
    center: Tensor,
    direction: Tensor,
    train_pairs: Tensor,
    train_labels: Tensor,
    template,
    spec,
    config: TransformerConfig,
) -> Tensor:
    parameter, _ = split_state(center)
    d_parameter, d_velocity = split_state(direction)
    hessian_direction = objective_hvp(
        parameter, d_parameter, train_pairs, train_labels, template, spec, config
    )
    next_d_velocity = config.momentum * d_velocity + hessian_direction
    next_d_parameter = d_parameter - config.learning_rate * next_d_velocity
    return torch.cat((next_d_parameter, next_d_velocity))


def optimizer_map_and_jvp(
    center: Tensor,
    direction: Tensor,
    train_pairs: Tensor,
    train_labels: Tensor,
    template,
    spec,
    config: TransformerConfig,
) -> tuple[Tensor, Tensor]:
    """Return the optimizer map and its JVP from one shared derivative graph."""

    parameter, velocity = split_state(center)
    d_parameter, d_velocity = split_state(direction)
    objective_gradient, hessian_direction = gradient_and_objective_hvp(
        parameter,
        d_parameter,
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )
    next_velocity = config.momentum * velocity + objective_gradient
    next_parameter = parameter - config.learning_rate * next_velocity
    next_d_velocity = config.momentum * d_velocity + hessian_direction
    next_d_parameter = d_parameter - config.learning_rate * next_d_velocity
    return (
        torch.cat((next_parameter, next_velocity)),
        torch.cat((next_d_parameter, next_d_velocity)),
    )


def replayable_anchor_optimizer(
    center: Tensor,
    train_pairs: Tensor,
    train_labels: Tensor,
    template,
    spec,
    config: TransformerConfig,
):
    """Return one anchor map value and a replayable anchor-JVP closure."""

    parameter, velocity = split_state(center)
    objective_gradient, parameter_hvp = replayable_gradient_and_hvp(
        parameter, train_pairs, train_labels, template, spec, config
    )
    next_velocity = config.momentum * velocity + objective_gradient
    mapped = torch.cat(
        (parameter - config.learning_rate * next_velocity, next_velocity)
    )

    def apply(direction: Tensor) -> Tensor:
        d_parameter, d_velocity = split_state(direction)
        next_d_velocity = config.momentum * d_velocity + parameter_hvp(d_parameter)
        return torch.cat(
            (
                d_parameter - config.learning_rate * next_d_velocity,
                next_d_velocity,
            )
        )

    return mapped, apply


def optimizer_map_jvp_quadratic(
    center: Tensor,
    direction: Tensor,
    train_pairs: Tensor,
    train_labels: Tensor,
    template,
    spec,
    config: TransformerConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fuse an unscaled optimizer map/JVP with scaled quadratic forcing.

    The first two returns use training coordinates ``(theta, velocity)`` so
    they can be inserted directly into the streaming recentering pipeline.
    The third return is ``D^2G[direction,direction]/2`` in certificate
    coordinates ``(theta, learning_rate * velocity)``.
    """

    parameter, velocity = split_state(center)
    d_parameter, d_velocity = split_state(direction)
    objective_gradient, hessian_direction, third_direction = (
        gradient_hvp_and_third_contraction(
            parameter,
            d_parameter,
            train_pairs,
            train_labels,
            template,
            spec,
            config,
        )
    )
    next_velocity = config.momentum * velocity + objective_gradient
    mapped = torch.cat(
        (parameter - config.learning_rate * next_velocity, next_velocity)
    )
    next_d_velocity = config.momentum * d_velocity + hessian_direction
    image = torch.cat(
        (
            d_parameter - config.learning_rate * next_d_velocity,
            next_d_velocity,
        )
    )
    if bool(torch.any(d_parameter != 0.0)):
        scaled_quadratic_velocity = 0.5 * config.learning_rate * third_direction
        quadratic = torch.cat(
            (-scaled_quadratic_velocity, scaled_quadratic_velocity)
        )
    else:
        quadratic = torch.zeros_like(direction)
    return mapped, image, quadratic


@torch.no_grad()
def affine_reference(
    anchor: Tensor,
    map_step,
    anchor_jvp,
    *,
    horizon: int,
) -> Tensor:
    displacement = torch.zeros_like(anchor)
    defect = map_step(anchor) - anchor
    rows = [anchor.clone()]
    for _ in range(horizon):
        displacement = anchor_jvp(displacement) + defect
        rows.append(anchor + displacement)
    return torch.stack(rows)


@torch.no_grad()
def first_persistent(values: np.ndarray, required: int, persistence: int) -> int | None:
    for start in range(0, len(values) - persistence + 1):
        if np.all(values[start : start + persistence] >= required):
            return int(start)
    return None


def load_run(seed: int, *, development: bool) -> tuple[dict, dict[str, np.ndarray]]:
    result_path, checkpoint_path = artifact_paths(seed, development=development)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoints = dict(np.load(checkpoint_path))
    return payload, checkpoints


def run(
    seed: int,
    anchor: int,
    *,
    horizon: int,
    sweeps: int,
    persistence: int,
    development: bool = True,
    evaluate_actual: bool = True,
) -> dict:
    payload, checkpoints = load_run(seed, development=development)
    config = TransformerConfig(**payload["config"])
    torch.set_num_threads(config.threads)
    template = make_template(config)
    spec = flat_spec(template)
    data = make_disjoint_split(config)
    train_pairs, train_labels = data[:2]
    cert_pairs, cert_labels = data[4:]
    parameter = torch.from_numpy(checkpoints[f"step_{anchor}"]).to(torch.float64)
    velocity = torch.from_numpy(checkpoints[f"velocity_{anchor}"]).to(torch.float64)
    anchor_state = torch.cat((parameter, velocity))

    def map_step(state: Tensor) -> Tensor:
        return optimizer_map(
            state, train_pairs, train_labels, template, spec, config
        )

    def jvp(center: Tensor, direction: Tensor) -> Tensor:
        return optimizer_jvp(
            center, direction, train_pairs, train_labels, template, spec, config
        )

    started = time.perf_counter()
    raw = affine_reference(
        anchor_state,
        map_step,
        lambda direction: jvp(anchor_state, direction),
        horizon=horizon,
    )
    corrected = raw
    sweep_rows = []
    for index in range(sweeps):
        corrected, diagnostic = signed_variational_recenter(
            corrected, map_step, jvp, numeric_cap=1e6
        )
        sweep_rows.append({"sweep": index + 1, **diagnostic})

    exact_tensor = None
    if evaluate_actual:
        exact = [anchor_state]
        for _ in range(horizon):
            exact.append(map_step(exact[-1]))
        exact_tensor = torch.stack(exact)

    residual_norm = np.asarray([
        float(torch.linalg.vector_norm(map_step(corrected[step]) - corrected[step + 1]))
        for step in range(len(corrected) - 1)
    ])

    def count_path(reference: Tensor) -> np.ndarray:
        rows = []
        for state in reference:
            theta, _ = split_state(state)
            rows.append(int((logits(theta, cert_pairs, template, spec).argmax(1) == cert_labels).sum()))
        return np.asarray(rows, dtype=np.int64)

    raw_count = count_path(raw)
    corrected_count = count_path(corrected)
    exact_count = None if exact_tensor is None else count_path(exact_tensor)
    errors = None
    if exact_tensor is not None:
        common = min(len(corrected), len(exact_tensor))
        errors = torch.linalg.vector_norm(corrected[:common] - exact_tensor[:common], dim=1)
    thresholds = (0.60, 0.70, 0.80, 0.90, 0.95)
    required = {f"{gate:.2f}": int(np.ceil(gate * len(cert_pairs))) for gate in thresholds}
    events = {
        label: {
            "required": count,
            "raw": first_persistent(raw_count, count, persistence),
            "recentered": first_persistent(corrected_count, count, persistence),
            "actual": (
                None if exact_count is None
                else first_persistent(exact_count, count, persistence)
            ),
        }
        for label, count in required.items()
    }
    result = {
        "status": (
            "development-only Transformer modal forecast"
            if development else "prospective blind Transformer modal forecast"
        ),
        "seed": seed,
        "anchor": anchor,
        "horizon": horizon,
        "persistence": persistence,
        "parameter_count": int(parameter.numel()),
        "optimizer_state_dimension": int(anchor_state.numel()),
        "dense_hessian_entries_formed": 0,
        "sweeps": sweep_rows,
        "events": events,
        "maximum_recentered_state_error": None if errors is None else float(errors.max()),
        "final_recentered_state_error": None if errors is None else float(errors[-1]),
        "maximum_recentered_residual_norm": float(residual_norm.max()),
        "elapsed_seconds": time.perf_counter() - started,
        "raw_count": raw_count.tolist(),
        "recentered_count": corrected_count.tolist(),
        "actual_count": None if exact_count is None else exact_count.tolist(),
        "state_error": None if errors is None else errors.numpy().tolist(),
        "recentered_residual_norm": residual_norm.tolist(),
    }
    kind = (
        "development"
        if development else ("prospective_audit" if evaluate_actual else "prospective_blind")
    )
    out = ROOT / "results" / f"transformer_modal_{kind}_seed_{seed}_anchor_{anchor}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["output"] = str(out)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--anchor", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--persistence", type=int, default=25)
    parser.add_argument("--prospective", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(
        args.seed,
        args.anchor,
        horizon=args.horizon,
        sweeps=args.sweeps,
        persistence=args.persistence,
        development=not args.prospective,
        evaluate_actual=not args.prospective,
    ), indent=2))


if __name__ == "__main__":
    main()
