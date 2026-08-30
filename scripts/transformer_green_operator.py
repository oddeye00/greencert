#!/usr/bin/env python3
"""Matrix-free finite-window Green operator for scaled momentum dynamics."""
from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import Tensor

from probe_jacobian_bound import ProbeConfig, ProbeRegistry, gram_norm_bound
from transformer_hvp_grokking import TransformerConfig
from transformer_optimizer_probe import make_scaled_optimizer_jvp_vjp


def make_causal_green_products(
    jvps: Sequence[Callable[[Tensor], Tensor]],
    vjps: Sequence[Callable[[Tensor], Tensor]],
    state_dimension: int,
) -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """Return `K` and `K^T` for a fixed nonautonomous Jacobian sequence."""
    if len(jvps) != len(vjps) or not jvps:
        raise ValueError("JVP/VJP sequences must have equal positive length")
    if state_dimension < 1:
        raise ValueError("state_dimension must be positive")
    horizon = len(jvps)

    def unpack(vector: Tensor) -> Tensor:
        if vector.numel() != horizon * state_dimension:
            raise ValueError("Green-operator vector has the wrong dimension")
        return vector.reshape(horizon, state_dimension)

    def apply(injection: Tensor) -> Tensor:
        rows = unpack(injection)
        state = torch.zeros_like(rows[0])
        output = []
        for step in range(horizon):
            state = jvps[step](state) + rows[step]
            output.append(state)
        return torch.stack(output).reshape(-1)

    def transpose(cotangent: Tensor) -> Tensor:
        rows = unpack(cotangent)
        adjoint = torch.zeros_like(rows[0])
        output = [None] * horizon
        for step in range(horizon - 1, -1, -1):
            adjoint = adjoint + rows[step]
            output[step] = adjoint
            adjoint = vjps[step](adjoint)
        return torch.stack(output).reshape(-1)

    return apply, transpose


def make_transformer_green_products(
    parameter_path: Tensor,
    train_pairs: Tensor,
    train_labels: Tensor,
    template,
    spec,
    config: TransformerConfig,
) -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """Build products for transitions based at `parameter_path[0:H]`."""
    if parameter_path.ndim != 2 or len(parameter_path) < 1:
        raise ValueError("parameter_path must contain at least one transition center")
    products = [
        make_scaled_optimizer_jvp_vjp(
            parameter, train_pairs, train_labels, template, spec, config
        )
        for parameter in parameter_path
    ]
    return make_causal_green_products(
        [row[0] for row in products],
        [row[1] for row in products],
        2 * parameter_path.shape[1],
    )


def green_norm_bound(
    parameter_path: Tensor,
    train_pairs: Tensor,
    train_labels: Tensor,
    template,
    spec,
    config: TransformerConfig,
    probe: ProbeConfig,
    identity: tuple[int, ...],
    registry: ProbeRegistry,
) -> dict:
    apply, transpose = make_transformer_green_products(
        parameter_path, train_pairs, train_labels, template, spec, config
    )

    def gram(vector: Tensor) -> Tensor:
        return transpose(apply(vector))

    horizon = len(parameter_path)
    state_dimension = 2 * parameter_path.shape[1]
    result = gram_norm_bound(
        gram,
        dimension=horizon * state_dimension,
        dtype=parameter_path.dtype,
        device=parameter_path.device,
        config=probe,
        identity=identity,
        registry=registry,
    )
    result.update(
        {
            "green_operator_norm_upper_bound": result["operator_norm_upper_bound"],
            "green_operator_norm_lower_estimate": result[
                "operator_norm_lower_estimate"
            ],
            "horizon": horizon,
            "state_dimension": state_dimension,
            "green_jvp_calls": result["gram_applications"],
            "green_vjp_calls": result["gram_applications"],
            "objective_hvp_calls": 2 * horizon * result["gram_applications"],
        }
    )
    return result
