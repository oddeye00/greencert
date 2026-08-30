#!/usr/bin/env python3
"""Batched matrix-free products for GreenCert probe blocks.

This module is a post-seal implementation acceleration.  It changes neither
the random-probe theorem nor the optimizer/output operators: the independent
Gaussian directions are generated in the same order and evaluated together
with PyTorch's batched reverse-mode interface.
"""
from __future__ import annotations

from math import sqrt
import time
from typing import Callable, Sequence

import torch
from torch import Tensor

from probe_jacobian_bound import ProbeConfig
from transformer_hvp_grokking import TransformerConfig, logits, objective


def objective_hvp_batch(
    parameter: Tensor,
    vectors: Tensor,
    pairs: Tensor,
    labels: Tensor,
    template,
    spec,
    config: TransformerConfig,
) -> Tensor:
    """Return one Hessian product per row of ``vectors``.

    ``is_grads_batched=True`` asks autograd to vectorize the reverse products;
    mathematically this is exactly a stack of ordinary Hessian-vector products.
    """
    if vectors.ndim != 2 or vectors.shape[1] != parameter.numel():
        raise ValueError("vectors must have shape (batch, parameter_count)")
    with torch.enable_grad():
        point = parameter.detach().requires_grad_(True)
        value = objective(point, pairs, labels, template, spec, config)
        (gradient,) = torch.autograd.grad(value, point, create_graph=True)
        (products,) = torch.autograd.grad(
            gradient,
            point,
            grad_outputs=vectors.detach(),
            is_grads_batched=True,
        )
    return products.detach()


def make_batched_scaled_optimizer_products(
    parameter: Tensor,
    train_pairs: Tensor,
    train_labels: Tensor,
    template,
    spec,
    config: TransformerConfig,
) -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """Return batched JVP/VJP maps for ``(theta, eta * velocity)``."""
    dimension = parameter.numel()
    eta, momentum = config.learning_rate, config.momentum

    def split(rows: Tensor) -> tuple[Tensor, Tensor]:
        if rows.ndim != 2 or rows.shape[1] != 2 * dimension:
            raise ValueError("optimizer rows have the wrong shape")
        return rows[:, :dimension], rows[:, dimension:]

    def hvp(rows: Tensor) -> Tensor:
        return objective_hvp_batch(
            parameter,
            rows,
            train_pairs,
            train_labels,
            template,
            spec,
            config,
        )

    def jvp(rows: Tensor) -> Tensor:
        d_parameter, d_velocity = split(rows)
        d_next_velocity = momentum * d_velocity + eta * hvp(d_parameter)
        return torch.cat((d_parameter - d_next_velocity, d_next_velocity), dim=1)

    def vjp(rows: Tensor) -> Tensor:
        parameter_cotangent, velocity_cotangent = split(rows)
        difference = velocity_cotangent - parameter_cotangent
        return torch.cat(
            (
                parameter_cotangent + eta * hvp(difference),
                momentum * difference,
            ),
            dim=1,
        )

    return jvp, vjp


def make_batched_causal_green_products(
    jvps: Sequence[Callable[[Tensor], Tensor]],
    vjps: Sequence[Callable[[Tensor], Tensor]],
    state_dimension: int,
) -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """Return batched products with ``K_H`` and ``K_H^T``."""
    if len(jvps) != len(vjps) or not jvps:
        raise ValueError("JVP/VJP sequences must have equal positive length")
    horizon = len(jvps)

    def unpack(rows: Tensor) -> Tensor:
        if rows.ndim != 2 or rows.shape[1] != horizon * state_dimension:
            raise ValueError("Green probe block has the wrong shape")
        return rows.reshape(rows.shape[0], horizon, state_dimension)

    def apply(rows: Tensor) -> Tensor:
        injections = unpack(rows)
        state = torch.zeros_like(injections[:, 0])
        output = []
        for step in range(horizon):
            state = jvps[step](state) + injections[:, step]
            output.append(state)
        return torch.stack(output, dim=1).reshape(rows.shape[0], -1)

    def transpose(rows: Tensor) -> Tensor:
        cotangents = unpack(rows)
        adjoint = torch.zeros_like(cotangents[:, 0])
        output: list[Tensor | None] = [None] * horizon
        for step in range(horizon - 1, -1, -1):
            adjoint = adjoint + cotangents[:, step]
            output[step] = adjoint
            adjoint = vjps[step](adjoint)
        return torch.stack(output, dim=1).reshape(rows.shape[0], -1)

    return apply, transpose


def make_batched_transformer_green_products(
    parameter_path: Tensor,
    train_pairs: Tensor,
    train_labels: Tensor,
    template,
    spec,
    config: TransformerConfig,
) -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """Build batched Green products along a fixed parameter path."""
    if parameter_path.ndim != 2 or len(parameter_path) < 1:
        raise ValueError("parameter_path must contain at least one transition")
    products = [
        make_batched_scaled_optimizer_products(
            parameter,
            train_pairs,
            train_labels,
            template,
            spec,
            config,
        )
        for parameter in parameter_path
    ]
    return make_batched_causal_green_products(
        [row[0] for row in products],
        [row[1] for row in products],
        2 * parameter_path.shape[1],
    )


def make_batched_output_gram_operator(
    parameter: Tensor,
    pairs: Tensor,
    template,
    spec,
) -> Callable[[Tensor], Tensor]:
    """Return batched rows ``v -> J^T J v`` for the flattened logits map."""

    def forward(theta: Tensor) -> Tensor:
        return logits(theta, pairs, template, spec).reshape(-1)

    def apply(vectors: Tensor) -> Tensor:
        if vectors.ndim != 2 or vectors.shape[1] != parameter.numel():
            raise ValueError("vectors must have shape (batch, parameter_count)")
        with torch.enable_grad():
            theta = parameter.detach().requires_grad_(True)
            output = forward(theta)
            auxiliary = torch.zeros(
                output.numel(),
                dtype=parameter.dtype,
                device=parameter.device,
                requires_grad=True,
            )
            (transposed,) = torch.autograd.grad(
                output,
                theta,
                grad_outputs=auxiliary,
                create_graph=True,
            )
            (jv,) = torch.autograd.grad(
                transposed,
                auxiliary,
                grad_outputs=vectors.detach(),
                is_grads_batched=True,
                retain_graph=True,
            )
            (products,) = torch.autograd.grad(
                output,
                theta,
                grad_outputs=jv.detach(),
                is_grads_batched=True,
            )
        return products.detach()

    return apply


@torch.no_grad()
def batched_gram_norm_bound(
    apply_gram: Callable[[Tensor], Tensor],
    *,
    dimension: int,
    dtype: torch.dtype,
    device: torch.device,
    config: ProbeConfig,
    seed: int,
) -> dict:
    """Evaluate the unchanged Gaussian Gram bound as one probe block."""
    if dimension < 1:
        raise ValueError("dimension must be positive")
    generator = torch.Generator(device=device).manual_seed(int(seed))
    # Generate rows separately to preserve the scalar implementation's exact
    # random stream and make scalar/batched regression comparisons meaningful.
    vectors = torch.stack(
        [
            torch.randn(
                dimension,
                generator=generator,
                dtype=dtype,
                device=device,
            )
            for _ in range(config.probes)
        ]
    )
    initial_norms = torch.linalg.vector_norm(vectors, dim=1)
    for _ in range(config.power):
        vectors = apply_gram(vectors)
    final_norms = torch.linalg.vector_norm(vectors, dim=1)
    best = float(final_norms.max())
    valid = (initial_norms > 0.0) & (final_norms > 0.0)
    if bool(valid.any()):
        ratios = final_norms[valid] / initial_norms[valid]
        lower = float(ratios.max() ** (1.0 / (2.0 * config.power)))
    else:
        lower = 0.0
    calibration = config.c_delta()
    bound = 0.0 if best <= 0.0 else (best / calibration) ** (
        1.0 / (2.0 * config.power)
    )
    return {
        "rng_seed": int(seed),
        "Y": best,
        "c_delta": calibration,
        "operator_norm_upper_bound": bound,
        "operator_norm_lower_estimate": lower,
        "logical_gram_applications": config.probes * config.power,
        "batched_gram_calls": config.power,
        "probe_batch_size": config.probes,
        "delta": config.delta,
        "probes": config.probes,
        "power": config.power,
    }


@torch.no_grad()
def progressive_batched_gram_norm_bounds(
    apply_gram: Callable[[Tensor], Tensor],
    *,
    dimension: int,
    dtype: torch.dtype,
    device: torch.device,
    config: ProbeConfig,
    seed: int,
) -> dict:
    """Return simultaneous bounds after every power from one probe block.

    On the single Gaussian event

        max_i |v_max^T g_i| >= c_delta,

    the usual Gram bound holds for *every* positive power.  Consequently a
    caller may inspect these rows in order and stop at the first power whose
    downstream deterministic closure succeeds without paying an additional
    probability union bound.  Probe directions and the maximum power remain
    fixed before any result is observed.
    """

    if dimension < 1:
        raise ValueError("dimension must be positive")
    generator = torch.Generator(device=device).manual_seed(int(seed))
    vectors = torch.stack(
        [
            torch.randn(
                dimension,
                generator=generator,
                dtype=dtype,
                device=device,
            )
            for _ in range(config.probes)
        ]
    )
    initial_norms = torch.linalg.vector_norm(vectors, dim=1)
    calibration = config.c_delta()
    rows = []
    cumulative_seconds = 0.0
    for power in range(1, config.power + 1):
        started = time.perf_counter()
        vectors = apply_gram(vectors)
        cumulative_seconds += time.perf_counter() - started
        final_norms = torch.linalg.vector_norm(vectors, dim=1)
        best = float(final_norms.max())
        valid = (initial_norms > 0.0) & (final_norms > 0.0)
        if bool(valid.any()):
            ratios = final_norms[valid] / initial_norms[valid]
            lower = float(ratios.max() ** (1.0 / (2.0 * power)))
        else:
            lower = 0.0
        bound = 0.0 if best <= 0.0 else (best / calibration) ** (
            1.0 / (2.0 * power)
        )
        rows.append(
            {
                "power": power,
                "Y": best,
                "c_delta": calibration,
                "operator_norm_upper_bound": bound,
                "operator_norm_lower_estimate": lower,
                "logical_gram_applications": config.probes * power,
                "batched_gram_calls": power,
                "cumulative_operator_seconds": cumulative_seconds,
            }
        )
    return {
        "rng_seed": int(seed),
        "delta": config.delta,
        "probes": config.probes,
        "maximum_power": config.power,
        "single_event_simultaneous_over_all_powers": True,
        "rows": rows,
    }


def relative_error(left: Tensor, right: Tensor) -> float:
    """Stable Euclidean relative error used by regression tests."""
    return float(torch.linalg.vector_norm(left - right)) / max(
        float(torch.linalg.vector_norm(right)), sqrt(torch.finfo(right.dtype).tiny)
    )
