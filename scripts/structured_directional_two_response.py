#!/usr/bin/env python3
"""Forcing-subspace and complementary-channel Green interfaces.

The sealed structured-Green implementations remain unchanged. This module
adds the second state channel ``C h=(h,h)`` and the scalar error accounting for
the post-v1.1.0 structured directional two-response theorem.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
from torch import Tensor


def _validate(
    jvps: Sequence[Callable[[Tensor], Tensor]],
    vjps: Sequence[Callable[[Tensor], Tensor]],
    parameter_dimension: int,
) -> int:
    if len(jvps) != len(vjps) or not jvps:
        raise ValueError("JVP/VJP sequences must have equal positive length")
    if parameter_dimension < 1:
        raise ValueError("parameter_dimension must be positive")
    return len(jvps)


def make_parameter_channel_green_products(
    jvps: Sequence[Callable[[Tensor], Tensor]],
    vjps: Sequence[Callable[[Tensor], Tensor]],
    parameter_dimension: int,
    parameter_scale: float,
    velocity_scale: float,
) -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """Return products with ``P K I`` and its transpose.

    The channel injection is ``I q=(parameter_scale*q, velocity_scale*q)``.
    Inputs and parameter outputs both have ``H*d`` coordinates.
    """

    horizon = _validate(jvps, vjps, parameter_dimension)
    left = float(parameter_scale)
    right = float(velocity_scale)
    if not math.isfinite(left) or not math.isfinite(right):
        raise ValueError("channel scales must be finite")

    def unpack(vector: Tensor) -> Tensor:
        if vector.numel() != horizon * parameter_dimension:
            raise ValueError("channel Green vector has the wrong dimension")
        return vector.reshape(horizon, parameter_dimension)

    def apply(forcing: Tensor) -> Tensor:
        rows = unpack(forcing)
        state = torch.zeros(
            2 * parameter_dimension, dtype=forcing.dtype, device=forcing.device
        )
        output = []
        for step in range(horizon):
            injection = torch.cat((left * rows[step], right * rows[step]))
            state = jvps[step](state) + injection
            output.append(state[:parameter_dimension])
        return torch.stack(output).reshape(-1)

    def transpose(parameter_cotangent: Tensor) -> Tensor:
        rows = unpack(parameter_cotangent)
        adjoint = torch.zeros(
            2 * parameter_dimension,
            dtype=parameter_cotangent.dtype,
            device=parameter_cotangent.device,
        )
        output: list[Tensor | None] = [None] * horizon
        for step in range(horizon - 1, -1, -1):
            adjoint = adjoint + torch.cat(
                (rows[step], torch.zeros_like(rows[step]))
            )
            parameter_adjoint, velocity_adjoint = adjoint.chunk(2)
            output[step] = left * parameter_adjoint + right * velocity_adjoint
            adjoint = vjps[step](adjoint)
        return torch.stack(output).reshape(-1)

    return apply, transpose


def make_batched_parameter_channel_green_products(
    jvps: Sequence[Callable[[Tensor], Tensor]],
    vjps: Sequence[Callable[[Tensor], Tensor]],
    parameter_dimension: int,
    parameter_scale: float,
    velocity_scale: float,
) -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """Row-batched products with a parameter-sized state-injection channel."""

    horizon = _validate(jvps, vjps, parameter_dimension)
    left = float(parameter_scale)
    right = float(velocity_scale)
    if not math.isfinite(left) or not math.isfinite(right):
        raise ValueError("channel scales must be finite")

    def unpack(rows: Tensor) -> Tensor:
        if rows.ndim != 2 or rows.shape[1] != horizon * parameter_dimension:
            raise ValueError("channel Green probe block has the wrong shape")
        return rows.reshape(rows.shape[0], horizon, parameter_dimension)

    def apply(rows: Tensor) -> Tensor:
        forcing = unpack(rows)
        state = torch.zeros(
            rows.shape[0],
            2 * parameter_dimension,
            dtype=rows.dtype,
            device=rows.device,
        )
        output = []
        for step in range(horizon):
            injection = torch.cat(
                (left * forcing[:, step], right * forcing[:, step]), dim=1
            )
            state = jvps[step](state) + injection
            output.append(state[:, :parameter_dimension])
        return torch.stack(output, dim=1).reshape(rows.shape[0], -1)

    def transpose(rows: Tensor) -> Tensor:
        cotangents = unpack(rows)
        adjoint = torch.zeros(
            rows.shape[0],
            2 * parameter_dimension,
            dtype=rows.dtype,
            device=rows.device,
        )
        output: list[Tensor | None] = [None] * horizon
        zeros = torch.zeros_like(cotangents[:, 0])
        for step in range(horizon - 1, -1, -1):
            adjoint = adjoint + torch.cat((cotangents[:, step], zeros), dim=1)
            parameter_adjoint = adjoint[:, :parameter_dimension]
            velocity_adjoint = adjoint[:, parameter_dimension:]
            output[step] = left * parameter_adjoint + right * velocity_adjoint
            adjoint = vjps[step](adjoint)
        return torch.stack(output, dim=1).reshape(rows.shape[0], -1)

    return apply, transpose


def split_scaled_momentum_channels(
    residual: Tensor,
    *,
    parameter_dimension: int,
    learning_rate: float,
) -> tuple[Tensor, Tensor]:
    """Split state blocks into ``B q + C h`` exactly.

    ``B q=(-eta*q,eta*q)`` and ``C h=(h,h)``. The returned tensors preserve
    all leading block dimensions except that the final state dimension is
    replaced by the parameter dimension.
    """

    eta = float(learning_rate)
    if not math.isfinite(eta) or eta <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if parameter_dimension < 1 or residual.shape[-1] != 2 * parameter_dimension:
        raise ValueError("residual has the wrong final dimension")
    parameter = residual[..., :parameter_dimension]
    velocity = residual[..., parameter_dimension:]
    structured = (velocity - parameter) / (2.0 * eta)
    complement = (parameter + velocity) / 2.0
    return structured, complement


def structured_scaled_momentum_taylor_sequence_error(
    objective_fourth_derivative_bounds: Sequence[float],
    parameter_direction_norms: Sequence[float],
) -> float:
    """Bound the parameter-forcing Taylor error ``||q-q_tilde||_2``.

    The scaled-state factor ``sqrt(2)*eta`` is intentionally absent: it is
    already represented by the structured injection ``B`` in
    ``P K B``.  This is the parameter-channel counterpart of
    ``scaled_momentum_taylor_sequence_error`` in the full-state interface.
    """

    if len(objective_fourth_derivative_bounds) != len(
        parameter_direction_norms
    ):
        raise ValueError("bound and direction sequences must have equal length")
    squared = 0.0
    for index, (bound, norm) in enumerate(
        zip(objective_fourth_derivative_bounds, parameter_direction_norms)
    ):
        bound = float(bound)
        norm = float(norm)
        if not math.isfinite(bound) or bound < 0.0:
            raise ValueError(
                f"objective_fourth_derivative_bounds[{index}] must be "
                "finite and nonnegative"
            )
        if not math.isfinite(norm) or norm < 0.0:
            raise ValueError(
                f"parameter_direction_norms[{index}] must be finite and "
                "nonnegative"
            )
        term = bound * norm**3 / 6.0
        squared += term * term
    return math.sqrt(squared)


def structured_directional_response_bound(
    *,
    approximate_parameter_response_norm: float,
    structured_green_bound: float,
    forcing_approximation_error_bound: float,
    structured_response_residual_bound: float,
    complement_green_bound: float,
    complement_path_defect_bound: float,
    complement_response_residual_bound: float,
) -> float:
    """Return the theorem-valid parameter response bound ``Y_theta``."""

    values = {
        "approximate_parameter_response_norm": approximate_parameter_response_norm,
        "structured_green_bound": structured_green_bound,
        "forcing_approximation_error_bound": forcing_approximation_error_bound,
        "structured_response_residual_bound": structured_response_residual_bound,
        "complement_green_bound": complement_green_bound,
        "complement_path_defect_bound": complement_path_defect_bound,
        "complement_response_residual_bound": complement_response_residual_bound,
    }
    checked: dict[str, float] = {}
    for name, value in values.items():
        scalar = float(value)
        if not math.isfinite(scalar) or scalar < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
        checked[name] = scalar
    return (
        checked["approximate_parameter_response_norm"]
        + checked["structured_green_bound"]
        * (
            checked["forcing_approximation_error_bound"]
            + checked["structured_response_residual_bound"]
        )
        + checked["complement_green_bound"]
        * (
            checked["complement_path_defect_bound"]
            + checked["complement_response_residual_bound"]
        )
    )
