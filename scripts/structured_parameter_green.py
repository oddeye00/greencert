#!/usr/bin/env python3
"""Structured parameter-only Green products for scaled momentum.

For ``G(theta, w) = (theta-r, r)`` with
``r = mu*w + eta*grad F(theta)``, every nonlinear Taylor remainder has the
form ``B q = (-eta*q, eta*q)`` and depends only on the parameter component.
This module applies

    T_H = P_theta K_H B

and its transpose without materializing ``K_H``, ``B``, or ``P_theta``.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
import math

import torch
from torch import Tensor


def _validate_products(
    jvps: Sequence[Callable[[Tensor], Tensor]],
    vjps: Sequence[Callable[[Tensor], Tensor]],
    parameter_dimension: int,
) -> int:
    if len(jvps) != len(vjps) or not jvps:
        raise ValueError("JVP/VJP sequences must have equal positive length")
    if parameter_dimension < 1:
        raise ValueError("parameter_dimension must be positive")
    return len(jvps)


def make_structured_parameter_green_products(
    jvps: Sequence[Callable[[Tensor], Tensor]],
    vjps: Sequence[Callable[[Tensor], Tensor]],
    parameter_dimension: int,
    injection_scale: float,
) -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """Return scalar products with ``T=P_theta K B`` and ``T^T``.

    ``B q=(-injection_scale*q, injection_scale*q)``.  Inputs and outputs have
    shape ``(horizon * parameter_dimension,)`` even though the internal
    optimizer state has twice that dimension.
    """

    horizon = _validate_products(jvps, vjps, parameter_dimension)
    scale = float(injection_scale)
    if not torch.isfinite(torch.tensor(scale)):
        raise ValueError("injection_scale must be finite")

    def unpack(vector: Tensor) -> Tensor:
        if vector.numel() != horizon * parameter_dimension:
            raise ValueError("structured Green vector has the wrong dimension")
        return vector.reshape(horizon, parameter_dimension)

    def apply(forcing: Tensor) -> Tensor:
        rows = unpack(forcing)
        state = torch.zeros(
            2 * parameter_dimension,
            dtype=forcing.dtype,
            device=forcing.device,
        )
        output = []
        for step in range(horizon):
            injection = torch.cat((-scale * rows[step], scale * rows[step]))
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
            output[step] = scale * (velocity_adjoint - parameter_adjoint)
            adjoint = vjps[step](adjoint)
        return torch.stack(output).reshape(-1)

    return apply, transpose


def make_batched_structured_parameter_green_products(
    jvps: Sequence[Callable[[Tensor], Tensor]],
    vjps: Sequence[Callable[[Tensor], Tensor]],
    parameter_dimension: int,
    injection_scale: float,
) -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """Return row-batched products with ``T=P_theta K B`` and ``T^T``."""

    horizon = _validate_products(jvps, vjps, parameter_dimension)
    scale = float(injection_scale)
    if not torch.isfinite(torch.tensor(scale)):
        raise ValueError("injection_scale must be finite")

    def unpack(rows: Tensor) -> Tensor:
        if rows.ndim != 2 or rows.shape[1] != horizon * parameter_dimension:
            raise ValueError("structured Green probe block has the wrong shape")
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
                (-scale * forcing[:, step], scale * forcing[:, step]), dim=1
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
            output[step] = scale * (velocity_adjoint - parameter_adjoint)
            adjoint = vjps[step](adjoint)
        return torch.stack(output, dim=1).reshape(rows.shape[0], -1)

    return apply, transpose


def structured_quadratic_root(
    response_bound: float,
    structured_gain: float,
    hessian_lipschitz: float,
) -> float | None:
    """Smaller root of ``Y + kappa*L*E^2/2 <= E``.

    Returns ``None`` when the nonnegative quadratic does not close.
    """

    response = float(response_bound)
    coefficient = float(structured_gain) * float(hessian_lipschitz)
    if response < 0.0 or coefficient < 0.0:
        raise ValueError("bounds must be nonnegative")
    # Do not round the Python float through PyTorch's default float32 dtype:
    # a valid binary64 bound above ~3e38 would otherwise be rejected as inf.
    if not all(math.isfinite(value) for value in (response, coefficient)):
        raise ValueError("bounds must be finite")
    if coefficient == 0.0:
        return response
    discriminant = 1.0 - 2.0 * coefficient * response
    if discriminant < 0.0:
        return None
    # Cancellation-safe form of (1-sqrt(discriminant))/coefficient.
    return 2.0 * response / (1.0 + discriminant**0.5)
