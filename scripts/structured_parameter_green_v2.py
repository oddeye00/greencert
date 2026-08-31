#!/usr/bin/env python3
"""Profiled forcing-subspace products extending structured_parameter_green.

The sealed v2 audit depends on the original module, so this extension lives in
a new file.  It applies ``T_L = P K B D_L`` and ``T_L^T`` without changing the
underlying causal Green implementation.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from structured_parameter_green import (
    make_batched_structured_parameter_green_products,
    make_structured_parameter_green_products,
)


def _profile_tensor(
    profile: Sequence[float],
    *,
    horizon: int,
    reference: Tensor,
) -> Tensor:
    if len(profile) != horizon:
        raise ValueError("curvature profile must have one value per step")
    values = torch.as_tensor(profile, dtype=reference.dtype, device=reference.device)
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("curvature profile must be finite and nonnegative")
    return values


def make_profiled_structured_parameter_green_products(
    jvps: Sequence[Callable[[Tensor], Tensor]],
    vjps: Sequence[Callable[[Tensor], Tensor]],
    parameter_dimension: int,
    injection_scale: float,
    curvature_profile: Sequence[float],
) -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """Return scalar products with ``T D_L`` and ``(T D_L)^T``."""

    horizon = len(jvps)
    base_apply, base_transpose = make_structured_parameter_green_products(
        jvps, vjps, parameter_dimension, injection_scale
    )

    def scale(vector: Tensor) -> Tensor:
        profile = _profile_tensor(
            curvature_profile, horizon=horizon, reference=vector
        )
        return (
            vector.reshape(horizon, parameter_dimension) * profile[:, None]
        ).reshape(-1)

    def apply(normalized_forcing: Tensor) -> Tensor:
        return base_apply(scale(normalized_forcing))

    def transpose(parameter_cotangent: Tensor) -> Tensor:
        return scale(base_transpose(parameter_cotangent))

    return apply, transpose


def make_batched_profiled_structured_parameter_green_products(
    jvps: Sequence[Callable[[Tensor], Tensor]],
    vjps: Sequence[Callable[[Tensor], Tensor]],
    parameter_dimension: int,
    injection_scale: float,
    curvature_profile: Sequence[float],
) -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """Return row-batched products with ``T D_L`` and ``(T D_L)^T``."""

    horizon = len(jvps)
    base_apply, base_transpose = make_batched_structured_parameter_green_products(
        jvps, vjps, parameter_dimension, injection_scale
    )

    def scale(rows: Tensor) -> Tensor:
        if rows.ndim != 2:
            raise ValueError("profiled Green probes must be a row block")
        profile = _profile_tensor(
            curvature_profile, horizon=horizon, reference=rows
        )
        return (
            rows.reshape(rows.shape[0], horizon, parameter_dimension)
            * profile[None, :, None]
        ).reshape(rows.shape[0], -1)

    def apply(normalized_forcing: Tensor) -> Tensor:
        return base_apply(scale(normalized_forcing))

    def transpose(parameter_cotangent: Tensor) -> Tensor:
        return scale(base_transpose(parameter_cotangent))

    return apply, transpose


def make_anchor_fixed_profiled_structured_parameter_green_products(
    jvps: Sequence[Callable[[Tensor], Tensor]],
    vjps: Sequence[Callable[[Tensor], Tensor]],
    parameter_dimension: int,
    injection_scale: float,
    curvature_profile: Sequence[float],
) -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """Return ``T D_L Q_0`` products for a fixed-anchor horizon.

    Inputs contain only forcing blocks 1 through ``H-1``.  The omitted update-0
    nonlinear remainder is exactly zero because the anchor error is zero.
    """

    horizon = len(jvps)
    base_apply, base_transpose = make_profiled_structured_parameter_green_products(
        jvps, vjps, parameter_dimension, injection_scale, curvature_profile
    )
    reduced_dimension = max(0, horizon - 1) * parameter_dimension

    def inject(vector: Tensor) -> Tensor:
        if vector.numel() != reduced_dimension:
            raise ValueError("anchor-fixed forcing vector has the wrong dimension")
        return torch.cat(
            (
                torch.zeros(
                    parameter_dimension, dtype=vector.dtype, device=vector.device
                ),
                vector.reshape(-1),
            )
        )

    def apply(reduced_forcing: Tensor) -> Tensor:
        return base_apply(inject(reduced_forcing))

    def transpose(parameter_cotangent: Tensor) -> Tensor:
        return base_transpose(parameter_cotangent).reshape(
            horizon, parameter_dimension
        )[1:].reshape(-1)

    return apply, transpose


def make_batched_anchor_fixed_profiled_structured_parameter_green_products(
    jvps: Sequence[Callable[[Tensor], Tensor]],
    vjps: Sequence[Callable[[Tensor], Tensor]],
    parameter_dimension: int,
    injection_scale: float,
    curvature_profile: Sequence[float],
) -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """Row-batched ``T D_L Q_0`` products for a fixed-anchor horizon."""

    horizon = len(jvps)
    base_apply, base_transpose = (
        make_batched_profiled_structured_parameter_green_products(
            jvps, vjps, parameter_dimension, injection_scale, curvature_profile
        )
    )
    reduced_dimension = max(0, horizon - 1) * parameter_dimension

    def inject(rows: Tensor) -> Tensor:
        if rows.ndim != 2 or rows.shape[1] != reduced_dimension:
            raise ValueError("anchor-fixed probe block has the wrong shape")
        return torch.cat(
            (
                torch.zeros(
                    rows.shape[0],
                    parameter_dimension,
                    dtype=rows.dtype,
                    device=rows.device,
                ),
                rows,
            ),
            dim=1,
        )

    def apply(reduced_forcing: Tensor) -> Tensor:
        return base_apply(inject(reduced_forcing))

    def transpose(parameter_cotangent: Tensor) -> Tensor:
        return base_transpose(parameter_cotangent).reshape(
            parameter_cotangent.shape[0], horizon, parameter_dimension
        )[:, 1:].reshape(parameter_cotangent.shape[0], -1)

    return apply, transpose


def profiled_quadratic_root(
    response_bound: float,
    profiled_quadratic_gain: float,
) -> float | None:
    """Smaller root of ``Y + kappa_L*E^2/2 <= E``."""

    response = float(response_bound)
    gain = float(profiled_quadratic_gain)
    values = torch.tensor((response, gain), dtype=torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("bounds must be finite")
    if response < 0.0 or gain < 0.0:
        raise ValueError("bounds must be nonnegative")
    if gain == 0.0:
        return response
    discriminant = 1.0 - 2.0 * gain * response
    if discriminant < 0.0:
        return None
    return 2.0 * response / (1.0 + discriminant**0.5)
