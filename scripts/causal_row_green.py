#!/usr/bin/env python3
"""Causal rowwise Green bounds and chronological nonlinear envelopes."""
from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist

import torch
from torch import Tensor


def simultaneous_row_direct_image_bounds(
    image_rows: Tensor,
    *,
    family_delta: float,
    row_budgets: Sequence[float] | None = None,
) -> tuple[Tensor, dict]:
    """Bound all causal row-operator norms from one Gaussian image block.

    ``image_rows`` has shape ``(probes, horizon, output_dimension)`` and stores
    ``K_i g_l``. The probes themselves must be independent standard Gaussians
    in the full input-sequence space. Equal row budgets are used by default.
    """

    if image_rows.ndim != 3:
        raise ValueError("image_rows must have shape (probes, horizon, dimension)")
    probes, horizon, output_dimension = map(int, image_rows.shape)
    if probes < 1 or horizon < 1 or output_dimension < 1:
        raise ValueError("every image_rows dimension must be positive")
    delta = float(family_delta)
    if not math.isfinite(delta) or not 0.0 < delta < 1.0:
        raise ValueError("family_delta must lie in (0,1)")
    if not bool(torch.isfinite(image_rows).all()):
        raise ValueError("image_rows must be finite")

    if row_budgets is None:
        budgets = [delta / horizon] * horizon
    else:
        if len(row_budgets) != horizon:
            raise ValueError("row_budgets must match the horizon")
        budgets = [float(value) for value in row_budgets]
        if any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in budgets):
            raise ValueError("row budgets must lie in (0,1)")
        if math.fsum(budgets) > delta * (1.0 + 1.0e-14):
            raise ValueError("row budgets exceed family_delta")

    maxima = torch.linalg.vector_norm(image_rows, dim=2).max(dim=0).values
    calibrations = []
    for budget in budgets:
        calibration = NormalDist().inv_cdf(
            0.5 * (1.0 + budget ** (1.0 / probes))
        )
        if not math.isfinite(calibration) or calibration <= 0.0:
            raise ValueError("row calibration is not finite and positive")
        calibrations.append(calibration)
    calibration_tensor = torch.tensor(
        calibrations, dtype=image_rows.dtype, device=image_rows.device
    )
    bounds = maxima / calibration_tensor
    return bounds, {
        "probes": probes,
        "horizon": horizon,
        "family_delta": delta,
        "row_budgets": budgets,
        "calibrations": calibrations,
        "union_bound_upper": math.fsum(budgets),
        "additional_green_passes": 0,
    }


def rowwise_signed_affine_bounds(
    signed_response_rows: Tensor,
    row_gain_bounds: Sequence[float] | Tensor,
    forcing_error_bounds: Sequence[float] | Tensor,
) -> Tensor:
    """Bound a signed response plus unresolved prefix forcing errors."""

    if signed_response_rows.ndim != 2:
        raise ValueError("signed_response_rows must be time by state")
    horizon = int(signed_response_rows.shape[0])
    if horizon < 1:
        raise ValueError("the response horizon must be positive")
    dtype, device = signed_response_rows.dtype, signed_response_rows.device

    def vector(values: Sequence[float] | Tensor, name: str) -> Tensor:
        result = torch.as_tensor(values, dtype=dtype, device=device)
        if result.ndim != 1 or result.numel() != horizon:
            raise ValueError(f"{name} must match the horizon")
        if not bool(torch.isfinite(result).all()) or bool((result < 0.0).any()):
            raise ValueError(f"{name} must be finite and nonnegative")
        return result

    gains = vector(row_gain_bounds, "row_gain_bounds")
    errors = vector(forcing_error_bounds, "forcing_error_bounds")
    signed = torch.linalg.vector_norm(signed_response_rows, dim=1)
    prefix_error = torch.sqrt(torch.cumsum(errors.square(), dim=0))
    return signed + gains * prefix_error


def causal_row_quadratic_envelope(
    affine_bounds: Sequence[float] | Tensor,
    row_gain_bounds: Sequence[float] | Tensor,
    curvature_bounds: Sequence[float] | Tensor,
) -> Tensor:
    """Evaluate the explicit causal row-Green radius recursion."""

    affine = torch.as_tensor(affine_bounds)
    if not affine.is_floating_point():
        affine = affine.to(torch.float64)
    if affine.ndim != 1 or affine.numel() < 1:
        raise ValueError("affine_bounds must be a nonempty vector")
    dtype, device = affine.dtype, affine.device
    horizon = int(affine.numel())

    def vector(values: Sequence[float] | Tensor, name: str) -> Tensor:
        result = torch.as_tensor(values, dtype=dtype, device=device)
        if result.ndim != 1 or result.numel() != horizon:
            raise ValueError(f"{name} must match affine_bounds")
        if not bool(torch.isfinite(result).all()) or bool((result < 0.0).any()):
            raise ValueError(f"{name} must be finite and nonnegative")
        return result

    if not bool(torch.isfinite(affine).all()) or bool((affine < 0.0).any()):
        raise ValueError("affine_bounds must be finite and nonnegative")
    gains = vector(row_gain_bounds, "row_gain_bounds")
    curvature = vector(curvature_bounds, "curvature_bounds")
    radii = torch.empty_like(affine)
    squared_forcing_sum = torch.zeros((), dtype=dtype, device=device)
    for output_step in range(horizon):
        radii[output_step] = (
            affine[output_step]
            + gains[output_step] * torch.sqrt(squared_forcing_sum)
        )
        if output_step + 1 < horizon:
            forcing = 0.5 * curvature[output_step + 1] * radii[output_step].square()
            squared_forcing_sum = squared_forcing_sum + forcing.square()
    return radii
