#!/usr/bin/env python3
"""Time-diagonal scaling wrappers for batched causal Green products."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor


def _weights(
    values: Sequence[float], *, horizon: int, dtype: torch.dtype, device: torch.device
) -> Tensor:
    result = torch.as_tensor(values, dtype=dtype, device=device)
    if result.shape != (horizon,):
        raise ValueError("time weights must have shape (horizon,)")
    if not bool(torch.isfinite(result).all()) or not bool((result > 0.0).all()):
        raise ValueError("time weights must be finite and positive")
    return result


def make_weighted_batched_green_products(
    apply_green: Callable[[Tensor], Tensor],
    apply_green_transpose: Callable[[Tensor], Tensor],
    *,
    state_weights: Sequence[float],
    injection_weights: Sequence[float],
    state_dimension: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """Return products for ``D_w K D_v^-1`` and its exact transpose."""

    if state_dimension < 1:
        raise ValueError("state_dimension must be positive")
    horizon = len(state_weights)
    if horizon < 1 or len(injection_weights) != horizon:
        raise ValueError("state and injection weights must have equal positive length")
    state = _weights(state_weights, horizon=horizon, dtype=dtype, device=device)
    injection = _weights(
        injection_weights, horizon=horizon, dtype=dtype, device=device
    )

    def reshape(rows: Tensor) -> Tensor:
        if rows.ndim != 2 or rows.shape[1] != horizon * state_dimension:
            raise ValueError("Green rows have the wrong flattened sequence shape")
        return rows.reshape(rows.shape[0], horizon, state_dimension)

    def apply(rows: Tensor) -> Tensor:
        unscaled = reshape(rows) / injection[None, :, None]
        response = reshape(apply_green(unscaled.reshape(rows.shape[0], -1)))
        return (response * state[None, :, None]).reshape(rows.shape[0], -1)

    def transpose(rows: Tensor) -> Tensor:
        weighted = reshape(rows) * state[None, :, None]
        response = reshape(
            apply_green_transpose(weighted.reshape(rows.shape[0], -1))
        )
        return (response / injection[None, :, None]).reshape(rows.shape[0], -1)

    return apply, transpose

