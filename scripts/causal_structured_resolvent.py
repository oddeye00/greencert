#!/usr/bin/env python3
"""Matrix-free interfaces for the causal structured resolvent theorem."""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from structured_directional_two_response import (
    make_batched_parameter_channel_green_products,
    make_parameter_channel_green_products,
)


def scalar_hessian_parameter_green_matrix(
    hessian_scalars: Sequence[float],
    *,
    learning_rate: float,
    momentum: float,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Return the exact temporal matrix ``P K_tilde B`` for ``H_j=lambda_j I``.

    The full parameter operator is this ``H x H`` matrix Kronecker the
    parameter-space identity, so its spectral norm is dimension-independent
    and can be computed without a neural-network HVP or randomized probe.
    """

    scalars = [float(value) for value in hessian_scalars]
    if not scalars:
        raise ValueError("hessian_scalars must be nonempty")
    eta = float(learning_rate)
    mu = float(momentum)
    if not math.isfinite(eta) or eta <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(mu):
        raise ValueError("momentum must be finite")
    if any(not math.isfinite(value) for value in scalars):
        raise ValueError("hessian scalars must be finite")
    horizon = len(scalars)
    output = torch.zeros(horizon, horizon, dtype=dtype)
    injection = torch.tensor((-eta, eta), dtype=dtype)
    for source in range(horizon):
        state = torch.zeros(2, dtype=dtype)
        for step, scalar in enumerate(scalars):
            jacobian = torch.tensor(
                (
                    (1.0 - eta * scalar, -mu),
                    (eta * scalar, mu),
                ),
                dtype=dtype,
            )
            state = jacobian @ state
            if step == source:
                state = state + injection
            output[step, source] = state[0]
    return output


def scalar_hessian_structured_gain(
    hessian_scalars: Sequence[float],
    *,
    learning_rate: float,
    momentum: float,
) -> float:
    """Return the exact ``2->2`` norm of the scalar-Hessian Green operator."""

    matrix = scalar_hessian_parameter_green_matrix(
        hessian_scalars,
        learning_rate=learning_rate,
        momentum=momentum,
    )
    return float(torch.linalg.matrix_norm(matrix, ord=2))


def make_batched_scalar_hessian_optimizer_products(
    *,
    parameter_dimension: int,
    learning_rate: float,
    momentum: float,
    hessian_scalar: float,
) -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """Return cheap batched JVP/VJP maps for ``H=lambda I`` scaled momentum."""

    dimension = int(parameter_dimension)
    eta = float(learning_rate)
    mu = float(momentum)
    scalar = float(hessian_scalar)
    if dimension < 1:
        raise ValueError("parameter_dimension must be positive")
    if not math.isfinite(eta) or eta <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(mu) or not math.isfinite(scalar):
        raise ValueError("momentum and hessian_scalar must be finite")

    def split(rows: Tensor) -> tuple[Tensor, Tensor]:
        if rows.ndim != 2 or rows.shape[1] != 2 * dimension:
            raise ValueError("optimizer rows have the wrong shape")
        return rows[:, :dimension], rows[:, dimension:]

    def jvp(rows: Tensor) -> Tensor:
        parameter, velocity = split(rows)
        next_velocity = mu * velocity + eta * scalar * parameter
        return torch.cat((parameter - next_velocity, next_velocity), dim=1)

    def vjp(rows: Tensor) -> Tensor:
        parameter, velocity = split(rows)
        difference = velocity - parameter
        return torch.cat(
            (parameter + eta * scalar * difference, mu * difference), dim=1
        )

    return jvp, vjp


def finite_geometric_sum(
    alpha: float, *, horizon: int, start_power: int = 0
) -> float:
    """Return ``sum(alpha**n, n=start_power,...,horizon-1)`` safely."""

    value = float(alpha)
    horizon = int(horizon)
    start_power = int(start_power)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("alpha must be finite and nonnegative")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if start_power < 0 or start_power > horizon:
        raise ValueError("start_power must lie in [0,horizon]")
    if start_power == horizon:
        return 0.0
    # A direct finite recurrence is stable at alpha=1 and preserves the exact
    # finite-horizon meaning when alpha exceeds one.
    term = 1.0
    total = 0.0
    for power in range(horizon):
        if power >= start_power:
            total += term
        term *= value
        if not math.isfinite(total) or not math.isfinite(term):
            return math.inf
    return total


def preconditioned_structured_gain_bound(
    *, approximate_gain_bound: float, mismatch_gain_bound: float, horizon: int
) -> float:
    """Return ``kappa_0 * g_H(alpha)`` from Corollary 1."""

    kappa = float(approximate_gain_bound)
    if not math.isfinite(kappa) or kappa < 0.0:
        raise ValueError("approximate_gain_bound must be finite and nonnegative")
    return kappa * finite_geometric_sum(
        mismatch_gain_bound, horizon=horizon, start_power=0
    )


def truncated_structured_response_error_bound(
    *,
    approximate_gain_bound: float,
    mismatch_gain_bound: float,
    horizon: int,
    maximum_neumann_power: int,
    approximate_forcing_norm: float,
    forcing_approximation_error_bound: float = 0.0,
    numerical_response_error_bound: float = 0.0,
) -> float:
    """Return the certified tail/error in Corollary 2.

    ``maximum_neumann_power=m`` means that powers 0 through ``m`` were
    included in the computed response.
    """

    kappa = float(approximate_gain_bound)
    alpha = float(mismatch_gain_bound)
    forcing = float(approximate_forcing_norm)
    approximation = float(forcing_approximation_error_bound)
    numerical = float(numerical_response_error_bound)
    for name, value in (
        ("approximate_gain_bound", kappa),
        ("mismatch_gain_bound", alpha),
        ("approximate_forcing_norm", forcing),
        ("forcing_approximation_error_bound", approximation),
        ("numerical_response_error_bound", numerical),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    maximum_neumann_power = int(maximum_neumann_power)
    if maximum_neumann_power < 0 or maximum_neumann_power >= int(horizon):
        raise ValueError("maximum_neumann_power must lie in [0,horizon)")
    whole = finite_geometric_sum(alpha, horizon=int(horizon), start_power=0)
    tail = finite_geometric_sum(
        alpha,
        horizon=int(horizon),
        start_power=maximum_neumann_power + 1,
    )
    return numerical + kappa * (whole * approximation + tail * forcing)


def _validate_delta_products(
    deltas: Sequence[Callable[[Tensor], Tensor]],
    delta_transposes: Sequence[Callable[[Tensor], Tensor]],
    horizon: int,
) -> None:
    if len(deltas) != horizon or len(delta_transposes) != horizon:
        raise ValueError("delta products must match the Green horizon")


def make_causal_structured_resolvent_products(
    approximate_jvps: Sequence[Callable[[Tensor], Tensor]],
    approximate_vjps: Sequence[Callable[[Tensor], Tensor]],
    delta_products: Sequence[Callable[[Tensor], Tensor]],
    delta_transposes: Sequence[Callable[[Tensor], Tensor]],
    parameter_dimension: int,
    learning_rate: float,
) -> tuple[
    Callable[[Tensor], Tensor],
    Callable[[Tensor], Tensor],
    Callable[[Tensor], Tensor],
    Callable[[Tensor], Tensor],
]:
    """Return products with ``T0``, ``T0.T``, ``A``, and ``A.T``.

    Here ``T0=P K_tilde B`` and ``A=D_delta S T0``. All vectors have
    ``H*d`` coordinates.
    """

    horizon = len(approximate_jvps)
    if horizon < 1 or len(approximate_vjps) != horizon:
        raise ValueError("approximate JVP/VJP sequences must have equal length")
    _validate_delta_products(delta_products, delta_transposes, horizon)
    dimension = int(parameter_dimension)
    eta = float(learning_rate)
    if dimension < 1:
        raise ValueError("parameter_dimension must be positive")
    if not math.isfinite(eta) or eta <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    t0, t0_transpose = make_parameter_channel_green_products(
        approximate_jvps,
        approximate_vjps,
        dimension,
        -eta,
        eta,
    )

    def unpack(vector: Tensor) -> Tensor:
        if vector.numel() != horizon * dimension:
            raise ValueError("resolvent vector has the wrong dimension")
        return vector.reshape(horizon, dimension)

    def apply_mismatch(vector: Tensor) -> Tensor:
        parameter_response = unpack(t0(vector))
        zero = torch.zeros_like(parameter_response[:1])
        shifted = torch.cat((zero, parameter_response[:-1]), dim=0)
        return torch.stack(
            [delta_products[step](shifted[step]) for step in range(horizon)]
        ).reshape(-1)

    def transpose_mismatch(vector: Tensor) -> Tensor:
        rows = unpack(vector)
        delta_back = torch.stack(
            [delta_transposes[step](rows[step]) for step in range(horizon)]
        )
        zero = torch.zeros_like(delta_back[:1])
        shift_back = torch.cat((delta_back[1:], zero), dim=0)
        return t0_transpose(shift_back.reshape(-1))

    return t0, t0_transpose, apply_mismatch, transpose_mismatch


def make_batched_causal_structured_resolvent_products(
    approximate_jvps: Sequence[Callable[[Tensor], Tensor]],
    approximate_vjps: Sequence[Callable[[Tensor], Tensor]],
    delta_products: Sequence[Callable[[Tensor], Tensor]],
    delta_transposes: Sequence[Callable[[Tensor], Tensor]],
    parameter_dimension: int,
    learning_rate: float,
) -> tuple[
    Callable[[Tensor], Tensor],
    Callable[[Tensor], Tensor],
    Callable[[Tensor], Tensor],
    Callable[[Tensor], Tensor],
]:
    """Row-batched products with ``T0``, ``T0.T``, ``A``, and ``A.T``."""

    horizon = len(approximate_jvps)
    if horizon < 1 or len(approximate_vjps) != horizon:
        raise ValueError("approximate JVP/VJP sequences must have equal length")
    _validate_delta_products(delta_products, delta_transposes, horizon)
    dimension = int(parameter_dimension)
    eta = float(learning_rate)
    if dimension < 1:
        raise ValueError("parameter_dimension must be positive")
    if not math.isfinite(eta) or eta <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    t0, t0_transpose = make_batched_parameter_channel_green_products(
        approximate_jvps,
        approximate_vjps,
        dimension,
        -eta,
        eta,
    )

    def unpack(rows: Tensor) -> Tensor:
        if rows.ndim != 2 or rows.shape[1] != horizon * dimension:
            raise ValueError("resolvent probe block has the wrong shape")
        return rows.reshape(rows.shape[0], horizon, dimension)

    def apply_mismatch(rows: Tensor) -> Tensor:
        parameter_response = unpack(t0(rows))
        zero = torch.zeros_like(parameter_response[:, :1])
        shifted = torch.cat((zero, parameter_response[:, :-1]), dim=1)
        return torch.stack(
            [delta_products[step](shifted[:, step]) for step in range(horizon)],
            dim=1,
        ).reshape(rows.shape[0], -1)

    def transpose_mismatch(rows: Tensor) -> Tensor:
        values = unpack(rows)
        delta_back = torch.stack(
            [delta_transposes[step](values[:, step]) for step in range(horizon)],
            dim=1,
        )
        zero = torch.zeros_like(delta_back[:, :1])
        shift_back = torch.cat((delta_back[:, 1:], zero), dim=1)
        return t0_transpose(shift_back.reshape(rows.shape[0], -1))

    return t0, t0_transpose, apply_mismatch, transpose_mismatch


def truncated_neumann_response(
    forcing: Tensor,
    *,
    apply_approximate_green: Callable[[Tensor], Tensor],
    apply_mismatch: Callable[[Tensor], Tensor],
    maximum_power: int,
) -> Tensor:
    """Compute ``T0 * sum(A**n forcing, n=0,...,maximum_power)``."""

    maximum_power = int(maximum_power)
    if maximum_power < 0:
        raise ValueError("maximum_power must be nonnegative")
    term = forcing
    accumulated = forcing.clone()
    for _ in range(maximum_power):
        term = apply_mismatch(term)
        accumulated = accumulated + term
    return apply_approximate_green(accumulated)
