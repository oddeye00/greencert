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


def causal_block_majorant(
    approximate_block_bounds: Tensor,
    mismatch_block_bounds: Sequence[float],
) -> tuple[Tensor, Tensor]:
    """Return the strict mismatch majorant M and exact-Green majorant C.

    approximate_block_bounds[i,k] bounds the norm of block (i,k) of
    T0=P K_tilde B. The mismatch sequence bounds norm(Delta_j).
    """

    bounds = approximate_block_bounds
    if bounds.ndim != 2 or bounds.shape[0] != bounds.shape[1]:
        raise ValueError("approximate_block_bounds must be square")
    horizon = int(bounds.shape[0])
    if horizon < 1 or len(mismatch_block_bounds) != horizon:
        raise ValueError("mismatch bounds must match a positive horizon")
    if not bool(torch.isfinite(bounds).all()) or bool((bounds < 0.0).any()):
        raise ValueError("approximate block bounds must be finite and nonnegative")
    # Row i may depend only on forcing blocks k<=i.
    if bool((torch.triu(bounds, diagonal=1) != 0.0).any()):
        raise ValueError("approximate block bounds must be lower triangular")
    mismatch = []
    for index, value in enumerate(mismatch_block_bounds):
        scalar = float(value)
        if not math.isfinite(scalar) or scalar < 0.0:
            raise ValueError(
                f"mismatch_block_bounds[{index}] must be finite and nonnegative"
            )
        mismatch.append(scalar)

    majorant = torch.zeros_like(bounds)
    for step in range(1, horizon):
        majorant[step] = mismatch[step] * bounds[step - 1]
    inverse_majorant = torch.eye(
        horizon, dtype=bounds.dtype, device=bounds.device
    )
    power = torch.eye(horizon, dtype=bounds.dtype, device=bounds.device)
    for _ in range(1, horizon):
        power = power @ majorant
        inverse_majorant = inverse_majorant + power
    exact_majorant = bounds @ inverse_majorant
    return majorant, exact_majorant


def causal_block_majorant_gain(
    approximate_block_bounds: Tensor,
    mismatch_block_bounds: Sequence[float],
) -> float:
    """Return the spectral norm of N times inverse(I-M)."""

    _, exact_majorant = causal_block_majorant(
        approximate_block_bounds, mismatch_block_bounds
    )
    return float(torch.linalg.matrix_norm(exact_majorant, ord=2))


def causal_profiled_block_majorant_gain(
    approximate_block_bounds: Tensor,
    mismatch_block_bounds: Sequence[float],
    curvature_profile: Sequence[float],
) -> float:
    """Return the profiled nonlinear coefficient norm(C D_(L,0))."""

    _, exact_majorant = causal_block_majorant(
        approximate_block_bounds, mismatch_block_bounds
    )
    horizon = int(exact_majorant.shape[0])
    if len(curvature_profile) != horizon:
        raise ValueError("curvature profile must match the horizon")
    curvature = []
    for index, value in enumerate(curvature_profile):
        scalar = float(value)
        if not math.isfinite(scalar) or scalar < 0.0:
            raise ValueError(
                f"curvature_profile[{index}] must be finite and nonnegative"
            )
        curvature.append(scalar)
    if horizon == 1:
        return 0.0
    injection = torch.zeros(
        horizon,
        horizon - 1,
        dtype=exact_majorant.dtype,
        device=exact_majorant.device,
    )
    for step in range(1, horizon):
        injection[step, step - 1] = curvature[step]
    return float(torch.linalg.matrix_norm(exact_majorant @ injection, ord=2))


def causal_block_majorant_response_bound(
    exact_majorant: Tensor,
    forcing_block_norms: Sequence[float],
) -> float:
    """Bound a specific response by the Euclidean norm of C v."""

    if exact_majorant.ndim != 2 or exact_majorant.shape[0] != exact_majorant.shape[1]:
        raise ValueError("exact_majorant must be square")
    horizon = int(exact_majorant.shape[0])
    if len(forcing_block_norms) != horizon:
        raise ValueError("forcing block norms must match the horizon")
    values = []
    for index, value in enumerate(forcing_block_norms):
        scalar = float(value)
        if not math.isfinite(scalar) or scalar < 0.0:
            raise ValueError(
                f"forcing_block_norms[{index}] must be finite and nonnegative"
            )
        values.append(scalar)
    vector = torch.tensor(
        values, dtype=exact_majorant.dtype, device=exact_majorant.device
    )
    return float(torch.linalg.vector_norm(exact_majorant @ vector))


def causal_directional_affine_bounds(
    known_parameter_response_norms: Sequence[float],
    structured_green_block_majorant: Tensor,
    structured_forcing_error_bounds: Sequence[float],
    *,
    complement_green_block_majorant: Tensor | None = None,
    complement_forcing_error_bounds: Sequence[float] | None = None,
) -> Tensor:
    """Return checkpointwise affine-response bounds.

    This is the blockwise counterpart of the two-response scalar bound.  The
    known signed response is retained checkpoint by checkpoint; only its
    unresolved B- and optional C-channel forcings pass through nonnegative
    Green block majorants.
    """

    structured = structured_green_block_majorant
    if structured.ndim != 2 or structured.shape[0] != structured.shape[1]:
        raise ValueError("structured Green block majorant must be square")
    horizon = int(structured.shape[0])
    if not bool(torch.isfinite(structured).all()) or bool(
        (structured < 0.0).any()
    ):
        raise ValueError(
            "structured Green block majorant must be finite and nonnegative"
        )
    if bool((torch.triu(structured, diagonal=1) != 0.0).any()):
        raise ValueError("structured Green block majorant must be lower triangular")

    def nonnegative_vector(values: Sequence[float], name: str) -> Tensor:
        if len(values) != horizon:
            raise ValueError(f"{name} must match the horizon")
        checked = []
        for index, value in enumerate(values):
            scalar = float(value)
            if not math.isfinite(scalar) or scalar < 0.0:
                raise ValueError(
                    f"{name}[{index}] must be finite and nonnegative"
                )
            checked.append(scalar)
        return torch.tensor(
            checked, dtype=structured.dtype, device=structured.device
        )

    known = nonnegative_vector(
        known_parameter_response_norms, "known_parameter_response_norms"
    )
    structured_error = nonnegative_vector(
        structured_forcing_error_bounds, "structured_forcing_error_bounds"
    )
    output = known + structured @ structured_error

    if (complement_green_block_majorant is None) != (
        complement_forcing_error_bounds is None
    ):
        raise ValueError(
            "complement majorant and forcing errors must be supplied together"
        )
    if complement_green_block_majorant is not None:
        complement = complement_green_block_majorant
        if complement.shape != structured.shape:
            raise ValueError("complement Green majorant must match the horizon")
        if not bool(torch.isfinite(complement).all()) or bool(
            (complement < 0.0).any()
        ):
            raise ValueError(
                "complement Green block majorant must be finite and nonnegative"
            )
        if bool((torch.triu(complement, diagonal=1) != 0.0).any()):
            raise ValueError("complement Green majorant must be lower triangular")
        complement_error = nonnegative_vector(
            complement_forcing_error_bounds,
            "complement_forcing_error_bounds",
        )
        output = output + complement @ complement_error
    return output


def causal_forward_quadratic_envelope(
    affine_parameter_bounds: Sequence[float] | Tensor,
    structured_green_block_majorant: Tensor,
    curvature_profile: Sequence[float],
) -> Tensor:
    """Propagate a triangular checkpointwise nonlinear radius envelope.

    If output row ``i`` bounds the parameter error at time ``i+1``, nonlinear
    forcing block ``k`` depends on the already bounded error at time ``k``.
    The update-zero forcing therefore vanishes exactly.  Causality makes this
    an explicit forward recurrence, with no scalar discriminant or fixed-point
    iteration.
    """

    majorant = structured_green_block_majorant
    if majorant.ndim != 2 or majorant.shape[0] != majorant.shape[1]:
        raise ValueError("structured Green block majorant must be square")
    horizon = int(majorant.shape[0])
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not bool(torch.isfinite(majorant).all()) or bool((majorant < 0.0).any()):
        raise ValueError(
            "structured Green block majorant must be finite and nonnegative"
        )
    if bool((torch.triu(majorant, diagonal=1) != 0.0).any()):
        raise ValueError("structured Green block majorant must be lower triangular")

    if isinstance(affine_parameter_bounds, Tensor):
        affine = affine_parameter_bounds.to(
            dtype=majorant.dtype, device=majorant.device
        )
        if affine.ndim != 1 or affine.numel() != horizon:
            raise ValueError("affine parameter bounds must match the horizon")
        if not bool(torch.isfinite(affine).all()) or bool((affine < 0.0).any()):
            raise ValueError("affine parameter bounds must be finite and nonnegative")
    else:
        if len(affine_parameter_bounds) != horizon:
            raise ValueError("affine parameter bounds must match the horizon")
        values = [float(value) for value in affine_parameter_bounds]
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("affine parameter bounds must be finite and nonnegative")
        affine = torch.tensor(
            values, dtype=majorant.dtype, device=majorant.device
        )

    if len(curvature_profile) != horizon:
        raise ValueError("curvature profile must match the horizon")
    curvature_values = [float(value) for value in curvature_profile]
    if any(
        not math.isfinite(value) or value < 0.0 for value in curvature_values
    ):
        raise ValueError("curvature profile must be finite and nonnegative")
    curvature = torch.tensor(
        curvature_values, dtype=majorant.dtype, device=majorant.device
    )

    radii = torch.empty_like(affine)
    nonlinear_forcing = torch.zeros_like(affine)
    for output_step in range(horizon):
        if output_step > 0:
            nonlinear_forcing[output_step] = (
                0.5
                * curvature[output_step]
                * radii[output_step - 1]
                * radii[output_step - 1]
            )
        radii[output_step] = affine[output_step] + torch.dot(
            majorant[output_step, : output_step + 1],
            nonlinear_forcing[: output_step + 1],
        )
    return radii


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
