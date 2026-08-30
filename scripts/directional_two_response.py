#!/usr/bin/env python3
"""Scalar error accounting for cancellation-safe quadratic two responses."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def directional_taylor_sequence_error(
    third_map_derivative_bounds: Sequence[float],
    state_direction_norms: Sequence[float],
) -> float:
    """Return ``||(L_j/6)||z_j||^3||_2`` for the Taylor remainder."""

    if len(third_map_derivative_bounds) != len(state_direction_norms):
        raise ValueError("bound and direction sequences must have equal length")
    squared = 0.0
    for index, (bound, norm) in enumerate(
        zip(third_map_derivative_bounds, state_direction_norms)
    ):
        bound = _nonnegative(f"third_map_derivative_bounds[{index}]", bound)
        norm = _nonnegative(f"state_direction_norms[{index}]", norm)
        term = bound * norm**3 / 6.0
        squared += term * term
    return math.sqrt(squared)


def scaled_momentum_taylor_sequence_error(
    *,
    learning_rate: float,
    objective_fourth_derivative_bounds: Sequence[float],
    parameter_direction_norms: Sequence[float],
) -> float:
    """Specialize the sequence remainder to ``(theta, eta*velocity)``."""

    eta = _nonnegative("learning_rate", learning_rate)
    map_bounds = [
        math.sqrt(2.0) * eta * _nonnegative(f"objective_bound[{i}]", bound)
        for i, bound in enumerate(objective_fourth_derivative_bounds)
    ]
    return directional_taylor_sequence_error(map_bounds, parameter_direction_norms)


def quadratic_two_response_bound(
    *,
    approximate_response_norm: float,
    green_operator_bound: float,
    taylor_remainder_bound: float,
    arithmetic_injection_error_bound: float,
    response_recurrence_residual_bound: float,
) -> float:
    """Return the theorem-valid ``beta`` for the quadratic second response."""

    response = _nonnegative("approximate_response_norm", approximate_response_norm)
    kappa = _nonnegative("green_operator_bound", green_operator_bound)
    taylor = _nonnegative("taylor_remainder_bound", taylor_remainder_bound)
    arithmetic = _nonnegative(
        "arithmetic_injection_error_bound", arithmetic_injection_error_bound
    )
    recurrence = _nonnegative(
        "response_recurrence_residual_bound", response_recurrence_residual_bound
    )
    return response + kappa * (taylor + arithmetic + recurrence)
