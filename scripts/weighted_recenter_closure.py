#!/usr/bin/env python3
"""Diagonally time-scaled response-centered Green closure.

Positive state weights ``w[j]`` and injection weights ``v[j]`` define

    ||x||_Xw = ||(w[j] x[j])_j||_2,
    ||u||_Uv = ||(v[j] u[j])_j||_2.

The causal operator queried by the randomized Gram routine is therefore
``diag(w) K diag(v)^-1``.  The closure below transports the nonlinear remainder
in those same norms and returns a separate pointwise radius at every state.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class WeightedClosure:
    green_operator_bound: float
    weighted_response_norm: float
    weighted_corrected_defect_norm_before_green: float
    weighted_linear_coefficient_before_green: float
    weighted_quadratic_coefficient_before_green: float
    response_error_bound: float
    corrected_defect_response_bound: float
    linearized_remainder_coefficient: float
    quadratic_remainder_coefficient: float
    discriminant: float
    weighted_remainder_radius: float | None
    pointwise_total_radii: list[float] | None
    maximum_pointwise_total_radius: float | None
    algebraic_closure_passed: bool
    domain_passed: bool
    closure_passed: bool
    closure_residual: float | None

    def as_dict(self) -> dict:
        return asdict(self)


def _nonnegative(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _positive_array(name: str, values: Iterable[float]) -> np.ndarray:
    result = np.asarray(list(values), dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError(f"{name} must contain finite positive values")
    return result


def _nonnegative_array(name: str, values: Iterable[float]) -> np.ndarray:
    result = np.asarray(list(values), dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative values")
    return result


def weighted_one_shot_closure(
    *,
    green_operator_bound: float,
    drift_bounds: Iterable[float],
    response_state_norms: Iterable[float],
    state_weights: Iterable[float],
    injection_weights: Iterable[float],
    response_error_bound: float = 0.0,
    domain_radii: Iterable[float] | None = None,
) -> WeightedClosure:
    """Close the response-centered tube in diagonal time-scaled norms.

    For horizon ``H``, ``response_state_norms`` and ``state_weights`` have
    length ``H`` (states 1..H), while ``injection_weights`` has length ``H``
    (transitions 0..H-1).  ``drift_bounds`` has length ``H-1`` and corresponds
    to transition-input states 1..H-1.  The exact anchor contributes no
    nonlinear term.

    If ``E`` is the weighted sequence error, then

        Q = 1/2 ||(v_j M_j p_j^2)_j||_2,
        A = max_j v_j M_j p_j / w_j,
        B = max_j v_j M_j / w_j^2,

    and ``alpha + kappa Q + kappa A E + kappa B E^2/2 <= E`` is sufficient.
    The returned state-j radius is ``p_j + E / w_j``.
    """

    kappa = _nonnegative("green_operator_bound", green_operator_bound)
    alpha = _nonnegative("response_error_bound", response_error_bound)
    drift = _nonnegative_array("drift_bounds", drift_bounds)
    response = _nonnegative_array("response_state_norms", response_state_norms)
    state = _positive_array("state_weights", state_weights)
    injection = _positive_array("injection_weights", injection_weights)
    horizon = response.size
    if state.size != horizon or injection.size != horizon:
        raise ValueError("response, state weights, and injection weights must agree")
    if drift.size != max(horizon - 1, 0):
        raise ValueError("drift_bounds must have length horizon-1")

    if drift.size:
        transition_response = response[:-1]
        transition_state = state[:-1]
        # Injection 0 is anchored and has no nonlinear remainder.  Drift at
        # state j enters injection j, hence the one-position shift.
        transition_injection = injection[1:]
        q_value = 0.5 * float(
            np.linalg.vector_norm(
                transition_injection * drift * transition_response**2
            )
        )
        a_value = float(
            np.max(
                transition_injection
                * drift
                * transition_response
                / transition_state
            )
        )
        b_value = float(
            np.max(transition_injection * drift / transition_state**2)
        )
    else:
        q_value = a_value = b_value = 0.0

    weighted_response = float(np.linalg.vector_norm(state * response))
    forcing = alpha + kappa * q_value
    linear = kappa * a_value
    quadratic = 0.5 * kappa * b_value
    if quadratic == 0.0:
        discriminant = (1.0 - linear) ** 2
        algebraic = linear < 1.0
        remainder = forcing / (1.0 - linear) if algebraic else None
    else:
        discriminant = (1.0 - linear) ** 2 - 4.0 * quadratic * forcing
        algebraic = linear < 1.0 and discriminant >= 0.0
        if algebraic:
            denominator = 1.0 - linear + math.sqrt(discriminant)
            remainder = 0.0 if forcing == 0.0 else 2.0 * forcing / denominator
        else:
            remainder = None

    pointwise = (
        None
        if remainder is None
        else (response + remainder / state).astype(float).tolist()
    )
    maximum = None if pointwise is None else max(pointwise)
    if domain_radii is None:
        domain_passed = pointwise is not None
    else:
        domain = _nonnegative_array("domain_radii", domain_radii)
        if domain.size != horizon:
            raise ValueError("domain_radii must have length horizon")
        domain_passed = pointwise is not None and bool(
            np.all(np.asarray(pointwise) <= domain)
        )
    closure = bool(algebraic and domain_passed)
    residual = None
    if remainder is not None:
        residual = (
            forcing + linear * remainder + quadratic * remainder**2 - remainder
        )
    return WeightedClosure(
        green_operator_bound=kappa,
        weighted_response_norm=weighted_response,
        weighted_corrected_defect_norm_before_green=q_value,
        weighted_linear_coefficient_before_green=a_value,
        weighted_quadratic_coefficient_before_green=b_value,
        response_error_bound=alpha,
        corrected_defect_response_bound=kappa * q_value,
        linearized_remainder_coefficient=linear,
        quadratic_remainder_coefficient=quadratic,
        discriminant=discriminant,
        weighted_remainder_radius=remainder,
        pointwise_total_radii=pointwise,
        maximum_pointwise_total_radius=maximum,
        algebraic_closure_passed=bool(algebraic),
        domain_passed=bool(domain_passed),
        closure_passed=closure,
        closure_residual=residual,
    )


def scale_causal_operator_rows(
    rows: np.ndarray,
    *,
    weights: Iterable[float],
    state_dimension: int,
    inverse: bool,
) -> np.ndarray:
    """Scale flattened row-batched sequences; useful for small theorem tests."""

    array = np.asarray(rows, dtype=np.float64)
    factors = _positive_array("weights", weights)
    expected = factors.size * int(state_dimension)
    if array.ndim != 2 or array.shape[1] != expected:
        raise ValueError("rows have the wrong flattened sequence shape")
    multiplier = (1.0 / factors if inverse else factors)[None, :, None]
    return (array.reshape(array.shape[0], factors.size, state_dimension) * multiplier).reshape(array.shape)

