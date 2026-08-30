#!/usr/bin/env python3
"""Generic amplified-secant identities for nonlinear response certification."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import sqrt

import numpy as np


Array = np.ndarray


def amplified_secant_defect(
    map_fn: Callable[[Array], Array],
    point: Array,
    direction: Array,
    jacobian_direction: Array,
    *,
    amplification: float,
) -> Array:
    """Return ``N(lambda*z)/lambda^2`` for ``N(u)=G(c+u)-G(c)-J u``."""

    point = np.asarray(point)
    direction = np.asarray(direction)
    jacobian_direction = np.asarray(jacobian_direction)
    if point.shape != direction.shape or point.shape != jacobian_direction.shape:
        raise ValueError("point, direction, and Jacobian product must have equal shape")
    lam = float(amplification)
    if not np.isfinite(lam) or lam <= 0.0:
        raise ValueError("amplification must be finite and positive")
    return (
        np.asarray(map_fn(point + lam * direction))
        - np.asarray(map_fn(point))
        - lam * jacobian_direction
    ) / (lam * lam)


def secant_taylor_error_upper(
    direction_norm: float,
    third_map_derivative_upper: float,
    *,
    amplification: float,
) -> float:
    """Bound ``||N(z)-N(lambda*z)/lambda^2||``.

    The result follows by writing both defects as weighted integrals of
    ``D^2G`` and integrating the ``D^3G`` drift along the common ray.
    """

    norm = float(direction_norm)
    derivative = float(third_map_derivative_upper)
    lam = float(amplification)
    if norm < 0.0 or derivative < 0.0:
        raise ValueError("norm and derivative bound must be nonnegative")
    if not np.isfinite(lam) or lam <= 0.0:
        raise ValueError("amplification must be finite and positive")
    return abs(lam - 1.0) * derivative * norm**3 / 6.0


def sequence_secant_taylor_error_upper(
    direction_norms: Sequence[float],
    third_map_derivative_uppers: Sequence[float],
    amplifications: Sequence[float],
) -> float:
    """Euclidean injection-space bound for checkpointwise secants."""

    if not (
        len(direction_norms)
        == len(third_map_derivative_uppers)
        == len(amplifications)
    ):
        raise ValueError("checkpointwise sequences must have equal length")
    terms = [
        secant_taylor_error_upper(norm, derivative, amplification=lam)
        for norm, derivative, lam in zip(
            direction_norms, third_map_derivative_uppers, amplifications
        )
    ]
    return sqrt(sum(term * term for term in terms))


def maximum_uniform_amplification(
    base_cubic_terms: Sequence[float],
    *,
    error_budget: float,
    domain_limits: Sequence[float] | None = None,
) -> float:
    """Largest uniform ``lambda >= 1`` allowed by Taylor and domain budgets.

    ``base_cubic_terms[j]`` is ``L_j ||z_j||^3 / 6``.  The secant discrepancy
    is ``(lambda-1) * ||base_cubic_terms||_2``.  ``domain_limits`` may provide
    checkpointwise upper limits on lambda from the certified radial domain.
    """

    terms = [float(value) for value in base_cubic_terms]
    if any(value < 0.0 for value in terms):
        raise ValueError("base cubic terms must be nonnegative")
    budget = float(error_budget)
    if budget < 0.0:
        raise ValueError("error budget must be nonnegative")
    base = sqrt(sum(value * value for value in terms))
    taylor_limit = np.inf if base == 0.0 else 1.0 + budget / base
    if domain_limits is None:
        domain_limit = np.inf
    else:
        limits = [float(value) for value in domain_limits]
        if len(limits) != len(terms):
            raise ValueError("domain limits must match the checkpoint count")
        if any(value < 1.0 for value in limits):
            raise ValueError("every domain limit must be at least one")
        domain_limit = min(limits, default=np.inf)
    return float(min(taylor_limit, domain_limit))


def optimal_amplification(
    cubic_coefficient: float,
    numerator_arithmetic_error_upper: float,
    *,
    maximum_amplification: float = np.inf,
) -> float:
    """Minimize ``A*(lambda-1) + epsilon/lambda^2`` for ``lambda >= 1``.

    ``A`` is ``L ||z||^3 / 6`` and ``epsilon`` bounds absolute arithmetic
    error in the *unscaled* secant numerator.  The returned value is clipped to
    a caller-supplied derivative-domain or protocol maximum.
    """

    coefficient = float(cubic_coefficient)
    arithmetic = float(numerator_arithmetic_error_upper)
    upper = float(maximum_amplification)
    if coefficient < 0.0 or arithmetic < 0.0:
        raise ValueError("error coefficients must be nonnegative")
    if upper < 1.0:
        raise ValueError("maximum amplification must be at least one")
    if coefficient == 0.0:
        candidate = upper
    elif arithmetic == 0.0:
        candidate = 1.0
    else:
        candidate = max(1.0, (2.0 * arithmetic / coefficient) ** (1.0 / 3.0))
    return float(min(candidate, upper))


def amplified_error_upper(
    cubic_coefficient: float,
    numerator_arithmetic_error_upper: float,
    *,
    amplification: float,
) -> float:
    """Combined analytic and numerator-arithmetic secant error."""

    coefficient = float(cubic_coefficient)
    arithmetic = float(numerator_arithmetic_error_upper)
    lam = float(amplification)
    if coefficient < 0.0 or arithmetic < 0.0:
        raise ValueError("error coefficients must be nonnegative")
    if not np.isfinite(lam) or lam <= 0.0:
        raise ValueError("amplification must be finite and positive")
    return coefficient * abs(lam - 1.0) + arithmetic / (lam * lam)
