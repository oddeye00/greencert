#!/usr/bin/env python3
"""Residual-corrected anytime Gaussian Gram bounds.

The routines here implement the scalar root from the inexact-power theorem.
They do not estimate a residual: callers must supply verified upper bounds on
the exact-real Gram defects of the computed power iterates.
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def _validate_nonnegative_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative, got {value}")
    return value


def gram_root_polynomial(
    gram_eigenvalue: float,
    *,
    terminal_norm: float,
    calibration: float,
    residual_norms: Sequence[float],
) -> float:
    """Evaluate c*l^q - Y - sum_l xi_l*l^(q-1-l)."""

    lam = _validate_nonnegative_finite("gram_eigenvalue", gram_eigenvalue)
    y = _validate_nonnegative_finite("terminal_norm", terminal_norm)
    c = float(calibration)
    if not math.isfinite(c) or c <= 0.0:
        raise ValueError("calibration must be finite and positive")
    xis = tuple(
        _validate_nonnegative_finite(f"residual_norms[{index}]", value)
        for index, value in enumerate(residual_norms)
    )
    if not xis:
        raise ValueError("at least one power residual is required")
    q = len(xis)
    return c * lam**q - y - sum(
        xi * lam ** (q - 1 - index) for index, xi in enumerate(xis)
    )


def inexact_gram_operator_upper_bound(
    *,
    terminal_norm: float,
    calibration: float,
    residual_norms: Sequence[float],
    relative_tolerance: float = 2.0e-15,
    maximum_iterations: int = 256,
) -> float:
    """Return the residual-corrected upper bound on ``||T||``.

    If ``v_{l+1}`` approximates ``A v_l`` and the exact-real residual satisfies
    ``||v_{l+1}-A v_l|| <= residual_norms[l]``, the returned value is the square
    root of the unique positive root (or zero when all data vanish) of

        c lambda^q = Y + sum_l xi_l lambda^(q-1-l).

    The floating-point root finder is for executable replay and sensitivity
    analysis.  A proof-producing implementation should outward-round this
    one-dimensional solve (or supply any outward upper supersolution).
    """

    y = _validate_nonnegative_finite("terminal_norm", terminal_norm)
    c = float(calibration)
    if not math.isfinite(c) or c <= 0.0:
        raise ValueError("calibration must be finite and positive")
    xis = tuple(
        _validate_nonnegative_finite(f"residual_norms[{index}]", value)
        for index, value in enumerate(residual_norms)
    )
    if not xis:
        raise ValueError("at least one power residual is required")
    if y == 0.0 and not any(xis):
        return 0.0
    if not 0.0 < relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must lie in (0,1)")
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")

    def value(lam: float) -> float:
        return gram_root_polynomial(
            lam,
            terminal_norm=y,
            calibration=c,
            residual_norms=xis,
        )

    lower = 0.0
    upper = max(1.0, (y + sum(xis)) / c)
    while value(upper) < 0.0:
        upper *= 2.0
        if not math.isfinite(upper):
            raise OverflowError("could not bracket the inexact Gram root")
    for _ in range(maximum_iterations):
        middle = 0.5 * (lower + upper)
        if value(middle) < 0.0:
            lower = middle
        else:
            upper = middle
        if upper - lower <= relative_tolerance * max(upper, 1.0e-300):
            break
    # Return the upper endpoint so roundoff in the bisection cannot turn this
    # diagnostic implementation into a downward root approximation.
    return math.sqrt(upper)


def q1_relative_terminal_residual_upper(
    *, terminal_norm: float, calibration: float, relative_residual: float
) -> float:
    """Convenience specialization for xi_0 <= relative_residual * Y."""

    y = _validate_nonnegative_finite("terminal_norm", terminal_norm)
    relative = _validate_nonnegative_finite(
        "relative_residual", relative_residual
    )
    return inexact_gram_operator_upper_bound(
        terminal_norm=y,
        calibration=calibration,
        residual_norms=(relative * y,),
    )
