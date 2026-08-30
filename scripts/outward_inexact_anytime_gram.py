#!/usr/bin/env python3
"""256-bit outward scalar solve for residual-corrected Gram bounds.

The neural/operator residuals must already be verified upper bounds.  This
module treats their binary64 values and the terminal norm as exact dyadics,
uses a conservative lower calibration, and returns an outward binary64 upper
bound for the operator norm.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from flint import arb, ctx

from inexact_anytime_gram import inexact_gram_operator_upper_bound


PRECISION_BITS = 256


def exact_arb(value: float) -> arb:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("exact Arb inputs must be finite")
    numerator, denominator = number.as_integer_ratio()
    return arb(numerator) / arb(denominator)


def _finite_nonnegative(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def folded_normal_calibration_lower(
    *, delta: float, probes: int, precision_bits: int = PRECISION_BITS
) -> float:
    """Return an outward binary64 lower bound on ``Phi^-1((1+d^(1/m))/2)``."""

    delta = float(delta)
    if not 0.0 < delta < 1.0 or probes < 1:
        raise ValueError("delta must lie in (0,1) and probes must be positive")
    old = ctx.prec
    ctx.prec = int(precision_bits)
    try:
        value = arb(2).sqrt() * exact_arb(delta).root(probes).erfinv()
        lower = float(np.nextafter(float(value.lower()), -math.inf))
        if not 0.0 < lower:
            raise RuntimeError("calibration lower bound is not positive")
        return lower
    finally:
        ctx.prec = old


def _arb_polynomial_value(
    x: arb,
    *,
    terminal: float,
    calibration: float,
    residuals: tuple[float, ...],
) -> arb:
    q = len(residuals)
    return (
        exact_arb(calibration) * x**q
        - exact_arb(terminal)
        - sum(
            (exact_arb(value) * x ** (q - 1 - index) for index, value in enumerate(residuals)),
            arb(0),
        )
    )


def outward_polynomial_value(
    gram_eigenvalue: float,
    *,
    terminal_norm: float,
    calibration_lower: float,
    residual_norms: Sequence[float],
) -> arb:
    """Evaluate the inexact-root polynomial in Arb on exact dyadic inputs."""

    lam = _finite_nonnegative("gram_eigenvalue", gram_eigenvalue)
    terminal = _finite_nonnegative("terminal_norm", terminal_norm)
    calibration = float(calibration_lower)
    if not math.isfinite(calibration) or calibration <= 0.0:
        raise ValueError("calibration_lower must be finite and positive")
    residuals = tuple(
        _finite_nonnegative(f"residual_norms[{index}]", value)
        for index, value in enumerate(residual_norms)
    )
    if not residuals:
        raise ValueError("at least one residual bound is required")
    return _arb_polynomial_value(
        exact_arb(lam),
        terminal=terminal,
        calibration=calibration,
        residuals=residuals,
    )


def outward_operator_supersolution_value(
    operator_bound: float,
    *,
    terminal_norm: float,
    calibration_lower: float,
    residual_norms: Sequence[float],
) -> arb:
    """Evaluate the polynomial at the exact square of a binary64 bound."""

    operator = _finite_nonnegative("operator_bound", operator_bound)
    terminal = _finite_nonnegative("terminal_norm", terminal_norm)
    calibration = float(calibration_lower)
    if not math.isfinite(calibration) or calibration <= 0.0:
        raise ValueError("calibration_lower must be finite and positive")
    residuals = tuple(
        _finite_nonnegative(f"residual_norms[{index}]", value)
        for index, value in enumerate(residual_norms)
    )
    if not residuals:
        raise ValueError("at least one residual bound is required")
    return _arb_polynomial_value(
        exact_arb(operator) ** 2,
        terminal=terminal,
        calibration=calibration,
        residuals=residuals,
    )


def outward_inexact_gram_operator_upper_bound(
    *,
    terminal_norm: float,
    calibration_lower: float,
    residual_norms: Sequence[float],
    precision_bits: int = PRECISION_BITS,
    bisection_iterations: int = 320,
) -> float:
    """Return a binary64 operator upper bound backed by an Arb supersolution."""

    terminal = _finite_nonnegative("terminal_norm", terminal_norm)
    residuals = tuple(
        _finite_nonnegative(f"residual_norms[{index}]", value)
        for index, value in enumerate(residual_norms)
    )
    if not residuals:
        raise ValueError("at least one residual bound is required")
    if terminal == 0.0 and not any(residuals):
        return 0.0
    calibration = float(calibration_lower)
    if not math.isfinite(calibration) or calibration <= 0.0:
        raise ValueError("calibration_lower must be finite and positive")

    old = ctx.prec
    ctx.prec = int(precision_bits)
    try:
        approximate = inexact_gram_operator_upper_bound(
            terminal_norm=terminal,
            calibration=calibration,
            residual_norms=residuals,
        )
        upper = float(np.nextafter(approximate * approximate, math.inf))

        def sign_interval(value: float) -> tuple[float, float]:
            enclosure = outward_polynomial_value(
                value,
                terminal_norm=terminal,
                calibration_lower=calibration,
                residual_norms=residuals,
            )
            return float(enclosure.lower()), float(enclosure.upper())

        lower = 0.0
        upper_sign = sign_interval(upper)
        while upper_sign[0] < 0.0:
            upper = float(np.nextafter(max(2.0 * upper, 1.0), math.inf))
            if not math.isfinite(upper):
                raise OverflowError("could not certify an inexact Gram supersolution")
            upper_sign = sign_interval(upper)
        for _ in range(int(bisection_iterations)):
            middle = 0.5 * (lower + upper)
            if middle == lower or middle == upper:
                break
            interval = sign_interval(middle)
            if interval[0] >= 0.0:
                upper = middle
            elif interval[1] <= 0.0:
                lower = middle
            else:
                # The sign cannot be certified at current precision.  The
                # existing upper endpoint remains a proved supersolution.
                break

        operator = float(np.nextafter(math.sqrt(upper), math.inf))
        # Certify that the returned float's exact square is no smaller than the
        # proved Gram-eigenvalue supersolution.
        while float((exact_arb(operator) ** 2 - exact_arb(upper)).lower()) < 0.0:
            operator = float(np.nextafter(operator, math.inf))
        final = outward_operator_supersolution_value(
            operator,
            terminal_norm=terminal,
            calibration_lower=calibration,
            residual_norms=residuals,
        )
        if float(final.lower()) < 0.0:
            raise AssertionError("returned operator bound is not an Arb supersolution")
        return operator
    finally:
        ctx.prec = old
