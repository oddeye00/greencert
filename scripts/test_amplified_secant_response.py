#!/usr/bin/env python3
"""Randomized checks of the amplified-secant discrepancy theorem."""

from __future__ import annotations

import numpy as np

from amplified_secant_response import (
    amplified_error_upper,
    amplified_secant_defect,
    maximum_uniform_amplification,
    optimal_amplification,
    secant_taylor_error_upper,
    sequence_secant_taylor_error_upper,
)


def polynomial_map(
    linear: np.ndarray,
    quadratic: np.ndarray,
    cubic: np.ndarray,
    point: np.ndarray,
) -> np.ndarray:
    return (
        linear @ point
        + 0.5 * np.einsum("iab,a,b->i", quadratic, point, point)
        + np.einsum("iabc,a,b,c->i", cubic, point, point, point) / 6.0
    )


def jacobian_direction(
    linear: np.ndarray,
    quadratic: np.ndarray,
    cubic: np.ndarray,
    point: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    return (
        linear @ direction
        + np.einsum("iab,a,b->i", quadratic, point, direction)
        + 0.5 * np.einsum(
            "iabc,a,b,c->i", cubic, point, point, direction
        )
    )


def main() -> None:
    rng = np.random.default_rng(20260828)
    cases = 4000
    worst_ratio = 0.0
    for _ in range(cases):
        dimension = int(rng.integers(1, 7))
        linear = rng.normal(size=(dimension, dimension))
        raw_q = rng.normal(size=(dimension, dimension, dimension))
        quadratic = 0.5 * (raw_q + raw_q.swapaxes(1, 2))
        raw_c = rng.normal(size=(dimension,) * 4)
        cubic = sum(
            raw_c.transpose((0,) + axes)
            for axes in (
                (1, 2, 3),
                (1, 3, 2),
                (2, 1, 3),
                (2, 3, 1),
                (3, 1, 2),
                (3, 2, 1),
            )
        ) / 6.0
        point = rng.normal(scale=0.3, size=dimension)
        direction = rng.normal(scale=0.05, size=dimension)
        lam = float(2.0 ** rng.integers(-3, 8))
        fn = lambda value: polynomial_map(linear, quadratic, cubic, value)
        jz = jacobian_direction(linear, quadratic, cubic, point, direction)
        exact = fn(point + direction) - fn(point) - jz
        secant = amplified_secant_defect(
            fn, point, direction, jz, amplification=lam
        )
        # For the cubic polynomial D^3G is the constant tensor ``cubic``.
        # Its Frobenius norm is a safe Euclidean operator-norm upper bound.
        derivative_upper = float(np.linalg.norm(cubic))
        upper = secant_taylor_error_upper(
            float(np.linalg.norm(direction)),
            derivative_upper,
            amplification=lam,
        )
        cubic_direction = np.einsum(
            "iabc,a,b,c->i", cubic, direction, direction, direction
        )
        analytic_difference = (1.0 - lam) * cubic_direction / 6.0
        analytic_error = float(np.linalg.norm(analytic_difference))
        if analytic_error > upper * (1.0 + 2.0e-13) + 1.0e-15:
            raise AssertionError((analytic_error, upper, lam))
        implementation_residual = float(
            np.linalg.norm((exact - secant) - analytic_difference)
        )
        if implementation_residual > 5.0e-11 * max(
            1.0, float(np.linalg.norm(exact)), float(np.linalg.norm(secant))
        ):
            raise AssertionError("amplified-secant implementation mismatch")
        if upper > 0.0:
            worst_ratio = max(worst_ratio, analytic_error / upper)
        if lam == 1.0 and float(np.linalg.norm(exact - secant)) > 5.0e-12:
            raise AssertionError("lambda=1 must reproduce the exact defect")

    norms = [0.2, 0.3]
    derivatives = [1.5, 2.0]
    lambdas = [4.0, 8.0]
    direct = sequence_secant_taylor_error_upper(norms, derivatives, lambdas)
    manual = np.sqrt(
        sum(
            secant_taylor_error_upper(n, d, amplification=l) ** 2
            for n, d, l in zip(norms, derivatives, lambdas)
        )
    )
    if not np.isclose(direct, manual, rtol=1.0e-15, atol=0.0):
        raise AssertionError("sequence aggregation mismatch")
    base = [0.01, 0.02]
    limit = maximum_uniform_amplification(
        base, error_budget=0.1, domain_limits=[8.0, 7.0]
    )
    expected = min(1.0 + 0.1 / np.linalg.norm(base), 7.0)
    if not np.isclose(limit, expected, rtol=1.0e-15, atol=0.0):
        raise AssertionError("uniform amplification policy mismatch")
    coefficient = 3.0e-8
    arithmetic = 7.0e-5
    optimum = optimal_amplification(
        coefficient, arithmetic, maximum_amplification=100.0
    )
    exact_optimum = (2.0 * arithmetic / coefficient) ** (1.0 / 3.0)
    if not np.isclose(optimum, exact_optimum, rtol=2.0e-15, atol=0.0):
        raise AssertionError("analytic amplification optimum mismatch")
    center_value = amplified_error_upper(
        coefficient, arithmetic, amplification=optimum
    )
    for factor in (0.8, 1.2):
        neighbor = max(1.0, min(100.0, factor * optimum))
        if center_value > amplified_error_upper(
            coefficient, arithmetic, amplification=neighbor
        ):
            raise AssertionError("reported amplification is not a minimizer")
    print(
        {
            "status": "amplified-secant theorem tests passed",
            "cases": cases,
            "worst_error_to_bound_ratio": worst_ratio,
        }
    )


if __name__ == "__main__":
    main()
