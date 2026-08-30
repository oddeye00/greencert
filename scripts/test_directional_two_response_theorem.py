#!/usr/bin/env python3
"""Random finite-dimensional checks of the directional two-response theorem."""

from __future__ import annotations

import math
import random

import numpy as np

from causal_response_residual import causal_response
from directional_two_response import (
    directional_taylor_sequence_error,
    quadratic_two_response_bound,
)


def sequence_norm(values: np.ndarray) -> float:
    return float(np.linalg.norm(values.reshape(-1)))


def main() -> None:
    rng = np.random.default_rng(20260828)
    scalar_rng = random.Random(20260828)
    cases = 0
    for _ in range(2000):
        horizon = int(rng.integers(1, 9))
        dimension = int(rng.integers(1, 7))
        jacobians = []
        quadratic = []
        cubic = []
        injections = []
        for _step in range(horizon):
            matrix = rng.normal(size=(dimension, dimension))
            matrix *= 0.55 / max(np.linalg.norm(matrix, ord=2), 1.0)
            jacobians.append(matrix)
            quadratic.append(rng.normal(scale=0.08, size=(dimension, dimension, dimension)))
            cubic.append(
                rng.normal(
                    scale=0.02,
                    size=(dimension, dimension, dimension, dimension),
                )
            )
            injections.append(rng.normal(scale=0.04, size=dimension))
        z = causal_response(jacobians, injections)
        q_exact = []
        q_second = []
        third_bounds = []
        for step in range(horizon):
            direction = z[step]
            second = 0.5 * np.einsum(
                "iab,a,b->i", quadratic[step], direction, direction
            )
            third = (1.0 / 6.0) * np.einsum(
                "iabc,a,b,c->i",
                cubic[step],
                direction,
                direction,
                direction,
            )
            q_second.append(second)
            q_exact.append(second + third)
            # Frobenius norm bounds the trilinear operator norm.
            third_bounds.append(float(np.linalg.norm(cubic[step])))
        q_exact_array = np.asarray(q_exact)
        q_second_array = np.asarray(q_second)
        sigma = directional_taylor_sequence_error(
            third_bounds,
            [float(np.linalg.norm(value)) for value in z[:-1]],
        )
        actual_injection_error = sequence_norm(q_exact_array - q_second_array)
        if actual_injection_error > sigma * (1.0 + 2.0e-13) + 1.0e-15:
            raise AssertionError("third-order Taylor sequence bound failed")

        y = causal_response(jacobians, q_second_array)[1:]
        exact_response = causal_response(jacobians, q_exact_array)[1:]
        # Dense construction of K supplies an exact norm upper bound for this test.
        columns = []
        for coordinate in range(horizon * dimension):
            basis = np.zeros((horizon, dimension))
            basis.reshape(-1)[coordinate] = 1.0
            columns.append(causal_response(jacobians, basis)[1:].reshape(-1))
        green = np.stack(columns, axis=1)
        kappa = float(np.linalg.norm(green, ord=2))
        beta = quadratic_two_response_bound(
            approximate_response_norm=sequence_norm(y),
            green_operator_bound=kappa,
            taylor_remainder_bound=sigma,
            arithmetic_injection_error_bound=0.0,
            response_recurrence_residual_bound=0.0,
        )
        if sequence_norm(exact_response) > beta * (1.0 + 3.0e-13) + 1.0e-14:
            raise AssertionError("directional two-response beta failed")

        # Positive synthetic arithmetic and recurrence budgets must add linearly.
        arithmetic = 10.0 ** scalar_rng.uniform(-12.0, -5.0)
        recurrence = 10.0 ** scalar_rng.uniform(-12.0, -5.0)
        enlarged = quadratic_two_response_bound(
            approximate_response_norm=sequence_norm(y),
            green_operator_bound=kappa,
            taylor_remainder_bound=sigma,
            arithmetic_injection_error_bound=arithmetic,
            response_recurrence_residual_bound=recurrence,
        )
        if enlarged + 1.0e-18 < beta:
            raise AssertionError("error budgets must not tighten beta")
        cases += 1

    print(
        {
            "status": "directional two-response theorem tests passed",
            "cases": cases,
        }
    )


if __name__ == "__main__":
    main()
