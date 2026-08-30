#!/usr/bin/env python3
"""Property checks for the directional quadratic two-response error bound."""

from __future__ import annotations

import math
import random

from directional_two_response import (
    directional_taylor_sequence_error,
    quadratic_two_response_bound,
    scaled_momentum_taylor_sequence_error,
)


def main() -> None:
    rng = random.Random(20260828)
    cases = 0
    for _ in range(4000):
        horizon = rng.randint(1, 20)
        bounds = [10.0 ** rng.uniform(-2.0, 3.0) for _ in range(horizon)]
        norms = [10.0 ** rng.uniform(-8.0, 1.0) for _ in range(horizon)]
        observed = directional_taylor_sequence_error(bounds, norms)
        expected = math.sqrt(
            sum((bound * norm**3 / 6.0) ** 2 for bound, norm in zip(bounds, norms))
        )
        if not math.isclose(observed, expected, rel_tol=2.0e-15, abs_tol=0.0):
            raise AssertionError("sequence Taylor remainder mismatch")

        eta = 10.0 ** rng.uniform(-5.0, 0.0)
        specialized = scaled_momentum_taylor_sequence_error(
            learning_rate=eta,
            objective_fourth_derivative_bounds=bounds,
            parameter_direction_norms=norms,
        )
        expected_specialized = math.sqrt(2.0) * eta * expected
        if not math.isclose(
            specialized, expected_specialized, rel_tol=3.0e-15, abs_tol=0.0
        ):
            raise AssertionError("scaled momentum specialization mismatch")

        response = rng.random()
        kappa = 10.0 ** rng.uniform(-1.0, 4.0)
        errors = [10.0 ** rng.uniform(-14.0, -2.0) for _ in range(3)]
        beta = quadratic_two_response_bound(
            approximate_response_norm=response,
            green_operator_bound=kappa,
            taylor_remainder_bound=errors[0],
            arithmetic_injection_error_bound=errors[1],
            response_recurrence_residual_bound=errors[2],
        )
        exact = response + kappa * sum(errors)
        if not math.isclose(beta, exact, rel_tol=2.0e-15, abs_tol=0.0):
            raise AssertionError("quadratic response beta mismatch")
        cases += 1

    print({"status": "directional two-response tests passed", "cases": cases})


if __name__ == "__main__":
    main()
