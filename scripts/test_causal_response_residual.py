#!/usr/bin/env python3
"""Property tests for the causal response-residual identity."""
from __future__ import annotations

import math

import numpy as np

from causal_response_residual import (
    causal_response,
    residual_corrected_response_error_bound,
    residual_corrected_response_norm_bound,
    response_recurrence_residuals,
)


def stacked_norm(path_without_anchor: np.ndarray) -> float:
    return float(np.linalg.norm(path_without_anchor.reshape(-1)))


def explicit_green_matrix(jacobians: list[np.ndarray]) -> np.ndarray:
    """Materialize the stacked causal operator only for theorem tests."""

    horizon = len(jacobians)
    dimension = jacobians[0].shape[0]
    columns = horizon * dimension
    matrix = np.empty((columns, columns), dtype=float)
    for column in range(columns):
        injections = np.zeros((horizon, dimension), dtype=float)
        injections.reshape(-1)[column] = 1.0
        response = causal_response(jacobians, injections)
        matrix[:, column] = response[1:].reshape(-1)
    return matrix


def main() -> None:
    generator = np.random.default_rng(20260827)
    cases = 0
    for dimension in (1, 2, 5, 9):
        for horizon in (1, 2, 4, 8):
            for _ in range(80):
                jacobians = [
                    generator.normal(scale=0.35, size=(dimension, dimension))
                    for _step in range(horizon)
                ]
                exact_injections = generator.normal(size=(horizon, dimension))
                approximate_injections = exact_injections + generator.normal(
                    scale=1.0e-4, size=(horizon, dimension)
                )

                # Use an arbitrary approximate path with an exact zero anchor;
                # it need not have been produced by the stated recurrence.
                approximate_path = generator.normal(
                    scale=0.5, size=(horizon + 1, dimension)
                )
                approximate_path[0] = 0.0
                residuals = response_recurrence_residuals(
                    jacobians, approximate_injections, approximate_path
                )

                exact_response = causal_response(jacobians, exact_injections)
                residual_response = causal_response(
                    jacobians,
                    exact_injections - approximate_injections - residuals,
                )
                observed_error = exact_response - approximate_path
                if not np.allclose(
                    observed_error,
                    residual_response,
                    rtol=2.0e-12,
                    atol=2.0e-12,
                ):
                    raise AssertionError("causal response-residual identity failed")

                green = explicit_green_matrix(jacobians)
                kappa = float(np.linalg.norm(green, ord=2))
                sigma = stacked_norm(exact_injections - approximate_injections)
                tau = stacked_norm(residuals)
                upper = residual_corrected_response_error_bound(
                    green_operator_bound=kappa,
                    defect_error_bound=sigma,
                    residual_bound=tau,
                )
                error_norm = stacked_norm(observed_error[1:])
                if error_norm > upper + 5.0e-11 * max(upper, 1.0):
                    raise AssertionError((error_norm, upper))

                # Independently exercise the second-response interface used
                # for K_H q.  The approximate path is intentionally arbitrary
                # so that its recurrence residual is nonzero.
                exact_second_injections = generator.normal(
                    size=(horizon, dimension)
                )
                approximate_second_injections = (
                    exact_second_injections
                    + generator.normal(scale=2.0e-4, size=(horizon, dimension))
                )
                approximate_second_path = generator.normal(
                    scale=0.4, size=(horizon + 1, dimension)
                )
                approximate_second_path[0] = 0.0
                second_residuals = response_recurrence_residuals(
                    jacobians,
                    approximate_second_injections,
                    approximate_second_path,
                )
                exact_second_response = causal_response(
                    jacobians, exact_second_injections
                )
                second_upper = residual_corrected_response_norm_bound(
                    approximate_response_norm=stacked_norm(
                        approximate_second_path[1:]
                    ),
                    green_operator_bound=kappa,
                    injection_error_bound=stacked_norm(
                        exact_second_injections
                        - approximate_second_injections
                    ),
                    residual_bound=stacked_norm(second_residuals),
                )
                exact_second_norm = stacked_norm(exact_second_response[1:])
                if exact_second_norm > second_upper + 5.0e-11 * max(
                    second_upper, 1.0
                ):
                    raise AssertionError((exact_second_norm, second_upper))
                cases += 1

    zero = residual_corrected_response_error_bound(
        green_operator_bound=3.0,
        defect_error_bound=0.0,
        residual_bound=0.0,
    )
    if not math.isclose(zero, 0.0):
        raise AssertionError("zero residual budget must give zero response error")
    exact_norm = residual_corrected_response_norm_bound(
        approximate_response_norm=2.5,
        green_operator_bound=3.0,
        injection_error_bound=0.0,
        residual_bound=0.0,
    )
    if not math.isclose(exact_norm, 2.5):
        raise AssertionError("zero correction must preserve the supplied norm")
    print({"status": "causal response-residual tests passed", "cases": cases})


if __name__ == "__main__":
    main()
