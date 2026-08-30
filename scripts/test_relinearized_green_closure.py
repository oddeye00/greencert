#!/usr/bin/env python3
"""Analytic and randomized tests for corrected-path Green closure."""

from __future__ import annotations

import math
import numpy as np

from relinearized_green_closure import exact_relinearized_closure


def causal_matrix(jacobians: np.ndarray) -> np.ndarray:
    horizon = len(jacobians)
    matrix = np.zeros((horizon, horizon), dtype=np.float64)
    for column in range(horizon):
        state = 0.0
        for step in range(horizon):
            state = jacobians[step] * state + (1.0 if step == column else 0.0)
            matrix[step, column] = state
    return matrix


def randomized_scalar_shadowing_trials() -> tuple[int, int]:
    rng = np.random.default_rng(20260828)
    attempted = 0
    closed = 0
    for _ in range(20_000):
        horizon = int(rng.integers(1, 9))
        linear = rng.uniform(-0.45, 0.45, size=horizon)
        quadratic = rng.uniform(-0.20, 0.20, size=horizon)
        offsets = rng.uniform(-2.0e-3, 2.0e-3, size=horizon)
        center = np.zeros(horizon + 1, dtype=np.float64)
        center[0] = rng.uniform(-0.05, 0.05)
        center[1:] = rng.uniform(-0.05, 0.05, size=horizon)

        def apply(step: int, value: float) -> float:
            return (
                linear[step] * value
                + offsets[step]
                + 0.5 * quadratic[step] * value * value
            )

        defect = np.asarray(
            [apply(j, center[j]) - center[j + 1] for j in range(horizon)]
        )
        old_jacobian = linear + quadratic * center[:-1]
        correction = np.zeros(horizon + 1, dtype=np.float64)
        for j in range(horizon):
            correction[j + 1] = old_jacobian[j] * correction[j] + defect[j]
        corrected = center + correction
        corrected_defect = np.asarray(
            [apply(j, corrected[j]) - corrected[j + 1] for j in range(horizon)]
        )

        # Exact identity: corrected defect = N_c(d) for an exact response d.
        expected = 0.5 * quadratic * correction[:-1] ** 2
        assert np.allclose(corrected_defect, expected, rtol=2e-12, atol=2e-17)

        new_jacobian = linear + quadratic * corrected[:-1]
        green = causal_matrix(new_jacobian)
        response = green @ corrected_defect
        kappa = float(np.linalg.svd(green, compute_uv=False)[0])
        forcing = float(np.linalg.norm(response))
        drift = float(np.max(np.abs(quadratic)))
        correction_max = float(np.max(np.abs(correction)))
        closure = exact_relinearized_closure(
            kappa=kappa,
            derivative_drift=drift,
            corrected_defect_response_bound=forcing,
            correction_max_state_norm=correction_max,
            domain_radius=1.0,
        )
        attempted += 1
        if not closure.closure_passed:
            continue
        closed += 1
        exact = np.zeros(horizon + 1, dtype=np.float64)
        exact[0] = center[0]
        for j in range(horizon):
            exact[j + 1] = apply(j, exact[j])
        error = float(np.linalg.norm(exact[1:] - corrected[1:]))
        assert closure.remainder_radius is not None
        assert error <= closure.remainder_radius * (1.0 + 2e-11) + 2e-15
        assert max(abs(exact - corrected)) <= closure.remainder_radius + 2e-15
    return attempted, closed


def approximate_response_identity_trials() -> int:
    rng = np.random.default_rng(20260829)
    trials = 10_000
    for _ in range(trials):
        c = rng.normal(scale=0.1)
        d = rng.normal(scale=0.1)
        a = rng.normal(scale=0.2)
        m = rng.normal(scale=0.2)
        s = rng.normal(scale=0.1)
        # Define the next reference so that s=G(c)-c_next.
        g = lambda x: a * x + 0.5 * m * x * x
        c_next = g(c) - s
        j = a + m * c
        d_next = rng.normal(scale=0.1)
        response_residual = d_next - j * d - s
        corrected_defect = g(c + d) - (c_next + d_next)
        nonlinear = g(c + d) - g(c) - j * d
        assert math.isclose(
            corrected_defect,
            nonlinear - response_residual,
            rel_tol=2e-12,
            abs_tol=2e-15,
        )
    return trials


def edge_cases() -> None:
    zero = exact_relinearized_closure(
        kappa=3.0,
        derivative_drift=2.0,
        corrected_defect_response_bound=0.0,
        correction_max_state_norm=0.1,
        domain_radius=0.1,
    )
    assert zero.closure_passed and zero.remainder_radius == 0.0
    linear = exact_relinearized_closure(
        kappa=0.0,
        derivative_drift=7.0,
        corrected_defect_response_bound=0.2,
        correction_max_state_norm=0.1,
        domain_radius=0.31,
    )
    assert linear.closure_passed
    assert math.isclose(float(linear.remainder_radius), 0.2)
    failure = exact_relinearized_closure(
        kappa=10.0,
        derivative_drift=10.0,
        corrected_defect_response_bound=0.01,
        correction_max_state_norm=0.0,
        domain_radius=1.0,
    )
    assert not failure.algebraic_closure_passed


def main() -> None:
    edge_cases()
    identity_trials = approximate_response_identity_trials()
    attempted, closed = randomized_scalar_shadowing_trials()
    print(
        {
            "status": "relinearized Green closure tests passed",
            "approximate_response_identity_trials": identity_trials,
            "randomized_shadowing_trials": attempted,
            "closed_shadowing_trials": closed,
        }
    )


if __name__ == "__main__":
    main()
