#!/usr/bin/env python3
"""Deterministic tests for time-resolved response-centered closure."""

from __future__ import annotations

import math

import numpy as np

from heterogeneous_recenter_closure import heterogeneous_one_shot_closure
from one_shot_recenter_closure import conservative_one_shot_closure


def _green(jacobians: np.ndarray, injections: np.ndarray) -> np.ndarray:
    state = 0.0
    output = []
    for jacobian, injection in zip(jacobians, injections):
        state = jacobian * state + injection
        output.append(state)
    return np.asarray(output, dtype=np.float64)


def _green_norm(jacobians: np.ndarray) -> float:
    horizon = len(jacobians)
    matrix = np.zeros((horizon, horizon), dtype=np.float64)
    for column in range(horizon):
        injection = np.zeros(horizon)
        injection[column] = 1.0
        matrix[:, column] = _green(jacobians, injection)
    return float(np.linalg.norm(matrix, ord=2))


def test_exact_time_varying_quadratics() -> None:
    rng = np.random.default_rng(20260826)
    for _ in range(250):
        horizon = 9
        linear = rng.uniform(0.45, 0.78, size=horizon)
        curvature = rng.uniform(-0.07, 0.07, size=horizon)
        offset = rng.uniform(-1.0e-4, 1.0e-4, size=horizon)
        center = [0.0]
        for j in range(horizon):
            center.append(linear[j] * center[-1] + offset[j])
        center = np.asarray(center)
        jacobians = linear + curvature * center[:-1]
        residual = (
            linear * center[:-1]
            + 0.5 * curvature * center[:-1] ** 2
            + offset
            - center[1:]
        )
        z = _green(jacobians, residual)
        closure = heterogeneous_one_shot_closure(
            kappa=_green_norm(jacobians),
            drift_bounds=np.abs(curvature[1:]),
            response_input_state_norms=np.abs(z[:-1]),
            response_sequence_norm=float(np.linalg.norm(z)),
            response_max_state_norm=float(np.max(np.abs(z))),
            domain_radius=1.0,
        )
        assert closure.closure_passed

        exact = [0.0]
        for j in range(horizon):
            value = exact[-1]
            exact.append(
                linear[j] * value + 0.5 * curvature[j] * value * value + offset[j]
            )
        error = np.asarray(exact[1:]) - center[1:] - z
        assert float(np.linalg.norm(error)) <= closure.remainder_radius + 1.0e-14


def test_dominates_scalar_corollary() -> None:
    rng = np.random.default_rng(7)
    for _ in range(500):
        response = rng.uniform(0.0, 0.02, size=12)
        drift = rng.uniform(0.0, 0.2, size=11)
        z_norm = float(np.linalg.norm(response))
        p = float(np.max(response))
        kappa = float(rng.uniform(1.0, 3.0))
        old = conservative_one_shot_closure(
            kappa=kappa,
            derivative_drift=float(np.max(drift)),
            response_sequence_norm=z_norm,
            response_max_state_norm=p,
            domain_radius=1.0,
        )
        new = heterogeneous_one_shot_closure(
            kappa=kappa,
            drift_bounds=drift,
            response_input_state_norms=response[:-1],
            response_sequence_norm=z_norm,
            response_max_state_norm=p,
            domain_radius=1.0,
        )
        assert (
            new.corrected_defect_response_bound
            <= old.corrected_defect_response_bound + 1.0e-15
        )
        assert (
            new.linearized_remainder_coefficient
            <= old.linearized_remainder_coefficient + 1.0e-15
        )
        if old.closure_passed:
            assert new.closure_passed
            assert new.remainder_radius <= old.remainder_radius + 1.0e-15


def test_anchor_and_linear_edge_cases() -> None:
    # H=1 has no nonlinear transition input after the exact anchor.  A
    # computed-response error alpha remains certifiable even when M=0.
    closure = heterogeneous_one_shot_closure(
        kappa=2.0,
        drift_bounds=[],
        response_input_state_norms=[],
        response_sequence_norm=0.2,
        response_max_state_norm=0.2,
        linear_response_error_bound=0.03,
        domain_radius=0.25,
    )
    assert closure.closure_passed
    assert math.isclose(closure.remainder_radius, 0.03)
    assert math.isclose(closure.total_pointwise_radius, 0.23)


def main() -> None:
    test_exact_time_varying_quadratics()
    test_dominates_scalar_corollary()
    test_anchor_and_linear_edge_cases()
    print(
        "PASS: 250 exact nonlinear trajectories, 500 dominance checks, "
        "and anchor/linear edge cases."
    )


if __name__ == "__main__":
    main()

