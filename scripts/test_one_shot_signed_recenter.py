#!/usr/bin/env python3
"""Deterministic tests for the one-shot recentered theorem and stored audit."""

from __future__ import annotations

import math

import numpy as np

from audit_one_shot_signed_recenter import build_audit
from one_shot_recenter_closure import (
    conservative_one_shot_closure,
    exact_one_shot_closure,
)


def _causal_green(jacobians: np.ndarray, injections: np.ndarray) -> np.ndarray:
    state = 0.0
    output = []
    for jacobian, injection in zip(jacobians, injections):
        state = jacobian * state + injection
        output.append(state)
    return np.asarray(output, dtype=np.float64)


def _explicit_green_norm(jacobians: np.ndarray) -> float:
    horizon = len(jacobians)
    matrix = np.zeros((horizon, horizon), dtype=np.float64)
    for column in range(horizon):
        injection = np.zeros(horizon, dtype=np.float64)
        injection[column] = 1.0
        matrix[:, column] = _causal_green(jacobians, injection)
    return float(np.linalg.norm(matrix, ord=2))


def test_scalar_quadratic_instances() -> None:
    """Check certified radii against exact nonlinear scalar trajectories."""

    rng = np.random.default_rng(20260825)
    passing = 0
    for _ in range(200):
        horizon = 8
        linear = float(rng.uniform(0.55, 0.82))
        curvature = float(rng.uniform(-0.08, 0.08))
        offset = float(rng.uniform(-2.0e-4, 2.0e-4))

        # The reference is the affine clock anchored at zero.  G is quadratic,
        # so |G'(x+u)-G'(x)| = |curvature| |u| exactly.
        center = [0.0]
        for _step in range(horizon):
            center.append(linear * center[-1] + offset)
        center = np.asarray(center)
        jacobians = linear + curvature * center[:-1]
        residual = (
            linear * center[:-1]
            + 0.5 * curvature * center[:-1] ** 2
            + offset
            - center[1:]
        )
        z = _causal_green(jacobians, residual)
        z_with_anchor = np.concatenate(([0.0], z))
        q = 0.5 * curvature * z_with_anchor[:-1] ** 2
        kzq = _causal_green(jacobians, q)
        kappa = _explicit_green_norm(jacobians)
        sequence_norm = float(np.linalg.norm(z))
        max_state_norm = float(np.max(np.abs(z)))
        y = float(np.linalg.norm(kzq))
        domain = 1.0
        closure = exact_one_shot_closure(
            kappa=kappa,
            derivative_drift=abs(curvature),
            response_sequence_norm=sequence_norm,
            response_max_state_norm=max_state_norm,
            corrected_defect_response_bound=y,
            domain_radius=domain,
        )
        if not closure.closure_passed:
            continue
        passing += 1

        exact = [0.0]
        for _step in range(horizon):
            value = exact[-1]
            exact.append(
                linear * value + 0.5 * curvature * value * value + offset
            )
        exact = np.asarray(exact)
        h = exact[1:] - center[1:]
        e = h - z
        assert float(np.linalg.norm(e)) <= closure.remainder_radius + 1.0e-14
        assert float(np.max(np.abs(h))) <= closure.total_pointwise_radius + 1.0e-14
        assert closure.closure_residual <= 1.0e-14
    assert passing == 200


def test_computed_response_error() -> None:
    """Exercise the explicit ||Ks-z_tilde|| term in the theorem."""

    horizon = 7
    linear = 0.7
    curvature = 0.06
    offset = 1.0e-4
    center = [0.0]
    for _ in range(horizon):
        center.append(linear * center[-1] + offset)
    center = np.asarray(center)
    jacobians = linear + curvature * center[:-1]
    residual = (
        linear * center[:-1]
        + 0.5 * curvature * center[:-1] ** 2
        + offset
        - center[1:]
    )
    z = _causal_green(jacobians, residual)
    perturbation = np.linspace(-1.0, 1.0, horizon) * 1.0e-8
    z_tilde = z + perturbation
    p = float(np.max(np.abs(z_tilde)))
    z_norm = float(np.linalg.norm(z_tilde))
    kappa = _explicit_green_norm(jacobians)
    alpha = float(np.linalg.norm(perturbation))
    closure = conservative_one_shot_closure(
        kappa=kappa,
        derivative_drift=abs(curvature),
        response_sequence_norm=z_norm,
        response_max_state_norm=p,
        linear_response_error_bound=alpha,
        domain_radius=1.0,
    )
    assert closure.closure_passed
    exact = [0.0]
    for _ in range(horizon):
        value = exact[-1]
        exact.append(linear * value + 0.5 * curvature * value * value + offset)
    error = np.asarray(exact[1:]) - center[1:] - z_tilde
    assert float(np.linalg.norm(error)) <= closure.remainder_radius + 1.0e-14


def test_edge_cases() -> None:
    zero = exact_one_shot_closure(
        kappa=1.0,
        derivative_drift=0.0,
        response_sequence_norm=2.0,
        response_max_state_norm=1.0,
        corrected_defect_response_bound=0.0,
        domain_radius=1.0,
    )
    assert zero.closure_passed
    assert zero.remainder_radius == 0.0

    failed = exact_one_shot_closure(
        kappa=1.0,
        derivative_drift=1.0,
        response_sequence_norm=2.0,
        response_max_state_norm=1.0,
        corrected_defect_response_bound=1.0,
        domain_radius=10.0,
    )
    assert not failed.closure_passed

    conservative = conservative_one_shot_closure(
        kappa=2.0,
        derivative_drift=0.1,
        response_sequence_norm=0.2,
        response_max_state_norm=0.1,
        domain_radius=0.4,
    )
    expected_y = 0.5 * 2.0 * 0.1 * 0.1 * 0.2
    assert math.isclose(conservative.corrected_defect_response_bound, expected_y)
    assert conservative.closure_passed


def test_stored_transformer_audit() -> None:
    audit = build_audit()
    assert audit["certificate_records"] == 23
    assert audit["green_evaluable_records"] == 18
    assert audit["old_issued"] == 9
    assert audit["new_issued"] == 13
    assert audit["new_covered"] == 13
    assert audit["converted_old_abstentions"] == 4
    assert audit["distinct_new_issuing_seeds"] == 9
    assert audit["observed_state_tube_violations"] == 0
    assert audit["maximum_new_to_old_radius_ratio"] < 0.15
    assert audit["response_lower_bound_early_abstentions"] == 4
    assert audit["early_abstentions_preserved_without_random_green_query"]
    converted = {
        (row["seed"], row["threshold"], row["anchor"])
        for row in audit["converted_candidates"]
    }
    assert converted == {
        (342, 0.70, 1480),
        (348, 0.90, 3240),
        (352, 0.80, 5320),
        (354, 0.70, 2120),
    }


def main() -> None:
    test_scalar_quadratic_instances()
    test_computed_response_error()
    test_edge_cases()
    test_stored_transformer_audit()
    print(
        "PASS: theorem algebra, 200 exact nonlinear trajectories, edge cases, "
        "and all stored Transformer records agree."
    )


if __name__ == "__main__":
    main()
