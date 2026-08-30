#!/usr/bin/env python3
"""Theorem/property tests for diagonally weighted Green closure."""
from __future__ import annotations

import numpy as np

from causal_response_residual import causal_response
from heterogeneous_recenter_closure import heterogeneous_one_shot_closure
from weighted_recenter_closure import weighted_one_shot_closure


def explicit_green(jacobians: list[np.ndarray]) -> np.ndarray:
    horizon = len(jacobians)
    dimension = jacobians[0].shape[0]
    matrix = np.empty((horizon * dimension, horizon * dimension))
    for column in range(horizon * dimension):
        injections = np.zeros((horizon, dimension))
        injections.reshape(-1)[column] = 1.0
        matrix[:, column] = causal_response(jacobians, injections)[1:].reshape(-1)
    return matrix


def block_diagonal(weights: np.ndarray, dimension: int) -> np.ndarray:
    return np.kron(np.diag(weights), np.eye(dimension))


def test_unweighted_reduces_to_time_resolved() -> int:
    generator = np.random.default_rng(20260827)
    cases = 0
    for horizon in (1, 2, 5, 11):
        for _ in range(250):
            response = generator.uniform(0.0, 0.02, size=horizon)
            drift = generator.uniform(0.0, 0.25, size=max(horizon - 1, 0))
            kappa = float(generator.uniform(0.5, 3.0))
            weighted = weighted_one_shot_closure(
                green_operator_bound=kappa,
                drift_bounds=drift,
                response_state_norms=response,
                state_weights=np.ones(horizon),
                injection_weights=np.ones(horizon),
                domain_radii=np.ones(horizon),
            )
            old = heterogeneous_one_shot_closure(
                kappa=kappa,
                drift_bounds=drift,
                response_input_state_norms=response[:-1],
                response_sequence_norm=float(np.linalg.norm(response)),
                response_max_state_norm=float(np.max(response)),
                domain_radius=1.0,
            )
            fields = (
                (weighted.corrected_defect_response_bound, old.corrected_defect_response_bound),
                (weighted.linearized_remainder_coefficient, old.linearized_remainder_coefficient),
                (weighted.quadratic_remainder_coefficient, old.quadratic_remainder_coefficient),
            )
            for left, right in fields:
                if not np.isclose(left, right, rtol=2.0e-13, atol=2.0e-15):
                    raise AssertionError((left, right))
            if weighted.closure_passed != old.closure_passed:
                raise AssertionError("unweighted closure disposition changed")
            if weighted.closure_passed and not np.isclose(
                weighted.weighted_remainder_radius,
                old.remainder_radius,
                rtol=2.0e-13,
                atol=2.0e-15,
            ):
                raise AssertionError("unweighted radius changed")
            cases += 1
    return cases


def test_weighted_nonlinear_trajectories() -> int:
    generator = np.random.default_rng(8272026)
    cases = 0
    for dimension in (1, 2, 4):
        for horizon in (2, 4, 7):
            for _ in range(100):
                jacobians = [
                    generator.normal(scale=0.12, size=(dimension, dimension))
                    for _step in range(horizon)
                ]
                # N_j(x)=M_j||x||^2 e_1/2 has Hessian bilinear form
                # M_j <u,v> e_1 and therefore exact Jacobian-Lipschitz
                # constant M_j.
                curvatures = generator.uniform(0.0, 0.04, size=horizon)
                defects = generator.normal(scale=1.0e-4, size=(horizon, dimension))
                z = causal_response(jacobians, defects)[1:]
                response_norms = np.linalg.norm(z, axis=1)
                state_weights = np.exp(generator.uniform(-0.8, 0.8, size=horizon))
                injection_weights = np.exp(generator.uniform(-0.8, 0.8, size=horizon))
                green = explicit_green(jacobians)
                weighted_green = (
                    block_diagonal(state_weights, dimension)
                    @ green
                    @ np.linalg.inv(block_diagonal(injection_weights, dimension))
                )
                kappa = float(np.linalg.norm(weighted_green, ord=2))
                closure = weighted_one_shot_closure(
                    green_operator_bound=kappa,
                    drift_bounds=curvatures[1:],
                    response_state_norms=response_norms,
                    state_weights=state_weights,
                    injection_weights=injection_weights,
                    domain_radii=np.full(horizon, 1.0),
                )
                if not closure.closure_passed:
                    continue

                exact = np.zeros((horizon + 1, dimension))
                for step in range(horizon):
                    value = exact[step]
                    radial = np.zeros(dimension)
                    radial[0] = 0.5 * curvatures[step] * float(value @ value)
                    exact[step + 1] = jacobians[step] @ value + defects[step] + radial
                error = exact[1:] - z
                limits = np.asarray(closure.pointwise_total_radii) - response_norms
                observed = np.linalg.norm(error, axis=1)
                if np.any(observed > limits + 2.0e-12):
                    raise AssertionError((observed, limits))
                cases += 1
    return cases


def test_global_scaling_invariance() -> int:
    response = np.array([0.004, 0.008, 0.003, 0.002])
    drift = np.array([0.2, 0.05, 0.1])
    state = np.array([0.8, 1.3, 0.7, 1.1])
    injection = np.array([1.2, 0.9, 1.4, 0.6])
    kappa = 2.4
    base = weighted_one_shot_closure(
        green_operator_bound=kappa,
        drift_bounds=drift,
        response_state_norms=response,
        state_weights=state,
        injection_weights=injection,
        domain_radii=np.ones(4),
    )
    # Multiplying both norms by c leaves K's induced norm unchanged, scales E
    # by c, and leaves every physical pointwise radius invariant.
    c = 7.0
    scaled = weighted_one_shot_closure(
        green_operator_bound=kappa,
        drift_bounds=drift,
        response_state_norms=response,
        state_weights=c * state,
        injection_weights=c * injection,
        domain_radii=np.ones(4),
    )
    if not np.allclose(
        base.pointwise_total_radii,
        scaled.pointwise_total_radii,
        rtol=2.0e-13,
        atol=2.0e-15,
    ):
        raise AssertionError("global norm scaling changed physical radii")
    return 1


def main() -> None:
    unweighted = test_unweighted_reduces_to_time_resolved()
    nonlinear = test_weighted_nonlinear_trajectories()
    invariant = test_global_scaling_invariance()
    print(
        {
            "status": "weighted response-centered closure tests passed",
            "unweighted_equivalence_cases": unweighted,
            "certified_nonlinear_cases": nonlinear,
            "scaling_invariance_cases": invariant,
        }
    )


if __name__ == "__main__":
    main()
