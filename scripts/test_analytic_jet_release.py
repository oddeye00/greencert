#!/usr/bin/env python3
"""Tests for deterministic analytic-jet release."""
from __future__ import annotations

import math

from analytic_jet_release import (
    analytic_jet_release,
    cross_entropy_objective_third_bound,
    logit_margin_radius,
    scaled_momentum_jacobian_drift,
)


def main() -> None:
    first, second, third, eta = 3.0, 5.0, 7.0, 0.01
    expected_third = 2.0 * first**3 + 1.5 * first * second + math.sqrt(2.0) * third
    assert math.isclose(
        cross_entropy_objective_third_bound(
            first=first, second=second, third=third
        ),
        expected_third,
        rel_tol=1.0e-15,
    )
    assert math.isclose(
        scaled_momentum_jacobian_drift(
            first=first, second=second, third=third, learning_rate=eta
        ),
        math.sqrt(2.0) * eta * expected_third,
        rel_tol=1.0e-15,
    )
    assert math.isclose(
        logit_margin_radius(first=4.0, state_radius=0.25),
        math.sqrt(2.0),
        rel_tol=1.0e-15,
    )

    released = analytic_jet_release(
        kappa=2.0,
        corrected_defect_response_bound=1.0e-4,
        correction_max_state_norm=1.0e-3,
        domain_radius=0.1,
        learning_rate=1.0e-3,
        transition_jets=[(1.0, 1.0, 1.0), (2.0, 1.5, 1.25)],
        output_first_bounds=[2.0, 3.0],
    )
    assert released.closure.closure_passed
    assert released.maximum_optimizer_jacobian_drift > 0.0
    assert released.maximum_output_first_derivative == 3.0
    assert released.maximum_margin_radius is not None
    assert math.isclose(
        released.maximum_margin_radius,
        math.sqrt(2.0)
        * released.maximum_output_first_derivative
        * released.state_radius_about_original_reference,
        rel_tol=1.0e-15,
    )

    for kwargs in (
        {"first": -1.0, "second": 0.0, "third": 0.0},
        {"first": 0.0, "state_radius": float("nan")},
    ):
        try:
            if "second" in kwargs:
                cross_entropy_objective_third_bound(**kwargs)
            else:
                logit_margin_radius(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid analytic-jet input was accepted")

    print("analytic jet release tests passed")


if __name__ == "__main__":
    main()
