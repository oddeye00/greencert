#!/usr/bin/env python3
"""Property tests for the certificate-aware precision budget."""
from __future__ import annotations

import math
import numpy as np

from precision_budget_controller import (
    OperatorCapBudget,
    equal_residual_contribution_budget,
)


def randomized_controller_tests() -> int:
    rng = np.random.default_rng(20260827)
    cases = 0
    for _ in range(1200):
        dimension = int(rng.integers(2, 10))
        probes = int(rng.integers(2, 9))
        power = int(rng.integers(1, 6))
        operator = rng.normal(size=(dimension, dimension)) / math.sqrt(dimension)
        gram = operator.T @ operator
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        spectral_square = float(eigenvalues[-1])
        top = eigenvectors[:, -1]
        block = rng.normal(size=(dimension, probes))
        projection = float(np.max(np.abs(top @ block)))
        calibration = max(0.2 * projection, 1.0e-8)

        iterates = block.copy()
        residual_uppers = []
        for _step in range(power):
            exact_next = gram @ iterates
            perturbation = 1.0e-8 * rng.normal(size=exact_next.shape)
            approximate_next = exact_next + perturbation
            residual_uppers.append(float(np.max(np.linalg.norm(perturbation, axis=0))))
            iterates = approximate_next
        terminal = float(np.max(np.linalg.norm(iterates, axis=0)))

        target = max(1.05 * spectral_square, 1.0e-5)
        while True:
            contributions = [
                residual_uppers[index] * target ** (power - 1 - index)
                for index in range(power)
            ]
            required = 1.000001 * (terminal + sum(contributions))
            if calibration * target**power >= required:
                break
            target *= 1.4
        available = calibration * target**power
        spare = available - terminal - sum(contributions)
        budget = OperatorCapBudget(
            target_squared_norm=target,
            calibration_lower=calibration,
            terminal_allowance=terminal + 0.25 * spare,
            residual_contribution_allowances=tuple(
                contribution + 0.5 * spare / power for contribution in contributions
            ),
        )
        result = budget.check(
            terminal_norm_upper=terminal,
            residual_norm_uppers=residual_uppers,
        )
        if not result["passed"]:
            raise AssertionError("valid randomized controller budget rejected")
        if spectral_square > target * (1.0 + 1.0e-11):
            raise AssertionError("controller passed below the true spectral square")

        failed_residuals = list(residual_uppers)
        failed_residuals[-1] = 1.01 * budget.residual_norm_allowances()[-1]
        if budget.check(
            terminal_norm_upper=terminal,
            residual_norm_uppers=failed_residuals,
        )["passed"]:
            raise AssertionError("controller accepted a residual above its local budget")
        cases += 2
    return cases


def deterministic_edge_tests() -> int:
    budget = equal_residual_contribution_budget(
        target_squared_norm=4.0,
        calibration_lower=0.5,
        terminal_allowance=1.0,
        power=2,
        spend_fraction=0.8,
    )
    allowances = budget.residual_norm_allowances()
    if not budget.check(
        terminal_norm_upper=1.0,
        residual_norm_uppers=allowances,
    )["passed"]:
        raise AssertionError("boundary-equality budget should pass")
    if budget.check(
        terminal_norm_upper=math.nextafter(1.0, math.inf),
        residual_norm_uppers=allowances,
    )["passed"]:
        raise AssertionError("terminal overflow should fail")
    for kwargs in (
        {"power": 0},
        {"spend_fraction": 1.1},
        {"terminal_allowance": 9.0},
    ):
        base = {
            "target_squared_norm": 4.0,
            "calibration_lower": 0.5,
            "terminal_allowance": 1.0,
            "power": 2,
            "spend_fraction": 0.8,
        }
        base.update(kwargs)
        try:
            equal_residual_contribution_budget(**base)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid controller input accepted: {kwargs}")
    return 5


def main() -> None:
    cases = deterministic_edge_tests() + randomized_controller_tests()
    print({"status": "precision budget controller tests passed", "cases": cases})


if __name__ == "__main__":
    main()
