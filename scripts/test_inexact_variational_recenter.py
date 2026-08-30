#!/usr/bin/env python3
"""Property tests for the inexact variational-sweep defect identity."""
from __future__ import annotations

import numpy as np

from inexact_variational_recenter import (
    inexact_sweep_defect_upper_bounds,
    sweep_recurrence_residuals,
)


def main() -> None:
    generator = np.random.default_rng(2026082702)
    cases = 0
    for dimension in (1, 2, 4, 7):
        for horizon in (1, 2, 5, 9):
            for _ in range(60):
                linear = generator.normal(scale=0.3, size=(horizon, dimension, dimension))
                bias = generator.normal(scale=0.2, size=(horizon, dimension))
                hessians = generator.normal(
                    scale=0.08,
                    size=(horizon, dimension, dimension, dimension),
                )
                hessians = 0.5 * (hessians + hessians.swapaxes(-1, -2))

                reference = generator.normal(
                    scale=0.4, size=(horizon + 1, dimension)
                )
                correction = generator.normal(
                    scale=0.15, size=(horizon + 1, dimension)
                )
                correction[0] = 0.0

                def apply(step: int, value: np.ndarray) -> np.ndarray:
                    quadratic = np.einsum(
                        "kab,a,b->k", hessians[step], value, value
                    )
                    return bias[step] + linear[step] @ value + 0.5 * quadratic

                exact_defects = np.stack(
                    [
                        apply(step, reference[step]) - reference[step + 1]
                        for step in range(horizon)
                    ]
                )
                defect_errors = generator.normal(
                    scale=1.0e-5, size=(horizon, dimension)
                )
                approximate_defects = exact_defects + defect_errors
                jacobians = []
                for step in range(horizon):
                    jacobians.append(
                        linear[step]
                        + np.einsum("kab,b->ka", hessians[step], reference[step])
                    )
                residuals = sweep_recurrence_residuals(
                    jacobians, approximate_defects, correction
                )

                new_defects = np.stack(
                    [
                        apply(step, reference[step] + correction[step])
                        - reference[step + 1]
                        - correction[step + 1]
                        for step in range(horizon)
                    ]
                )
                nonlinear = np.stack(
                    [
                        apply(step, reference[step] + correction[step])
                        - apply(step, reference[step])
                        - jacobians[step] @ correction[step]
                        for step in range(horizon)
                    ]
                )
                identity = nonlinear + (exact_defects - approximate_defects) - residuals
                if not np.allclose(new_defects, identity, rtol=3.0e-11, atol=3.0e-11):
                    raise AssertionError("inexact sweep identity failed")

                # The Frobenius norm of the Hessian tensor dominates its
                # bilinear operator norm, so it is a valid (loose) M_j.
                drift = np.linalg.norm(hessians.reshape(horizon, -1), axis=1)
                upper = inexact_sweep_defect_upper_bounds(
                    correction_norms=np.linalg.norm(correction[:-1], axis=1),
                    jacobian_drift_bounds=drift,
                    defect_error_bounds=np.linalg.norm(defect_errors, axis=1),
                    recurrence_residual_bounds=np.linalg.norm(residuals, axis=1),
                )
                observed = np.linalg.norm(new_defects, axis=1)
                if np.any(observed > upper + 2.0e-11 * np.maximum(upper, 1.0)):
                    raise AssertionError((observed, upper))
                cases += 1

    print({"status": "inexact variational recenter tests passed", "cases": cases})


if __name__ == "__main__":
    main()
