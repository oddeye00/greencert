#!/usr/bin/env python3
"""Local residual accounting for an inexact variational recentering sweep."""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def sweep_recurrence_residuals(
    jacobians: Sequence[np.ndarray],
    approximate_defects: Sequence[np.ndarray],
    correction: np.ndarray,
) -> np.ndarray:
    """Return ``d[j]=z[j+1]-J[j]z[j]-r_tilde[j]``."""

    if len(jacobians) != len(approximate_defects) or not jacobians:
        raise ValueError("jacobians and defects must have equal positive length")
    path = np.asarray(correction, dtype=float)
    dimension = np.asarray(approximate_defects[0], dtype=float).size
    if path.shape != (len(jacobians) + 1, dimension):
        raise ValueError("correction has the wrong shape")
    if not np.array_equal(path[0], np.zeros(dimension)):
        raise ValueError("an anchor-fixed correction must start exactly at zero")
    result = np.empty((len(jacobians), dimension), dtype=float)
    for index, (jacobian, defect) in enumerate(
        zip(jacobians, approximate_defects)
    ):
        matrix = np.asarray(jacobian, dtype=float)
        vector = np.asarray(defect, dtype=float)
        if matrix.shape != (dimension, dimension) or vector.shape != (dimension,):
            raise ValueError(f"invalid shape at transition {index}")
        result[index] = path[index + 1] - matrix @ path[index] - vector
    return result


def inexact_sweep_defect_upper_bounds(
    *,
    correction_norms: Sequence[float],
    jacobian_drift_bounds: Sequence[float],
    defect_error_bounds: Sequence[float],
    recurrence_residual_bounds: Sequence[float],
) -> np.ndarray:
    """Return ``M_j ||z_j||^2/2 + sigma_j + tau_j`` checkpointwise."""

    rows = tuple(
        zip(
            correction_norms,
            jacobian_drift_bounds,
            defect_error_bounds,
            recurrence_residual_bounds,
        )
    )
    sizes = {
        len(correction_norms),
        len(jacobian_drift_bounds),
        len(defect_error_bounds),
        len(recurrence_residual_bounds),
    }
    if len(sizes) != 1 or not rows:
        raise ValueError("all bound sequences must have equal positive length")
    result = []
    for index, values in enumerate(rows):
        checked = []
        for value in values:
            number = float(value)
            if not math.isfinite(number) or number < 0.0:
                raise ValueError(f"bound at transition {index} is invalid")
            checked.append(number)
        z, drift, defect_error, recurrence_error = checked
        result.append(0.5 * drift * z * z + defect_error + recurrence_error)
    return np.asarray(result, dtype=float)
