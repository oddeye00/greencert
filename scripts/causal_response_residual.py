#!/usr/bin/env python3
"""Local residual accounting for a causal variational response.

The routines implement the exact identity behind the response-residual
interface in ``INEXACT_OPERATOR_GREENCERT_THEOREM.md``.  They are diagnostic
helpers, not outward-arithmetic routines: a proof-producing caller must pass
verified upper bounds for the defect and recurrence residuals.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def causal_response(
    jacobians: Sequence[np.ndarray], injections: Sequence[np.ndarray]
) -> np.ndarray:
    """Return ``x`` from ``x[0]=0`` and ``x[j+1]=J[j]x[j]+u[j]``."""

    if len(jacobians) != len(injections):
        raise ValueError("jacobians and injections must have the same length")
    if len(injections) == 0:
        raise ValueError("at least one injection is required")
    first = np.asarray(injections[0], dtype=float)
    if first.ndim != 1:
        raise ValueError("injections must be vectors")
    dimension = first.size
    path = np.zeros((len(injections) + 1, dimension), dtype=float)
    for index, (jacobian, injection) in enumerate(zip(jacobians, injections)):
        matrix = np.asarray(jacobian, dtype=float)
        vector = np.asarray(injection, dtype=float)
        if matrix.shape != (dimension, dimension):
            raise ValueError(f"jacobians[{index}] has the wrong shape")
        if vector.shape != (dimension,):
            raise ValueError(f"injections[{index}] has the wrong shape")
        path[index + 1] = matrix @ path[index] + vector
    return path


def response_recurrence_residuals(
    jacobians: Sequence[np.ndarray],
    approximate_injections: Sequence[np.ndarray],
    approximate_path: np.ndarray,
) -> np.ndarray:
    """Return ``d[j]=z[j+1]-J[j]z[j]-s_tilde[j]``."""

    if len(jacobians) != len(approximate_injections):
        raise ValueError(
            "jacobians and approximate_injections must have the same length"
        )
    if len(approximate_injections) == 0:
        raise ValueError("at least one injection is required")
    path = np.asarray(approximate_path, dtype=float)
    dimension = np.asarray(approximate_injections[0], dtype=float).size
    if path.shape != (len(jacobians) + 1, dimension):
        raise ValueError("approximate_path has the wrong shape")
    if not np.array_equal(path[0], np.zeros(dimension, dtype=float)):
        raise ValueError("the causal approximate path must start exactly at zero")
    residuals = np.empty((len(jacobians), dimension), dtype=float)
    for index, (jacobian, injection) in enumerate(
        zip(jacobians, approximate_injections)
    ):
        matrix = np.asarray(jacobian, dtype=float)
        vector = np.asarray(injection, dtype=float)
        if matrix.shape != (dimension, dimension):
            raise ValueError(f"jacobians[{index}] has the wrong shape")
        if vector.shape != (dimension,):
            raise ValueError(f"approximate_injections[{index}] has the wrong shape")
        residuals[index] = path[index + 1] - matrix @ path[index] - vector
    return residuals


def residual_corrected_response_error_bound(
    *, green_operator_bound: float, defect_error_bound: float, residual_bound: float
) -> float:
    """Return ``kappa * (sigma + tau)`` after validating scalar inputs."""

    values = {
        "green_operator_bound": green_operator_bound,
        "defect_error_bound": defect_error_bound,
        "residual_bound": residual_bound,
    }
    checked: dict[str, float] = {}
    for name, value in values.items():
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
        checked[name] = number
    return checked["green_operator_bound"] * (
        checked["defect_error_bound"] + checked["residual_bound"]
    )


def residual_corrected_response_norm_bound(
    *,
    approximate_response_norm: float,
    green_operator_bound: float,
    injection_error_bound: float,
    residual_bound: float,
) -> float:
    """Bound an exact causal response from an approximate response.

    If ``y_tilde = K(u_tilde + d)`` exactly as an algebraic identity, then
    ``||K u|| <= ||y_tilde|| + kappa (||u-u_tilde|| + ||d||)``.  A
    proof-producing caller must supply verified upper bounds for all four
    scalar inputs.
    """

    values = {
        "approximate_response_norm": approximate_response_norm,
        "green_operator_bound": green_operator_bound,
        "injection_error_bound": injection_error_bound,
        "residual_bound": residual_bound,
    }
    checked: dict[str, float] = {}
    for name, value in values.items():
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
        checked[name] = number
    return checked["approximate_response_norm"] + checked[
        "green_operator_bound"
    ] * (checked["injection_error_bound"] + checked["residual_bound"])
