#!/usr/bin/env python3
"""Time-resolved one-shot response-centered closure.

The scalar GREENCERT closure replaces every checkpoint drift bound by one
maximum ``M``.  This module retains the aligned pairs

    (M_j, ||z_j||),    j = 1, ..., H - 1,

where ``z = K s`` is the signed response.  The anchor term is absent because
``z_0 = e_0 = 0``; the terminal response ``z_H`` affects the deployed state
radius but is not an input to a transition remainder.

No model evaluation or randomized query is performed here.  The calculation
is a deterministic post-processing of drift bounds and the signed response
already required by the response-centered certificate.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class HeterogeneousClosure:
    """Result of a time-resolved quadratic closure calculation."""

    kappa: float
    response_sequence_norm: float
    response_max_state_norm: float
    maximum_derivative_drift: float
    aligned_linear_coefficient_before_green: float
    corrected_defect_norm_before_green: float
    linear_response_error_bound: float
    corrected_defect_response_bound: float
    linearized_remainder_coefficient: float
    quadratic_remainder_coefficient: float
    discriminant: float
    remainder_radius: float | None
    total_pointwise_radius: float | None
    domain_radius: float | None
    algebraic_closure_passed: bool
    domain_passed: bool
    closure_passed: bool
    closure_residual: float | None

    def as_dict(self) -> dict:
        return asdict(self)


def _finite_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative, got {value}")
    return value


def _array(name: str, values: Iterable[float]) -> np.ndarray:
    result = np.asarray(list(values), dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative values")
    return result


def heterogeneous_one_shot_closure(
    *,
    kappa: float,
    drift_bounds: Iterable[float],
    response_input_state_norms: Iterable[float],
    response_sequence_norm: float,
    response_max_state_norm: float,
    linear_response_error_bound: float = 0.0,
    domain_radius: float | None = None,
) -> HeterogeneousClosure:
    """Solve the time-resolved response-centered closure.

    ``drift_bounds[j]`` bounds the Jacobian drift at the same transition input
    as ``response_input_state_norms[j]``.  For an anchor-fixed horizon-H
    recurrence these are states 1 through H-1.  The exact anchor contributes
    no nonlinear remainder, and the terminal state is not a transition input.

    With ``d_j = ||z_j||`` define

        Q = 1/2 * ||(M_j d_j^2)_j||_2,
        A = max_j M_j d_j,
        B = max_j M_j.

    Taylor's theorem and ``||K|| <= kappa`` give the sufficient condition

        alpha + kappa Q + kappa A E + (kappa B / 2) E^2 <= E.

    This dominates the scalar-max corollary because
    ``Q <= B p ||z||_X / 2`` and ``A <= B p``.
    """

    kappa = _finite_nonnegative("kappa", kappa)
    response_sequence_norm = _finite_nonnegative(
        "response_sequence_norm", response_sequence_norm
    )
    response_max_state_norm = _finite_nonnegative(
        "response_max_state_norm", response_max_state_norm
    )
    linear_response_error_bound = _finite_nonnegative(
        "linear_response_error_bound", linear_response_error_bound
    )
    if response_max_state_norm > response_sequence_norm:
        raise ValueError("max-state response norm cannot exceed sequence norm")
    if domain_radius is not None:
        domain_radius = _finite_nonnegative("domain_radius", domain_radius)

    drift = _array("drift_bounds", drift_bounds)
    response = _array("response_input_state_norms", response_input_state_norms)
    if drift.shape != response.shape:
        raise ValueError("drift and response arrays must have identical shapes")

    if drift.size:
        maximum_drift = float(np.max(drift))
        aligned_linear = float(np.max(drift * response))
        corrected_defect = 0.5 * float(
            np.linalg.vector_norm(drift * response * response)
        )
    else:
        maximum_drift = 0.0
        aligned_linear = 0.0
        corrected_defect = 0.0

    forcing = linear_response_error_bound + kappa * corrected_defect
    linear = kappa * aligned_linear
    quadratic = 0.5 * kappa * maximum_drift

    # Solve forcing + linear E + quadratic E^2 <= E.  The linear case is
    # handled separately, including nonzero computed-response error.
    if quadratic == 0.0:
        discriminant = (1.0 - linear) ** 2
        algebraic = linear < 1.0
        remainder = forcing / (1.0 - linear) if algebraic else None
    else:
        discriminant = (1.0 - linear) ** 2 - 4.0 * quadratic * forcing
        algebraic = linear < 1.0 and discriminant >= 0.0
        if algebraic:
            denominator = 1.0 - linear + math.sqrt(discriminant)
            remainder = 0.0 if forcing == 0.0 else 2.0 * forcing / denominator
        else:
            remainder = None

    total = None if remainder is None else response_max_state_norm + remainder
    domain_passed = total is not None and (
        domain_radius is None or total <= domain_radius
    )
    closure_passed = bool(algebraic and domain_passed)
    residual = None
    if remainder is not None:
        residual = forcing + linear * remainder + quadratic * remainder**2 - remainder

    return HeterogeneousClosure(
        kappa=kappa,
        response_sequence_norm=response_sequence_norm,
        response_max_state_norm=response_max_state_norm,
        maximum_derivative_drift=maximum_drift,
        aligned_linear_coefficient_before_green=aligned_linear,
        corrected_defect_norm_before_green=corrected_defect,
        linear_response_error_bound=linear_response_error_bound,
        corrected_defect_response_bound=kappa * corrected_defect,
        linearized_remainder_coefficient=linear,
        quadratic_remainder_coefficient=quadratic,
        discriminant=discriminant,
        remainder_radius=remainder,
        total_pointwise_radius=total,
        domain_radius=domain_radius,
        algebraic_closure_passed=bool(algebraic),
        domain_passed=bool(domain_passed),
        closure_passed=closure_passed,
        closure_residual=residual,
    )

