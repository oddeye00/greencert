#!/usr/bin/env python3
"""Scalar closure tests for one-shot signed defect recentering.

The exact trajectory error relative to a reference path obeys

    h = z + K N(h),       z = K s.

Writing h = z + e and q = N(z) gives a sharper fixed-point problem for e.
This module contains only the resulting scalar inequalities; it performs no
model evaluation and no randomized operator query.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class OneShotClosure:
    """Result of a one-shot recentered quadratic closure calculation."""

    kappa: float
    derivative_drift: float
    response_sequence_norm: float
    response_max_state_norm: float
    linear_response_error_bound: float
    corrected_defect_response_bound: float
    linearized_remainder_coefficient: float
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


def _validate_nonnegative_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative, got {value}")
    return value


def exact_one_shot_closure(
    *,
    kappa: float,
    derivative_drift: float,
    response_sequence_norm: float,
    response_max_state_norm: float,
    corrected_defect_response_bound: float,
    linear_response_error_bound: float = 0.0,
    domain_radius: float | None = None,
) -> OneShotClosure:
    """Solve the recentered closure with supplied forcing-error bounds.

    Let ``b = kappa * derivative_drift``, ``p = max_j ||z_j||``, and let
    ``Y`` bound the total recentered forcing.  In exact arithmetic this is
    ``||K N(z)||_X``.  With a computed response, it also includes a bound on
    ``||K s - z_tilde||_X``.  The radius ``E`` is the smaller root of

        Y + b p E + (b/2) E^2 <= E.

    The derivative-drift envelope must hold on the pointwise ball of radius
    ``p + E`` around the original reference path.  When ``domain_radius`` is
    supplied, that containment is checked explicitly.
    """

    kappa = _validate_nonnegative_finite("kappa", kappa)
    derivative_drift = _validate_nonnegative_finite(
        "derivative_drift", derivative_drift
    )
    response_sequence_norm = _validate_nonnegative_finite(
        "response_sequence_norm", response_sequence_norm
    )
    response_max_state_norm = _validate_nonnegative_finite(
        "response_max_state_norm", response_max_state_norm
    )
    corrected_defect_response_bound = _validate_nonnegative_finite(
        "corrected_defect_response_bound", corrected_defect_response_bound
    )
    linear_response_error_bound = _validate_nonnegative_finite(
        "linear_response_error_bound", linear_response_error_bound
    )
    if response_max_state_norm > response_sequence_norm:
        raise ValueError("max-state response norm cannot exceed the sequence norm")
    if domain_radius is not None:
        domain_radius = _validate_nonnegative_finite("domain_radius", domain_radius)

    b = kappa * derivative_drift
    linear = b * response_max_state_norm
    forcing = linear_response_error_bound + corrected_defect_response_bound

    if b == 0.0:
        # Vanishing derivative drift makes N identically zero on the domain.
        algebraic_passed = forcing == 0.0
        remainder_radius = 0.0 if algebraic_passed else None
        discriminant = 1.0 if algebraic_passed else -math.inf
    else:
        discriminant = (1.0 - linear) ** 2 - 2.0 * b * forcing
        algebraic_passed = linear < 1.0 and discriminant >= 0.0
        if algebraic_passed:
            # This form avoids cancellation when Y is tiny.
            denominator = 1.0 - linear + math.sqrt(discriminant)
            remainder_radius = (
                0.0
                if forcing == 0.0
                else 2.0 * forcing / denominator
            )
        else:
            remainder_radius = None

    total_radius = (
        None
        if remainder_radius is None
        else response_max_state_norm + remainder_radius
    )
    domain_passed = (
        total_radius is not None
        and (domain_radius is None or total_radius <= domain_radius)
    )
    closure_passed = algebraic_passed and domain_passed
    closure_residual = None
    if remainder_radius is not None:
        closure_residual = (
            forcing
            + linear * remainder_radius
            + 0.5 * b * remainder_radius * remainder_radius
            - remainder_radius
        )

    return OneShotClosure(
        kappa=kappa,
        derivative_drift=derivative_drift,
        response_sequence_norm=response_sequence_norm,
        response_max_state_norm=response_max_state_norm,
        linear_response_error_bound=linear_response_error_bound,
        corrected_defect_response_bound=corrected_defect_response_bound,
        linearized_remainder_coefficient=linear,
        discriminant=discriminant,
        remainder_radius=remainder_radius,
        total_pointwise_radius=total_radius,
        domain_radius=domain_radius,
        algebraic_closure_passed=algebraic_passed,
        domain_passed=domain_passed,
        closure_passed=closure_passed,
        closure_residual=closure_residual,
    )


def conservative_one_shot_closure(
    *,
    kappa: float,
    derivative_drift: float,
    response_sequence_norm: float,
    response_max_state_norm: float,
    linear_response_error_bound: float = 0.0,
    domain_radius: float | None = None,
) -> OneShotClosure:
    """Close after recentering without evaluating the corrected defect.

    Taylor's theorem gives

        ||K N(z)||_X <= (kappa M / 2) ||z||_{infinity,2} ||z||_X.

    Thus this conservative variant reuses quantities already present in a
    signed-Green record and requires no additional model or operator call.
    """

    kappa = _validate_nonnegative_finite("kappa", kappa)
    derivative_drift = _validate_nonnegative_finite(
        "derivative_drift", derivative_drift
    )
    response_sequence_norm = _validate_nonnegative_finite(
        "response_sequence_norm", response_sequence_norm
    )
    response_max_state_norm = _validate_nonnegative_finite(
        "response_max_state_norm", response_max_state_norm
    )
    linear_response_error_bound = _validate_nonnegative_finite(
        "linear_response_error_bound", linear_response_error_bound
    )
    y_bound = (
        0.5
        * kappa
        * derivative_drift
        * response_max_state_norm
        * response_sequence_norm
    )
    return exact_one_shot_closure(
        kappa=kappa,
        derivative_drift=derivative_drift,
        response_sequence_norm=response_sequence_norm,
        response_max_state_norm=response_max_state_norm,
        corrected_defect_response_bound=y_bound,
        linear_response_error_bound=linear_response_error_bound,
        domain_radius=domain_radius,
    )
