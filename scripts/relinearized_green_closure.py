#!/usr/bin/env python3
"""Stable scalar closure for a Green operator rebuilt on a corrected path.

If ``b = c + d`` is an anchor-fixed correction of a reference path, then the
defect of ``b`` is already second order when ``d`` solves the variational
response.  Rebuilding the Jacobian sequence at ``b`` removes the mixed
``kappa * M * max_norm(d) * E`` term from the unknown-error tube.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


def _nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


@dataclass(frozen=True)
class RelinearizedClosure:
    kappa: float
    derivative_drift: float
    corrected_defect_response_bound: float
    correction_max_state_norm: float
    domain_radius: float
    discriminant: float
    remainder_radius: float | None
    total_radius_about_original_reference: float | None
    algebraic_closure_passed: bool
    domain_passed: bool
    closure_passed: bool
    closure_residual: float | None

    def as_dict(self) -> dict:
        return asdict(self)


def exact_relinearized_closure(
    *,
    kappa: float,
    derivative_drift: float,
    corrected_defect_response_bound: float,
    correction_max_state_norm: float,
    domain_radius: float,
) -> RelinearizedClosure:
    """Return the smaller root of ``Y + kappa*M*E^2/2 <= E``.

    ``corrected_defect_response_bound`` is an upper bound on
    ``||K_bar s_bar||`` including arithmetic/residual error.  The derivative
    drift bound must hold on the original-reference ball of radius
    ``domain_radius``; consequently ``correction_max_state_norm + E`` must stay
    in that ball.
    """

    kappa = _nonnegative("kappa", kappa)
    drift = _nonnegative("derivative_drift", derivative_drift)
    forcing = _nonnegative(
        "corrected_defect_response_bound", corrected_defect_response_bound
    )
    correction = _nonnegative(
        "correction_max_state_norm", correction_max_state_norm
    )
    domain = _nonnegative("domain_radius", domain_radius)
    coefficient = kappa * drift
    discriminant = 1.0 - 2.0 * coefficient * forcing
    algebraic = discriminant >= 0.0
    radius: float | None = None
    residual: float | None = None
    if algebraic:
        if forcing == 0.0:
            radius = 0.0
        elif coefficient == 0.0:
            radius = forcing
        else:
            # Stable form of (1-sqrt(discriminant))/coefficient.
            radius = 2.0 * forcing / (1.0 + math.sqrt(discriminant))
        residual = forcing + 0.5 * coefficient * radius * radius - radius
    domain_passed = radius is not None and correction + radius <= domain
    return RelinearizedClosure(
        kappa=kappa,
        derivative_drift=drift,
        corrected_defect_response_bound=forcing,
        correction_max_state_norm=correction,
        domain_radius=domain,
        discriminant=discriminant,
        remainder_radius=radius,
        total_radius_about_original_reference=(
            None if radius is None else correction + radius
        ),
        algebraic_closure_passed=algebraic,
        domain_passed=domain_passed,
        closure_passed=algebraic and domain_passed,
        closure_residual=residual,
    )


def mixed_term_closure_statistic(
    *, kappa: float, derivative_drift: float, correction_max_state_norm: float
) -> float:
    """The first-order coefficient paid by the unrelinearized theorem."""

    return (
        _nonnegative("kappa", kappa)
        * _nonnegative("derivative_drift", derivative_drift)
        * _nonnegative("correction_max_state_norm", correction_max_state_norm)
    )

