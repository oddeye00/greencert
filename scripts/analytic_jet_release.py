#!/usr/bin/env python3
"""Deterministic neural-jet release for corrected-path certificates.

The corrected-path Green closure needs a Lipschitz bound for the optimizer
Jacobian and an output-margin transport bound.  If a ball-valid neural jet is
already available, both obligations can be discharged without a randomized
output-Jacobian query.  A verifier may try this deterministic route first and
fall back to a sharper output probe only when it does not close.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from relinearized_green_closure import RelinearizedClosure, exact_relinearized_closure


def _finite_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def cross_entropy_objective_third_bound(
    *, first: float, second: float, third: float
) -> float:
    """Bound ``||D^3(CE o f)||`` from a ball-valid output jet.

    The constants use ``||D CE|| <= sqrt(2)``, ``||D^2 CE|| <= 1/2``, and
    ``||D^3 CE|| <= 2``.  Averaging examples cannot increase the bound, and
    quadratic weight decay has zero third derivative.
    """

    first = _finite_nonnegative("first", first)
    second = _finite_nonnegative("second", second)
    third = _finite_nonnegative("third", third)
    return 2.0 * first**3 + 1.5 * first * second + math.sqrt(2.0) * third


def scaled_momentum_jacobian_drift(
    *, first: float, second: float, third: float, learning_rate: float
) -> float:
    """Ball-valid Lipschitz constant for the scaled-momentum map Jacobian."""

    learning_rate = _finite_nonnegative("learning_rate", learning_rate)
    return (
        math.sqrt(2.0)
        * learning_rate
        * cross_entropy_objective_third_bound(
            first=first, second=second, third=third
        )
    )


def logit_margin_radius(*, first: float, state_radius: float) -> float:
    """Uniform true-versus-competitor margin radius from a global first jet."""

    first = _finite_nonnegative("first", first)
    state_radius = _finite_nonnegative("state_radius", state_radius)
    return math.sqrt(2.0) * first * state_radius


@dataclass(frozen=True)
class AnalyticJetRelease:
    maximum_optimizer_jacobian_drift: float
    maximum_output_first_derivative: float
    closure: RelinearizedClosure
    state_radius_about_original_reference: float | None
    maximum_margin_radius: float | None

    def as_dict(self) -> dict:
        result = asdict(self)
        result["closure"] = self.closure.as_dict()
        return result


def analytic_jet_release(
    *,
    kappa: float,
    corrected_defect_response_bound: float,
    correction_max_state_norm: float,
    domain_radius: float,
    learning_rate: float,
    transition_jets: list[tuple[float, float, float]],
    output_first_bounds: list[float],
) -> AnalyticJetRelease:
    """Try a corrected-path release with no randomized output operator.

    ``transition_jets`` contains ball-valid ``(first, second, third)`` output
    derivative bounds for every transition input with nonzero unknown error.
    ``output_first_bounds`` contains the corresponding first-derivative bounds
    at event-query states.  Both families must be valid on the declared outer
    ball about the original reference path.
    """

    if not transition_jets:
        maximum_drift = 0.0
    else:
        maximum_drift = max(
            scaled_momentum_jacobian_drift(
                first=first,
                second=second,
                third=third,
                learning_rate=learning_rate,
            )
            for first, second, third in transition_jets
        )
    maximum_first = max(
        (_finite_nonnegative("output first", value) for value in output_first_bounds),
        default=0.0,
    )
    closure = exact_relinearized_closure(
        kappa=kappa,
        derivative_drift=maximum_drift,
        corrected_defect_response_bound=corrected_defect_response_bound,
        correction_max_state_norm=correction_max_state_norm,
        domain_radius=domain_radius,
    )
    state_radius = closure.total_radius_about_original_reference
    margin = (
        None
        if state_radius is None
        else logit_margin_radius(first=maximum_first, state_radius=state_radius)
    )
    return AnalyticJetRelease(
        maximum_optimizer_jacobian_drift=maximum_drift,
        maximum_output_first_derivative=maximum_first,
        closure=closure,
        state_radius_about_original_reference=state_radius,
        maximum_margin_radius=margin,
    )
