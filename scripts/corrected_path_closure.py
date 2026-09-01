#!/usr/bin/env python3
"""Scalar closure for a theorem centered entirely at a corrected path."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


def _nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


@dataclass(frozen=True)
class CorrectedPathClosure:
    kappa: float
    derivative_drift: float
    defect_response_bound: float
    domain_radius: float
    discriminant: float
    remainder_radius: float | None
    algebraic_closure_passed: bool
    domain_passed: bool
    closure_passed: bool
    closure_residual: float | None

    def as_dict(self) -> dict:
        return asdict(self)


def exact_corrected_path_closure(
    *,
    kappa: float,
    derivative_drift: float,
    defect_response_bound: float,
    domain_radius: float,
) -> CorrectedPathClosure:
    """Solve ``B + kappa*M*E^2/2 <= E`` at its smaller root."""

    kappa = _nonnegative("kappa", kappa)
    drift = _nonnegative("derivative_drift", derivative_drift)
    forcing = _nonnegative("defect_response_bound", defect_response_bound)
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
            radius = 2.0 * forcing / (1.0 + math.sqrt(discriminant))
        residual = forcing + 0.5 * coefficient * radius * radius - radius
    domain_passed = radius is not None and radius <= domain
    return CorrectedPathClosure(
        kappa=kappa,
        derivative_drift=drift,
        defect_response_bound=forcing,
        domain_radius=domain,
        discriminant=discriminant,
        remainder_radius=radius,
        algebraic_closure_passed=algebraic,
        domain_passed=domain_passed,
        closure_passed=algebraic and domain_passed,
        closure_residual=residual,
    )
