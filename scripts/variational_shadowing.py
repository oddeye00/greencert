#!/usr/bin/env python3
"""Defect-corrected nonautonomous shadowing for discrete dynamical systems.

For a true iteration ``x[j+1] = G(x[j])`` and any checkpoint-computable
reference path ``bar[j]``, this module propagates the *vector* response to the
known reference defect before bounding the genuinely nonlinear uncertainty.
That retains directional cancellation discarded by a scalar defect norm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class StepLinearization:
    mapped_center: Tensor
    jacobian: Tensor
    jacobian_operator_norm: float
    jacobian_lipschitz: Callable[[float], float]


@dataclass(frozen=True)
class VariationalTube:
    corrected_reference: Tensor
    correction: Tensor
    remainder_radius: np.ndarray
    total_error_radius: np.ndarray
    reference_defect_norm: np.ndarray
    correction_norm: np.ndarray
    jacobian_norm: np.ndarray
    nonlinear_injection: np.ndarray
    reached_horizon: int


@dataclass(frozen=True)
class ReferenceCorrection:
    """One signed variational refinement of an arbitrary reference path."""

    corrected_reference: Tensor
    correction: Tensor
    reference_defect_norm: np.ndarray
    correction_norm: np.ndarray
    jacobian_norm: np.ndarray
    reached_horizon: int


@dataclass(frozen=True)
class ResidualTube:
    """A scalar shadowing tube centered on a fully specified reference."""

    reference: Tensor
    error_radius: np.ndarray
    reference_defect_norm: np.ndarray
    jacobian_norm: np.ndarray
    nonlinear_injection: np.ndarray
    reached_horizon: int


@torch.no_grad()
def variational_reference_correction(
    reference: Tensor,
    linearize: Callable[[Tensor], StepLinearization],
    numeric_cap: float = 1e6,
) -> ReferenceCorrection:
    """Apply one signed Newton--Picard/variational sweep to ``reference``.

    The refinement is

        z[0] = 0,
        z[j+1] = DG(reference[j]) z[j] + G(reference[j]) - reference[j+1].

    This routine only constructs the refined centerline.  In particular, it
    does not truncate merely because a conservative nonlinear tube around the
    *old* centerline becomes large.  A certificate should subsequently be
    built around ``reference + z`` with :func:`residual_centered_tube`.
    """
    if reference.ndim != 2 or len(reference) < 1:
        raise ValueError("reference must have shape [horizon+1, dimension]")
    horizon = len(reference) - 1
    dimension = reference.shape[1]
    correction = torch.zeros_like(reference)
    defects = np.zeros(horizon, dtype=np.float64)
    correction_norm = np.zeros(horizon + 1, dtype=np.float64)
    jacobian_norm = np.zeros(horizon, dtype=np.float64)
    reached = 0
    for step in range(horizon):
        geometry = linearize(reference[step])
        if geometry.jacobian.shape != (dimension, dimension):
            raise ValueError("linearization Jacobian has the wrong shape")
        defect = geometry.mapped_center - reference[step + 1]
        next_correction = geometry.jacobian @ correction[step] + defect
        next_norm = float(torch.linalg.vector_norm(next_correction))
        defects[step] = float(torch.linalg.vector_norm(defect))
        jacobian_norm[step] = float(geometry.jacobian_operator_norm)
        if not np.isfinite(next_norm) or next_norm > numeric_cap:
            break
        correction[step + 1] = next_correction
        correction_norm[step + 1] = next_norm
        reached = step + 1
    keep = reached + 1
    return ReferenceCorrection(
        corrected_reference=reference[:keep] + correction[:keep],
        correction=correction[:keep],
        reference_defect_norm=defects[:reached],
        correction_norm=correction_norm[:keep],
        jacobian_norm=jacobian_norm[:reached],
        reached_horizon=reached,
    )


@torch.no_grad()
def residual_centered_tube(
    reference: Tensor,
    linearize: Callable[[Tensor], StepLinearization],
    numeric_cap: float = 1e6,
) -> ResidualTube:
    """Certify a standard nonlinear tube around an explicit reference path.

    With ``r[j] = G(reference[j]) - reference[j+1]`` and
    ``epsilon[0] = 0``, the recursion is

        epsilon[j+1] = beta[j] epsilon[j] + ||r[j]||
                         + 0.5 M[j](epsilon[j]) epsilon[j]**2.

    When ``reference`` is the output of
    :func:`variational_reference_correction`, its new defects are exact
    second-order residuals.  Evaluating those residuals before taking norms
    avoids charging the tube for the much larger known first correction.
    """
    if reference.ndim != 2 or len(reference) < 1:
        raise ValueError("reference must have shape [horizon+1, dimension]")
    horizon = len(reference) - 1
    dimension = reference.shape[1]
    epsilon = np.zeros(horizon + 1, dtype=np.float64)
    defects = np.zeros(horizon, dtype=np.float64)
    jacobian_norm = np.zeros(horizon, dtype=np.float64)
    injection = np.zeros(horizon, dtype=np.float64)
    reached = 0
    for step in range(horizon):
        current = float(epsilon[step])
        if not np.isfinite(current) or current > numeric_cap:
            break
        geometry = linearize(reference[step])
        if geometry.jacobian.shape != (dimension, dimension):
            raise ValueError("linearization Jacobian has the wrong shape")
        defect = float(
            torch.linalg.vector_norm(geometry.mapped_center - reference[step + 1])
        )
        beta = float(geometry.jacobian_operator_norm)
        nonlinear = 0.5 * float(geometry.jacobian_lipschitz(current)) * current**2
        next_radius = beta * current + defect + nonlinear
        defects[step] = defect
        jacobian_norm[step] = beta
        injection[step] = nonlinear
        if not np.isfinite(next_radius) or next_radius > numeric_cap:
            break
        epsilon[step + 1] = next_radius
        reached = step + 1
    keep = reached + 1
    return ResidualTube(
        reference=reference[:keep],
        error_radius=epsilon[:keep],
        reference_defect_norm=defects[:reached],
        jacobian_norm=jacobian_norm[:reached],
        nonlinear_injection=injection[:reached],
        reached_horizon=reached,
    )


@torch.no_grad()
def defect_corrected_variational_tube(
    reference: Tensor,
    linearize: Callable[[Tensor], StepLinearization],
    numeric_cap: float = 1e6,
) -> VariationalTube:
    """Certify a tube around a first variational correction of ``reference``.

    ``linearize(center)`` must return ``G(center)``, ``DG(center)``, a rigorous
    operator-norm bound for that Jacobian, and a function ``M(radius)`` that
    bounds the Jacobian Lipschitz constant on the Euclidean ball of the given
    radius around ``center``.

    If ``z`` is the exact linear response to the known reference defects and
    ``omega`` bounds the remaining nonlinear error, the recursion is

        z[j+1] = J[j] z[j] + G(bar[j]) - bar[j+1],
        omega[j+1] = beta[j] omega[j]
                         + 0.5 M[j](||z[j]|| + omega[j])
                               (||z[j]|| + omega[j])**2.

    The returned certificate is ``||x[j]-(bar[j]+z[j])|| <= omega[j]``.
    """
    if reference.ndim != 2 or len(reference) < 1:
        raise ValueError("reference must have shape [horizon+1, dimension]")
    horizon = len(reference) - 1
    dimension = reference.shape[1]
    correction = torch.zeros_like(reference)
    remainder = np.zeros(horizon + 1, dtype=np.float64)
    total = np.zeros(horizon + 1, dtype=np.float64)
    defects = np.zeros(horizon, dtype=np.float64)
    correction_norm = np.zeros(horizon + 1, dtype=np.float64)
    jacobian_norm = np.zeros(horizon, dtype=np.float64)
    injection = np.zeros(horizon, dtype=np.float64)
    reached = 0

    for step in range(horizon):
        omega = float(remainder[step])
        if not np.isfinite(omega) or omega > numeric_cap:
            break
        geometry = linearize(reference[step])
        if geometry.jacobian.shape != (dimension, dimension):
            raise ValueError("linearization Jacobian has the wrong shape")
        defect = geometry.mapped_center - reference[step + 1]
        defects[step] = float(torch.linalg.vector_norm(defect))
        jacobian_norm[step] = float(geometry.jacobian_operator_norm)
        radius = float(torch.linalg.vector_norm(correction[step])) + omega
        local_lipschitz = float(geometry.jacobian_lipschitz(radius))
        nonlinear = 0.5 * local_lipschitz * radius**2
        next_omega = jacobian_norm[step] * omega + nonlinear
        next_correction = geometry.jacobian @ correction[step] + defect
        next_total = float(torch.linalg.vector_norm(next_correction)) + next_omega
        injection[step] = nonlinear
        if (
            not np.isfinite(next_omega)
            or not np.isfinite(next_total)
            or next_omega > numeric_cap
            or next_total > numeric_cap
        ):
            break
        correction[step + 1] = next_correction
        correction_norm[step + 1] = float(torch.linalg.vector_norm(next_correction))
        remainder[step + 1] = next_omega
        total[step + 1] = next_total
        reached = step + 1

    keep = reached + 1
    return VariationalTube(
        corrected_reference=reference[:keep] + correction[:keep],
        correction=correction[:keep],
        remainder_radius=remainder[:keep],
        total_error_radius=total[:keep],
        reference_defect_norm=defects[:reached],
        correction_norm=correction_norm[:keep],
        jacobian_norm=jacobian_norm[:reached],
        nonlinear_injection=injection[:reached],
        reached_horizon=reached,
    )


@torch.no_grad()
def uncorrected_variational_tube(
    reference: Tensor,
    linearize: Callable[[Tensor], StepLinearization],
    numeric_cap: float = 1e6,
) -> np.ndarray:
    """Scalar comparison tube around the uncorrected reference path."""
    horizon = len(reference) - 1
    epsilon = np.zeros(horizon + 1, dtype=np.float64)
    reached = 0
    for step in range(horizon):
        radius = float(epsilon[step])
        if not np.isfinite(radius) or radius > numeric_cap:
            break
        geometry = linearize(reference[step])
        defect = float(
            torch.linalg.vector_norm(geometry.mapped_center - reference[step + 1])
        )
        nonlinear = 0.5 * float(geometry.jacobian_lipschitz(radius)) * radius**2
        next_radius = geometry.jacobian_operator_norm * radius + defect + nonlinear
        if not np.isfinite(next_radius) or next_radius > numeric_cap:
            break
        epsilon[step + 1] = next_radius
        reached = step + 1
    return epsilon[: reached + 1]
