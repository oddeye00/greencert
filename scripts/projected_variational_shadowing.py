#!/usr/bin/env python3
"""Matrix-free projected variational shadowing.

The module implements a fixed-subspace two-radius certificate for an
arbitrary reference path of a discrete map.  Only products of the Jacobian
with vectors are needed by a model-specific caller.  For symmetric Jacobians,
the two cross-block norms coincide.

It also provides a Gaussian power-probe upper bound for a symmetric operator.
That bound is probabilistic, with an explicit user-supplied failure
probability; the state-tube theorem itself is deterministic conditional on the
reported block bounds.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Callable

import numpy as np
import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class ProjectedStepGeometry:
    """Certified geometry of one Jacobian in a fixed orthogonal splitting."""

    mapped_center: Tensor
    active_block_norm: float
    complement_to_active_norm: float
    active_to_complement_norm: float
    complement_block_norm: float
    jacobian_lipschitz: Callable[[float], float]


@dataclass(frozen=True)
class ProjectedResidualTube:
    reference: Tensor
    active_radius: np.ndarray
    complement_radius: np.ndarray
    total_radius: np.ndarray
    active_defect_norm: np.ndarray
    complement_defect_norm: np.ndarray
    active_block_norm: np.ndarray
    cross_block_norm: np.ndarray
    complement_block_norm: np.ndarray
    nonlinear_injection: np.ndarray
    reached_horizon: int


def orthogonal_residual(vector: Tensor, basis: Tensor) -> Tensor:
    """Return ``(I-UU^T) vector`` for a column-orthonormal ``basis=U``."""

    return vector - basis @ (basis.T @ vector)


@torch.no_grad()
def projected_residual_tube(
    reference: Tensor,
    basis: Tensor,
    geometry: Callable[[int, Tensor], ProjectedStepGeometry],
    *,
    numeric_cap: float = 1e6,
) -> ProjectedResidualTube:
    """Certify active and orthogonal errors around an explicit reference.

    Let ``P=UU^T`` and ``Q=I-P`` for the fixed orthonormal basis ``U``.  If
    ``h[j]`` is the error around ``reference[j]``, this routine propagates

        ||P h[j]|| <= a[j],       ||Q h[j]|| <= b[j].

    The caller supplies upper bounds for all four Jacobian blocks and a
    Lipschitz modulus for the full Jacobian.  The nonlinear Taylor residual is
    bounded in the full Euclidean radius ``sqrt(a**2+b**2)`` and injected into
    both projected recurrences.  No dense Jacobian is required.
    """

    if reference.ndim != 2 or len(reference) < 1:
        raise ValueError("reference must have shape [horizon+1, dimension]")
    if basis.ndim != 2 or basis.shape[0] != reference.shape[1]:
        raise ValueError("basis must have shape [dimension, rank]")
    gram = basis.T @ basis
    eye = torch.eye(basis.shape[1], dtype=basis.dtype, device=basis.device)
    if not torch.allclose(gram, eye, rtol=1e-9, atol=1e-11):
        raise ValueError("basis columns must be orthonormal")

    horizon = len(reference) - 1
    active = np.zeros(horizon + 1, dtype=np.float64)
    complement = np.zeros(horizon + 1, dtype=np.float64)
    total = np.zeros(horizon + 1, dtype=np.float64)
    active_defect = np.zeros(horizon, dtype=np.float64)
    complement_defect = np.zeros(horizon, dtype=np.float64)
    active_norm = np.zeros(horizon, dtype=np.float64)
    cross_norm = np.zeros(horizon, dtype=np.float64)
    complement_norm = np.zeros(horizon, dtype=np.float64)
    nonlinear = np.zeros(horizon, dtype=np.float64)
    reached = 0

    for step in range(horizon):
        a = float(active[step])
        b = float(complement[step])
        rho = float(np.hypot(a, b))
        if not np.isfinite(rho) or rho > numeric_cap:
            break

        row = geometry(step, reference[step])
        block_values = (
            row.active_block_norm,
            row.complement_to_active_norm,
            row.active_to_complement_norm,
            row.complement_block_norm,
        )
        if any((not np.isfinite(value) or value < 0.0) for value in block_values):
            raise ValueError("all block bounds must be finite and nonnegative")

        defect = row.mapped_center - reference[step + 1]
        defect_active = basis.T @ defect
        defect_complement = defect - basis @ defect_active
        p_defect = float(torch.linalg.vector_norm(defect_active))
        q_defect = float(torch.linalg.vector_norm(defect_complement))
        injection = 0.5 * float(row.jacobian_lipschitz(rho)) * rho**2

        next_active = (
            row.active_block_norm * a
            + row.complement_to_active_norm * b
            + p_defect
            + injection
        )
        next_complement = (
            row.active_to_complement_norm * a
            + row.complement_block_norm * b
            + q_defect
            + injection
        )
        next_total = float(np.hypot(next_active, next_complement))

        active_defect[step] = p_defect
        complement_defect[step] = q_defect
        active_norm[step] = row.active_block_norm
        cross_norm[step] = max(
            row.complement_to_active_norm, row.active_to_complement_norm
        )
        complement_norm[step] = row.complement_block_norm
        nonlinear[step] = injection

        if (
            not np.isfinite(next_active)
            or not np.isfinite(next_complement)
            or not np.isfinite(next_total)
            or next_total > numeric_cap
        ):
            break
        active[step + 1] = next_active
        complement[step + 1] = next_complement
        total[step + 1] = next_total
        reached = step + 1

    keep = reached + 1
    return ProjectedResidualTube(
        reference=reference[:keep],
        active_radius=active[:keep],
        complement_radius=complement[:keep],
        total_radius=total[:keep],
        active_defect_norm=active_defect[:reached],
        complement_defect_norm=complement_defect[:reached],
        active_block_norm=active_norm[:reached],
        cross_block_norm=cross_norm[:reached],
        complement_block_norm=complement_norm[:reached],
        nonlinear_injection=nonlinear[:reached],
        reached_horizon=reached,
    )


@torch.no_grad()
def projected_scalar_error_bound(
    gradient: Tensor,
    basis: Tensor,
    active_radius: float,
    complement_radius: float,
    hessian_operator_bound: float,
) -> float:
    """Second-order scalar-output bound on a projected state cylinder."""

    active_gradient = basis.T @ gradient
    complement_gradient = gradient - basis @ active_gradient
    rho = float(np.hypot(active_radius, complement_radius))
    return float(
        torch.linalg.vector_norm(active_gradient) * active_radius
        + torch.linalg.vector_norm(complement_gradient) * complement_radius
        + 0.5 * hessian_operator_bound * rho**2
    )


@torch.no_grad()
def gaussian_power_operator_upper(
    apply: Callable[[Tensor], Tensor],
    dimension: int,
    *,
    power: int,
    probes: int,
    failure_probability: float,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    projector_basis: Tensor | None = None,
) -> tuple[float, dict]:
    """High-probability upper bound for a fixed symmetric operator norm.

    For independent standard Gaussian probes ``g_i`` and a top unit
    eigenvector ``v`` of a fixed symmetric operator ``A``,

        ||A^q g_i|| >= ||A||^q |v^T g_i|.

    The exact folded-normal quantile therefore converts the largest observed
    power norm into an upper bound with the requested failure probability.
    When ``projector_basis`` is supplied, probes are projected to its
    orthogonal complement and ``apply`` must map that complement to itself.
    """

    if dimension < 1 or power < 1 or probes < 1:
        raise ValueError("dimension, power, and probes must be positive")
    if not (0.0 < failure_probability < 1.0):
        raise ValueError("failure_probability must lie strictly between 0 and 1")

    inside = failure_probability ** (1.0 / probes)
    quantile = NormalDist().inv_cdf(0.5 * (1.0 + inside))
    if not np.isfinite(quantile) or quantile <= 0.0:
        raise ValueError("invalid folded-normal quantile")

    maximum_power_norm = 0.0
    for _ in range(probes):
        vector = torch.randn(
            dimension, generator=generator, dtype=dtype, device=device
        )
        if projector_basis is not None:
            vector = orthogonal_residual(vector, projector_basis)
        for _ in range(power):
            vector = apply(vector)
            if projector_basis is not None:
                vector = orthogonal_residual(vector, projector_basis)
        maximum_power_norm = max(
            maximum_power_norm, float(torch.linalg.vector_norm(vector))
        )

    upper = 0.0
    if maximum_power_norm > 0.0:
        upper = (maximum_power_norm / quantile) ** (1.0 / power)
    return float(upper), {
        "power": int(power),
        "probes": int(probes),
        "failure_probability": float(failure_probability),
        "folded_normal_quantile": float(quantile),
        "maximum_power_norm": float(maximum_power_norm),
    }
