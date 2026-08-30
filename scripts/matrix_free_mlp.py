#!/usr/bin/env python3
"""HVP, Gauss-Newton product, and Krylov utilities for the smooth MLP."""
from __future__ import annotations

from collections.abc import Callable

import torch

from smooth_mlp_modular_grokking import Config, logits, objective


Tensor = torch.Tensor


def objective_hvp(
    parameter: Tensor,
    vector: Tensor,
    pairs: Tensor,
    labels: Tensor,
    config: Config,
) -> Tensor:
    """Exact reverse-over-reverse Hessian-vector product."""

    with torch.enable_grad():
        point = parameter.detach().requires_grad_(True)
        value = objective(point, pairs, labels, config)
        (gradient,) = torch.autograd.grad(value, point, create_graph=True)
        (product,) = torch.autograd.grad(torch.dot(gradient, vector.detach()), point)
    return product.detach()


def gauss_newton_vp(
    parameter: Tensor,
    vector: Tensor,
    pairs: Tensor,
    config: Config,
) -> Tensor:
    """Product with the data Gauss-Newton matrix, excluding weight decay."""

    with torch.enable_grad():
        point = parameter.detach().requires_grad_(True)

        def forward(value: Tensor) -> Tensor:
            return logits(value, pairs, config)

        _, tangent = torch.autograd.functional.jvp(
            forward, point, vector.detach(), create_graph=False, strict=True
        )
        output = forward(point)
        (product,) = torch.autograd.grad(
            torch.sum(output * tangent.detach()) / output.numel(), point
        )
    return product.detach()


def residual_curvature_vp(
    parameter: Tensor,
    vector: Tensor,
    pairs: Tensor,
    labels: Tensor,
    config: Config,
) -> Tensor:
    """Residual-weighted curvature, excluding Gauss-Newton and L2 terms."""

    return (
        objective_hvp(parameter, vector, pairs, labels, config)
        - gauss_newton_vp(parameter, vector, pairs, config)
        - config.weight_decay * vector
    )


@torch.no_grad()
def block_apply(apply: Callable[[Tensor], Tensor], vectors: Tensor) -> Tensor:
    """Apply a vector operator to every column of a dense thin block."""

    if vectors.ndim != 2:
        raise ValueError("vectors must have shape [dimension, block]")
    return torch.stack([apply(vectors[:, j]) for j in range(vectors.shape[1])], dim=1)


@torch.no_grad()
def block_krylov_basis(
    apply: Callable[[Tensor], Tensor],
    starts: Tensor,
    *,
    rank: int,
    tolerance: float = 1e-11,
) -> tuple[Tensor, dict]:
    """Build a fully reorthogonalized block Krylov basis using only HVPs."""

    if starts.ndim == 1:
        starts = starts[:, None]
    if starts.ndim != 2 or rank < 1:
        raise ValueError("starts must be a vector/block and rank must be positive")

    q, r = torch.linalg.qr(starts, mode="reduced")
    keep = torch.abs(torch.diag(r)) > tolerance
    q = q[:, keep]
    if q.shape[1] == 0:
        raise ValueError("all starting directions were numerically zero")
    q = q[:, :rank]
    basis_blocks = [q]
    frontier = q
    calls = 0

    while sum(block.shape[1] for block in basis_blocks) < rank:
        candidate = block_apply(apply, frontier)
        calls += frontier.shape[1]
        basis = torch.cat(basis_blocks, dim=1)
        # Two passes suppress loss of orthogonality in nearly invariant spaces.
        for _ in range(2):
            candidate = candidate - basis @ (basis.T @ candidate)
        q_new, r_new = torch.linalg.qr(candidate, mode="reduced")
        diagonal = torch.abs(torch.diag(r_new))
        q_new = q_new[:, diagonal > tolerance]
        if q_new.shape[1] == 0:
            break
        remaining = rank - basis.shape[1]
        frontier = q_new[:, :remaining]
        basis_blocks.append(frontier)

    basis = torch.cat(basis_blocks, dim=1)[:, :rank]
    # Final polar cleanup makes the fixed projector numerically stable.
    basis, _ = torch.linalg.qr(basis, mode="reduced")
    return basis, {
        "requested_rank": int(rank),
        "achieved_rank": int(basis.shape[1]),
        "operator_calls_during_basis": int(calls),
        "orthogonality_error": float(
            torch.linalg.matrix_norm(
                basis.T @ basis
                - torch.eye(basis.shape[1], dtype=basis.dtype, device=basis.device),
                ord=2,
            )
        ),
    }


@torch.no_grad()
def projected_affine_reference(
    parameter: Tensor,
    gradient: Tensor,
    basis: Tensor,
    projected_hessian: Tensor,
    *,
    learning_rate: float,
    horizon: int,
) -> Tensor:
    """HVP-Krylov approximation of the frozen local quadratic trajectory."""

    coordinate = torch.zeros(basis.shape[1], dtype=parameter.dtype)
    projected_gradient = basis.T @ gradient
    jacobian = (
        torch.eye(basis.shape[1], dtype=parameter.dtype)
        - learning_rate * projected_hessian
    )
    rows = [parameter.clone()]
    for _ in range(horizon):
        coordinate = jacobian @ coordinate - learning_rate * projected_gradient
        rows.append(parameter + basis @ coordinate)
    return torch.stack(rows)


@torch.no_grad()
def hvp_affine_reference(
    parameter: Tensor,
    gradient: Tensor,
    hessian_apply: Callable[[Tensor], Tensor],
    *,
    learning_rate: float,
    horizon: int,
) -> Tensor:
    """Exact frozen-quadratic trajectory using HVPs and no dense Hessian.

    If ``d_j = theta_j - parameter``, this advances

        d_{j+1} = d_j - eta (gradient + H d_j).

    Thus it is algebraically identical to propagating every Hessian mode, but
    its memory cost is linear in the parameter dimension.  The Krylov basis is
    deliberately *not* used here: projection is reserved for splitting the
    certification uncertainty, rather than truncating the predictive clock.
    """

    if horizon < 0:
        raise ValueError("horizon must be nonnegative")
    displacement = torch.zeros_like(parameter)
    rows = [parameter.clone()]
    for _ in range(horizon):
        displacement = displacement - learning_rate * (
            gradient + hessian_apply(displacement)
        )
        rows.append(parameter + displacement)
    return torch.stack(rows)


@torch.no_grad()
def signed_variational_recenter(
    reference: Tensor,
    map_step: Callable[[Tensor], Tensor],
    jacobian_vector: Callable[[Tensor, Tensor], Tensor],
    *,
    numeric_cap: float = 1e4,
) -> tuple[Tensor, dict]:
    """One full-space signed correction sweep using one HVP per time step."""

    correction = torch.zeros_like(reference)
    defect_norms: list[float] = []
    correction_norms: list[float] = [0.0]
    reached = 0
    for step in range(len(reference) - 1):
        mapped = map_step(reference[step])
        defect = mapped - reference[step + 1]
        next_correction = (
            jacobian_vector(reference[step], correction[step]) + defect
        )
        norm = float(torch.linalg.vector_norm(next_correction))
        defect_norms.append(float(torch.linalg.vector_norm(defect)))
        if not torch.isfinite(next_correction).all() or norm > numeric_cap:
            break
        correction[step + 1] = next_correction
        correction_norms.append(norm)
        reached = step + 1
    keep = reached + 1
    return reference[:keep] + correction[:keep], {
        "reached_horizon": int(reached),
        "maximum_uncorrected_defect_norm": max(defect_norms, default=0.0),
        "maximum_correction_norm": max(correction_norms, default=0.0),
        "hvp_calls": int(reached),
    }
