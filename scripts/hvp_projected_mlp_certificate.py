#!/usr/bin/env python3
"""HVP-only projected certificates for the one-hidden-layer tanh MLP.

The implementation never forms a Hessian.  A thin block Krylov basis defines
the active state, one HVP per step performs signed variational recentering, and
Gaussian power probes bound the data-curvature complement with an explicit
family-wise failure probability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from matrix_free_mlp import (
    block_apply,
    block_krylov_basis,
    gauss_newton_vp,
    hvp_affine_reference,
    objective_hvp,
    residual_curvature_vp,
    signed_variational_recenter,
)
from modular_accuracy_certificate import persistent_event_bracket
from projected_variational_shadowing import (
    ProjectedStepGeometry,
    gaussian_power_operator_upper,
    orthogonal_residual,
    projected_residual_tube,
)
from smooth_mlp_certificate import TANH_D2, objective_hessian_lipschitz
from smooth_mlp_modular_grokking import Config, analytic_gradient, logits, unpack


Tensor = torch.Tensor
X_NORM = np.sqrt(3.0)


@dataclass(frozen=True)
class HVPProjectedResult:
    basis: Tensor
    raw_reference: Tensor
    corrected_reference: Tensor
    tube: object
    probe_diagnostics: tuple[dict, ...]
    construction_diagnostics: dict


@torch.no_grad()
def margin_gradient_matrix(
    parameter: Tensor,
    pairs: Tensor,
    labels: Tensor,
    config: Config,
) -> Tensor:
    """All true-vs-competitor margin gradients as ``[N,p,d]``."""

    p, h = config.modulus, config.width
    w, q, v, _ = unpack(parameter, config)
    preactivation = w[:, pairs[:, 0]].T + w[:, p + pairs[:, 1]].T + q
    hidden = torch.tanh(preactivation)
    first = 1.0 - hidden.square()
    n = len(pairs)
    total = parameter.numel()
    gradients = torch.zeros((n, p, total), dtype=parameter.dtype)
    w_end = h * 2 * p
    q_end = w_end + h
    v_end = q_end + p * h

    for sample in range(n):
        label = int(labels[sample])
        left = int(pairs[sample, 0])
        right = p + int(pairs[sample, 1])
        for competitor in range(p):
            if competitor == label:
                continue
            pregradient = (v[label] - v[competitor]) * first[sample]
            reshaped_w = gradients[sample, competitor, :w_end].reshape(h, 2 * p)
            reshaped_w[:, left] = pregradient
            reshaped_w[:, right] = pregradient
            gradients[sample, competitor, w_end:q_end] = pregradient
            reshaped_v = gradients[sample, competitor, q_end:v_end].reshape(p, h)
            reshaped_v[label] = hidden[sample]
            reshaped_v[competitor] = -hidden[sample]
            gradients[sample, competitor, v_end + label] = 1.0
            gradients[sample, competitor, v_end + competitor] = -1.0
    return gradients


@torch.no_grad()
def margin_matrix(
    parameter: Tensor,
    pairs: Tensor,
    labels: Tensor,
    config: Config,
) -> Tensor:
    values = logits(parameter, pairs, config)
    true = values[torch.arange(len(pairs)), labels]
    return true[:, None] - values


@torch.no_grad()
def active_start_block(
    parameter: Tensor,
    gradient: Tensor,
    certificate_pairs: Tensor,
    certificate_labels: Tensor,
    config: Config,
    *,
    margin_starts: int,
) -> tuple[Tensor, dict]:
    if margin_starts < 0:
        raise ValueError("margin_starts must be nonnegative")
    gradients = margin_gradient_matrix(
        parameter, certificate_pairs, certificate_labels, config
    )
    margins = margin_matrix(parameter, certificate_pairs, certificate_labels, config)
    rows = torch.arange(len(certificate_pairs))
    margins[rows, certificate_labels] = torch.inf
    flat_order = torch.argsort(torch.abs(margins).reshape(-1))
    chosen: list[Tensor] = [gradient]
    selected: list[dict] = []
    if margin_starts == 0:
        return torch.stack(chosen, dim=1), {"selected_margin_starts": selected}
    for flat in flat_order.tolist():
        sample = flat // config.modulus
        competitor = flat % config.modulus
        vector = gradients[sample, competitor]
        if float(torch.linalg.vector_norm(vector)) <= 1e-14:
            continue
        chosen.append(vector)
        selected.append(
            {
                "sample": int(sample),
                "label": int(certificate_labels[sample]),
                "competitor": int(competitor),
                "absolute_margin": abs(float(margins[sample, competitor])),
            }
        )
        if len(selected) >= margin_starts:
            break
    return torch.stack(chosen, dim=1), {"selected_margin_starts": selected}


@torch.no_grad()
def build_active_basis(
    parameter: Tensor,
    train_pairs: Tensor,
    train_labels: Tensor,
    certificate_pairs: Tensor,
    certificate_labels: Tensor,
    config: Config,
    *,
    rank: int,
    margin_starts: int,
) -> tuple[Tensor, Tensor, Tensor, dict]:
    gradient = analytic_gradient(parameter, train_pairs, train_labels, config)
    starts, start_diagnostic = active_start_block(
        parameter,
        gradient,
        certificate_pairs,
        certificate_labels,
        config,
        margin_starts=margin_starts,
    )

    def hessian_apply(vector: Tensor) -> Tensor:
        return objective_hvp(parameter, vector, train_pairs, train_labels, config)

    basis, krylov_diagnostic = block_krylov_basis(
        hessian_apply, starts, rank=rank
    )
    hessian_basis = block_apply(hessian_apply, basis)
    projected_hessian = 0.5 * (
        basis.T @ hessian_basis + hessian_basis.T @ basis
    )
    data_basis = hessian_basis - config.weight_decay * basis
    base = 1.0 - config.learning_rate * config.weight_decay
    jacobian_basis = base * basis - config.learning_rate * data_basis
    anchor_active = float(torch.linalg.matrix_norm(
        0.5 * (basis.T @ jacobian_basis + jacobian_basis.T @ basis), ord=2
    ))
    anchor_cross = config.learning_rate * float(torch.linalg.matrix_norm(
        orthogonal_residual(data_basis, basis), ord=2
    ))
    return basis, projected_hessian, gradient, {
        **start_diagnostic,
        **krylov_diagnostic,
        "hvp_calls_for_projected_hessian": int(basis.shape[1]),
        "anchor_active_block_norm": anchor_active,
        "anchor_cross_block_norm": anchor_cross,
    }


@torch.no_grad()
def recentered_hvp_reference(
    parameter: Tensor,
    train_pairs: Tensor,
    train_labels: Tensor,
    config: Config,
    *,
    horizon: int,
    recenter_sweeps: int,
    numeric_cap: float = 1e4,
    anchor_gradient: Tensor | None = None,
) -> tuple[Tensor, Tensor, dict]:
    """Construct the full-space HVP clock and apply fixed signed sweeps."""

    if horizon < 1 or recenter_sweeps < 1:
        raise ValueError("horizon and recenter_sweeps must be positive")
    gradient = (
        analytic_gradient(parameter, train_pairs, train_labels, config)
        if anchor_gradient is None
        else anchor_gradient
    )

    def anchor_hessian_apply(vector: Tensor) -> Tensor:
        return objective_hvp(
            parameter, vector, train_pairs, train_labels, config
        )

    raw_reference = hvp_affine_reference(
        parameter,
        gradient,
        anchor_hessian_apply,
        learning_rate=config.learning_rate,
        horizon=horizon,
    )

    def map_step(center: Tensor) -> Tensor:
        return center - config.learning_rate * analytic_gradient(
            center, train_pairs, train_labels, config
        )

    def jacobian_vector(center: Tensor, vector: Tensor) -> Tensor:
        return vector - config.learning_rate * objective_hvp(
            center, vector, train_pairs, train_labels, config
        )

    corrected = raw_reference
    recenter_diagnostics: list[dict] = []
    for sweep in range(recenter_sweeps):
        corrected, diagnostic = signed_variational_recenter(
            corrected, map_step, jacobian_vector, numeric_cap=numeric_cap
        )
        recenter_diagnostics.append({"sweep": sweep + 1, **diagnostic})
        if len(corrected) <= 1:
            break
    return raw_reference, corrected, {
        "recenter_sweeps_requested": int(recenter_sweeps),
        "recenter_sweeps_completed": len(recenter_diagnostics),
        "recenter_diagnostics": recenter_diagnostics,
        "reference_hvp_calls": int(horizon),
        "reference_projection_rank": None,
    }


@torch.no_grad()
def _probe_geometry(
    center: Tensor,
    basis: Tensor,
    train_pairs: Tensor,
    train_labels: Tensor,
    config: Config,
    *,
    power: int,
    probes: int,
    component_failure: float,
    random_seed: int,
) -> dict:
    eta = config.learning_rate
    base = 1.0 - eta * config.weight_decay

    def full_hessian(vector: Tensor) -> Tensor:
        return objective_hvp(center, vector, train_pairs, train_labels, config)

    hessian_basis = block_apply(full_hessian, basis)
    data_basis = hessian_basis - config.weight_decay * basis
    jacobian_basis = base * basis - eta * data_basis
    active_matrix = 0.5 * (
        basis.T @ jacobian_basis + jacobian_basis.T @ basis
    )
    active = float(torch.linalg.matrix_norm(active_matrix, ord=2))
    cross = eta * float(
        torch.linalg.matrix_norm(orthogonal_residual(data_basis, basis), ord=2)
    )

    def projected_gn(vector: Tensor) -> Tensor:
        q_vector = orthogonal_residual(vector, basis)
        return orthogonal_residual(
            gauss_newton_vp(center, q_vector, train_pairs, config), basis
        )

    def projected_residual(vector: Tensor) -> Tensor:
        q_vector = orthogonal_residual(vector, basis)
        return orthogonal_residual(
            residual_curvature_vp(
                center, q_vector, train_pairs, train_labels, config
            ),
            basis,
        )

    gn_generator = torch.Generator().manual_seed(random_seed)
    residual_generator = torch.Generator().manual_seed(random_seed + 1_000_003)
    gn_upper, gn_diagnostic = gaussian_power_operator_upper(
        projected_gn,
        center.numel(),
        power=power,
        probes=probes,
        failure_probability=component_failure,
        generator=gn_generator,
        dtype=center.dtype,
        device=center.device,
        projector_basis=basis,
    )
    residual_upper, residual_diagnostic = gaussian_power_operator_upper(
        projected_residual,
        center.numel(),
        power=power,
        probes=probes,
        failure_probability=component_failure,
        generator=residual_generator,
        dtype=center.dtype,
        device=center.device,
        projector_basis=basis,
    )
    complement = max(
        abs(base + eta * residual_upper),
        abs(base - eta * (gn_upper + residual_upper)),
    )
    safety = 1.0 + 5e-12
    return {
        "active_block": active * safety,
        "cross_block": cross * safety,
        "complement_block": complement * safety,
        "gauss_newton_complement_upper": gn_upper,
        "residual_complement_upper": residual_upper,
        "gauss_newton_probe": gn_diagnostic,
        "residual_probe": residual_diagnostic,
        "hvp_calls": int(basis.shape[1] + 2 * power * probes),
    }


@torch.no_grad()
def build_hvp_projected_certificate(
    parameter: Tensor,
    train_pairs: Tensor,
    train_labels: Tensor,
    certificate_pairs: Tensor,
    certificate_labels: Tensor,
    config: Config,
    *,
    horizon: int,
    rank: int = 24,
    margin_starts: int = 4,
    geometry_stride: int = 25,
    power: int = 24,
    probes: int = 8,
    total_failure_probability: float = 1e-6,
    random_seed: int = 20260822,
    numeric_cap: float = 1e4,
    recenter_sweeps: int = 1,
) -> HVPProjectedResult:
    if horizon < 1 or geometry_stride < 1 or recenter_sweeps < 1:
        raise ValueError("horizon, geometry_stride, and recenter_sweeps must be positive")
    basis, projected_hessian, gradient, basis_diagnostic = build_active_basis(
        parameter,
        train_pairs,
        train_labels,
        certificate_pairs,
        certificate_labels,
        config,
        rank=rank,
        margin_starts=margin_starts,
    )
    raw_reference, corrected, reference_diagnostic = recentered_hvp_reference(
        parameter,
        train_pairs,
        train_labels,
        config,
        horizon=horizon,
        recenter_sweeps=recenter_sweeps,
        numeric_cap=numeric_cap,
        anchor_gradient=gradient,
    )

    def map_step(center: Tensor) -> Tensor:
        return center - config.learning_rate * analytic_gradient(
            center, train_pairs, train_labels, config
        )

    def jacobian_vector(center: Tensor, vector: Tensor) -> Tensor:
        return vector - config.learning_rate * objective_hvp(
            center, vector, train_pairs, train_labels, config
        )

    effective_horizon = len(corrected) - 1
    maximum_probe_count = max(
        (effective_horizon + geometry_stride - 1) // geometry_stride, 1
    )
    component_failure = total_failure_probability / (2 * maximum_probe_count)
    probe_cache: dict[int, dict] = {}

    def get_probe(step: int) -> dict:
        probe_step = (step // geometry_stride) * geometry_stride
        if probe_step in probe_cache:
            return probe_cache[probe_step]
        index = probe_step // geometry_stride
        row = _probe_geometry(
            corrected[probe_step],
            basis,
            train_pairs,
            train_labels,
            config,
            power=power,
            probes=probes,
            component_failure=component_failure,
            random_seed=random_seed + 10_007 * index,
        )
        row["step"] = int(probe_step)
        probe_cache[probe_step] = row
        return row

    def geometry(step: int, center: Tensor) -> ProjectedStepGeometry:
        probe = get_probe(step)
        probe_center = corrected[probe["step"]]
        gap = float(torch.linalg.vector_norm(center - probe_center))
        drift = (
            config.learning_rate
            * objective_hessian_lipschitz(probe_center, config, gap)
            * gap
        )
        return ProjectedStepGeometry(
            mapped_center=map_step(center),
            active_block_norm=probe["active_block"] + drift,
            complement_to_active_norm=probe["cross_block"] + drift,
            active_to_complement_norm=probe["cross_block"] + drift,
            complement_block_norm=probe["complement_block"] + drift,
            jacobian_lipschitz=lambda radius, center=center: (
                config.learning_rate
                * objective_hessian_lipschitz(center, config, radius)
            ),
        )

    tube = projected_residual_tube(
        corrected, basis, geometry, numeric_cap=numeric_cap
    )
    probe_rows = [probe_cache[key] for key in sorted(probe_cache)]
    construction = {
        **basis_diagnostic,
        **reference_diagnostic,
        "requested_horizon": int(horizon),
        "geometry_stride": int(geometry_stride),
        "geometry_probe_count": len(probe_rows),
        "maximum_geometry_probe_count": int(maximum_probe_count),
        "power": int(power),
        "probes_per_component": int(probes),
        "familywise_failure_probability": float(total_failure_probability),
        "component_failure_probability": float(component_failure),
        "hvp_calls_geometry": int(sum(row["hvp_calls"] for row in probe_rows)),
        "dense_hessian_entries_formed": 0,
    }
    return HVPProjectedResult(
        basis=basis,
        raw_reference=raw_reference,
        corrected_reference=corrected,
        tube=tube,
        probe_diagnostics=tuple(probe_rows),
        construction_diagnostics=construction,
    )


@torch.no_grad()
def projected_certified_counts(
    result: HVPProjectedResult,
    certificate_pairs: Tensor,
    certificate_labels: Tensor,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    steps = len(result.tube.reference)
    n = len(certificate_pairs)
    p = config.modulus
    guaranteed = np.zeros(steps, dtype=np.int64)
    possible = np.zeros(steps, dtype=np.int64)
    center_correct = np.zeros(steps, dtype=np.int64)
    max_error = np.zeros(steps, dtype=np.float64)
    rows = torch.arange(n)

    for step in range(steps):
        center = result.tube.reference[step]
        values = logits(center, certificate_pairs, config)
        margins = values[rows, certificate_labels][:, None] - values
        center_correct[step] = int(
            torch.sum(torch.argmax(values, dim=1) == certificate_labels)
        )
        gradients = margin_gradient_matrix(
            center, certificate_pairs, certificate_labels, config
        )
        active_gradient = gradients @ result.basis
        active_norm = torch.linalg.vector_norm(active_gradient, dim=2)
        full_squared = torch.sum(gradients.square(), dim=2)
        active_squared = torch.sum(active_gradient.square(), dim=2)
        complement_norm = torch.sqrt(torch.clamp(full_squared - active_squared, min=0.0))

        a = float(result.tube.active_radius[step])
        b = float(result.tube.complement_radius[step])
        rho = float(np.hypot(a, b))
        _, _, v, _ = unpack(center, config)
        row_difference = torch.linalg.vector_norm(
            v[:, None, :] - v[None, :, :], dim=2
        ).numpy() + np.sqrt(2.0) * rho
        aa = row_difference * TANH_D2 * X_NORM**2
        ab = np.sqrt(2.0) * X_NORM
        b2 = 0.5 * (np.abs(aa) + np.sqrt(aa * aa + 4.0 * ab * ab))
        errors = (
            active_norm.numpy() * a
            + complement_norm.numpy() * b
            + 0.5 * b2[certificate_labels.numpy(), :] * rho**2
        )
        lower = margins.numpy() - errors
        upper = margins.numpy() + errors
        lower[np.arange(n), certificate_labels.numpy()] = np.inf
        upper[np.arange(n), certificate_labels.numpy()] = np.inf
        guaranteed[step] = int(np.sum(np.all(lower > 0.0, axis=1)))
        possible[step] = int(n - np.sum(np.any(upper < 0.0, axis=1)))
        max_error[step] = float(np.max(errors))
    return guaranteed, possible, center_correct, max_error


def brackets_for_thresholds(
    guaranteed: np.ndarray,
    possible: np.ndarray,
    required: dict[float, int],
    *,
    persistence: int = 1,
) -> dict[str, list[int] | None]:
    return {
        f"{threshold:.2f}": persistent_event_bracket(
            guaranteed, possible, count, persistence
        )
        for threshold, count in required.items()
    }
