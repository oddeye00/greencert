#!/usr/bin/env python3
"""Fresh-probe upper certificates for hidden vectors and Green images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from probe_jacobian_bound import c_delta
from transformer_hvp_grokking import TransformerConfig, objective


@dataclass(frozen=True)
class ResidualProbeCertificate:
    projection_upper: float
    calibration: float
    residual_norm_upper: float
    probes: int
    delta: float


@dataclass(frozen=True)
class AmplifiedSecantProbeBudget:
    """Closure contribution supplied by a fresh scalar-probe block."""

    projection_certificate: ResidualProbeCertificate
    analytic_discrepancy: float
    green_gain: float
    beta_upper: float


def residual_norm_upper_from_projection_intervals(
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    delta: float,
) -> ResidualProbeCertificate:
    """Convert outward Gaussian projection intervals to a norm upper bound.

    The intervals must correspond to fresh independent standard-Gaussian
    probes.  Conditional on every quantity fixed before those probes, the
    returned bound fails with probability at most ``delta``.
    """

    if len(lower) != len(upper) or len(lower) == 0:
        raise ValueError("nonempty lower/upper sequences must have equal length")
    projection_upper = 0.0
    for lo, hi in zip(lower, upper):
        lo = float(lo)
        hi = float(hi)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
            raise ValueError("each projection interval must be finite and ordered")
        projection_upper = max(projection_upper, abs(lo), abs(hi))
    calibration = c_delta(float(delta), len(lower))
    return ResidualProbeCertificate(
        projection_upper=projection_upper,
        calibration=calibration,
        residual_norm_upper=projection_upper / calibration,
        probes=len(lower),
        delta=float(delta),
    )


def green_image_norm_upper_from_projection_intervals(
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    delta: float,
) -> ResidualProbeCertificate:
    """Bound ``||K e||`` from intervals for ``<K.T g_i, e>``.

    The probes ``g_i`` must be fresh standard Gaussians in the *output* space
    of ``K``.  Since ``<K.T g_i,e> = <g_i,K e>``, the same anti-concentration
    calibration applies directly to the propagated residual and avoids a
    worst-case multiplication by ``||K||``.
    """

    return residual_norm_upper_from_projection_intervals(
        lower, upper, delta=delta
    )


def response_free_amplified_secant_beta(
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    delta: float,
    green_gain: float,
    analytic_discrepancy: float,
) -> AmplifiedSecantProbeBudget:
    """Bound the second-response term without constructing that response.

    Intervals must enclose fresh Gaussian projections of the exact amplified
    secant forcing ``q^[lambda]``.  The resulting theorem contribution is

        beta <= ||K|| (sigma_sec + ||q^[lambda]||).

    This option is useful when the amplified forcing is much smaller than the
    analytic ray discrepancy, as in the sealed horizon-52 audit.
    """

    kappa = float(green_gain)
    sigma = float(analytic_discrepancy)
    if not np.isfinite(kappa) or kappa < 0.0:
        raise ValueError("green gain must be finite and nonnegative")
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("analytic discrepancy must be finite and nonnegative")
    certificate = residual_norm_upper_from_projection_intervals(
        lower, upper, delta=delta
    )
    return AmplifiedSecantProbeBudget(
        projection_certificate=certificate,
        analytic_discrepancy=sigma,
        green_gain=kappa,
        beta_upper=kappa * (sigma + certificate.residual_norm_upper),
    )


def green_image_amplified_secant_beta(
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    delta: float,
    analytic_response_upper: float,
    computed_response_norm: float,
) -> AmplifiedSecantProbeBudget:
    """Bound arithmetic/recurrence error after Green propagation.

    Here the intervals enclose ``<g_i, K e_ar>`` (equivalently
    ``<K.T g_i,e_ar>``), while ``analytic_response_upper`` is a separate bound
    on ``||K(q-q^[lambda])||``.  Unlike a forcing-space residual bound, this
    interface does not multiply numerical error by a worst-case Green gain.
    """

    analytic = float(analytic_response_upper)
    center = float(computed_response_norm)
    if not np.isfinite(analytic) or analytic < 0.0:
        raise ValueError("analytic response bound must be finite and nonnegative")
    if not np.isfinite(center) or center < 0.0:
        raise ValueError("computed response norm must be finite and nonnegative")
    certificate = green_image_norm_upper_from_projection_intervals(
        lower, upper, delta=delta
    )
    return AmplifiedSecantProbeBudget(
        projection_certificate=certificate,
        analytic_discrepancy=analytic,
        green_gain=1.0,
        beta_upper=center + analytic + certificate.residual_norm_upper,
    )


def required_projection_radius(
    residual_norm_budget: float,
    *,
    delta: float,
    probes: int,
) -> float:
    """Largest symmetric per-probe interval radius that fits a norm budget."""

    budget = float(residual_norm_budget)
    if budget < 0.0:
        raise ValueError("residual norm budget must be nonnegative")
    return c_delta(float(delta), int(probes)) * budget


def required_unscaled_secant_projection_radius(
    residual_norm_budget: float,
    *,
    delta: float,
    probes: int,
    amplification: float,
    learning_rate: float,
) -> float:
    """Translate a state-forcing norm budget to a scalar objective-jet budget."""

    lam = float(amplification)
    eta = float(learning_rate)
    if lam <= 0.0 or eta <= 0.0:
        raise ValueError("amplification and learning rate must be positive")
    return (
        required_projection_radius(
            residual_norm_budget, delta=delta, probes=probes
        )
        * lam**2
        / eta
    )


def required_unscaled_green_image_secant_projection_radius(
    propagated_residual_budget: float,
    *,
    delta: float,
    probes: int,
    amplification: float,
    learning_rate: float,
) -> float:
    """Scalar-jet radius for direct probes of a Green-propagated residual.

    The probe directions at each checkpoint are adjoint Green responses.  The
    returned number is therefore an aggregate, weighted sequence-projection
    tolerance; it is not a per-checkpoint interval width.
    """

    return required_unscaled_secant_projection_radius(
        propagated_residual_budget,
        delta=delta,
        probes=probes,
        amplification=amplification,
        learning_rate=learning_rate,
    )


def optimizer_amplified_secant_scalar_projection(
    parameter: Tensor,
    parameter_direction: Tensor,
    state_probe: Tensor,
    train_pairs: Tensor,
    train_labels: Tensor,
    template,
    spec,
    config: TransformerConfig,
    *,
    amplification: float,
) -> Tensor:
    """Evaluate one scalar projection of the amplified optimizer secant.

    If ``state_probe=(g_theta,g_w)``, only ``w=g_w-g_theta`` is needed:

      eta/lambda^2 [D_w F(theta+lambda*a) - D_w F(theta)
                    - lambda D^2 F(theta)[w,a]].

    This is the scalar bivariate-jet obligation that an outward implementation
    must enclose; it avoids an interval for the complete gradient/HVP vector.
    """

    if parameter.ndim != 1 or parameter_direction.shape != parameter.shape:
        raise ValueError("parameter and direction must be equal flat vectors")
    if state_probe.ndim != 1 or state_probe.numel() != 2 * parameter.numel():
        raise ValueError("state probe has the wrong shape")
    lam = float(amplification)
    if not np.isfinite(lam) or lam <= 0.0:
        raise ValueError("amplification must be finite and positive")
    g_theta, g_scaled_velocity = state_probe.chunk(2)
    probe_direction = (g_scaled_velocity - g_theta).detach()
    a = parameter_direction.detach()
    with torch.enable_grad():
        base = parameter.detach().requires_grad_(True)
        base_value = objective(
            base, train_pairs, train_labels, template, spec, config
        )
        (base_gradient,) = torch.autograd.grad(
            base_value, base, create_graph=True
        )
        base_directional = torch.dot(base_gradient, probe_direction)
        (mixed_vector,) = torch.autograd.grad(base_directional, base)
        mixed = torch.dot(mixed_vector, a)

        shifted = (parameter.detach() + lam * a).requires_grad_(True)
        shifted_value = objective(
            shifted, train_pairs, train_labels, template, spec, config
        )
        (shifted_gradient,) = torch.autograd.grad(shifted_value, shifted)
        shifted_directional = torch.dot(shifted_gradient, probe_direction)
    return config.learning_rate * (
        shifted_directional - base_directional.detach() - lam * mixed.detach()
    ) / (lam * lam)
