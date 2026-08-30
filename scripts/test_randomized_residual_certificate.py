#!/usr/bin/env python3
"""Tests for fresh-probe residual norm and scalar optimizer projection."""

from __future__ import annotations

import math

import numpy as np
import torch

from randomized_residual_certificate import (
    green_image_amplified_secant_beta,
    green_image_norm_upper_from_projection_intervals,
    optimizer_amplified_secant_scalar_projection,
    required_projection_radius,
    required_unscaled_green_image_secant_projection_radius,
    required_unscaled_secant_projection_radius,
    response_free_amplified_secant_beta,
    residual_norm_upper_from_projection_intervals,
)
from transformer_hvp_grokking import (
    TransformerConfig,
    flat_spec,
    flatten_parameters,
    make_disjoint_split,
    make_template,
)
from transformer_two_response import optimizer_amplified_secant_defect


def main() -> None:
    rng = np.random.default_rng(20260830)
    delta = 0.05
    probes = 8
    trials = 20_000
    misses = 0
    for _ in range(trials):
        residual = rng.normal(size=11)
        gaussian = rng.normal(size=(probes, residual.size))
        projections = gaussian @ residual
        certificate = residual_norm_upper_from_projection_intervals(
            projections, projections, delta=delta
        )
        misses += float(np.linalg.norm(residual)) > certificate.residual_norm_upper
    observed = misses / trials
    standard_error = math.sqrt(delta * (1.0 - delta) / trials)
    if abs(observed - delta) > 5.0 * standard_error + 0.002:
        raise AssertionError((observed, delta, standard_error))

    # The same event controls a propagated residual directly.  This exercises
    # the adjoint identity <K.T g,e>=<g,K e>, not the looser ||K||||e|| route.
    image_trials = 10_000
    image_misses = 0
    for _ in range(image_trials):
        operator = rng.normal(size=(13, 9))
        residual = rng.normal(size=9)
        image = operator @ residual
        gaussian = rng.normal(size=(probes, image.size))
        left = gaussian @ image
        right = np.einsum("mi,ij,j->m", gaussian, operator, residual)
        if not np.allclose(left, right, rtol=2.0e-14, atol=2.0e-14):
            raise AssertionError("adjoint projection identity changed")
        certificate = green_image_norm_upper_from_projection_intervals(
            right, right, delta=delta
        )
        image_misses += (
            float(np.linalg.norm(image)) > certificate.residual_norm_upper
        )
    image_observed = image_misses / image_trials
    image_standard_error = math.sqrt(delta * (1.0 - delta) / image_trials)
    if abs(image_observed - delta) > 5.0 * image_standard_error + 0.003:
        raise AssertionError((image_observed, delta, image_standard_error))

    values = [-0.2, 0.3, -0.1]
    intervals = residual_norm_upper_from_projection_intervals(
        [value - 0.01 for value in values],
        [value + 0.02 for value in values],
        delta=1.0e-4,
    )
    if intervals.projection_upper != 0.32:
        raise AssertionError("interval endpoint maximum is wrong")
    budget = 6.014744428490773e-20
    state_radius = required_projection_radius(
        budget, delta=1.0e-6, probes=16
    )
    unscaled = required_unscaled_secant_projection_radius(
        budget,
        delta=1.0e-6,
        probes=16,
        amplification=4096.0,
        learning_rate=1.0,
    )
    if not math.isclose(
        state_radius, 3.3433828306300715e-20, rel_tol=2.0e-15
    ):
        raise AssertionError("state projection budget changed")
    if not math.isclose(
        unscaled, 5.609265592017213e-13, rel_tol=2.0e-15
    ):
        raise AssertionError("unscaled scalar-jet budget changed")
    propagated_budget = 4804.639273433786 * budget
    image_unscaled = required_unscaled_green_image_secant_projection_radius(
        propagated_budget,
        delta=1.0e-6,
        probes=16,
        amplification=4096.0,
        learning_rate=1.0,
    )
    if not math.isclose(
        image_unscaled, 2.6950497758526715e-9, rel_tol=2.0e-15
    ):
        raise AssertionError("Green-image scalar-jet budget changed")

    response_free = response_free_amplified_secant_beta(
        [-1.0e-22, 2.0e-22],
        [1.5e-22, 2.5e-22],
        delta=1.0e-6,
        green_gain=4804.639273433786,
        analytic_discrepancy=2.568911437370008e-21,
    )
    expected_response_free = 4804.639273433786 * (
        2.568911437370008e-21 + 2.5e-22 / response_free.projection_certificate.calibration
    )
    if not math.isclose(response_free.beta_upper, expected_response_free):
        raise AssertionError("response-free beta formula changed")
    image_beta = green_image_amplified_secant_beta(
        [-2.0e-17, 1.0e-17],
        [2.5e-17, 1.5e-17],
        delta=1.0e-6,
        analytic_response_upper=1.0e-17,
        computed_response_norm=2.0e-22,
    )
    if image_beta.beta_upper <= 1.0e-17:
        raise AssertionError("Green-image beta omitted a contribution")

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    config = TransformerConfig(
        modulus=5,
        model_dim=8,
        hidden_dim=16,
        heads=2,
        depth=1,
        train_fraction=0.60,
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=0.01,
        steps=1,
        seed=20260830,
        threads=1,
        dtype="float64",
        loss="cross_entropy",
        normalization="none",
    )
    template = make_template(config)
    spec = flat_spec(template)
    parameter = flatten_parameters(template)
    train_pairs, train_labels, _, _, _, _ = make_disjoint_split(config)
    generator = torch.Generator().manual_seed(413)
    state_direction = torch.randn(
        2 * parameter.numel(), generator=generator
    ) * 1.0e-3
    state_probe = torch.randn(2 * parameter.numel(), generator=generator)
    lam = 8.0
    vector = optimizer_amplified_secant_defect(
        parameter,
        state_direction,
        train_pairs,
        train_labels,
        template,
        spec,
        config,
        amplification=lam,
    )
    projected_vector = torch.dot(state_probe, vector)
    scalar = optimizer_amplified_secant_scalar_projection(
        parameter,
        state_direction[: parameter.numel()],
        state_probe,
        train_pairs,
        train_labels,
        template,
        spec,
        config,
        amplification=lam,
    )
    if not torch.allclose(projected_vector, scalar, rtol=2.0e-9, atol=2.0e-15):
        raise AssertionError((float(projected_vector), float(scalar)))

    print(
        {
            "status": "randomized residual certificate tests passed",
            "monte_carlo_trials": trials,
            "observed_miss_rate": observed,
            "green_image_observed_miss_rate": image_observed,
            "target_miss_rate": delta,
            "scalar_projection_identity": True,
            "unscaled_scalar_jet_budget": unscaled,
            "green_image_unscaled_scalar_jet_budget": image_unscaled,
        }
    )


if __name__ == "__main__":
    main()
