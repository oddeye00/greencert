#!/usr/bin/env python3
"""Deterministic checks for cancellation-safe second-response ingredients."""

from __future__ import annotations

import torch

from transformer_hvp_grokking import (
    TransformerConfig,
    flat_spec,
    flatten_parameters,
    gradient,
    make_disjoint_split,
    make_template,
    objective_hvp,
)
from transformer_two_response import (
    optimizer_amplified_secant_defect,
    third_gradient_contraction,
)


def main() -> None:
    torch.set_default_dtype(torch.float64)
    point = torch.tensor([0.7, -1.1])
    direction = torch.tensor([0.03, -0.02])

    def quartic(value: torch.Tensor) -> torch.Tensor:
        # Mixed and coordinatewise quartics exercise all contracted indices.
        return (
            0.25 * value[0] ** 4
            + 0.5 * value[0] ** 2 * value[1] ** 2
            + 0.125 * value[1] ** 4
        )

    observed = third_gradient_contraction(quartic, point, direction)
    variable = point.detach().requires_grad_(True)
    (quartic_gradient,) = torch.autograd.grad(
        quartic(variable), variable, create_graph=True
    )
    (hvp,) = torch.autograd.grad(
        torch.dot(quartic_gradient, direction), variable, create_graph=True
    )
    (expected,) = torch.autograd.grad(torch.dot(hvp, direction), variable)
    if not torch.allclose(observed, expected, rtol=0.0, atol=0.0):
        raise AssertionError("third-gradient contraction mismatch")

    zero = third_gradient_contraction(quartic, point, torch.zeros_like(point))
    if not torch.equal(zero, torch.zeros_like(point)):
        raise AssertionError("zero direction must produce an exact zero")

    # Scalar x^4/4 has D^3F(x)[d,d] = 6*x*d^2.
    scalar_point = torch.tensor([1.25])
    scalar_direction = torch.tensor([-0.4])
    scalar = third_gradient_contraction(
        lambda value: 0.25 * value[0] ** 4,
        scalar_point,
        scalar_direction,
    )
    exact = 6.0 * scalar_point * scalar_direction.square()
    if not torch.allclose(scalar, exact, rtol=2.0e-15, atol=0.0):
        raise AssertionError("analytic quartic identity failed")

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
        seed=20260828,
        threads=1,
        dtype="float64",
        loss="cross_entropy",
        normalization="none",
    )
    template = make_template(config)
    spec = flat_spec(template)
    parameter = flatten_parameters(template)
    train_pairs, train_labels, _, _, _, _ = make_disjoint_split(config)
    generator = torch.Generator().manual_seed(91)
    state_direction = torch.randn(
        2 * parameter.numel(), generator=generator, dtype=parameter.dtype
    ) * 1.0e-3
    amplification = 8.0
    base = gradient(
        parameter, train_pairs, train_labels, template, spec, config
    )
    observed_secant = optimizer_amplified_secant_defect(
        parameter,
        state_direction,
        train_pairs,
        train_labels,
        template,
        spec,
        config,
        amplification=amplification,
        base_gradient=base,
    )
    a = state_direction[: parameter.numel()]
    shifted = gradient(
        parameter + amplification * a,
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )
    linear = objective_hvp(
        parameter, a, train_pairs, train_labels, template, spec, config
    )
    remainder = (
        shifted - base - amplification * linear
    ) / amplification**2
    expected_secant = torch.cat(
        (-config.learning_rate * remainder, config.learning_rate * remainder)
    )
    if not torch.equal(observed_secant, expected_secant):
        raise AssertionError("Transformer amplified-secant implementation mismatch")

    print({"status": "transformer two-response unit checks passed", "cases": 4})


if __name__ == "__main__":
    main()
