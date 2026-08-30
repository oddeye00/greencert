#!/usr/bin/env python3
"""Analytic and regression checks for the independent fourth-order jet."""

from __future__ import annotations

import math
import random

import torch

from transformer_fourth_jet_bound import (
    Jet4,
    objective_fourth_derivative_bound,
    product,
    smooth_map,
    transformer_output_fourth_jet_bound,
)
from transformer_hvp_grokking import (
    TransformerConfig,
    flat_spec,
    flatten_parameters,
    make_template,
    objective,
)
from transformer_jet_bound import transformer_output_jet_bound


def derivatives(function, point: torch.Tensor) -> list[float]:
    variable = point.detach().requires_grad_(True)
    value = function(variable)
    out = [float(value.detach())]
    for _ in range(4):
        (value,) = torch.autograd.grad(value, variable, create_graph=True)
        out.append(float(value.detach()))
    return out


def polynomial(coefficients: list[float], value: torch.Tensor) -> torch.Tensor:
    return sum(
        coefficient * value**order / math.factorial(order)
        for order, coefficient in enumerate(coefficients)
    )


def fourth_directional_derivative(function, point: torch.Tensor) -> float:
    variable = point.detach().requires_grad_(True)
    value = function(variable)
    for _ in range(4):
        (value,) = torch.autograd.grad(value, variable, create_graph=True)
    return float(value.detach())


def main() -> None:
    torch.set_default_dtype(torch.float64)
    rng = random.Random(20260828)
    algebra_cases = 0
    for _ in range(50):
        left = [rng.uniform(-1.0, 1.0) for _ in range(5)]
        right = [rng.uniform(-1.0, 1.0) for _ in range(5)]
        observed_product = product(Jet4(*left), Jet4(*right))
        expected_product = derivatives(
            lambda t: polynomial(left, t) * polynomial(right, t),
            torch.tensor(0.0),
        )
        for observed, expected in zip(
            observed_product.__dict__.values(), expected_product
        ):
            if not math.isclose(
                observed, expected, rel_tol=3.0e-13, abs_tol=3.0e-13
            ):
                raise AssertionError("fourth-jet product identity failed")

        source = [rng.uniform(-0.5, 0.5) for _ in range(5)]
        center = source[0]
        outer = math.exp(center)
        observed_map = smooth_map(
            Jet4(*source),
            value=outer,
            first=outer,
            second=outer,
            third=outer,
            fourth=outer,
        )
        expected_map = derivatives(
            lambda t: torch.exp(polynomial(source, t)), torch.tensor(0.0)
        )
        for observed, expected in zip(
            observed_map.__dict__.values(), expected_map
        ):
            if not math.isclose(
                observed, expected, rel_tol=5.0e-13, abs_tol=5.0e-13
            ):
                raise AssertionError("fourth-jet composition identity failed")
        algebra_cases += 1

    config = TransformerConfig(
        modulus=5,
        model_dim=8,
        hidden_dim=16,
        heads=2,
        depth=1,
        seed=91,
        dtype="float64",
        loss="cross_entropy",
        normalization="none",
    )
    template = make_template(config)
    spec = flat_spec(template)
    parameter = flatten_parameters(template)
    radius = 1.0e-5
    old = transformer_output_jet_bound(
        parameter, template, spec, config, radius=radius
    )
    new = transformer_output_fourth_jet_bound(
        parameter, template, spec, config, radius=radius
    )
    for name in ("value", "first", "second", "third"):
        if not math.isclose(
            float(getattr(new, name)),
            float(getattr(old, name)),
            rel_tol=3.0e-15,
            abs_tol=0.0,
        ):
            raise AssertionError(f"lower-order jet regression failed at {name}")
    fourth = objective_fourth_derivative_bound(
        parameter, template, spec, config, radius=radius
    )
    if not math.isfinite(fourth) or fourth <= 0.0:
        raise AssertionError("fourth objective envelope must be positive and finite")

    # Exercise the complete parameter-to-cross-entropy graph, not only the
    # scalar jet algebra.  Every sampled point remains inside the ball used by
    # the deterministic envelope and every direction has Euclidean norm one.
    pairs = torch.tensor(
        [(left, right) for left in range(config.modulus) for right in range(config.modulus)],
        dtype=torch.long,
    )
    labels = (pairs[:, 0] + pairs[:, 1]) % config.modulus
    generator = torch.Generator().manual_seed(20260828)
    worst_directional_ratio = 0.0
    directional_cases = 12
    for _ in range(directional_cases):
        direction = torch.randn(
            parameter.shape, generator=generator, dtype=parameter.dtype
        )
        direction /= torch.linalg.vector_norm(direction)
        offset = radius * (2.0 * float(torch.rand((), generator=generator)) - 1.0)
        observed = abs(
            fourth_directional_derivative(
                lambda t: objective(
                    parameter + (offset + t) * direction,
                    pairs,
                    labels,
                    template,
                    spec,
                    config,
                ),
                torch.tensor(0.0, dtype=parameter.dtype),
            )
        )
        ratio = observed / fourth
        worst_directional_ratio = max(worst_directional_ratio, ratio)
        if observed > fourth * (1.0 + 2.0e-12):
            raise AssertionError("actual fourth directional derivative exceeds envelope")

    print(
        {
            "status": "transformer fourth-jet checks passed",
            "algebra_cases": algebra_cases,
            "directional_cases": directional_cases,
            "worst_directional_to_bound_ratio": worst_directional_ratio,
            "objective_fourth_bound": fourth,
        }
    )


if __name__ == "__main__":
    main()
