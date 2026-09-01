#!/usr/bin/env python3
"""Equivalence, autodiff, and runtime gates for the linear-cost mixed jet."""
from __future__ import annotations

import math
import time

import torch

from transformer_directional_fourth_bound import directional_objective_fourth_bound
from transformer_hvp_grokking import (
    TransformerConfig,
    flat_spec,
    flatten_parameters,
    make_template,
    objective,
)
from transformer_mixed_directional_jet import (
    mixed_directional_objective_fourth_bound,
    parameter_block_radii,
)


def mixed_fourth(function, point: torch.Tensor, directions: list[torch.Tensor]) -> float:
    variable = point.detach().requires_grad_(True)
    value = function(variable)
    for direction in directions:
        (gradient,) = torch.autograd.grad(value, variable, create_graph=True)
        value = torch.dot(gradient, direction)
    return abs(float(value.detach()))


def main() -> None:
    torch.set_default_dtype(torch.float64)
    config = TransformerConfig(
        modulus=3,
        model_dim=4,
        hidden_dim=6,
        heads=1,
        depth=1,
        seed=227,
        dtype="float64",
        loss="cross_entropy",
        normalization="none",
    )
    template = make_template(config)
    spec = flat_spec(template)
    parameter = flatten_parameters(template)
    pairs = torch.cartesian_prod(
        torch.arange(config.modulus), torch.arange(config.modulus)
    ).long()
    labels = (pairs[:, 0] + pairs[:, 1]) % config.modulus
    generator = torch.Generator().manual_seed(20260831)

    equality_cases = 0
    autodiff_cases = 0
    worst_autodiff_ratio = 0.0
    polynomial_seconds = 0.0
    mixed_seconds = 0.0
    for scale in (1.0e-6, 2.0e-5, 1.0e-4):
        direction = torch.randn(parameter.shape, generator=generator)
        direction *= scale / torch.linalg.vector_norm(direction)
        radii = parameter_block_radii(direction, spec, config)
        if not math.isclose(
            float(torch.linalg.vector_norm(direction)),
            float((radii @ radii) ** 0.5),
            rel_tol=2.0e-13,
        ):
            raise AssertionError("independent block partition failed")

        started = time.perf_counter()
        polynomial = directional_objective_fourth_bound(
            parameter, direction, spec, config
        )
        polynomial_seconds += time.perf_counter() - started
        started = time.perf_counter()
        mixed = mixed_directional_objective_fourth_bound(
            parameter, direction, spec, config
        )
        mixed_seconds += time.perf_counter() - started

        for key in (
            "mixed_fourth_derivative_upper",
            "gradient_taylor_remainder_upper",
            "maximum_stage_inflation",
        ):
            left = float(polynomial[key])
            right = float(mixed[key])
            if not math.isclose(left, right, rel_tol=2.0e-12, abs_tol=1.0e-300):
                raise AssertionError(f"mixed/polynomial mismatch at {key}: {left} != {right}")
        equality_cases += 1

        upper = float(mixed["mixed_fourth_derivative_upper"])
        for fraction in (0.0, 0.37, 1.0):
            point = parameter + fraction * direction
            dual = torch.randn(parameter.shape, generator=generator)
            dual /= torch.linalg.vector_norm(dual)
            observed = mixed_fourth(
                lambda theta: objective(
                    theta, pairs, labels, template, spec, config
                ),
                point,
                [direction, direction, direction, dual],
            )
            worst_autodiff_ratio = max(worst_autodiff_ratio, observed / upper)
            if observed > upper * (1.0 + 2.0e-11):
                raise AssertionError("autodiff mixed derivative exceeds mixed-jet bound")
            autodiff_cases += 1

    if not mixed_seconds < polynomial_seconds:
        raise AssertionError("linear-cost mixed jet did not beat polynomial implementation")
    print(
        {
            "status": "mixed directional jet checks passed",
            "polynomial_equivalence_cases": equality_cases,
            "autodiff_cases": autodiff_cases,
            "worst_autodiff_to_bound_ratio": worst_autodiff_ratio,
            "polynomial_seconds": polynomial_seconds,
            "mixed_seconds": mixed_seconds,
            "speedup": polynomial_seconds / mixed_seconds,
        }
    )


if __name__ == "__main__":
    main()
