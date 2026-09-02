#!/usr/bin/env python3
"""Autodiff and v1-equivalence tests for the mixed directional v2 jet."""
from __future__ import annotations

import math

import torch

from transformer_hvp_grokking import (
    TransformerConfig,
    flat_spec,
    flatten_parameters,
    make_template,
    objective,
)
from transformer_mixed_directional_jet import mixed_directional_objective_fourth_bound
from transformer_mixed_directional_jet_v2 import mixed_directional_objective_bounds


def contracted(function, point, directions) -> float:
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
        seed=941,
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
    generator = torch.Generator().manual_seed(20_260_901)
    third_cases = 0
    fourth_cases = 0
    worst_third_ratio = 0.0
    worst_fourth_ratio = 0.0

    for scale in (1.0e-6, 2.0e-5, 1.0e-4):
        direction = torch.randn(parameter.shape, generator=generator)
        direction *= scale / torch.linalg.vector_norm(direction)
        record = mixed_directional_objective_bounds(
            parameter, direction, spec, config
        )
        released = mixed_directional_objective_fourth_bound(
            parameter, direction, spec, config
        )
        for key in (
            "mixed_fourth_derivative_upper",
            "gradient_taylor_remainder_upper",
            "maximum_stage_inflation",
        ):
            if not math.isclose(
                float(record[key]),
                float(released[key]),
                rel_tol=2.0e-12,
                abs_tol=1.0e-300,
            ):
                raise AssertionError(f"v1/v2 mismatch at {key}")

        for fraction in (0.0, 0.37, 1.0):
            point = parameter + fraction * direction
            dual = torch.randn(parameter.shape, generator=generator)
            dual /= torch.linalg.vector_norm(dual)
            function = lambda theta: objective(
                theta, pairs, labels, template, spec, config
            )
            third = contracted(function, point, [direction, direction, dual])
            fourth = contracted(
                function, point, [direction, direction, direction, dual]
            )
            third_upper = float(record["mixed_third_derivative_upper"])
            fourth_upper = float(record["mixed_fourth_derivative_upper"])
            worst_third_ratio = max(worst_third_ratio, third / third_upper)
            worst_fourth_ratio = max(worst_fourth_ratio, fourth / fourth_upper)
            if third > third_upper * (1.0 + 2.0e-11):
                raise AssertionError("autodiff mixed third derivative exceeds bound")
            if fourth > fourth_upper * (1.0 + 2.0e-11):
                raise AssertionError("autodiff mixed fourth derivative exceeds bound")
            third_cases += 1
            fourth_cases += 1

    print(
        {
            "status": "mixed directional v2 checks passed",
            "third_autodiff_cases": third_cases,
            "fourth_autodiff_cases": fourth_cases,
            "worst_third_to_bound_ratio": worst_third_ratio,
            "worst_fourth_to_bound_ratio": worst_fourth_ratio,
        }
    )


if __name__ == "__main__":
    main()
