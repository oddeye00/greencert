#!/usr/bin/env python3
"""Validity gates for the real-data MLP gradient and HVP."""
from __future__ import annotations

import torch

from real_dataset_mlp import (
    RealMLPConfig,
    analytic_gradient,
    analytic_objective_hvp,
    autograd_objective_hvp,
    initialize,
    make_split,
    objective,
    parameter_spec,
)


def main() -> None:
    config = RealMLPConfig(seed=17, width=8, learning_rate=0.005)
    data = make_split(config)
    spec = parameter_spec(config)
    parameter = initialize(config)
    generator = torch.Generator().manual_seed(401)
    direction = torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
    direction /= torch.linalg.vector_norm(direction)

    with torch.enable_grad():
        point = parameter.detach().requires_grad_(True)
        loss = objective(point, data["train_x"], data["train_y"], spec, config)
        (reference_gradient,) = torch.autograd.grad(loss, point)
    got_gradient = analytic_gradient(
        parameter, data["train_x"], data["train_y"], spec, config
    )
    gradient_relative = float(
        torch.linalg.vector_norm(got_gradient - reference_gradient)
        / torch.linalg.vector_norm(reference_gradient)
    )
    assert gradient_relative < 1e-12, gradient_relative

    reference_hvp = autograd_objective_hvp(
        parameter, direction, data["train_x"], data["train_y"], spec, config
    )
    got_hvp = analytic_objective_hvp(
        parameter, direction, data["train_x"], data["train_y"], spec, config
    )
    hvp_relative = float(
        torch.linalg.vector_norm(got_hvp - reference_hvp)
        / torch.linalg.vector_norm(reference_hvp)
    )
    assert hvp_relative < 1e-12, hvp_relative

    step = 1e-5
    plus = analytic_gradient(
        parameter + step * direction,
        data["train_x"],
        data["train_y"],
        spec,
        config,
    )
    minus = analytic_gradient(
        parameter - step * direction,
        data["train_x"],
        data["train_y"],
        spec,
        config,
    )
    finite = (plus - minus) / (2 * step)
    finite_relative = float(
        torch.linalg.vector_norm(got_hvp - finite)
        / torch.linalg.vector_norm(got_hvp)
    )
    assert finite_relative < 1e-8, finite_relative
    print(
        "PASS real-data MLP gradient/HVP "
        f"relative errors {gradient_relative:.2e}/{hvp_relative:.2e}/{finite_relative:.2e}"
    )


if __name__ == "__main__":
    main()
