#!/usr/bin/env python3
"""Exact-product tests for the AdamW optimizer map."""
from __future__ import annotations

import torch

from adamw_optimizer_probe import (
    AdamWSettings,
    adamw_step_from_gradient,
    make_adamw_jvp_vjp,
    split_adamw_state,
)
from batched_green_operator import relative_error
from transformer_hvp_grokking import (
    TransformerConfig,
    flat_spec,
    flatten_parameters,
    make_disjoint_split,
    make_template,
    objective,
)


def main() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = TransformerConfig(
        modulus=5,
        model_dim=8,
        hidden_dim=16,
        heads=2,
        depth=2,
        train_fraction=0.60,
        learning_rate=0.001,
        momentum=0.0,
        weight_decay=0.0,
        steps=1,
        seed=20260825,
        threads=1,
        dtype="float64",
        loss="cross_entropy",
        normalization="layernorm",
    )
    settings = AdamWSettings(
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=1e-2,
        step=7,
    )
    template = make_template(config)
    spec = flat_spec(template)
    parameter = flatten_parameters(template)
    pairs, labels, _, _, _, _ = make_disjoint_split(config)
    generator = torch.Generator().manual_seed(407)
    first_moment = 1e-3 * torch.randn(
        parameter.shape, generator=generator, dtype=parameter.dtype
    )
    second_moment = 1e-3 + 1e-3 * torch.rand(
        parameter.shape, generator=generator, dtype=parameter.dtype
    )
    state = torch.cat((parameter, first_moment, second_moment))
    direction = torch.randn(state.shape, generator=generator, dtype=state.dtype)
    cotangent = torch.randn(state.shape, generator=generator, dtype=state.dtype)
    jvp, vjp = make_adamw_jvp_vjp(
        parameter,
        first_moment,
        second_moment,
        pairs,
        labels,
        template,
        spec,
        config,
        settings,
    )
    manual_jvp = jvp(direction)
    manual_vjp = vjp(cotangent)
    dimension = parameter.numel()

    def map_fn(full_state: torch.Tensor) -> torch.Tensor:
        theta, first, second = split_adamw_state(full_state, dimension)
        value = objective(theta, pairs, labels, template, spec, config)
        (objective_gradient,) = torch.autograd.grad(value, theta, create_graph=True)
        next_theta, next_first, next_second = adamw_step_from_gradient(
            theta, first, second, objective_gradient, settings
        )
        return torch.cat((next_theta, next_first, next_second))

    state_for_jvp = state.detach().requires_grad_(True)
    _, automatic_jvp = torch.autograd.functional.jvp(
        map_fn, state_for_jvp, direction, create_graph=False
    )
    jvp_error = relative_error(manual_jvp, automatic_jvp)
    state_for_vjp = state.detach().requires_grad_(True)
    output = map_fn(state_for_vjp)
    (automatic_vjp,) = torch.autograd.grad(torch.dot(output, cotangent), state_for_vjp)
    vjp_error = relative_error(manual_vjp, automatic_vjp)
    adjoint_left = float(torch.dot(manual_jvp, cotangent))
    adjoint_right = float(torch.dot(direction, manual_vjp))
    adjoint_error = abs(adjoint_left - adjoint_right) / max(
        abs(adjoint_left), abs(adjoint_right), 1.0
    )
    assert jvp_error < 2e-11, jvp_error
    assert vjp_error < 2e-11, vjp_error
    assert adjoint_error < 2e-12, adjoint_error
    print(
        "PASS: two-block LayerNorm Transformer AdamW JVP/VJP match autograd "
        f"(relative errors {jvp_error:.3e}/{vjp_error:.3e}, adjoint {adjoint_error:.3e})."
    )


if __name__ == "__main__":
    main()
