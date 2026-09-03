#!/usr/bin/env python3
"""Matrix-free Gram enclosure for the scaled momentum optimizer Jacobian."""
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from probe_jacobian_bound import ProbeConfig, ProbeRegistry, gram_norm_bound
from transformer_hvp_grokking_v15 import (
    TransformerConfig,
    gradient_and_objective_hvp,
    gradient_hvp_and_third_contraction,
    objective_hvp,
)


def split_scaled_state(vector: Tensor) -> tuple[Tensor, Tensor]:
    if vector.numel() % 2:
        raise ValueError("scaled optimizer state must have even dimension")
    return vector.chunk(2)


def make_scaled_optimizer_jvp_vjp(
    parameter: Tensor,
    train_pairs: Tensor,
    train_labels: Tensor,
    template,
    spec,
    config: TransformerConfig,
) -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """Return exact JVP/VJP for state ``(theta, w=eta*v)``.

    With ``r = mu*w + eta*grad F(theta)``, the map is
    ``(theta,w) -> (theta-r,r)``.  Its two products each require one objective
    HVP because the objective Hessian is symmetric.
    """
    eta, mu = config.learning_rate, config.momentum

    def hvp(vector: Tensor) -> Tensor:
        return objective_hvp(
            parameter, vector, train_pairs, train_labels, template, spec, config
        )

    def jvp(direction: Tensor) -> Tensor:
        d_parameter, d_scaled_velocity = split_scaled_state(direction)
        d_next_velocity = mu * d_scaled_velocity + eta * hvp(d_parameter)
        return torch.cat((d_parameter - d_next_velocity, d_next_velocity))

    def vjp(cotangent: Tensor) -> Tensor:
        parameter_cotangent, velocity_cotangent = split_scaled_state(cotangent)
        difference = velocity_cotangent - parameter_cotangent
        return torch.cat(
            (
                parameter_cotangent + eta * hvp(difference),
                mu * difference,
            )
        )

    return jvp, vjp


def scaled_optimizer_map_and_jvp(
    state: Tensor,
    direction: Tensor,
    train_pairs: Tensor,
    train_labels: Tensor,
    template,
    spec,
    config: TransformerConfig,
) -> tuple[Tensor, Tensor]:
    """Return the scaled optimizer map and scaled JVP from one graph.

    ``state`` uses the training coordinates ``(theta, velocity)`` while
    ``direction`` and both returned tensors use certificate coordinates
    ``(theta, learning_rate * velocity)``.  This is the fused primitive needed
    when the signed-response recurrence evaluates the map and Jacobian at the
    same checkpoint.
    """

    mapped, image, _, _ = scaled_optimizer_map_jvp_differentials(
        state,
        direction,
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )
    return mapped, image


def scaled_optimizer_map_jvp_differentials(
    state: Tensor,
    direction: Tensor,
    train_pairs: Tensor,
    train_labels: Tensor,
    template,
    spec,
    config: TransformerConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Also expose the shared objective gradient and Hessian direction."""

    parameter, velocity = state.chunk(2)
    d_parameter, d_scaled_velocity = split_scaled_state(direction)
    objective_gradient, hessian_direction = gradient_and_objective_hvp(
        parameter,
        d_parameter,
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )
    eta, mu = config.learning_rate, config.momentum
    next_velocity = mu * velocity + objective_gradient
    scaled_next_velocity = eta * next_velocity
    mapped = torch.cat((parameter - scaled_next_velocity, scaled_next_velocity))
    d_next_scaled_velocity = mu * d_scaled_velocity + eta * hessian_direction
    image = torch.cat(
        (d_parameter - d_next_scaled_velocity, d_next_scaled_velocity)
    )
    return mapped, image, objective_gradient, hessian_direction


def scaled_optimizer_map_jvp_quadratic(
    state: Tensor,
    direction: Tensor,
    train_pairs: Tensor,
    train_labels: Tensor,
    template,
    spec,
    config: TransformerConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fuse map, scaled JVP, and center-quadratic nonlinear forcing.

    The third return is ``D^2 G(state)[direction,direction] / 2`` in scaled
    optimizer coordinates.  Momentum is affine, so only the parameter
    component enters the contracted third derivative of the objective.
    """

    parameter, velocity = state.chunk(2)
    d_parameter, d_scaled_velocity = split_scaled_state(direction)
    objective_gradient, hessian_direction, third_direction = (
        gradient_hvp_and_third_contraction(
            parameter,
            d_parameter,
            train_pairs,
            train_labels,
            template,
            spec,
            config,
        )
    )
    eta, mu = config.learning_rate, config.momentum
    next_velocity = mu * velocity + objective_gradient
    scaled_next_velocity = eta * next_velocity
    mapped = torch.cat((parameter - scaled_next_velocity, scaled_next_velocity))
    d_next_scaled_velocity = mu * d_scaled_velocity + eta * hessian_direction
    image = torch.cat(
        (d_parameter - d_next_scaled_velocity, d_next_scaled_velocity)
    )
    if bool(torch.any(d_parameter != 0.0)):
        quadratic_velocity = 0.5 * eta * third_direction
        quadratic = torch.cat((-quadratic_velocity, quadratic_velocity))
    else:
        # Canonical positive zeros preserve byte identity with the standalone
        # quadratic-surrogate construction's explicit initial row.
        quadratic = torch.zeros_like(direction)
    return mapped, image, quadratic


def scaled_optimizer_norm_bound(
    parameter: Tensor,
    train_pairs: Tensor,
    train_labels: Tensor,
    template,
    spec,
    config: TransformerConfig,
    probe: ProbeConfig,
    identity: tuple[int, ...],
    registry: ProbeRegistry,
) -> dict:
    jvp, vjp = make_scaled_optimizer_jvp_vjp(
        parameter, train_pairs, train_labels, template, spec, config
    )

    def gram(vector: Tensor) -> Tensor:
        return vjp(jvp(vector))

    result = gram_norm_bound(
        gram,
        dimension=2 * parameter.numel(),
        dtype=parameter.dtype,
        device=parameter.device,
        config=probe,
        identity=identity,
        registry=registry,
    )
    result.update(
        {
            "optimizer_jacobian_norm_upper_bound": result[
                "operator_norm_upper_bound"
            ],
            "optimizer_jvp_calls": result["gram_applications"],
            "optimizer_vjp_calls": result["gram_applications"],
            "objective_hvp_calls": 2 * result["gram_applications"],
            "state_coordinates": "(theta, learning_rate * velocity)",
        }
    )
    return result
