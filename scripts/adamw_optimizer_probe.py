#!/usr/bin/env python3
"""Matrix-free JVP/VJP for a fixed-step AdamW optimizer map."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from transformer_hvp_grokking import (
    TransformerConfig,
    gradient,
    objective_hvp,
)


@dataclass(frozen=True)
class AdamWSettings:
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 1e-2
    step: int = 1

    def validate(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("AdamW betas must lie in [0,1)")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if self.step < 1:
            raise ValueError("step must be positive")


def split_adamw_state(rows: Tensor, dimension: int) -> tuple[Tensor, Tensor, Tensor]:
    if rows.shape[-1] != 3 * dimension:
        raise ValueError("AdamW state has the wrong final dimension")
    return rows[..., :dimension], rows[..., dimension : 2 * dimension], rows[..., 2 * dimension :]


def adamw_step_from_gradient(
    parameter: Tensor,
    first_moment: Tensor,
    second_moment: Tensor,
    objective_gradient: Tensor,
    settings: AdamWSettings,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply the standard bias-corrected AdamW recurrence."""
    settings.validate()
    first = settings.beta1 * first_moment + (1.0 - settings.beta1) * objective_gradient
    second = settings.beta2 * second_moment + (1.0 - settings.beta2) * objective_gradient.square()
    first_hat = first / (1.0 - settings.beta1**settings.step)
    second_hat = second / (1.0 - settings.beta2**settings.step)
    if bool((second_hat <= 0.0).any()):
        raise ValueError("AdamW derivative requires a strictly positive second moment")
    denominator = second_hat.sqrt() + settings.epsilon
    next_parameter = (
        (1.0 - settings.learning_rate * settings.weight_decay) * parameter
        - settings.learning_rate * first_hat / denominator
    )
    return next_parameter, first, second


def make_adamw_jvp_vjp(
    parameter: Tensor,
    first_moment: Tensor,
    second_moment: Tensor,
    pairs: Tensor,
    labels: Tensor,
    template,
    spec,
    config: TransformerConfig,
    settings: AdamWSettings,
) -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    """Return exact matrix-free products for one deterministic AdamW step.

    The only neural second-order primitive is an objective HVP.  No optimizer
    Jacobian or Hessian is formed.
    """
    settings.validate()
    dimension = parameter.numel()
    if first_moment.shape != parameter.shape or second_moment.shape != parameter.shape:
        raise ValueError("AdamW moments must match the parameter vector")
    objective_gradient = gradient(
        parameter, pairs, labels, template, spec, config
    )
    first = settings.beta1 * first_moment + (1.0 - settings.beta1) * objective_gradient
    second = settings.beta2 * second_moment + (1.0 - settings.beta2) * objective_gradient.square()
    first_scale = 1.0 - settings.beta1**settings.step
    second_scale = 1.0 - settings.beta2**settings.step
    first_hat = first / first_scale
    second_hat = second / second_scale
    if bool((second_hat <= 0.0).any()):
        raise ValueError("AdamW derivative requires a strictly positive second moment")
    root = second_hat.sqrt()
    denominator = root + settings.epsilon
    decay = 1.0 - settings.learning_rate * settings.weight_decay

    def hvp(vector: Tensor) -> Tensor:
        return objective_hvp(
            parameter,
            vector,
            pairs,
            labels,
            template,
            spec,
            config,
        )

    def jvp(direction: Tensor) -> Tensor:
        d_parameter, d_first_moment, d_second_moment = split_adamw_state(
            direction, dimension
        )
        d_gradient = hvp(d_parameter)
        d_first = settings.beta1 * d_first_moment + (1.0 - settings.beta1) * d_gradient
        d_second = (
            settings.beta2 * d_second_moment
            + 2.0 * (1.0 - settings.beta2) * objective_gradient * d_gradient
        )
        d_first_hat = d_first / first_scale
        d_second_hat = d_second / second_scale
        d_denominator = 0.5 * d_second_hat / root
        d_ratio = (
            d_first_hat / denominator
            - first_hat * d_denominator / denominator.square()
        )
        d_next_parameter = decay * d_parameter - settings.learning_rate * d_ratio
        return torch.cat((d_next_parameter, d_first, d_second))

    def vjp(cotangent: Tensor) -> Tensor:
        next_parameter_bar, first_bar, second_bar = split_adamw_state(
            cotangent, dimension
        )
        parameter_bar = decay * next_parameter_bar
        ratio_bar = -settings.learning_rate * next_parameter_bar
        first_hat_bar = ratio_bar / denominator
        denominator_bar = -ratio_bar * first_hat / denominator.square()
        second_hat_bar = 0.5 * denominator_bar / root
        first_bar = first_bar + first_hat_bar / first_scale
        second_bar = second_bar + second_hat_bar / second_scale
        first_moment_bar = settings.beta1 * first_bar
        second_moment_bar = settings.beta2 * second_bar
        gradient_bar = (
            (1.0 - settings.beta1) * first_bar
            + 2.0 * (1.0 - settings.beta2) * objective_gradient * second_bar
        )
        parameter_bar = parameter_bar + hvp(gradient_bar)
        return torch.cat((parameter_bar, first_moment_bar, second_moment_bar))

    return jvp, vjp
