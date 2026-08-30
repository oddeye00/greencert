#!/usr/bin/env python3
"""Deterministic fourth-order scalar jet for the sealed smooth Transformer.

This module is deliberately separate from every v3-sealed source file. It
extends the shipped scalar derivative envelope by one order so that the
cancellation-safe quadratic second response has an explicit Taylor remainder.
The constants are conservative global Euclidean bounds:

* row-wise softmax derivatives through order four: 1/2, 2, 6, 150;
* GELU derivatives through order four: 1.13, 0.80, 2, 6;
* cross-entropy logit derivatives through order four:
  sqrt(2), 1/2, 2, 26.

The softmax fourth constant follows by dualizing to a fifth derivative of
log-sum-exp and applying the joint-cumulant partition bound
sum_k S(5,k)(k-1)! = 150. The cross-entropy fourth constant similarly uses
the order-four bound 26. These constants prioritize auditability over
tightness; the post-recentering directions are cubically small.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor

from transformer_hvp_grokking import (
    FlatSpec,
    SmoothModularTransformer,
    TransformerConfig,
    unflatten_parameters,
)


@dataclass(frozen=True)
class Jet4:
    value: float
    first: float
    second: float
    third: float
    fourth: float


def add(left: Jet4, right: Jet4) -> Jet4:
    return Jet4(
        left.value + right.value,
        left.first + right.first,
        left.second + right.second,
        left.third + right.third,
        left.fourth + right.fourth,
    )


def product(left: Jet4, right: Jet4, *, scale: float = 1.0) -> Jet4:
    return Jet4(
        scale * left.value * right.value,
        scale * (left.first * right.value + left.value * right.first),
        scale
        * (
            left.second * right.value
            + 2.0 * left.first * right.first
            + left.value * right.second
        ),
        scale
        * (
            left.third * right.value
            + 3.0 * left.second * right.first
            + 3.0 * left.first * right.second
            + left.value * right.third
        ),
        scale
        * (
            left.fourth * right.value
            + 4.0 * left.third * right.first
            + 6.0 * left.second * right.second
            + 4.0 * left.first * right.third
            + left.value * right.fourth
        ),
    )


def smooth_map(
    source: Jet4,
    *,
    value: float,
    first: float,
    second: float,
    third: float,
    fourth: float,
) -> Jet4:
    return Jet4(
        value,
        first * source.first,
        second * source.first**2 + first * source.second,
        third * source.first**3
        + 3.0 * second * source.first * source.second
        + first * source.third,
        fourth * source.first**4
        + 6.0 * third * source.first**2 * source.second
        + 3.0 * second * source.second**2
        + 4.0 * second * source.first * source.third
        + first * source.fourth,
    )


def affine_parameter(
    source: Jet4,
    *,
    weight_operator: float,
    bias_norm: float,
    bias_repetitions: int,
) -> Jet4:
    bias_scale = sqrt(float(bias_repetitions))
    weight = Jet4(weight_operator, 1.0, 0.0, 0.0, 0.0)
    bias = Jet4(bias_scale * bias_norm, bias_scale, 0.0, 0.0, 0.0)
    return add(product(source, weight), bias)


def _operator(value: Tensor, radius: float) -> float:
    return float(torch.linalg.matrix_norm(value, ord=2)) + radius


def _vector(value: Tensor, radius: float) -> float:
    return float(torch.linalg.vector_norm(value)) + radius


@torch.no_grad()
def transformer_output_fourth_jet_bound(
    parameter: Tensor,
    template: SmoothModularTransformer,
    spec: FlatSpec,
    config: TransformerConfig,
    *,
    radius: float,
) -> Jet4:
    """Return a ball-valid scalar output jet through derivative order four."""

    if radius < 0.0:
        raise ValueError("radius must be nonnegative")
    if config.normalization != "none" or config.depth != 1:
        raise ValueError(
            "the analytic fourth jet covers one normalization-free block"
        )
    rows = unflatten_parameters(parameter, spec)
    p, d, length = config.modulus, config.model_dim, 3

    token = rows["token_embedding.weight"]
    position = rows["position_embedding"]
    max_center = 0.0
    for left in range(p):
        for right in range(p):
            indices = torch.tensor((left, right, p), dtype=torch.long)
            hidden_value = token[indices] + position
            max_center = max(
                max_center, float(torch.linalg.vector_norm(hidden_value))
            )
    input_first = sqrt(3.0)
    hidden = Jet4(
        max_center + radius * input_first,
        input_first,
        0.0,
        0.0,
        0.0,
    )

    prefix = "blocks.0."
    in_weight = rows[prefix + "attention.in_proj_weight"]
    in_bias = rows[prefix + "attention.in_proj_bias"]
    qkv: list[Jet4] = []
    for index in range(3):
        sl = slice(index * d, (index + 1) * d)
        qkv.append(
            affine_parameter(
                hidden,
                weight_operator=_operator(in_weight[sl], radius),
                bias_norm=_vector(in_bias[sl], radius),
                bias_repetitions=length,
            )
        )
    query, key, value = qkv
    head_dim = d // config.heads
    score = product(query, key, scale=1.0 / sqrt(float(head_dim)))
    attention_weight = smooth_map(
        score,
        value=sqrt(float(config.heads * length)),
        first=0.5,
        second=2.0,
        third=6.0,
        fourth=150.0,
    )
    attended = product(attention_weight, value)
    attended = affine_parameter(
        attended,
        weight_operator=_operator(
            rows[prefix + "attention.out_proj.weight"], radius
        ),
        bias_norm=_vector(rows[prefix + "attention.out_proj.bias"], radius),
        bias_repetitions=length,
    )
    hidden = add(hidden, attended)
    feedforward = affine_parameter(
        hidden,
        weight_operator=_operator(rows[prefix + "linear1.weight"], radius),
        bias_norm=_vector(rows[prefix + "linear1.bias"], radius),
        bias_repetitions=length,
    )
    feedforward = smooth_map(
        feedforward,
        value=feedforward.value + sqrt(float(length * config.hidden_dim)),
        first=1.13,
        second=0.80,
        third=2.0,
        fourth=6.0,
    )
    feedforward = affine_parameter(
        feedforward,
        weight_operator=_operator(rows[prefix + "linear2.weight"], radius),
        bias_norm=_vector(rows[prefix + "linear2.bias"], radius),
        bias_repetitions=length,
    )
    hidden = add(hidden, feedforward)
    return affine_parameter(
        hidden,
        weight_operator=_operator(rows["readout.weight"], radius),
        bias_norm=0.0,
        bias_repetitions=1,
    )


def cross_entropy_objective_fourth_derivative_bound(jet: Jet4) -> float:
    """Bound the fourth derivative norm of cross entropy composed with logits."""

    return (
        26.0 * jet.first**4
        + 12.0 * jet.first**2 * jet.second
        + 1.5 * jet.second**2
        + 2.0 * jet.first * jet.third
        + sqrt(2.0) * jet.fourth
    )


@torch.no_grad()
def objective_fourth_derivative_bound(
    parameter: Tensor,
    template: SmoothModularTransformer,
    spec: FlatSpec,
    config: TransformerConfig,
    *,
    radius: float,
) -> float:
    """Ball-valid fourth objective derivative bound for cross entropy."""

    if config.loss != "cross_entropy":
        raise ValueError("the current fourth-objective formula covers cross entropy")
    jet = transformer_output_fourth_jet_bound(
        parameter, template, spec, config, radius=radius
    )
    return cross_entropy_objective_fourth_derivative_bound(jet)
