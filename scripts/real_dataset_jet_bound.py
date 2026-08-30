#!/usr/bin/env python3
"""Deterministic analytic derivative envelopes for the real-data tanh MLP."""
from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor

from block_jet_bound import (
    BlockJet,
    add,
    affine_parameter,
    block_linear,
    constant,
    product,
    smooth_map,
)
from real_dataset_mlp import ParameterSpec, unpack


B_W1, B_B1, B_W2, B_B2 = range(4)
TANH_FIRST = 1.0
TANH_SECOND = 4.0 / (3.0 * sqrt(3.0))
TANH_THIRD = 2.0


def _input_norm(features: Tensor) -> float:
    return float(torch.linalg.vector_norm(features, dim=1).max())


@torch.no_grad()
def hidden_jet(
    parameter: Tensor,
    features: Tensor,
    spec: ParameterSpec,
    radius: float,
) -> BlockJet:
    """Uniform hidden-state jet over all supplied finite inputs and a ball."""
    w1, b1, _, _ = unpack(parameter, spec)
    source = constant(_input_norm(features))
    preactivation = affine_parameter(
        source,
        weight_operator=float(torch.linalg.matrix_norm(w1, ord=2)) + radius,
        weight_block=B_W1,
        bias_norm=float(torch.linalg.vector_norm(b1)) + radius,
        bias_block=B_B1,
        bias_repetitions=1,
    )
    return smooth_map(
        preactivation,
        value=sqrt(float(spec.width)),
        first=TANH_FIRST,
        second=TANH_SECOND,
        third=TANH_THIRD,
    )


@torch.no_grad()
def output_jet_bound(
    parameter: Tensor,
    features: Tensor,
    spec: ParameterSpec,
    radius: float,
) -> dict[str, float]:
    """Uniform first/second/third logit-map derivative bounds."""
    _, _, w2, b2 = unpack(parameter, spec)
    hidden = hidden_jet(parameter, features, spec, radius)
    out = affine_parameter(
        hidden,
        weight_operator=float(torch.linalg.matrix_norm(w2, ord=2)) + radius,
        weight_block=B_W2,
        bias_norm=float(torch.linalg.vector_norm(b2)) + radius,
        bias_block=B_B2,
        bias_repetitions=1,
    )
    return {
        "value": out.value,
        "first": out.sphere(1),
        "second": out.sphere(2),
        "third": out.sphere(3),
    }


@torch.no_grad()
def margin_jet_bound(
    parameter: Tensor,
    features: Tensor,
    spec: ParameterSpec,
    label: int,
    competitor: int,
    radius: float,
) -> dict[str, float]:
    """Derivative bounds for one true-minus-competitor logit margin."""
    if label == competitor:
        raise ValueError("a margin requires two distinct classes")
    _, _, w2, b2 = unpack(parameter, spec)
    hidden = hidden_jet(parameter, features, spec, radius)
    row = block_linear(
        float(torch.linalg.vector_norm(w2[label] - w2[competitor]))
        + sqrt(2.0) * radius,
        B_W2,
        sqrt(2.0),
    )
    bias = block_linear(
        abs(float(b2[label] - b2[competitor])) + sqrt(2.0) * radius,
        B_B2,
        sqrt(2.0),
    )
    margin = add(product(hidden, row), bias)
    return {
        "value": margin.value,
        "first": margin.sphere(1),
        "second": margin.sphere(2),
        "third": margin.sphere(3),
    }


def cross_entropy_hessian_lipschitz(jet: dict[str, float]) -> float:
    """Uniform objective-Hessian Lipschitz bound from a logit jet."""
    first, second, third = jet["first"], jet["second"], jet["third"]
    return 2.0 * first**3 + 1.5 * first * second + sqrt(2.0) * third
