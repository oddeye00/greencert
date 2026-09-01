#!/usr/bin/env python3
"""Directional block-majorant for the Transformer fourth-order remainder.

The scalar fourth-order envelope bounds ``||D^4 F|| ||z||^3`` after forgetting
the block geometry of the three *known* correction directions.  This module
retains that geometry.  It constructs a nonnegative homogeneous polynomial
``P4(s)`` that coefficientwise majorizes the fourth derivative of the
cross-entropy objective, where ``s_b`` is the norm assigned to parameter block
``b``.  If ``r_b = ||z_b||``, symmetry and multilinearity give

    sup_{||u||_2 <= 1} |D^4 F(theta)[z,z,z,u]|
        <= ||grad P4(r)||_2 / 4.

The factor ``1/4`` appears because differentiating the diagonal fourth form
inserts the free direction in each of four slots.  Thus the cubic Taylor
remainder of the objective gradient is at most ``||grad P4(r)||_2 / 24``.

Unlike the global ball envelope, values and parameter operator norms are
inflated only over the realized segment ``theta + t z, 0 <= t <= 1``.  A cheap
first-order fixed point establishes simultaneous stage-value bounds; the full
fourth-order polynomial is built only once after that fixed point converges.
No random quantity or future training outcome is used here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Dict, Mapping, Tuple

import numpy as np
import torch
from torch import Tensor

from transformer_block_envelope import (
    BLOCK_COUNT,
    B_BK,
    B_BQ,
    B_BV,
    B_EMBED,
    B_LIN1_B,
    B_LIN1_W,
    B_LIN2_B,
    B_LIN2_W,
    B_OUT_B,
    B_OUT_W,
    B_READOUT,
    B_WK,
    B_WQ,
    B_WV,
    exact_stage_values,
)
from transformer_hvp_grokking import FlatSpec, TransformerConfig, unflatten_parameters


Monomial = Tuple[int, ...]
Polynomial = Dict[Monomial, float]


def _merge(target: Polynomial, source: Polynomial, scale: float) -> None:
    for key, coefficient in source.items():
        value = scale * coefficient
        if value != 0.0:
            target[key] = target.get(key, 0.0) + value


def _convolve(left: Polynomial, right: Polynomial, scale: float = 1.0) -> Polynomial:
    out: Polynomial = {}
    for left_key, left_value in left.items():
        if left_value == 0.0:
            continue
        for right_key, right_value in right.items():
            if right_value == 0.0:
                continue
            key = tuple(sorted(left_key + right_key))
            out[key] = out.get(key, 0.0) + scale * left_value * right_value
    return out


def evaluate_polynomial(polynomial: Mapping[Monomial, float], radii: np.ndarray) -> float:
    total = 0.0
    for monomial, coefficient in polynomial.items():
        term = coefficient
        for block in monomial:
            term *= float(radii[block])
        total += term
    return float(total)


def homogeneous_gradient(
    polynomial: Mapping[Monomial, float], radii: np.ndarray, block_count: int
) -> np.ndarray:
    """Gradient of a nonnegative homogeneous polynomial, including zero radii."""

    gradient = np.zeros(block_count, dtype=np.float64)
    for monomial, coefficient in polynomial.items():
        if len(monomial) == 0:
            continue
        counts: dict[int, int] = {}
        for block in monomial:
            counts[block] = counts.get(block, 0) + 1
        for differentiated, multiplicity in counts.items():
            term = coefficient * multiplicity
            for block, exponent in counts.items():
                remaining = exponent - (1 if block == differentiated else 0)
                if remaining:
                    term *= float(radii[block]) ** remaining
            gradient[differentiated] += term
    return gradient


@dataclass
class FirstJet:
    value: float
    first: np.ndarray


def _first_constant(value: float) -> FirstJet:
    return FirstJet(value, np.zeros(BLOCK_COUNT, dtype=np.float64))


def _first_block(value: float, block: int, coefficient: float = 1.0) -> FirstJet:
    first = np.zeros(BLOCK_COUNT, dtype=np.float64)
    first[block] = coefficient
    return FirstJet(value, first)


def _first_add(left: FirstJet, right: FirstJet) -> FirstJet:
    return FirstJet(left.value + right.value, left.first + right.first)


def _first_product(left: FirstJet, right: FirstJet, scale: float = 1.0) -> FirstJet:
    return FirstJet(
        scale * left.value * right.value,
        scale * (left.first * right.value + right.first * left.value),
    )


def _first_smooth(source: FirstJet, value: float, first: float) -> FirstJet:
    return FirstJet(value, first * source.first)


def _first_affine(
    source: FirstJet,
    *,
    weight_operator: float,
    weight_block: int,
    bias_norm: float,
    bias_block: int,
    bias_repetitions: int,
) -> FirstJet:
    bias_scale = sqrt(float(bias_repetitions))
    weight = _first_block(weight_operator, weight_block)
    bias = _first_block(bias_scale * bias_norm, bias_block, bias_scale)
    return _first_add(_first_product(source, weight), bias)


@dataclass
class BlockJet4:
    value: float
    p1: Polynomial = field(default_factory=dict)
    p2: Polynomial = field(default_factory=dict)
    p3: Polynomial = field(default_factory=dict)
    p4: Polynomial = field(default_factory=dict)


def constant(value: float) -> BlockJet4:
    return BlockJet4(value)


def block_linear(value: float, block: int, coefficient: float = 1.0) -> BlockJet4:
    return BlockJet4(value, {(block,): coefficient})


def add(left: BlockJet4, right: BlockJet4) -> BlockJet4:
    out = BlockJet4(left.value + right.value)
    for destination, left_source, right_source in (
        (out.p1, left.p1, right.p1),
        (out.p2, left.p2, right.p2),
        (out.p3, left.p3, right.p3),
        (out.p4, left.p4, right.p4),
    ):
        _merge(destination, left_source, 1.0)
        _merge(destination, right_source, 1.0)
    return out


def product(left: BlockJet4, right: BlockJet4, scale: float = 1.0) -> BlockJet4:
    out = BlockJet4(scale * left.value * right.value)
    _merge(out.p1, left.p1, scale * right.value)
    _merge(out.p1, right.p1, scale * left.value)

    _merge(out.p2, left.p2, scale * right.value)
    _merge(out.p2, right.p2, scale * left.value)
    _merge(out.p2, _convolve(left.p1, right.p1), 2.0 * scale)

    _merge(out.p3, left.p3, scale * right.value)
    _merge(out.p3, right.p3, scale * left.value)
    _merge(out.p3, _convolve(left.p2, right.p1), 3.0 * scale)
    _merge(out.p3, _convolve(left.p1, right.p2), 3.0 * scale)

    _merge(out.p4, left.p4, scale * right.value)
    _merge(out.p4, right.p4, scale * left.value)
    _merge(out.p4, _convolve(left.p3, right.p1), 4.0 * scale)
    _merge(out.p4, _convolve(left.p1, right.p3), 4.0 * scale)
    _merge(out.p4, _convolve(left.p2, right.p2), 6.0 * scale)
    return out


def smooth_map(
    source: BlockJet4,
    *,
    value: float,
    first: float,
    second: float,
    third: float,
    fourth: float,
) -> BlockJet4:
    out = BlockJet4(value)
    _merge(out.p1, source.p1, first)

    _merge(out.p2, source.p2, first)
    _merge(out.p2, _convolve(source.p1, source.p1), second)

    _merge(out.p3, source.p3, first)
    _merge(out.p3, _convolve(source.p1, source.p2), 3.0 * second)
    _merge(out.p3, _convolve(_convolve(source.p1, source.p1), source.p1), third)

    _merge(out.p4, source.p4, first)
    _merge(out.p4, _convolve(source.p1, source.p3), 4.0 * second)
    _merge(out.p4, _convolve(source.p2, source.p2), 3.0 * second)
    _merge(
        out.p4,
        _convolve(_convolve(source.p1, source.p1), source.p2),
        6.0 * third,
    )
    _merge(
        out.p4,
        _convolve(_convolve(_convolve(source.p1, source.p1), source.p1), source.p1),
        fourth,
    )
    return out


def affine_parameter(
    source: BlockJet4,
    *,
    weight_operator: float,
    weight_block: int,
    bias_norm: float,
    bias_block: int,
    bias_repetitions: int,
) -> BlockJet4:
    bias_scale = sqrt(float(bias_repetitions))
    weight = block_linear(weight_operator, weight_block)
    bias = block_linear(bias_scale * bias_norm, bias_block, bias_scale)
    return add(product(source, weight), bias)


def parameter_block_radii(
    direction: Tensor, spec: FlatSpec, config: TransformerConfig
) -> np.ndarray:
    """Exact Euclidean norms of the fourteen disjoint analytic blocks."""

    rows = unflatten_parameters(direction, spec)
    prefix = "blocks.0."
    d = config.model_dim
    radii = np.zeros(BLOCK_COUNT, dtype=np.float64)

    def squared(*values: Tensor) -> float:
        return sum(float(torch.dot(value.reshape(-1), value.reshape(-1))) for value in values)

    radii[B_EMBED] = sqrt(
        squared(rows["position_embedding"], rows["token_embedding.weight"])
    )
    in_weight = rows[prefix + "attention.in_proj_weight"]
    in_bias = rows[prefix + "attention.in_proj_bias"]
    for index, (weight_block, bias_block) in enumerate(
        ((B_WQ, B_BQ), (B_WK, B_BK), (B_WV, B_BV))
    ):
        sl = slice(index * d, (index + 1) * d)
        radii[weight_block] = float(torch.linalg.vector_norm(in_weight[sl]))
        radii[bias_block] = float(torch.linalg.vector_norm(in_bias[sl]))
    radii[B_OUT_W] = float(
        torch.linalg.vector_norm(rows[prefix + "attention.out_proj.weight"])
    )
    radii[B_OUT_B] = float(
        torch.linalg.vector_norm(rows[prefix + "attention.out_proj.bias"])
    )
    radii[B_LIN1_W] = float(torch.linalg.vector_norm(rows[prefix + "linear1.weight"]))
    radii[B_LIN1_B] = float(torch.linalg.vector_norm(rows[prefix + "linear1.bias"]))
    radii[B_LIN2_W] = float(torch.linalg.vector_norm(rows[prefix + "linear2.weight"]))
    radii[B_LIN2_B] = float(torch.linalg.vector_norm(rows[prefix + "linear2.bias"]))
    radii[B_READOUT] = float(torch.linalg.vector_norm(rows["readout.weight"]))
    total = float(torch.linalg.vector_norm(direction))
    if not np.isclose(float(np.linalg.norm(radii)), total, rtol=2.0e-13, atol=1.0e-30):
        raise AssertionError("analytic parameter blocks do not partition the direction")
    return radii


def _parameter_values(parameter: Tensor, spec: FlatSpec) -> dict[str, Tensor]:
    return dict(unflatten_parameters(parameter, spec))


def _compose_first(
    parameter: Tensor,
    spec: FlatSpec,
    config: TransformerConfig,
    centre: Mapping[str, float],
    inflation: Mapping[str, float],
    radii: np.ndarray,
) -> dict[str, FirstJet]:
    rows = _parameter_values(parameter, spec)
    d = config.model_dim
    prefix = "blocks.0."
    stages: dict[str, FirstJet] = {}

    def op(value: Tensor, block: int) -> float:
        return float(torch.linalg.matrix_norm(value, ord=2)) + float(radii[block])

    def vec(value: Tensor, block: int) -> float:
        return float(torch.linalg.vector_norm(value)) + float(radii[block])

    def val(name: str) -> float:
        return float(centre[name]) + float(inflation.get(name, 0.0))

    def stage(name: str, jet: FirstJet) -> FirstJet:
        jet.value = val(name)
        stages[name] = jet
        return jet

    hidden = stage("embedding", _first_block(val("embedding"), B_EMBED, sqrt(3.0)))
    in_weight = rows[prefix + "attention.in_proj_weight"]
    in_bias = rows[prefix + "attention.in_proj_bias"]
    parts = []
    for index, (weight_block, bias_block, name) in enumerate(
        ((B_WQ, B_BQ, "query"), (B_WK, B_BK, "key"), (B_WV, B_BV, "value"))
    ):
        sl = slice(index * d, (index + 1) * d)
        parts.append(
            stage(
                name,
                _first_affine(
                    hidden,
                    weight_operator=op(in_weight[sl], weight_block),
                    weight_block=weight_block,
                    bias_norm=vec(in_bias[sl], bias_block),
                    bias_block=bias_block,
                    bias_repetitions=3,
                ),
            )
        )
    query, key, value = parts
    score = stage(
        "score", _first_product(query, key, 1.0 / sqrt(float(d // config.heads)))
    )
    weights = stage(
        "softmax",
        _first_smooth(score, val("softmax"), 0.5),
    )
    attended = stage("attended", _first_product(weights, value))
    attended = stage(
        "out_proj",
        _first_affine(
            attended,
            weight_operator=op(rows[prefix + "attention.out_proj.weight"], B_OUT_W),
            weight_block=B_OUT_W,
            bias_norm=vec(rows[prefix + "attention.out_proj.bias"], B_OUT_B),
            bias_block=B_OUT_B,
            bias_repetitions=3,
        ),
    )
    hidden = stage("residual_1", _first_add(hidden, attended))
    feedforward = stage(
        "linear1",
        _first_affine(
            hidden,
            weight_operator=op(rows[prefix + "linear1.weight"], B_LIN1_W),
            weight_block=B_LIN1_W,
            bias_norm=vec(rows[prefix + "linear1.bias"], B_LIN1_B),
            bias_block=B_LIN1_B,
            bias_repetitions=3,
        ),
    )
    feedforward = stage("gelu", _first_smooth(feedforward, val("gelu"), 1.13))
    feedforward = stage(
        "linear2",
        _first_affine(
            feedforward,
            weight_operator=op(rows[prefix + "linear2.weight"], B_LIN2_W),
            weight_block=B_LIN2_W,
            bias_norm=vec(rows[prefix + "linear2.bias"], B_LIN2_B),
            bias_block=B_LIN2_B,
            bias_repetitions=3,
        ),
    )
    hidden = stage("residual_2", _first_add(hidden, feedforward))
    stage(
        "readout",
        _first_affine(
            hidden,
            weight_operator=op(rows["readout.weight"], B_READOUT),
            weight_block=B_READOUT,
            bias_norm=0.0,
            bias_block=B_READOUT,
            bias_repetitions=1,
        ),
    )
    return stages


def _compose_fourth(
    parameter: Tensor,
    spec: FlatSpec,
    config: TransformerConfig,
    centre: Mapping[str, float],
    inflation: Mapping[str, float],
    radii: np.ndarray,
) -> BlockJet4:
    rows = _parameter_values(parameter, spec)
    d = config.model_dim
    prefix = "blocks.0."

    def op(value: Tensor, block: int) -> float:
        return float(torch.linalg.matrix_norm(value, ord=2)) + float(radii[block])

    def vec(value: Tensor, block: int) -> float:
        return float(torch.linalg.vector_norm(value)) + float(radii[block])

    def val(name: str) -> float:
        return float(centre[name]) + float(inflation.get(name, 0.0))

    def recenter(name: str, jet: BlockJet4) -> BlockJet4:
        jet.value = val(name)
        return jet

    hidden = block_linear(val("embedding"), B_EMBED, sqrt(3.0))
    in_weight = rows[prefix + "attention.in_proj_weight"]
    in_bias = rows[prefix + "attention.in_proj_bias"]
    parts = []
    for index, (weight_block, bias_block, name) in enumerate(
        ((B_WQ, B_BQ, "query"), (B_WK, B_BK, "key"), (B_WV, B_BV, "value"))
    ):
        sl = slice(index * d, (index + 1) * d)
        parts.append(
            recenter(
                name,
                affine_parameter(
                    hidden,
                    weight_operator=op(in_weight[sl], weight_block),
                    weight_block=weight_block,
                    bias_norm=vec(in_bias[sl], bias_block),
                    bias_block=bias_block,
                    bias_repetitions=3,
                ),
            )
        )
    query, key, value = parts
    score = recenter(
        "score", product(query, key, 1.0 / sqrt(float(d // config.heads)))
    )
    weights = smooth_map(
        score,
        value=val("softmax"),
        first=0.5,
        second=2.0,
        third=6.0,
        fourth=150.0,
    )
    attended = recenter("attended", product(weights, value))
    attended = recenter(
        "out_proj",
        affine_parameter(
            attended,
            weight_operator=op(rows[prefix + "attention.out_proj.weight"], B_OUT_W),
            weight_block=B_OUT_W,
            bias_norm=vec(rows[prefix + "attention.out_proj.bias"], B_OUT_B),
            bias_block=B_OUT_B,
            bias_repetitions=3,
        ),
    )
    hidden = recenter("residual_1", add(hidden, attended))
    feedforward = recenter(
        "linear1",
        affine_parameter(
            hidden,
            weight_operator=op(rows[prefix + "linear1.weight"], B_LIN1_W),
            weight_block=B_LIN1_W,
            bias_norm=vec(rows[prefix + "linear1.bias"], B_LIN1_B),
            bias_block=B_LIN1_B,
            bias_repetitions=3,
        ),
    )
    feedforward = smooth_map(
        feedforward,
        value=val("gelu"),
        first=1.13,
        second=0.80,
        third=2.0,
        fourth=6.0,
    )
    feedforward = recenter(
        "linear2",
        affine_parameter(
            feedforward,
            weight_operator=op(rows[prefix + "linear2.weight"], B_LIN2_W),
            weight_block=B_LIN2_W,
            bias_norm=vec(rows[prefix + "linear2.bias"], B_LIN2_B),
            bias_block=B_LIN2_B,
            bias_repetitions=3,
        ),
    )
    hidden = recenter("residual_2", add(hidden, feedforward))
    return recenter(
        "readout",
        affine_parameter(
            hidden,
            weight_operator=op(rows["readout.weight"], B_READOUT),
            weight_block=B_READOUT,
            bias_norm=0.0,
            bias_block=B_READOUT,
            bias_repetitions=1,
        ),
    )


def objective_fourth_polynomial(output: BlockJet4) -> Polynomial:
    """Fourth cross-entropy derivative majorant, coefficient by coefficient."""

    out: Polynomial = {}
    p1_squared = _convolve(output.p1, output.p1)
    _merge(out, _convolve(p1_squared, p1_squared), 26.0)
    _merge(out, _convolve(p1_squared, output.p2), 12.0)
    _merge(out, _convolve(output.p2, output.p2), 1.5)
    _merge(out, _convolve(output.p1, output.p3), 2.0)
    _merge(out, output.p4, sqrt(2.0))
    if any(len(monomial) != 4 for monomial in out):
        raise AssertionError("objective fourth polynomial is not homogeneous")
    if any(coefficient < 0.0 or not np.isfinite(coefficient) for coefficient in out.values()):
        raise AssertionError("objective fourth polynomial is not a finite majorant")
    return out


@torch.no_grad()
def directional_objective_fourth_bound(
    parameter: Tensor,
    direction: Tensor,
    spec: FlatSpec,
    config: TransformerConfig,
    *,
    fixed_point_iterations: int = 64,
) -> dict:
    """Bound ``sup_t ||D4 F(theta+t z)[z,z,z,.]||`` on ``0<=t<=1``."""

    if config.loss != "cross_entropy":
        raise ValueError("the directional fourth bound currently covers cross entropy")
    if config.depth != 1 or config.normalization != "none":
        raise ValueError("the directional fourth bound covers one normalization-free block")
    radii = parameter_block_radii(direction, spec, config)
    centre = exact_stage_values(parameter, spec, config)
    inflation = {name: 0.0 for name in centre}
    history: list[float] = []
    consistent = False
    for _ in range(fixed_point_iterations):
        stages = _compose_first(parameter, spec, config, centre, inflation, radii)
        proposed = {
            name: float(np.dot(stage.first, radii)) for name, stage in stages.items()
        }
        history.append(max(proposed.values(), default=0.0))
        if all(
            proposed[name] <= inflation[name] * (1.0 + 1.0e-12) + 1.0e-18
            for name in centre
        ):
            consistent = True
            break
        inflation = {
            name: max(inflation[name], proposed[name]) for name in centre
        }
    if not consistent:
        stages = _compose_first(parameter, spec, config, centre, inflation, radii)
        proposed = {
            name: float(np.dot(stage.first, radii)) for name, stage in stages.items()
        }
        consistent = all(
            proposed[name] <= inflation[name] * (1.0 + 1.0e-9) + 1.0e-18
            for name in centre
        )
    if not consistent:
        raise RuntimeError("directional stage-value fixed point did not close")

    output = _compose_fourth(parameter, spec, config, centre, inflation, radii)
    polynomial = objective_fourth_polynomial(output)
    gradient = homogeneous_gradient(polynomial, radii, BLOCK_COUNT)
    mixed_bound = float(np.linalg.norm(gradient) / 4.0)
    taylor_remainder = mixed_bound / 6.0
    return {
        "mixed_fourth_derivative_upper": mixed_bound,
        "gradient_taylor_remainder_upper": taylor_remainder,
        "block_radii": radii.tolist(),
        "direction_norm": float(np.linalg.norm(radii)),
        "objective_polynomial_at_direction": evaluate_polynomial(polynomial, radii),
        "objective_polynomial_terms": len(polynomial),
        "stage_value_inflation": inflation,
        "fixed_point_iterations_used": len(history),
        "fixed_point_consistent": True,
        "maximum_stage_inflation": max(inflation.values(), default=0.0),
    }

