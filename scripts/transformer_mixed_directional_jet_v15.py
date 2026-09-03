#!/usr/bin/env python3
"""Linear-cost mixed jet for the directional Transformer remainder.

This is an implementation independent of the degree-four block polynomial.
It propagates derivatives in two formal directions:

* ``t`` is the known correction ``z``;
* ``epsilon`` is one unknown dual direction, represented by a vector of
  coefficients over the orthogonal parameter blocks.

Only ``D_t^k`` and ``D_t^k D_epsilon`` for ``k <= 3`` are retained.  The final
``D_t^3 D_epsilon`` cross-entropy bound is exactly the three-known/one-free
contraction required by the gradient Taylor remainder.  Runtime and storage
are linear in the number of blocks rather than in the number of degree-four
monomials.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb, sqrt
from typing import Mapping

import numpy as np
import torch
from torch import Tensor

from transformer_block_envelope_v15 import (
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
    directionally_shifted_parameter_geometry,
    exact_stage_values,
    parameter_geometry,
)
from transformer_hvp_grokking import FlatSpec, TransformerConfig, unflatten_parameters


@dataclass
class MixedJet:
    value: float
    a1: float
    a2: float
    a3: float
    b0: np.ndarray
    b1: np.ndarray
    b2: np.ndarray
    b3: np.ndarray


def _zeros() -> np.ndarray:
    return np.zeros(BLOCK_COUNT, dtype=np.float64)


def constant(value: float) -> MixedJet:
    return MixedJet(value, 0.0, 0.0, 0.0, _zeros(), _zeros(), _zeros(), _zeros())


def linear_parameter(value: float, known: float, block: int, free: float = 1.0) -> MixedJet:
    b0 = _zeros()
    b0[block] = free
    return MixedJet(value, known, 0.0, 0.0, b0, _zeros(), _zeros(), _zeros())


def add(left: MixedJet, right: MixedJet) -> MixedJet:
    return MixedJet(
        left.value + right.value,
        left.a1 + right.a1,
        left.a2 + right.a2,
        left.a3 + right.a3,
        left.b0 + right.b0,
        left.b1 + right.b1,
        left.b2 + right.b2,
        left.b3 + right.b3,
    )


def product(left: MixedJet, right: MixedJet, scale: float = 1.0) -> MixedJet:
    left_a = (left.value, left.a1, left.a2, left.a3)
    right_a = (right.value, right.a1, right.a2, right.a3)
    left_b = (left.b0, left.b1, left.b2, left.b3)
    right_b = (right.b0, right.b1, right.b2, right.b3)
    out_a = []
    out_b = []
    for order in range(4):
        scalar = 0.0
        vector = _zeros()
        for split in range(order + 1):
            coefficient = float(comb(order, split))
            scalar += coefficient * left_a[split] * right_a[order - split]
            vector += coefficient * (
                left_b[split] * right_a[order - split]
                + left_a[split] * right_b[order - split]
            )
        out_a.append(scale * scalar)
        out_b.append(scale * vector)
    return MixedJet(
        out_a[0], out_a[1], out_a[2], out_a[3],
        out_b[0], out_b[1], out_b[2], out_b[3],
    )


def smooth_map(
    source: MixedJet,
    *,
    value: float,
    first: float,
    second: float,
    third: float,
    fourth: float,
) -> MixedJet:
    x1, x2, x3 = source.a1, source.a2, source.a3
    b0, b1, b2, b3 = source.b0, source.b1, source.b2, source.b3
    a1 = first * x1
    a2 = second * x1**2 + first * x2
    a3 = third * x1**3 + 3.0 * second * x1 * x2 + first * x3
    out_b0 = first * b0
    out_b1 = second * x1 * b0 + first * b1
    out_b2 = (
        third * x1**2 * b0
        + second * x2 * b0
        + 2.0 * second * x1 * b1
        + first * b2
    )
    out_b3 = (
        fourth * x1**3 * b0
        + 3.0 * third * x1**2 * b1
        + 3.0 * third * x1 * x2 * b0
        + 3.0 * second * x2 * b1
        + 3.0 * second * x1 * b2
        + second * x3 * b0
        + first * b3
    )
    return MixedJet(value, a1, a2, a3, out_b0, out_b1, out_b2, out_b3)


def affine_parameter(
    source: MixedJet,
    *,
    weight_operator: float,
    weight_known: float,
    weight_block: int,
    bias_norm: float,
    bias_known: float,
    bias_block: int,
    bias_repetitions: int,
) -> MixedJet:
    bias_scale = sqrt(float(bias_repetitions))
    weight = linear_parameter(
        weight_operator, weight_known, weight_block, free=1.0
    )
    bias = linear_parameter(
        bias_scale * bias_norm,
        bias_scale * bias_known,
        bias_block,
        free=bias_scale,
    )
    return add(product(source, weight), bias)


def parameter_block_radii(
    direction: Tensor, spec: FlatSpec, config: TransformerConfig
) -> np.ndarray:
    """Independent exact partition into the fourteen analytic blocks."""

    rows = unflatten_parameters(direction, spec)
    d = config.model_dim
    prefix = "blocks.0."
    radii = _zeros()
    radii[B_EMBED] = float(
        torch.sqrt(
            torch.dot(rows["position_embedding"].reshape(-1), rows["position_embedding"].reshape(-1))
            + torch.dot(rows["token_embedding.weight"].reshape(-1), rows["token_embedding.weight"].reshape(-1))
        )
    )
    weight = rows[prefix + "attention.in_proj_weight"]
    bias = rows[prefix + "attention.in_proj_bias"]
    for index, (weight_block, bias_block) in enumerate(
        ((B_WQ, B_BQ), (B_WK, B_BK), (B_WV, B_BV))
    ):
        sl = slice(index * d, (index + 1) * d)
        radii[weight_block] = float(torch.linalg.vector_norm(weight[sl]))
        radii[bias_block] = float(torch.linalg.vector_norm(bias[sl]))
    assignments = (
        (B_OUT_W, prefix + "attention.out_proj.weight"),
        (B_OUT_B, prefix + "attention.out_proj.bias"),
        (B_LIN1_W, prefix + "linear1.weight"),
        (B_LIN1_B, prefix + "linear1.bias"),
        (B_LIN2_W, prefix + "linear2.weight"),
        (B_LIN2_B, prefix + "linear2.bias"),
        (B_READOUT, "readout.weight"),
    )
    for block, name in assignments:
        radii[block] = float(torch.linalg.vector_norm(rows[name]))
    total = float(torch.linalg.vector_norm(direction))
    if not np.isclose(np.linalg.norm(radii), total, rtol=2.0e-13, atol=1.0e-30):
        raise AssertionError("mixed-jet parameter blocks do not partition direction")
    return radii


def _compose(
    parameter: Tensor,
    spec: FlatSpec,
    config: TransformerConfig,
    centre: Mapping[str, float],
    inflation: Mapping[str, float],
    radii: np.ndarray,
    geometry: Mapping[str, float] | None = None,
) -> tuple[MixedJet, dict[str, MixedJet]]:
    if geometry is None:
        geometry = parameter_geometry(parameter, spec, config)
    d = config.model_dim
    stages: dict[str, MixedJet] = {}

    def op(name: str, block: int) -> float:
        return float(geometry[name]) + float(radii[block])

    def vec(name: str, block: int) -> float:
        return float(geometry[name]) + float(radii[block])

    def val(name: str) -> float:
        return float(centre[name]) + float(inflation.get(name, 0.0))

    def stage(name: str, jet: MixedJet) -> MixedJet:
        jet.value = val(name)
        stages[name] = jet
        return jet

    hidden = stage(
        "embedding",
        linear_parameter(
            val("embedding"), sqrt(3.0) * radii[B_EMBED], B_EMBED, sqrt(3.0)
        ),
    )
    parts = []
    for index, (weight_block, bias_block, name) in enumerate(
        ((B_WQ, B_BQ, "query"), (B_WK, B_BK, "key"), (B_WV, B_BV, "value"))
    ):
        parts.append(
            stage(
                name,
                affine_parameter(
                    hidden,
                    weight_operator=op(f"{name}_weight", weight_block),
                    weight_known=float(radii[weight_block]),
                    weight_block=weight_block,
                    bias_norm=vec(f"{name}_bias", bias_block),
                    bias_known=float(radii[bias_block]),
                    bias_block=bias_block,
                    bias_repetitions=3,
                ),
            )
        )
    query, key, value = parts
    score = stage(
        "score", product(query, key, 1.0 / sqrt(float(d // config.heads)))
    )
    weights = stage(
        "softmax",
        smooth_map(
            score,
            value=val("softmax"),
            first=0.5,
            second=2.0,
            third=6.0,
            fourth=150.0,
        ),
    )
    attended = stage("attended", product(weights, value))
    attended = stage(
        "out_proj",
        affine_parameter(
            attended,
            weight_operator=op("out_proj_weight", B_OUT_W),
            weight_known=float(radii[B_OUT_W]),
            weight_block=B_OUT_W,
            bias_norm=vec("out_proj_bias", B_OUT_B),
            bias_known=float(radii[B_OUT_B]),
            bias_block=B_OUT_B,
            bias_repetitions=3,
        ),
    )
    hidden = stage("residual_1", add(hidden, attended))
    feedforward = stage(
        "linear1",
        affine_parameter(
            hidden,
            weight_operator=op("linear1_weight", B_LIN1_W),
            weight_known=float(radii[B_LIN1_W]),
            weight_block=B_LIN1_W,
            bias_norm=vec("linear1_bias", B_LIN1_B),
            bias_known=float(radii[B_LIN1_B]),
            bias_block=B_LIN1_B,
            bias_repetitions=3,
        ),
    )
    feedforward = stage(
        "gelu",
        smooth_map(
            feedforward,
            value=val("gelu"),
            first=1.13,
            second=0.80,
            third=2.0,
            fourth=6.0,
        ),
    )
    feedforward = stage(
        "linear2",
        affine_parameter(
            feedforward,
            weight_operator=op("linear2_weight", B_LIN2_W),
            weight_known=float(radii[B_LIN2_W]),
            weight_block=B_LIN2_W,
            bias_norm=vec("linear2_bias", B_LIN2_B),
            bias_known=float(radii[B_LIN2_B]),
            bias_block=B_LIN2_B,
            bias_repetitions=3,
        ),
    )
    hidden = stage("residual_2", add(hidden, feedforward))
    output = stage(
        "readout",
        affine_parameter(
            hidden,
            weight_operator=op("readout_weight", B_READOUT),
            weight_known=float(radii[B_READOUT]),
            weight_block=B_READOUT,
            bias_norm=0.0,
            # Match the shipped block jet's conservative convention: its
            # generic affine helper charges the readout block once more even
            # though this particular layer is bias-free.  A future tightening
            # may remove that harmless overcount under a separate protocol.
            bias_known=float(radii[B_READOUT]),
            bias_block=B_READOUT,
            bias_repetitions=1,
        ),
    )
    return output, stages


def objective_mixed_vector(output: MixedJet) -> np.ndarray:
    """Cross-entropy ``D_t^3 D_epsilon`` coefficient majorant."""

    a1, a2, a3 = output.a1, output.a2, output.a3
    return (
        26.0 * a1**3 * output.b0
        + 6.0 * a1**2 * output.b1
        + 6.0 * a1 * a2 * output.b0
        + 1.5 * a2 * output.b1
        + 1.5 * a1 * output.b2
        + 0.5 * a3 * output.b0
        + sqrt(2.0) * output.b3
    )


def directionally_transported_envelope_inputs(
    centre_values: Mapping[str, float],
    parameter_norms: Mapping[str, float],
    mixed_record: Mapping[str, object],
    *,
    path_fraction: float = 1.0,
) -> tuple[dict[str, float], dict[str, float]]:
    """Transport envelope inputs from ``c`` to the known center ``c + z``.

    The mixed jet certifies an upper bound on every activation's change along
    ``c + t z``, ``0 <= t <= 1``.  Its disjoint block radii likewise majorize
    the change in each parameter operator/vector norm.  Multiplying both by
    ``path_fraction`` supplies majorants at ``c + path_fraction * z``.  This is
    useful when one amplified direction ``lambda * z`` serves both a secant
    error bound and transport to the physical center at fraction ``1/lambda``.
    No neural evaluation at the shifted center is needed.
    """

    fraction = float(path_fraction)
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("path_fraction must lie in [0, 1]")
    inflation_raw = mixed_record.get("stage_value_inflation")
    radii_raw = mixed_record.get("block_radii")
    if not isinstance(inflation_raw, Mapping) or radii_raw is None:
        raise ValueError("mixed record lacks directional transport data")
    if set(inflation_raw) != set(centre_values):
        raise ValueError("mixed stage registry does not match centre values")
    shifted_values = {
        name: float(value) + fraction * float(inflation_raw[name])
        for name, value in centre_values.items()
    }
    if any(not np.isfinite(value) or value < 0.0 for value in shifted_values.values()):
        raise ValueError("transported stage values must be finite and nonnegative")
    shifted_geometry = directionally_shifted_parameter_geometry(
        parameter_norms,
        [fraction * float(value) for value in radii_raw],  # type: ignore[union-attr]
    )
    return shifted_values, shifted_geometry


@torch.no_grad()
def mixed_directional_objective_fourth_bound(
    parameter: Tensor,
    direction: Tensor,
    spec: FlatSpec,
    config: TransformerConfig,
    *,
    fixed_point_iterations: int = 64,
    centre_values: Mapping[str, float] | None = None,
    parameter_norms: Mapping[str, float] | None = None,
    reuse_geometry: bool = True,
) -> dict:
    if config.loss != "cross_entropy":
        raise ValueError("mixed directional jet currently covers cross entropy")
    if config.depth != 1 or config.normalization != "none":
        raise ValueError("mixed directional jet covers one normalization-free block")
    radii = parameter_block_radii(direction, spec, config)
    centre = (
        exact_stage_values(parameter, spec, config)
        if centre_values is None
        else dict(centre_values)
    )
    if parameter_norms is not None and not reuse_geometry:
        raise ValueError("parameter_norms requires reuse_geometry=True")
    geometry = None
    if reuse_geometry:
        geometry = (
            parameter_geometry(parameter, spec, config)
            if parameter_norms is None
            else dict(parameter_norms)
        )
    inflation = {name: 0.0 for name in centre}
    history = []
    consistent = False
    for _ in range(fixed_point_iterations):
        _, stages = _compose(
            parameter, spec, config, centre, inflation, radii, geometry
        )
        proposed = {name: stage.a1 for name, stage in stages.items()}
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
        _, stages = _compose(
            parameter, spec, config, centre, inflation, radii, geometry
        )
        proposed = {name: stage.a1 for name, stage in stages.items()}
        consistent = all(
            proposed[name] <= inflation[name] * (1.0 + 1.0e-9) + 1.0e-18
            for name in centre
        )
    if not consistent:
        raise RuntimeError("mixed-jet stage-value fixed point did not close")
    output, _ = _compose(
        parameter, spec, config, centre, inflation, radii, geometry
    )
    vector = objective_mixed_vector(output)
    mixed = float(np.linalg.norm(vector))
    return {
        "mixed_fourth_derivative_upper": mixed,
        "gradient_taylor_remainder_upper": mixed / 6.0,
        "block_radii": radii.tolist(),
        "direction_norm": float(np.linalg.norm(radii)),
        "mixed_block_coefficients": vector.tolist(),
        "fixed_point_iterations_used": len(history),
        "fixed_point_consistent": True,
        "maximum_stage_inflation": max(inflation.values(), default=0.0),
        "stage_value_inflation": inflation,
    }
