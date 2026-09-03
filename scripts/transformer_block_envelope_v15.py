#!/usr/bin/env python3
"""Block-aware, ball-valid derivative envelope for the smooth Transformer.

Three deterministic ingredients, no probability anywhere in this module:

1.  **Block-aware jet** (`block_jet_bound`).  Each parameter block carries its
    own perturbation radius and the bound is maximised over the sphere
    ``sum_b s_b^2 <= 1`` instead of setting every block to the full radius.

2.  **Ball-valid value chain.**  The Transformer input set is finite and
    discrete (all ``p^2`` token pairs with fixed positions), so the exact
    maximum activation norm at the centre is computable.  A centre value is not
    valid on ``B(c, eps)``, so each stage value is inflated:

        V_ball  <=  V_centre + first_ball * eps

    where ``first_ball`` is this module's own first-derivative bound.  The
    dependence is monotone, so the chain is iterated upward from the centre
    values to a fixed point; any fixed point is a valid simultaneous solution
    and the iteration only ever increases values.

3.  **Domination.**  Setting every ``s_b = 1`` and every value to the shipped
    worst-case magnitude reproduces the shipped scalar jet exactly, so the new
    envelope never exceeds the old one.  Asserted in
    ``test_block_jet_bound.py``.

The parameter-Jacobian norm at the centre is *not* probe-bounded here; that is
the separate probabilistic ingredient and lives elsewhere.  This module reports
the deterministic envelope only.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite, sqrt
from typing import Dict

import torch
from torch import Tensor

from block_jet_bound import BlockJet, add, affine_parameter, block_linear, product, smooth_map
from transformer_hvp_grokking import TransformerConfig, unflatten_parameters

# Block indices.  q/k/v slices of in_proj are disjoint sub-blocks, so giving
# them separate radii is valid: their squared norms sum into the in_proj block,
# which is itself part of the total budget.
B_EMBED = 0          # token + position, treated jointly (matches the shipped sqrt(3))
B_WQ, B_WK, B_WV = 1, 2, 3
B_BQ, B_BK, B_BV = 4, 5, 6
B_OUT_W, B_OUT_B = 7, 8
B_LIN1_W, B_LIN1_B = 9, 10
B_LIN2_W, B_LIN2_B = 11, 12
B_READOUT = 13
BLOCK_COUNT = 14

# Geometry entries and the disjoint parameter block that can change them.
# A displacement in a matrix block changes its spectral norm by at most the
# block's Frobenius norm; for vector blocks this is the ordinary triangle
# inequality.  Keeping this registry next to the block convention prevents a
# transported envelope from silently using a different partition.
GEOMETRY_BLOCK = {
    "query_weight": B_WQ,
    "query_bias": B_BQ,
    "key_weight": B_WK,
    "key_bias": B_BK,
    "value_weight": B_WV,
    "value_bias": B_BV,
    "out_proj_weight": B_OUT_W,
    "out_proj_bias": B_OUT_B,
    "linear1_weight": B_LIN1_W,
    "linear1_bias": B_LIN1_B,
    "linear2_weight": B_LIN2_W,
    "linear2_bias": B_LIN2_B,
    "readout_weight": B_READOUT,
}

TANH_SOFTMAX = dict(first=0.5, second=2.0, third=6.0)
GELU = dict(first=1.13, second=0.80, third=2.0)


@torch.no_grad()
def exact_stage_values(
    parameter: Tensor, spec, config: TransformerConfig
) -> Dict[str, float]:
    """Exact max-over-inputs activation norm at the centre, all p^2 pairs."""
    rows = unflatten_parameters(parameter, spec)
    p, d = config.modulus, config.model_dim
    prefix = "blocks.0."
    pairs = torch.cartesian_prod(torch.arange(p), torch.arange(p)).long()
    a, b = pairs[:, 0], pairs[:, 1]
    n = len(pairs)
    idx = torch.stack((a, b, torch.full_like(a, p)), dim=1)
    hidden = rows["token_embedding.weight"][idx] + rows["position_embedding"].unsqueeze(0)

    out: Dict[str, float] = {}

    def rec(name: str, tensor: Tensor) -> None:
        out[name] = float(
            torch.linalg.vector_norm(tensor.reshape(tensor.shape[0], -1), dim=1).max()
        )

    rec("embedding", hidden)
    in_w = rows[prefix + "attention.in_proj_weight"]
    in_b = rows[prefix + "attention.in_proj_bias"]
    proj = [hidden @ in_w[i * d:(i + 1) * d].T + in_b[i * d:(i + 1) * d] for i in range(3)]
    query, key, value = proj
    rec("query", query)
    rec("key", key)
    rec("value", value)

    heads, hd = config.heads, d // config.heads

    def split(t: Tensor) -> Tensor:
        return t.reshape(n, 3, heads, hd).transpose(1, 2)

    q, k, v = split(query), split(key), split(value)
    score = (q @ k.transpose(-1, -2)) / sqrt(float(hd))
    rec("score", score.reshape(n, -1))
    mask = torch.full((3, 3), float("-inf")).triu(1)
    weights = torch.softmax(score + mask, dim=-1)
    rec("softmax", weights.reshape(n, -1))
    attended = (weights @ v).transpose(1, 2).reshape(n, 3, d)
    rec("attended", attended)
    attended = attended @ rows[prefix + "attention.out_proj.weight"].T + rows[
        prefix + "attention.out_proj.bias"
    ]
    rec("out_proj", attended)
    hidden = hidden + attended
    rec("residual_1", hidden)
    ff = hidden @ rows[prefix + "linear1.weight"].T + rows[prefix + "linear1.bias"]
    rec("linear1", ff)
    ff = torch.nn.functional.gelu(ff)
    rec("gelu", ff)
    ff = ff @ rows[prefix + "linear2.weight"].T + rows[prefix + "linear2.bias"]
    rec("linear2", ff)
    hidden = hidden + ff
    rec("residual_2", hidden)
    rec("readout", hidden[:, -1, :] @ rows["readout.weight"].T)
    return out


@torch.no_grad()
def parameter_geometry(
    parameter: Tensor, spec, config: TransformerConfig
) -> Dict[str, float]:
    """Cache parameter operator/vector norms shared by envelope iterations.

    The geometry is independent of the fixed-point stage-value inflation and
    of the enclosing radius.  Computing it once avoids repeating the same
    spectral decompositions in every monotone envelope iteration.  Callers
    may also share the returned dictionary with compatible directional jets.
    """

    rows = unflatten_parameters(parameter, spec)
    d = config.model_dim
    prefix = "blocks.0."
    in_weight = rows[prefix + "attention.in_proj_weight"]
    in_bias = rows[prefix + "attention.in_proj_bias"]
    geometry: Dict[str, float] = {}
    for index, name in enumerate(("query", "key", "value")):
        sl = slice(index * d, (index + 1) * d)
        geometry[f"{name}_weight"] = float(
            torch.linalg.matrix_norm(in_weight[sl], ord=2)
        )
        geometry[f"{name}_bias"] = float(torch.linalg.vector_norm(in_bias[sl]))
    for name, key, matrix in (
        ("out_proj_weight", prefix + "attention.out_proj.weight", True),
        ("out_proj_bias", prefix + "attention.out_proj.bias", False),
        ("linear1_weight", prefix + "linear1.weight", True),
        ("linear1_bias", prefix + "linear1.bias", False),
        ("linear2_weight", prefix + "linear2.weight", True),
        ("linear2_bias", prefix + "linear2.bias", False),
        ("readout_weight", "readout.weight", True),
    ):
        geometry[name] = float(
            torch.linalg.matrix_norm(rows[key], ord=2)
            if matrix
            else torch.linalg.vector_norm(rows[key])
        )
    return geometry


def directionally_shifted_parameter_geometry(
    centre_geometry: Mapping[str, float],
    block_displacement_radii: Sequence[float],
) -> Dict[str, float]:
    """Majorize parameter geometry after one known signed displacement.

    If ``z_b`` is the displacement in a disjoint matrix block, then
    ``||W + z_b||_2 <= ||W||_2 + ||z_b||_F``.  The same formula is the
    Euclidean triangle inequality for bias vectors.  Thus the returned
    dictionary is valid at the shifted center without another spectral
    decomposition.
    """

    if set(centre_geometry) != set(GEOMETRY_BLOCK):
        missing = sorted(set(GEOMETRY_BLOCK) - set(centre_geometry))
        extra = sorted(set(centre_geometry) - set(GEOMETRY_BLOCK))
        raise ValueError(
            f"unexpected geometry registry (missing={missing}, extra={extra})"
        )
    if len(block_displacement_radii) != BLOCK_COUNT:
        raise ValueError(
            f"expected {BLOCK_COUNT} block radii, got {len(block_displacement_radii)}"
        )
    radii = [float(value) for value in block_displacement_radii]
    if any(not isfinite(value) or value < 0.0 for value in radii):
        raise ValueError("block displacement radii must be finite and nonnegative")
    result: Dict[str, float] = {}
    for name, block in GEOMETRY_BLOCK.items():
        value = float(centre_geometry[name])
        if not isfinite(value) or value < 0.0:
            raise ValueError("centre geometry must be finite and nonnegative")
        result[name] = value + radii[block]
    return result


@torch.no_grad()
def anchor_majorized_parameter_geometry(
    parameter: Tensor,
    anchor_parameter: Tensor,
    spec,
    config: TransformerConfig,
    *,
    anchor_norms: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """Upper-bound checkpoint geometry from one anchor eigensolve.

    For a matrix block ``W``, ``||W||_2 <= ||W0||_2 + ||W-W0||_F``; for a
    vector block the same statement is the ordinary triangle inequality.
    Thus every post-anchor spectral norm can be replaced by one cached anchor
    norm plus a linear-cost displacement norm.  The returned entries dominate
    :func:`parameter_geometry` entrywise and can be passed directly to the
    nonnegative block-jet majorants.
    """

    if parameter.shape != anchor_parameter.shape or parameter.ndim != 1:
        raise ValueError("parameter and anchor_parameter must be equally shaped vectors")
    base = (
        parameter_geometry(anchor_parameter, spec, config)
        if anchor_norms is None
        else dict(anchor_norms)
    )
    current = unflatten_parameters(parameter, spec)
    anchor = unflatten_parameters(anchor_parameter, spec)
    d = config.model_dim
    prefix = "blocks.0."
    current_in_weight = current[prefix + "attention.in_proj_weight"]
    anchor_in_weight = anchor[prefix + "attention.in_proj_weight"]
    current_in_bias = current[prefix + "attention.in_proj_bias"]
    anchor_in_bias = anchor[prefix + "attention.in_proj_bias"]
    geometry: Dict[str, float] = {}

    def displacement(key: str) -> float:
        return float(torch.linalg.vector_norm((current[key] - anchor[key]).reshape(-1)))

    for index, name in enumerate(("query", "key", "value")):
        sl = slice(index * d, (index + 1) * d)
        geometry[f"{name}_weight"] = base[f"{name}_weight"] + float(
            torch.linalg.vector_norm(
                (current_in_weight[sl] - anchor_in_weight[sl]).reshape(-1)
            )
        )
        geometry[f"{name}_bias"] = base[f"{name}_bias"] + float(
            torch.linalg.vector_norm(current_in_bias[sl] - anchor_in_bias[sl])
        )
    for name, key in (
        ("out_proj_weight", prefix + "attention.out_proj.weight"),
        ("out_proj_bias", prefix + "attention.out_proj.bias"),
        ("linear1_weight", prefix + "linear1.weight"),
        ("linear1_bias", prefix + "linear1.bias"),
        ("linear2_weight", prefix + "linear2.weight"),
        ("linear2_bias", prefix + "linear2.bias"),
        ("readout_weight", "readout.weight"),
    ):
        geometry[name] = base[name] + displacement(key)
    return geometry


@torch.no_grad()
def _compose(
    parameter: Tensor,
    spec,
    config: TransformerConfig,
    centre: Dict[str, float],
    inflation: Dict[str, float],
    radius: float,
    *,
    exact_values: bool,
    sphere: bool,
    stage_jets: Dict[str, BlockJet] | None = None,
    geometry: Dict[str, float] | None = None,
) -> BlockJet:
    """One pass of the block jet with a given value assignment."""
    if geometry is None:
        geometry = parameter_geometry(parameter, spec, config)
    d = config.model_dim

    def op(name: str) -> float:
        return geometry[name] + radius

    def vec(name: str) -> float:
        return geometry[name] + radius

    def val(name: str, fallback: float) -> float:
        if not exact_values:
            return fallback
        return centre[name] + inflation.get(name, 0.0)

    def stage(name: str, jet: BlockJet) -> BlockJet:
        if stage_jets is not None:
            stage_jets[name] = jet
        return jet

    hidden = stage(
        "embedding",
        block_linear(
            val("embedding", centre["embedding"] + radius * sqrt(3.0)),
            B_EMBED,
            sqrt(3.0),
        ),
    )

    parts = []
    for index, (wb, bb, name) in enumerate(
        ((B_WQ, B_BQ, "query"), (B_WK, B_BK, "key"), (B_WV, B_BV, "value"))
    ):
        jet = affine_parameter(
            hidden,
            weight_operator=op(f"{name}_weight"),
            weight_block=wb,
            bias_norm=vec(f"{name}_bias"),
            bias_block=bb,
            bias_repetitions=3,
        )
        if exact_values:
            jet = BlockJet(val(name, jet.value), jet.p1, jet.p2, jet.p3)
        parts.append(stage(name, jet))
    query, key, value = parts

    head_dim = d // config.heads
    score = product(query, key, scale=1.0 / sqrt(float(head_dim)))
    if exact_values:
        score = BlockJet(val("score", score.value), score.p1, score.p2, score.p3)
    score = stage("score", score)

    weights = smooth_map(
        score, value=val("softmax", sqrt(float(config.heads * 3))), **TANH_SOFTMAX
    )
    weights = stage("softmax", weights)
    attended = product(weights, value)
    if exact_values:
        attended = BlockJet(val("attended", attended.value), attended.p1, attended.p2, attended.p3)
    attended = stage("attended", attended)
    attended = affine_parameter(
        attended,
        weight_operator=op("out_proj_weight"),
        weight_block=B_OUT_W,
        bias_norm=vec("out_proj_bias"),
        bias_block=B_OUT_B,
        bias_repetitions=3,
    )
    if exact_values:
        attended = BlockJet(val("out_proj", attended.value), attended.p1, attended.p2, attended.p3)
    attended = stage("out_proj", attended)
    hidden = add(hidden, attended)
    if exact_values:
        hidden = BlockJet(val("residual_1", hidden.value), hidden.p1, hidden.p2, hidden.p3)
    hidden = stage("residual_1", hidden)

    ff = affine_parameter(
        hidden,
        weight_operator=op("linear1_weight"),
        weight_block=B_LIN1_W,
        bias_norm=vec("linear1_bias"),
        bias_block=B_LIN1_B,
        bias_repetitions=3,
    )
    if exact_values:
        ff = BlockJet(val("linear1", ff.value), ff.p1, ff.p2, ff.p3)
    ff = stage("linear1", ff)
    ff = smooth_map(
        ff,
        value=val("gelu", ff.value + sqrt(float(3 * config.hidden_dim))),
        **GELU,
    )
    ff = stage("gelu", ff)
    ff = affine_parameter(
        ff,
        weight_operator=op("linear2_weight"),
        weight_block=B_LIN2_W,
        bias_norm=vec("linear2_bias"),
        bias_block=B_LIN2_B,
        bias_repetitions=3,
    )
    if exact_values:
        ff = BlockJet(val("linear2", ff.value), ff.p1, ff.p2, ff.p3)
    ff = stage("linear2", ff)
    hidden = add(hidden, ff)
    if exact_values:
        hidden = BlockJet(val("residual_2", hidden.value), hidden.p1, hidden.p2, hidden.p3)
    hidden = stage("residual_2", hidden)

    out = affine_parameter(
        hidden,
        weight_operator=op("readout_weight"),
        weight_block=B_READOUT,
        bias_norm=0.0,
        bias_block=B_READOUT,
        bias_repetitions=1,
    )
    if exact_values:
        out = BlockJet(val("readout", out.value), out.p1, out.p2, out.p3)
    return stage("readout", out)


@torch.no_grad()
def ball_valid_envelope(
    parameter: Tensor,
    spec,
    config: TransformerConfig,
    *,
    epsilon: float,
    exact_values: bool = True,
    sphere: bool = True,
    iterations: int = 64,
    centre_values: Dict[str, float] | None = None,
    parameter_norms: Dict[str, float] | None = None,
    reuse_geometry: bool = True,
) -> dict:
    """Fixed-point inflation of center majorants over ``B(c, epsilon)``.

    By default, exact activation maxima and parameter norms are computed at
    ``parameter``.  Supplied ``centre_values`` and ``parameter_norms`` may
    instead be any certified entrywise upper bounds at the intended center.
    The update ``V_ball <= V_centre + first_ball * epsilon`` is iterated upward.
    Values only increase, so the iterate is monotone; the returned assignment
    is checked to be a valid simultaneous bound.  The majorant interface lets
    a known directional correction be transported into the center without a
    second forward pass or spectral decomposition.
    """
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
    reduce = (lambda j, o: j.sphere(o)) if sphere else (lambda j, o: j.scalar(o))

    history = []
    for _ in range(iterations):
        stages: Dict[str, BlockJet] = {}
        jet = _compose(
            parameter, spec, config, centre, inflation, epsilon,
            exact_values=exact_values, sphere=sphere,
            stage_jets=stages,
            geometry=geometry,
        )
        stage_first = {name: reduce(stages[name], 1) for name in centre}
        first = stage_first["readout"]
        new_inflation = {
            name: stage_first[name] * epsilon for name in centre
        }
        history.append(max(stage_first.values()))
        if all(
            new_inflation[name] <= inflation[name] * (1 + 1e-12) + 1e-18
            for name in centre
        ):
            break
        inflation = {
            name: max(inflation[name], new_inflation[name]) for name in centre
        }

    stages = {}
    jet = _compose(
        parameter, spec, config, centre, inflation, epsilon,
        exact_values=exact_values, sphere=sphere,
        stage_jets=stages,
        geometry=geometry,
    )
    first = reduce(jet, 1)
    second = reduce(jet, 2)
    third = reduce(jet, 3)
    stage_first = {name: reduce(stages[name], 1) for name in centre}
    consistent = all(
        stage_first[name] * epsilon
        <= inflation[name] * (1 + 1e-9) + 1e-18
        for name in centre
    )
    return {
        "value": jet.value,
        "first": first,
        "second": second,
        "third": third,
        "centre_values": centre,
        "inflation": inflation,
        "stage_first": stage_first,
        "fixed_point_consistent": bool(consistent),
        "first_iterations": history,
        "fixed_point_iterations_used": len(history),
        "jet": jet,
    }


def objective_hessian_lipschitz(first: float, second: float, third: float) -> float:
    """Cross-entropy composition, identical in form to the shipped bound."""
    return 2.0 * first**3 + 1.5 * first * second + sqrt(2.0) * third
