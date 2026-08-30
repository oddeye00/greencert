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

from math import sqrt
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
) -> BlockJet:
    """One pass of the block jet with a given value assignment."""
    rows = unflatten_parameters(parameter, spec)
    d = config.model_dim
    prefix = "blocks.0."

    def op(t: Tensor) -> float:
        return float(torch.linalg.matrix_norm(t, ord=2)) + radius

    def vec(t: Tensor) -> float:
        return float(torch.linalg.vector_norm(t)) + radius

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

    in_w = rows[prefix + "attention.in_proj_weight"]
    in_b = rows[prefix + "attention.in_proj_bias"]
    parts = []
    for index, (wb, bb, name) in enumerate(
        ((B_WQ, B_BQ, "query"), (B_WK, B_BK, "key"), (B_WV, B_BV, "value"))
    ):
        sl = slice(index * d, (index + 1) * d)
        jet = affine_parameter(
            hidden,
            weight_operator=op(in_w[sl]),
            weight_block=wb,
            bias_norm=vec(in_b[sl]),
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
        weight_operator=op(rows[prefix + "attention.out_proj.weight"]),
        weight_block=B_OUT_W,
        bias_norm=vec(rows[prefix + "attention.out_proj.bias"]),
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
        weight_operator=op(rows[prefix + "linear1.weight"]),
        weight_block=B_LIN1_W,
        bias_norm=vec(rows[prefix + "linear1.bias"]),
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
        weight_operator=op(rows[prefix + "linear2.weight"]),
        weight_block=B_LIN2_W,
        bias_norm=vec(rows[prefix + "linear2.bias"]),
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
        weight_operator=op(rows["readout.weight"]),
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
) -> dict:
    """Fixed-point inflation of the exact centre values over ``B(c, epsilon)``.

    ``V_ball <= V_centre + first_ball * epsilon``, iterated upward.  Values only
    increase, so the iterate is monotone; the returned assignment is checked to
    be a valid (self-consistent) inflation.
    """
    centre = exact_stage_values(parameter, spec, config)
    inflation = {name: 0.0 for name in centre}
    reduce = (lambda j, o: j.sphere(o)) if sphere else (lambda j, o: j.scalar(o))

    history = []
    for _ in range(iterations):
        stages: Dict[str, BlockJet] = {}
        jet = _compose(
            parameter, spec, config, centre, inflation, epsilon,
            exact_values=exact_values, sphere=sphere,
            stage_jets=stages,
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
