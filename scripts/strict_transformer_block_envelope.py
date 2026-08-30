#!/usr/bin/env python3
"""Strict-binary64 post-fixed Transformer derivative envelopes.

This is a post-seal hardening layer.  It keeps the analytic block jet unchanged
but inflates the monotone activation-value fixed point until every displayed
post-fixed inequality is literally true under the executed binary64
comparisons.  It is not a substitute for outward-rounded real arithmetic.
"""
from __future__ import annotations

import math
from typing import Dict

import torch
from torch import Tensor

from transformer_block_envelope import _compose, exact_stage_values
from transformer_hvp_grokking import TransformerConfig


@torch.no_grad()
def strict_ball_valid_envelope(
    parameter: Tensor,
    spec,
    config: TransformerConfig,
    *,
    epsilon: float,
    exact_values: bool = True,
    sphere: bool = True,
    safety_factor: float = 1e-12,
    iterations: int = 256,
) -> dict:
    if epsilon < 0.0:
        raise ValueError("epsilon must be nonnegative")
    if safety_factor <= 0.0:
        raise ValueError("safety_factor must be positive")

    centre = exact_stage_values(parameter, spec, config)
    inflation: Dict[str, float] = {name: 0.0 for name in centre}
    reduce = (lambda jet, order: jet.sphere(order)) if sphere else (
        lambda jet, order: jet.scalar(order)
    )
    history = []

    for _ in range(iterations):
        stages = {}
        jet = _compose(
            parameter,
            spec,
            config,
            centre,
            inflation,
            epsilon,
            exact_values=exact_values,
            sphere=sphere,
            stage_jets=stages,
        )
        stage_first = {name: reduce(stages[name], 1) for name in centre}
        targets = {name: stage_first[name] * epsilon for name in centre}
        maximum_deficit = max(targets[name] - inflation[name] for name in centre)
        history.append(maximum_deficit)
        if all(targets[name] <= inflation[name] for name in centre):
            first = reduce(jet, 1)
            second = reduce(jet, 2)
            third = reduce(jet, 3)
            return {
                "value": jet.value,
                "first": first,
                "second": second,
                "third": third,
                "centre_values": centre,
                "inflation": inflation,
                "stage_first": stage_first,
                "fixed_point_consistent": True,
                "strict_binary64_postfixed": True,
                "first_iterations": history,
                "fixed_point_iterations_used": len(history),
                "jet": jet,
                "safety_factor": safety_factor,
            }
        inflation = {
            name: math.nextafter(
                max(inflation[name], targets[name]) * (1.0 + safety_factor),
                math.inf,
            )
            for name in centre
        }

    raise RuntimeError(
        f"strict post-fixed inflation did not close in {iterations} iterations"
    )
