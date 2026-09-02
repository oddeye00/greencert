#!/usr/bin/env python3
"""Two-known and three-known mixed directional Transformer bounds.

This v2 module leaves the released v1.3 mixed-jet source byte-for-byte intact.
It exposes, from the same linear-cost jet, both

* ``D^3 F(theta+t z)[z,z,.]`` for the complete nonlinear optimizer defect;
* ``D^4 F(theta+t z)[z,z,z,.]`` for the residual after a quadratic response.
"""
from __future__ import annotations

from math import sqrt

import numpy as np
import torch
from torch import Tensor

from transformer_block_envelope import exact_stage_values
from transformer_hvp_grokking import FlatSpec, TransformerConfig
from transformer_mixed_directional_jet import (
    _compose,
    objective_mixed_vector,
    parameter_block_radii,
)


def objective_mixed_third_vector(output) -> np.ndarray:
    """Cross-entropy ``D_t^2 D_epsilon`` coefficient majorant.

    The constants ``1, 1/2, sqrt(2)`` bound the third, second, and first
    derivatives of mean cross entropy in the same convention as the released
    fourth-order mixed jet.
    """

    return (
        output.a1**2 * output.b0
        + output.a1 * output.b1
        + 0.5 * output.a2 * output.b0
        + sqrt(2.0) * output.b2
    )


@torch.no_grad()
def mixed_directional_objective_bounds(
    parameter: Tensor,
    direction: Tensor,
    spec: FlatSpec,
    config: TransformerConfig,
    *,
    fixed_point_iterations: int = 64,
) -> dict:
    """Return segment-valid mixed third- and fourth-derivative bounds."""

    if config.loss != "cross_entropy":
        raise ValueError("mixed directional jet currently covers cross entropy")
    if config.depth != 1 or config.normalization != "none":
        raise ValueError("mixed directional jet covers one normalization-free block")
    radii = parameter_block_radii(direction, spec, config)
    centre = exact_stage_values(parameter, spec, config)
    inflation = {name: 0.0 for name in centre}
    history = []
    consistent = False
    for _ in range(fixed_point_iterations):
        _, stages = _compose(parameter, spec, config, centre, inflation, radii)
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
        _, stages = _compose(parameter, spec, config, centre, inflation, radii)
        proposed = {name: stage.a1 for name, stage in stages.items()}
        consistent = all(
            proposed[name] <= inflation[name] * (1.0 + 1.0e-9) + 1.0e-18
            for name in centre
        )
    if not consistent:
        raise RuntimeError("mixed-jet stage-value fixed point did not close")

    output, _ = _compose(parameter, spec, config, centre, inflation, radii)
    third_vector = objective_mixed_third_vector(output)
    fourth_vector = objective_mixed_vector(output)
    third = float(np.linalg.norm(third_vector))
    fourth = float(np.linalg.norm(fourth_vector))
    return {
        "mixed_third_derivative_upper": third,
        "gradient_nonlinear_remainder_upper": third / 2.0,
        "mixed_fourth_derivative_upper": fourth,
        "gradient_taylor_remainder_upper": fourth / 6.0,
        "block_radii": radii.tolist(),
        "direction_norm": float(np.linalg.norm(radii)),
        "mixed_third_block_coefficients": third_vector.tolist(),
        "mixed_fourth_block_coefficients": fourth_vector.tolist(),
        "fixed_point_iterations_used": len(history),
        "fixed_point_consistent": True,
        "maximum_stage_inflation": max(inflation.values(), default=0.0),
        "stage_value_inflation": inflation,
    }
