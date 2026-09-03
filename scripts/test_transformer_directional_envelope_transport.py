#!/usr/bin/env python3
"""Regression tests for directional transport of corrected-center envelopes."""
from __future__ import annotations

import math

import torch

from transformer_block_envelope_v15 import (
    ball_valid_envelope,
    exact_stage_values,
    parameter_geometry,
)
from transformer_hvp_grokking import (
    TransformerConfig,
    flat_spec,
    flatten_parameters,
    make_template,
)
from transformer_mixed_directional_jet_v15 import (
    directionally_transported_envelope_inputs,
    mixed_directional_objective_fourth_bound,
)


def dominated(upper: float, lower: float) -> bool:
    return float(upper) >= float(lower) * (1.0 - 3.0e-13) - 1.0e-14


def main() -> None:
    torch.set_default_dtype(torch.float64)
    config = TransformerConfig(
        modulus=3,
        model_dim=4,
        hidden_dim=6,
        heads=1,
        depth=1,
        seed=229,
        dtype="float64",
        loss="cross_entropy",
        normalization="none",
    )
    template = make_template(config)
    spec = flat_spec(template)
    parameter = flatten_parameters(template)
    centre = exact_stage_values(parameter, spec, config)
    geometry = parameter_geometry(parameter, spec, config)
    generator = torch.Generator().manual_seed(20260902)
    checks = 0
    maximum_stage_inflation = 1.0
    maximum_jet_inflation = 1.0

    for scale in (1.0e-7, 3.0e-6, 1.0e-4):
        direction = torch.randn(
            parameter.shape, dtype=parameter.dtype, generator=generator
        )
        direction *= scale / torch.linalg.vector_norm(direction)
        mixed = mixed_directional_objective_fourth_bound(
            parameter,
            direction,
            spec,
            config,
            centre_values=centre,
            parameter_norms=geometry,
        )
        transported_values, transported_geometry = (
            directionally_transported_envelope_inputs(centre, geometry, mixed)
        )
        amplification = 8.0
        amplified_mixed = mixed_directional_objective_fourth_bound(
            parameter,
            amplification * direction,
            spec,
            config,
            centre_values=centre,
            parameter_norms=geometry,
        )
        amplified_values, amplified_geometry = (
            directionally_transported_envelope_inputs(
                centre,
                geometry,
                amplified_mixed,
                path_fraction=1.0 / amplification,
            )
        )
        shifted = parameter + direction
        exact_shifted_values = exact_stage_values(shifted, spec, config)
        exact_shifted_geometry = parameter_geometry(shifted, spec, config)
        for name, observed in exact_shifted_values.items():
            upper = transported_values[name]
            if not dominated(upper, observed):
                raise AssertionError(
                    f"transported activation failed at {name}: {upper} < {observed}"
                )
            maximum_stage_inflation = max(
                maximum_stage_inflation,
                upper / max(observed, torch.finfo(torch.float64).tiny),
            )
            checks += 1
            if not dominated(amplified_values[name], observed):
                raise AssertionError(
                    f"fractional activation transport failed at {name}"
                )
            checks += 1
        for name, observed in exact_shifted_geometry.items():
            upper = transported_geometry[name]
            if not dominated(upper, observed):
                raise AssertionError(
                    f"transported geometry failed at {name}: {upper} < {observed}"
                )
            checks += 1
            if not dominated(amplified_geometry[name], observed):
                raise AssertionError(
                    f"fractional geometry transport failed at {name}"
                )
            checks += 1

        epsilon = 2.0e-6
        exact_envelope = ball_valid_envelope(
            shifted,
            spec,
            config,
            epsilon=epsilon,
            centre_values=exact_shifted_values,
            parameter_norms=exact_shifted_geometry,
        )
        transported_envelope = ball_valid_envelope(
            parameter,
            spec,
            config,
            epsilon=epsilon,
            centre_values=transported_values,
            parameter_norms=transported_geometry,
        )
        amplified_envelope = ball_valid_envelope(
            parameter,
            spec,
            config,
            epsilon=epsilon,
            centre_values=amplified_values,
            parameter_norms=amplified_geometry,
        )
        if not transported_envelope["fixed_point_consistent"]:
            raise AssertionError("transported envelope fixed point failed")
        for name in ("first", "second", "third"):
            upper = float(transported_envelope[name])
            observed = float(exact_envelope[name])
            if not dominated(upper, observed):
                raise AssertionError(
                    f"transported {name} envelope failed: {upper} < {observed}"
                )
            maximum_jet_inflation = max(
                maximum_jet_inflation,
                upper / max(observed, torch.finfo(torch.float64).tiny),
            )
            checks += 1
            if not dominated(float(amplified_envelope[name]), observed):
                raise AssertionError(
                    f"fractional transported {name} envelope failed"
                )
            checks += 1

    malformed = dict(geometry)
    malformed.pop("query_weight")
    try:
        directionally_transported_envelope_inputs(
            centre,
            malformed,
            mixed,
        )
    except ValueError:
        checks += 1
    else:
        raise AssertionError("malformed geometry registry was accepted")

    if not math.isfinite(maximum_jet_inflation):
        raise AssertionError("nonfinite transport inflation")
    print(
        {
            "status": "directional envelope transport checks passed",
            "checks": checks,
            "maximum_stage_majorant_ratio": maximum_stage_inflation,
            "maximum_derivative_envelope_ratio": maximum_jet_inflation,
        }
    )


if __name__ == "__main__":
    main()
