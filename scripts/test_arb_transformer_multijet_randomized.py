#!/usr/bin/env python3
"""Randomized architecture-level cross-checks for the frozen Arb multijet."""

from __future__ import annotations

import math

import torch
from flint import ctx

from arb_transformer_multijet import arb_transformer_objective_jet, make_parameter_jet
from arb_transformer_objective import arb_transformer_objective
from transformer_hvp_grokking import (
    TransformerConfig,
    flat_spec,
    flatten_parameters,
    make_disjoint_split,
    make_template,
    objective,
    objective_hvp,
)


def main() -> None:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    ctx.prec = 192
    cases = 12
    maximum_objective_error = 0.0
    maximum_jet_error = 0.0
    maximum_ball_radius = 0.0
    for case in range(cases):
        config = TransformerConfig(
            modulus=5 + case % 2,
            model_dim=8,
            hidden_dim=12 + 4 * (case % 2),
            heads=2,
            depth=1,
            train_fraction=0.55 + 0.05 * (case % 2),
            learning_rate=0.01,
            momentum=0.9,
            weight_decay=0.003 + 0.001 * case,
            steps=1,
            seed=7000 + case,
            threads=1,
            dtype="float64",
            loss="cross_entropy",
            normalization="none",
        )
        template = make_template(config)
        spec = flat_spec(template)
        parameter = flatten_parameters(template)
        train_pairs, train_labels, *_ = make_disjoint_split(config)
        generator = torch.Generator().manual_seed(8000 + case)
        x = torch.randn(parameter.shape, generator=generator) * (1.0e-3 + case * 1.0e-5)
        ys = [torch.randn(parameter.shape, generator=generator) for _ in range(3)]
        point = parameter.detach().requires_grad_(True)
        value = objective(point, train_pairs, train_labels, template, spec, config)
        (gradient,) = torch.autograd.grad(value, point, create_graph=True)
        expected_objective = float(value.detach())
        expected_x = float(torch.dot(gradient, x).detach())
        expected_y = [float(torch.dot(gradient, row).detach()) for row in ys]
        hvp = objective_hvp(
            parameter, x, train_pairs, train_labels, template, spec, config
        )
        expected_xy = [float(torch.dot(hvp, row)) for row in ys]
        scalar = arb_transformer_objective(
            parameter.tolist(), train_pairs, train_labels, spec, config
        )
        jet = arb_transformer_objective_jet(
            make_parameter_jet(
                parameter.tolist(),
                [row.tolist() for row in ys],
                spec,
                x_direction=x.tolist(),
            ),
            train_pairs,
            train_labels,
            config,
        )
        objective_error = max(
            abs(float(scalar.mid()) - expected_objective),
            abs(float(jet.value.mid()) - expected_objective),
        )
        comparisons = [(float(jet.x.mid()), expected_x)]  # type: ignore[union-attr]
        comparisons.extend((float(a.mid()), b) for a, b in zip(jet.y, expected_y))
        comparisons.extend((float(a.mid()), b) for a, b in zip(jet.xy or [], expected_xy))
        jet_error = max(abs(a - b) for a, b in comparisons)
        radii = [float(jet.value.rad())]
        radii.extend(float(value.rad()) for value in jet.y)
        radii.extend(float(value.rad()) for value in jet.xy or [])
        maximum_objective_error = max(maximum_objective_error, objective_error)
        maximum_jet_error = max(maximum_jet_error, jet_error)
        maximum_ball_radius = max(maximum_ball_radius, *radii)
        if objective_error > 5.0e-13:
            raise AssertionError((case, objective_error))
        scale = max(1.0, *(abs(value) for _, value in comparisons))
        if jet_error > 5.0e-11 * scale:
            raise AssertionError((case, jet_error, scale))
    print(
        {
            "status": "randomized Arb Transformer multijet tests passed",
            "cases": cases,
            "probe_directions_per_case": 3,
            "maximum_objective_midpoint_error": maximum_objective_error,
            "maximum_jet_midpoint_error": maximum_jet_error,
            "maximum_ball_radius": maximum_ball_radius,
            "finite": all(
                math.isfinite(value)
                for value in (
                    maximum_objective_error,
                    maximum_jet_error,
                    maximum_ball_radius,
                )
            ),
        }
    )


if __name__ == "__main__":
    main()
