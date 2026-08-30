#!/usr/bin/env python3
"""Verify outward Transformer multi-jets against PyTorch autograd."""

from __future__ import annotations

import time

import torch
from flint import arb, ctx

from arb_transformer_multijet import (
    arb_transformer_objective_jet,
    make_parameter_jet,
)
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
    ctx.prec = 160
    config = TransformerConfig(
        modulus=5,
        model_dim=8,
        hidden_dim=16,
        heads=2,
        depth=1,
        train_fraction=0.60,
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=0.01,
        steps=1,
        seed=20260831,
        threads=1,
        dtype="float64",
        loss="cross_entropy",
        normalization="none",
    )
    template = make_template(config)
    spec = flat_spec(template)
    parameter = flatten_parameters(template)
    train_pairs, train_labels, _, _, _, _ = make_disjoint_split(config)
    generator = torch.Generator().manual_seed(9182)
    x = torch.randn(parameter.shape, generator=generator) * 1.0e-3
    ys = [torch.randn(parameter.shape, generator=generator) for _ in range(2)]
    point = parameter.detach().requires_grad_(True)
    value = objective(point, train_pairs, train_labels, template, spec, config)
    (gradient,) = torch.autograd.grad(value, point, create_graph=True)
    expected_y = [float(torch.dot(gradient, row)) for row in ys]
    expected_x = float(torch.dot(gradient, x))
    expected_xy = [
        float(torch.dot(objective_hvp(parameter, x, train_pairs, train_labels, template, spec, config), row))
        for row in ys
    ]
    started = time.perf_counter()
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
    elapsed = time.perf_counter() - started
    checks = [(float(jet.x.mid()), expected_x)]  # type: ignore[union-attr]
    checks.extend((float(got.mid()), expected) for got, expected in zip(jet.y, expected_y))
    checks.extend((float(got.mid()), expected) for got, expected in zip(jet.xy or [], expected_xy))
    for got, expected in checks:
        if abs(got - expected) > 2.0e-11 * max(1.0, abs(expected)):
            raise AssertionError((got, expected))
    print(
        {
            "status": "Arb Transformer multijet test passed",
            "probes": 2,
            "mixed": True,
            "elapsed_seconds": elapsed,
            "maximum_midpoint_error": max(abs(a - b) for a, b in checks),
            "maximum_ball_radius": max(
                [float(value.rad()) for value in jet.y]
                + [float(value.rad()) for value in jet.xy or []]
            ),
        }
    )

    # Exact linear combinations must be formed inside Arb rather than rounded
    # once in binary64.  This is the state-probe split used by the full audit.
    exact_terms = make_parameter_jet(
        parameter.tolist(),
        [],
        spec,
        y_direction_terms=[[(1.0, ys[0].tolist()), (-1.0, ys[1].tolist())]],
    )
    first = exact_terms.flat[0].y[0]
    expected_difference = arb(float(ys[0][0])) - arb(float(ys[1][0]))
    if first != expected_difference:
        raise AssertionError("Arb probe difference was rounded before enclosure")


if __name__ == "__main__":
    main()
