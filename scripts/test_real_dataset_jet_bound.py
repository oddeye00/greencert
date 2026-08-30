#!/usr/bin/env python3
"""Validity gates for the analytic real-data MLP derivative envelope."""
from __future__ import annotations

import torch
from torch.func import jvp

from real_dataset_jet_bound import (
    cross_entropy_hessian_lipschitz,
    margin_jet_bound,
    output_jet_bound,
)
from real_dataset_mlp import (
    RealMLPConfig,
    initialize,
    logits,
    make_split,
    objective_hvp,
    parameter_spec,
)


def directional_orders(function, point, direction):
    first_fn = lambda theta: jvp(function, (theta,), (direction,))[1]
    second_fn = lambda theta: jvp(first_fn, (theta,), (direction,))[1]
    first = first_fn(point)
    second = second_fn(point)
    third = jvp(second_fn, (point,), (direction,))[1]
    return first, second, third


def main() -> None:
    config = RealMLPConfig(seed=19, width=8, learning_rate=0.005)
    data = make_split(config)
    spec = parameter_spec(config)
    anchor = initialize(config)
    radius = 1e-3
    output_bound = output_jet_bound(
        anchor, data["train_x"], spec, radius
    )
    objective_bound = cross_entropy_hessian_lipschitz(output_bound)
    generator = torch.Generator().manual_seed(812)
    worst = {"first": 0.0, "second": 0.0, "third": 0.0, "drift": 0.0}

    for trial in range(8):
        offset = torch.randn(anchor.shape, generator=generator, dtype=anchor.dtype)
        offset /= torch.linalg.vector_norm(offset)
        offset *= radius * (trial + 1) / 8.0
        point = anchor + offset
        direction = torch.randn(anchor.shape, generator=generator, dtype=anchor.dtype)
        direction /= torch.linalg.vector_norm(direction)
        sample = trial % len(data["train_x"])
        feature = data["train_x"][sample : sample + 1]

        function = lambda theta: logits(theta, feature, spec).reshape(-1)
        first, second, third = directional_orders(function, point, direction)
        observed = {
            "first": float(torch.linalg.vector_norm(first)),
            "second": float(torch.linalg.vector_norm(second)),
            "third": float(torch.linalg.vector_norm(third)),
        }
        for name in ("first", "second", "third"):
            worst[name] = max(worst[name], observed[name] / output_bound[name])
            assert observed[name] <= output_bound[name] * (1 + 1e-10)

        label, competitor = 1, 0
        margin_bound = margin_jet_bound(
            anchor, data["train_x"], spec, label, competitor, radius
        )
        margin_function = lambda theta: (
            logits(theta, feature, spec)[0, label]
            - logits(theta, feature, spec)[0, competitor]
        ).reshape(1)
        margin_orders = directional_orders(margin_function, point, direction)
        for name, value in zip(("first", "second", "third"), margin_orders):
            assert float(torch.linalg.vector_norm(value)) <= margin_bound[name] * (1 + 1e-10)

        probe = torch.randn(anchor.shape, generator=generator, dtype=anchor.dtype)
        probe /= torch.linalg.vector_norm(probe)
        displacement = offset
        if torch.linalg.vector_norm(displacement) > 0:
            h0 = objective_hvp(
                anchor,
                probe,
                data["train_x"],
                data["train_y"],
                spec,
                config,
            )
            h1 = objective_hvp(
                point,
                probe,
                data["train_x"],
                data["train_y"],
                spec,
                config,
            )
            ratio = float(torch.linalg.vector_norm(h1 - h0) / torch.linalg.vector_norm(displacement))
            worst["drift"] = max(worst["drift"], ratio / objective_bound)
            assert ratio <= objective_bound * (1 + 1e-10)

    print(
        "PASS real-data jet; worst observed/bound ratios "
        + ", ".join(f"{key}={value:.3e}" for key, value in worst.items())
    )


if __name__ == "__main__":
    main()
