#!/usr/bin/env python3
"""Regression tests for causal prefix-local variational recentering."""
from __future__ import annotations

import torch

from matrix_free_mlp import signed_variational_recenter
from streaming_variational_centerline import streaming_signed_recentered_reference
from transformer_modal_forecast import affine_reference


def main() -> None:
    torch.manual_seed(20260829)
    dtype = torch.float64
    dimension = 11
    anchor = torch.randn(dimension, dtype=dtype) * 0.1
    matrix = torch.randn(dimension, dimension, dtype=dtype) * 0.03
    quadratic = torch.randn(dimension, dtype=dtype) * 0.01

    def map_step(state: torch.Tensor) -> torch.Tensor:
        return matrix @ state + quadratic * state.square() + 0.8 * state

    def jvp(center: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        return matrix @ direction + 2.0 * quadratic * center * direction + 0.8 * direction

    for horizon in (0, 1, 7, 20):
        raw = affine_reference(
            anchor, map_step, lambda direction: jvp(anchor, direction), horizon=horizon
        )
        batch = raw
        for _ in range(4):
            batch, diagnostic = signed_variational_recenter(
                batch, map_step, jvp, numeric_cap=1.0e6
            )
            assert diagnostic["reached_horizon"] == horizon
        streamed, diagnostics = streaming_signed_recentered_reference(
            anchor,
            map_step,
            jvp,
            lambda direction: jvp(anchor, direction),
            maximum_horizon=horizon,
            sweeps=4,
            numeric_cap=1.0e6,
        )
        assert torch.equal(streamed, batch)
        assert all(row["reached_horizon"] == horizon for row in diagnostics)

    full, _ = streaming_signed_recentered_reference(
        anchor,
        map_step,
        jvp,
        lambda direction: jvp(anchor, direction),
        maximum_horizon=20,
        sweeps=4,
        numeric_cap=1.0e6,
    )
    stopped, _ = streaming_signed_recentered_reference(
        anchor,
        map_step,
        jvp,
        lambda direction: jvp(anchor, direction),
        maximum_horizon=20,
        sweeps=4,
        numeric_cap=1.0e6,
        stop_when=lambda step, _: step == 7,
    )
    assert torch.equal(stopped, full[:8])
    print(
        {
            "status": "streaming variational centerline tests passed",
            "batch_stream_bitwise_cases": 4,
            "causal_stop_prefix": 7,
        }
    )


if __name__ == "__main__":
    main()
