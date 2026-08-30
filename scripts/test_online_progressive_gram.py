#!/usr/bin/env python3
"""Regression test for resumable anytime Gram bounds."""

from __future__ import annotations

import math

import torch

from batched_green_operator import progressive_batched_gram_norm_bounds
from online_progressive_gram import OnlineGramState
from probe_jacobian_bound import ProbeConfig


def main() -> None:
    torch.set_default_dtype(torch.float64)
    generator = torch.Generator().manual_seed(2381)
    matrix = torch.randn(19, 19, generator=generator)
    gram = matrix.T @ matrix

    def apply(rows: torch.Tensor) -> torch.Tensor:
        return rows @ gram.T

    config = ProbeConfig(probes=9, power=7, delta=2.0e-6)
    seed = 88771
    expected = progressive_batched_gram_norm_bounds(
        apply,
        dimension=19,
        dtype=torch.float64,
        device=torch.device("cpu"),
        config=config,
        seed=seed,
    )
    state = OnlineGramState.initialize(
        dimension=19,
        dtype=torch.float64,
        device=torch.device("cpu"),
        config=config,
        seed=seed,
    )
    observed = [state.step(apply) for _ in range(config.power)]
    for left, right in zip(observed, expected["rows"]):
        for key in (
            "power",
            "Y",
            "c_delta",
            "operator_norm_upper_bound",
            "operator_norm_lower_estimate",
            "logical_gram_applications",
            "batched_gram_calls",
        ):
            assert left[key] == right[key], (key, left[key], right[key])
    true_norm = float(torch.linalg.matrix_norm(matrix, ord=2))
    assert all(row["operator_norm_upper_bound"] >= true_norm for row in observed)
    assert state.allocated_bytes == config.probes * 19 * 8
    assert math.isfinite(state.cumulative_operator_seconds)
    print("PASS: online q=1..7 rows exactly reproduce the sealed progressive kernel.")


if __name__ == "__main__":
    main()
