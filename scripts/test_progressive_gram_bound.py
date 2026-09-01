#!/usr/bin/env python3
"""Regression tests for simultaneous progressive Gaussian Gram bounds."""

from __future__ import annotations

import math

import torch

from batched_green_operator import (
    batched_gram_norm_bound,
    progressive_batched_gram_norm_bounds,
)
from probe_jacobian_bound import ProbeConfig


def main() -> None:
    torch.set_default_dtype(torch.float64)
    generator = torch.Generator().manual_seed(991)
    matrix = torch.randn(17, 17, generator=generator)
    gram = matrix.T @ matrix

    def apply(rows: torch.Tensor) -> torch.Tensor:
        return rows @ gram.T

    config = ProbeConfig(probes=8, power=6, delta=1.0e-5)
    seed = 44109
    progressive = progressive_batched_gram_norm_bounds(
        apply,
        dimension=17,
        dtype=torch.float64,
        device=torch.device("cpu"),
        config=config,
        seed=seed,
    )
    ordinary = batched_gram_norm_bound(
        apply,
        dimension=17,
        dtype=torch.float64,
        device=torch.device("cpu"),
        config=config,
        seed=seed,
    )
    final = progressive["rows"][-1]
    assert final["power"] == config.power
    assert final["Y"] == ordinary["Y"]
    assert final["operator_norm_upper_bound"] == ordinary[
        "operator_norm_upper_bound"
    ]
    assert final["operator_norm_lower_estimate"] == ordinary[
        "operator_norm_lower_estimate"
    ]
    assert progressive["single_event_simultaneous_over_all_powers"]
    assert all(
        row["logical_gram_applications"] == config.probes * row["power"]
        for row in progressive["rows"]
    )
    true_norm = float(torch.linalg.matrix_norm(matrix, ord=2))
    # This seeded numerical example is not the probabilistic proof, but guards
    # the exponent and operator orientation at every returned power.
    assert all(
        math.isfinite(row["operator_norm_upper_bound"])
        and row["operator_norm_upper_bound"] >= true_norm
        for row in progressive["rows"]
    )
    print("PASS: progressive and ordinary q=6 bounds match exactly.")


if __name__ == "__main__":
    main()
