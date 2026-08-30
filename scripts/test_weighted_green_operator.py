#!/usr/bin/env python3
"""Adjoint and explicit-matrix tests for time-scaled Green products."""
from __future__ import annotations

import numpy as np
import torch

from batched_green_operator import make_batched_causal_green_products
from weighted_green_operator import make_weighted_batched_green_products


def main() -> None:
    generator = np.random.default_rng(20260827)
    cases = 0
    for dimension in (1, 2, 5):
        for horizon in (1, 3, 7):
            for _ in range(80):
                matrices = [
                    torch.tensor(
                        generator.normal(scale=0.25, size=(dimension, dimension)),
                        dtype=torch.float64,
                    )
                    for _step in range(horizon)
                ]

                def row_apply(matrix: torch.Tensor):
                    return lambda rows: rows @ matrix.T

                def row_transpose(matrix: torch.Tensor):
                    return lambda rows: rows @ matrix

                green, green_t = make_batched_causal_green_products(
                    [row_apply(matrix) for matrix in matrices],
                    [row_transpose(matrix) for matrix in matrices],
                    dimension,
                )
                state = np.exp(generator.uniform(-1.0, 1.0, size=horizon))
                injection = np.exp(generator.uniform(-1.0, 1.0, size=horizon))
                weighted, weighted_t = make_weighted_batched_green_products(
                    green,
                    green_t,
                    state_weights=state,
                    injection_weights=injection,
                    state_dimension=dimension,
                    dtype=torch.float64,
                    device=torch.device("cpu"),
                )

                columns = horizon * dimension
                eye = torch.eye(columns, dtype=torch.float64)
                observed = weighted(eye).numpy().T
                raw = green(eye).numpy().T
                expected = (
                    np.kron(np.diag(state), np.eye(dimension))
                    @ raw
                    @ np.kron(np.diag(1.0 / injection), np.eye(dimension))
                )
                if not np.allclose(observed, expected, rtol=2.0e-13, atol=2.0e-13):
                    raise AssertionError("weighted Green matrix mismatch")

                left = torch.tensor(
                    generator.normal(size=(4, columns)), dtype=torch.float64
                )
                right = torch.tensor(
                    generator.normal(size=(4, columns)), dtype=torch.float64
                )
                inner_left = torch.sum(weighted(left) * right, dim=1)
                inner_right = torch.sum(left * weighted_t(right), dim=1)
                if not torch.allclose(
                    inner_left, inner_right, rtol=2.0e-13, atol=2.0e-13
                ):
                    raise AssertionError("weighted Green adjoint identity failed")
                cases += 1
    print({"status": "weighted Green operator tests passed", "cases": cases})


if __name__ == "__main__":
    main()
