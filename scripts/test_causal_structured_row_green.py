#!/usr/bin/env python3
"""Randomized tests for the scaled-momentum row-Green specialization."""
from __future__ import annotations

import torch

from causal_row_green import causal_row_quadratic_envelope


DTYPE = torch.float64


def structured_rows(jacobians: list[torch.Tensor], dimension: int, eta: float) -> list[torch.Tensor]:
    horizon = len(jacobians)
    rows = []
    injection = torch.cat(
        (-eta * torch.eye(dimension, dtype=DTYPE), eta * torch.eye(dimension, dtype=DTYPE)),
        dim=0,
    )
    for output_step in range(horizon):
        row = torch.zeros(dimension, horizon * dimension, dtype=DTYPE)
        for source in range(output_step + 1):
            state = injection
            for transition in range(source + 1, output_step + 1):
                state = jacobians[transition] @ state
            row[:, source * dimension : (source + 1) * dimension] = state[:dimension]
        rows.append(row)
    return rows


def main() -> None:
    generator = torch.Generator().manual_seed(9_731_104)
    cases = 0
    for horizon in (1, 2, 4, 7):
        for dimension in (1, 2, 4):
            eta = 0.03
            for _trial in range(30):
                jacobians = [
                    0.18
                    * torch.randn(
                        2 * dimension,
                        2 * dimension,
                        generator=generator,
                        dtype=DTYPE,
                    )
                    for _ in range(horizon)
                ]
                rows = structured_rows(jacobians, dimension, eta)
                gains = torch.tensor(
                    [torch.linalg.matrix_norm(row, ord=2) for row in rows],
                    dtype=DTYPE,
                )
                curvature = 0.05 + torch.rand(
                    horizon, generator=generator, dtype=DTYPE
                )
                affine_state = 2.0e-3 * torch.randn(
                    horizon, 2 * dimension, generator=generator, dtype=DTYPE
                )

                # Exact signed affine response through the full state dynamics.
                affine_parameter = []
                state = torch.zeros(2 * dimension, dtype=DTYPE)
                for step in range(horizon):
                    state = jacobians[step] @ state + affine_state[step]
                    affine_parameter.append(state[:dimension].clone())
                affine = torch.linalg.vector_norm(
                    torch.stack(affine_parameter), dim=1
                )
                radii = causal_row_quadratic_envelope(
                    affine, gains, curvature
                )

                # Realized nonlinear recurrence. The forcing direction is
                # arbitrary but has exactly the permitted quadratic norm.
                state = torch.zeros(2 * dimension, dtype=DTYPE)
                observed = []
                for step in range(horizon):
                    parameter = state[:dimension]
                    raw = torch.randn(
                        dimension, generator=generator, dtype=DTYPE
                    )
                    raw /= max(float(torch.linalg.vector_norm(raw)), 1.0)
                    q = (
                        0.5
                        * curvature[step]
                        * torch.linalg.vector_norm(parameter).square()
                        * raw
                    )
                    nonlinear = torch.cat((-eta * q, eta * q))
                    state = jacobians[step] @ state + affine_state[step] + nonlinear
                    observed.append(torch.linalg.vector_norm(state[:dimension]))
                observed_tensor = torch.stack(observed)
                if not bool(
                    (
                        observed_tensor
                        <= radii * (1.0 + 3.0e-13) + 3.0e-15
                    ).all()
                ):
                    raise AssertionError("structured causal radius violated")
                cases += 1
    print(f"PASS: structured causal row-Green theorem ({cases} randomized systems)")


if __name__ == "__main__":
    main()
