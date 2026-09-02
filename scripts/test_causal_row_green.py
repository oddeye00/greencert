#!/usr/bin/env python3
"""Independent tests for the causal row-Green theorem implementation."""
from __future__ import annotations

import math

import torch

from causal_row_green import (
    causal_row_quadratic_envelope,
    rowwise_signed_affine_bounds,
    simultaneous_row_direct_image_bounds,
)


DTYPE = torch.float64


def randomized_nonlinear_tests() -> int:
    generator = torch.Generator().manual_seed(4_710_291)
    cases = 0
    for horizon in (1, 2, 4, 7):
        for dimension in (1, 2, 5):
            for _trial in range(40):
                # Explicit causal row operators T_i from sequence forcing to
                # one state. Their norms are computed exactly by SVD.
                rows = []
                for output_step in range(horizon):
                    row = torch.zeros(
                        dimension,
                        horizon * dimension,
                        dtype=DTYPE,
                    )
                    row[:, : (output_step + 1) * dimension] = (
                        0.15
                        * torch.randn(
                            dimension,
                            (output_step + 1) * dimension,
                            generator=generator,
                            dtype=DTYPE,
                        )
                    )
                    rows.append(row)
                gains = torch.tensor(
                    [torch.linalg.matrix_norm(row, ord=2) for row in rows],
                    dtype=DTYPE,
                )
                affine_vectors = 0.01 * torch.randn(
                    horizon, dimension, generator=generator, dtype=DTYPE
                )
                affine = torch.linalg.vector_norm(affine_vectors, dim=1)
                curvature = 0.1 + torch.rand(
                    horizon, generator=generator, dtype=DTYPE
                )

                exact = []
                nonlinear = torch.zeros(horizon, dimension, dtype=DTYPE)
                directions = []
                for _ in range(horizon):
                    raw = torch.randn(
                        dimension, dimension, generator=generator, dtype=DTYPE
                    )
                    directions.append(
                        raw / max(float(torch.linalg.matrix_norm(raw, ord=2)), 1.0)
                    )
                for output_step in range(horizon):
                    value = affine_vectors[output_step] + rows[output_step] @ nonlinear.reshape(-1)
                    exact.append(value)
                    if output_step + 1 < horizon:
                        nonlinear[output_step + 1] = (
                            0.5
                            * curvature[output_step + 1]
                            * torch.linalg.vector_norm(value)
                            * (directions[output_step + 1] @ value)
                        )
                radii = causal_row_quadratic_envelope(affine, gains, curvature)
                observed = torch.linalg.vector_norm(torch.stack(exact), dim=1)
                if not bool((observed <= radii * (1.0 + 2.0e-13) + 2.0e-15).all()):
                    raise AssertionError("causal row radius violated")
                cases += 1
    return cases


def separation_and_interface_tests() -> None:
    affine = torch.tensor((1.0e-2, 1.0e-8, 1.0e-2), dtype=DTYPE)
    gains = torch.tensor((1.0, 1.0e-10, 1.0e-10), dtype=DTYPE)
    curvature = torch.tensor((0.0, 1.0e8, 1.0e8), dtype=DTYPE)
    radii = causal_row_quadratic_envelope(affine, gains, curvature)
    global_discriminant = 1.0 - 2.0 * float(curvature.max()) * float(
        torch.linalg.vector_norm(affine)
    )
    assert global_discriminant < 0.0
    assert bool(torch.isfinite(radii).all())
    assert float(radii.max()) < 0.010001

    signed = torch.tensor(((3.0, 4.0), (0.0, 12.0)), dtype=DTYPE)
    row_gain = torch.tensor((2.0, 3.0), dtype=DTYPE)
    errors = torch.tensor((0.1, 0.2), dtype=DTYPE)
    observed = rowwise_signed_affine_bounds(signed, row_gain, errors)
    expected = torch.tensor(
        (5.0 + 0.2, 12.0 + 3.0 * math.sqrt(0.05)), dtype=DTYPE
    )
    assert torch.allclose(observed, expected, rtol=2.0e-15, atol=2.0e-15)


def direct_image_reduction_test() -> None:
    images = torch.tensor(
        (
            ((3.0, 4.0), (0.0, 6.0), (5.0, 12.0)),
            ((0.0, 2.0), (8.0, 15.0), (0.0, 1.0)),
        ),
        dtype=DTYPE,
    )
    bounds, audit = simultaneous_row_direct_image_bounds(
        images, family_delta=3.0e-4
    )
    maxima = torch.tensor((5.0, 17.0, 13.0), dtype=DTYPE)
    calibrations = torch.tensor(audit["calibrations"], dtype=DTYPE)
    assert torch.allclose(bounds, maxima / calibrations, rtol=0.0, atol=0.0)
    assert math.isclose(audit["union_bound_upper"], 3.0e-4, rel_tol=2.0e-15)
    assert audit["additional_green_passes"] == 0


def validation_tests() -> None:
    try:
        simultaneous_row_direct_image_bounds(
            torch.zeros(2, 3, 4, dtype=DTYPE),
            family_delta=1.0e-4,
            row_budgets=(1.0e-4, 1.0e-4, 1.0e-4),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("excess row budget was accepted")
    try:
        causal_row_quadratic_envelope((1.0,), (1.0,), (-1.0,))
    except ValueError:
        pass
    else:
        raise AssertionError("negative curvature was accepted")


def main() -> None:
    cases = randomized_nonlinear_tests()
    separation_and_interface_tests()
    direct_image_reduction_test()
    validation_tests()
    print(f"PASS: causal row-Green theorem ({cases} randomized systems)")


if __name__ == "__main__":
    main()
