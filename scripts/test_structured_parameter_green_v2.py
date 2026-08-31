#!/usr/bin/env python3
"""Tests for profiled curvature and unstructured-residual closure."""
from __future__ import annotations

import math

import torch

from structured_parameter_green import (
    make_structured_parameter_green_products,
)
from structured_parameter_green_v2 import (
    make_anchor_fixed_profiled_structured_parameter_green_products,
    make_batched_anchor_fixed_profiled_structured_parameter_green_products,
    make_batched_profiled_structured_parameter_green_products,
    make_profiled_structured_parameter_green_products,
    profiled_quadratic_root,
)
from test_structured_parameter_green import (
    DTYPE,
    explicit_causal_green,
    momentum_jacobian,
    momentum_map,
)


def explicit_projection_injection(
    horizon: int, dimension: int, eta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    projection = torch.zeros(
        horizon * dimension, horizon * 2 * dimension, dtype=DTYPE
    )
    injection = torch.zeros(
        horizon * 2 * dimension, horizon * dimension, dtype=DTYPE
    )
    identity = torch.eye(dimension, dtype=DTYPE)
    for step in range(horizon):
        parameter = slice(step * dimension, (step + 1) * dimension)
        state = step * 2 * dimension
        projection[parameter, state : state + dimension] = identity
        injection[state : state + dimension, parameter] = -eta * identity
        injection[state + dimension : state + 2 * dimension, parameter] = eta * identity
    return projection, injection


def profiled_product_tests() -> None:
    generator = torch.Generator().manual_seed(834921)
    strict = 0
    for horizon in (1, 3, 6):
        for dimension in (1, 2, 4):
            eta = 0.017
            jacobians = []
            for _ in range(horizon):
                raw = torch.randn(
                    2 * dimension, 2 * dimension, generator=generator, dtype=DTYPE
                )
                jacobians.append(raw / (3.0 + float(torch.linalg.matrix_norm(raw))))
            profile = (0.1 + 1.9 * torch.rand(horizon, generator=generator)).tolist()
            jvps = [lambda value, matrix=row: matrix @ value for row in jacobians]
            vjps = [lambda value, matrix=row: matrix.T @ value for row in jacobians]
            apply, transpose = make_profiled_structured_parameter_green_products(
                jvps, vjps, dimension, eta, profile
            )
            full = explicit_causal_green(jacobians)
            projection, injection = explicit_projection_injection(
                horizon, dimension, eta
            )
            diagonal = torch.diag(
                torch.tensor(profile, dtype=DTYPE).repeat_interleave(dimension)
            )
            explicit = projection @ full @ injection @ diagonal
            vector = torch.randn(
                horizon * dimension, generator=generator, dtype=DTYPE
            )
            cotangent = torch.randn(
                horizon * dimension, generator=generator, dtype=DTYPE
            )
            assert torch.allclose(apply(vector), explicit @ vector, atol=3e-14, rtol=3e-14)
            assert torch.allclose(
                transpose(cotangent), explicit.T @ cotangent, atol=3e-14, rtol=3e-14
            )
            assert math.isclose(
                float(cotangent @ apply(vector)),
                float(transpose(cotangent) @ vector),
                rel_tol=3e-14,
                abs_tol=3e-14,
            )

            base = projection @ full @ injection
            profiled_norm = float(torch.linalg.matrix_norm(explicit, ord=2))
            scalar_norm = float(torch.linalg.matrix_norm(base, ord=2)) * max(profile)
            assert profiled_norm <= scalar_norm + 3e-14
            if profiled_norm < 0.99 * scalar_norm:
                strict += 1

            batched_jvps = [
                lambda rows, matrix=row: rows @ matrix.T for row in jacobians
            ]
            batched_vjps = [
                lambda rows, matrix=row: rows @ matrix for row in jacobians
            ]
            batch_apply, batch_transpose = (
                make_batched_profiled_structured_parameter_green_products(
                    batched_jvps,
                    batched_vjps,
                    dimension,
                    eta,
                    profile,
                )
            )
            block = torch.randn(
                5, horizon * dimension, generator=generator, dtype=DTYPE
            )
            assert torch.allclose(
                batch_apply(block),
                torch.stack([apply(row) for row in block]),
                atol=3e-14,
                rtol=3e-14,
            )
            assert torch.allclose(
                batch_transpose(block),
                torch.stack([transpose(row) for row in block]),
                atol=3e-14,
                rtol=3e-14,
            )

            anchor_apply, anchor_transpose = (
                make_anchor_fixed_profiled_structured_parameter_green_products(
                    jvps, vjps, dimension, eta, profile
                )
            )
            reduced_explicit = explicit[:, dimension:]
            reduced_vector = torch.randn(
                max(0, horizon - 1) * dimension,
                generator=generator,
                dtype=DTYPE,
            )
            assert torch.allclose(
                anchor_apply(reduced_vector),
                reduced_explicit @ reduced_vector,
                atol=3e-14,
                rtol=3e-14,
            )
            assert torch.allclose(
                anchor_transpose(cotangent),
                reduced_explicit.T @ cotangent,
                atol=3e-14,
                rtol=3e-14,
            )
            anchor_norm = float(torch.linalg.matrix_norm(reduced_explicit, ord=2))
            assert anchor_norm <= profiled_norm + 3e-14

            batch_anchor_apply, batch_anchor_transpose = (
                make_batched_anchor_fixed_profiled_structured_parameter_green_products(
                    batched_jvps,
                    batched_vjps,
                    dimension,
                    eta,
                    profile,
                )
            )
            reduced_block = torch.randn(
                5,
                max(0, horizon - 1) * dimension,
                generator=generator,
                dtype=DTYPE,
            )
            assert torch.allclose(
                batch_anchor_apply(reduced_block),
                torch.stack([anchor_apply(row) for row in reduced_block]),
                atol=3e-14,
                rtol=3e-14,
            )
            assert torch.allclose(
                batch_anchor_transpose(block),
                torch.stack([anchor_transpose(row) for row in block]),
                atol=3e-14,
                rtol=3e-14,
            )
    assert strict >= 5


def split_residual_nonlinear_tests() -> None:
    generator = torch.Generator().manual_seed(479102)
    issued = 0
    for _ in range(300):
        dimension = 2
        horizon = 5
        eta = 0.02
        momentum = 0.72
        raw = torch.randn(dimension, dimension, generator=generator, dtype=DTYPE)
        matrix = 0.06 * (raw + raw.T)
        cubic = 0.03 + 0.04 * torch.rand(
            dimension, generator=generator, dtype=DTYPE
        )
        initial = 0.04 * torch.randn(
            2 * dimension, generator=generator, dtype=DTYPE
        )

        truth = [initial]
        for _step in range(horizon):
            truth.append(momentum_map(truth[-1], matrix, cubic, eta, momentum))
        reference = [initial]
        for step in range(1, horizon + 1):
            reference.append(
                truth[step]
                + 1.5e-4
                * torch.randn(
                    2 * dimension, generator=generator, dtype=DTYPE
                )
            )

        jacobians = [
            momentum_jacobian(reference[step], matrix, cubic, eta, momentum)
            for step in range(horizon)
        ]
        defects = torch.stack(
            [
                momentum_map(reference[step], matrix, cubic, eta, momentum)
                - reference[step + 1]
                for step in range(horizon)
            ]
        )
        full = explicit_causal_green(jacobians)
        projection, injection = explicit_projection_injection(
            horizon, dimension, eta
        )
        structured = projection @ full @ injection

        # Orthogonally split each arbitrary full-state defect into a component
        # in range(B) and a generally nonzero unstructured residual.
        theta_defect = defects[:, :dimension]
        velocity_defect = defects[:, dimension:]
        r = (velocity_defect - theta_defect) / (2.0 * eta)
        structured_defect = (injection @ r.reshape(-1)).reshape(
            horizon, 2 * dimension
        )
        unstructured = defects - structured_defect
        assert float(torch.linalg.vector_norm(unstructured)) > 0.0

        exact_parameter_response = projection @ full @ defects.reshape(-1)
        split_parameter_response = (
            structured @ r.reshape(-1)
            + projection @ full @ unstructured.reshape(-1)
        )
        assert torch.allclose(
            exact_parameter_response,
            split_parameter_response,
            atol=3e-14,
            rtol=3e-14,
        )
        split_triangle_bound = float(
            torch.linalg.vector_norm(structured @ r.reshape(-1))
            + torch.linalg.vector_norm(
                projection @ full @ unstructured.reshape(-1)
            )
        )
        assert split_triangle_bound + 3e-14 >= float(
            torch.linalg.vector_norm(exact_parameter_response)
        )

        jvps = [lambda value, matrix=row: matrix @ value for row in jacobians]
        vjps = [lambda value, matrix=row: matrix.T @ value for row in jacobians]
        apply, _ = make_structured_parameter_green_products(
            jvps, vjps, dimension, eta
        )
        explicit_columns = []
        for index in range(horizon * dimension):
            basis = torch.zeros(horizon * dimension, dtype=DTYPE)
            basis[index] = 1.0
            explicit_columns.append(apply(basis))
        assert torch.allclose(
            torch.stack(explicit_columns, dim=1),
            structured,
            atol=3e-14,
            rtol=3e-14,
        )

        lipschitz = float(cubic.max())
        # The true nonlinear forcing starts at update 1; update 0 is exactly
        # zero at the fixed anchor.  Use the corresponding restricted gain.
        profiled_gain = float(
            torch.linalg.matrix_norm(structured[:, dimension:], ord=2)
        ) * lipschitz
        radius = profiled_quadratic_root(split_triangle_bound, profiled_gain)
        assert radius is not None
        issued += 1
        parameter_error = torch.stack(
            [
                (truth[step] - reference[step])[:dimension]
                for step in range(1, horizon + 1)
            ]
        )
        assert float(torch.linalg.vector_norm(parameter_error)) <= radius * (
            1.0 + 3e-12
        )
    assert issued == 300


def root_edge_tests() -> None:
    assert profiled_quadratic_root(0.25, 0.0) == 0.25
    assert profiled_quadratic_root(1.0, 1.0) is None
    assert math.isclose(profiled_quadratic_root(0.5, 1.0), 1.0)
    for bad in ((-1.0, 1.0), (1.0, -1.0), (math.inf, 1.0)):
        try:
            profiled_quadratic_root(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid bound was accepted")


def main() -> None:
    profiled_product_tests()
    split_residual_nonlinear_tests()
    root_edge_tests()
    print(
        "PASS: profiled structured Green products, exact gain dominance, "
        "and 300 closures with nonzero unstructured residuals"
    )


if __name__ == "__main__":
    main()
