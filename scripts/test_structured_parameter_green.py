#!/usr/bin/env python3
"""Algebraic and nonlinear tests for the structured parameter Green theorem."""
from __future__ import annotations

import math

import torch

from structured_parameter_green import (
    make_batched_structured_parameter_green_products,
    make_structured_parameter_green_products,
    structured_quadratic_root,
)


DTYPE = torch.float64


def explicit_causal_green(jacobians: list[torch.Tensor]) -> torch.Tensor:
    horizon = len(jacobians)
    state_dimension = jacobians[0].shape[0]
    columns = []
    for index in range(horizon * state_dimension):
        forcing = torch.zeros(horizon, state_dimension, dtype=DTYPE)
        forcing.reshape(-1)[index] = 1.0
        state = torch.zeros(state_dimension, dtype=DTYPE)
        output = []
        for step in range(horizon):
            state = jacobians[step] @ state + forcing[step]
            output.append(state)
        columns.append(torch.stack(output).reshape(-1))
    return torch.stack(columns, dim=1)


def linear_product_tests() -> None:
    generator = torch.Generator().manual_seed(20260830)
    for horizon in (1, 2, 5):
        for dimension in (1, 3):
            eta = 0.03
            jacobians = []
            for _ in range(horizon):
                raw = torch.randn(2 * dimension, 2 * dimension, generator=generator, dtype=DTYPE)
                jacobians.append(0.25 * raw / max(float(torch.linalg.matrix_norm(raw)), 1.0))
            jvps = [lambda value, matrix=matrix: matrix @ value for matrix in jacobians]
            vjps = [lambda value, matrix=matrix: matrix.T @ value for matrix in jacobians]
            apply, transpose = make_structured_parameter_green_products(
                jvps, vjps, dimension, eta
            )

            full = explicit_causal_green(jacobians)
            projection = torch.zeros(horizon * dimension, horizon * 2 * dimension, dtype=DTYPE)
            injection = torch.zeros(horizon * 2 * dimension, horizon * dimension, dtype=DTYPE)
            for step in range(horizon):
                projection[
                    step * dimension : (step + 1) * dimension,
                    step * 2 * dimension : step * 2 * dimension + dimension,
                ] = torch.eye(dimension, dtype=DTYPE)
                injection[
                    step * 2 * dimension : step * 2 * dimension + dimension,
                    step * dimension : (step + 1) * dimension,
                ] = -eta * torch.eye(dimension, dtype=DTYPE)
                injection[
                    step * 2 * dimension + dimension : (step + 1) * 2 * dimension,
                    step * dimension : (step + 1) * dimension,
                ] = eta * torch.eye(dimension, dtype=DTYPE)
            explicit = projection @ full @ injection

            vector = torch.randn(horizon * dimension, generator=generator, dtype=DTYPE)
            cotangent = torch.randn(horizon * dimension, generator=generator, dtype=DTYPE)
            assert torch.allclose(apply(vector), explicit @ vector, atol=2e-14, rtol=2e-14)
            assert torch.allclose(transpose(cotangent), explicit.T @ cotangent, atol=2e-14, rtol=2e-14)
            assert math.isclose(
                float(cotangent @ apply(vector)),
                float(transpose(cotangent) @ vector),
                rel_tol=2e-14,
                abs_tol=2e-14,
            )
            assert float(torch.linalg.matrix_norm(explicit, ord=2)) <= (
                math.sqrt(2.0) * eta * float(torch.linalg.matrix_norm(full, ord=2))
                + 2e-14
            )

            batched_jvps = [lambda rows, matrix=matrix: rows @ matrix.T for matrix in jacobians]
            batched_vjps = [lambda rows, matrix=matrix: rows @ matrix for matrix in jacobians]
            batch_apply, batch_transpose = make_batched_structured_parameter_green_products(
                batched_jvps, batched_vjps, dimension, eta
            )
            block = torch.randn(4, horizon * dimension, generator=generator, dtype=DTYPE)
            assert torch.allclose(
                batch_apply(block), torch.stack([apply(row) for row in block]), atol=2e-14, rtol=2e-14
            )
            assert torch.allclose(
                batch_transpose(block),
                torch.stack([transpose(row) for row in block]),
                atol=2e-14,
                rtol=2e-14,
            )


def momentum_map(
    state: torch.Tensor,
    matrix: torch.Tensor,
    cubic: torch.Tensor,
    eta: float,
    momentum: float,
) -> torch.Tensor:
    theta, velocity = state.chunk(2)
    gradient = matrix @ theta + 0.5 * cubic * theta.square()
    next_velocity = momentum * velocity + eta * gradient
    return torch.cat((theta - next_velocity, next_velocity))


def momentum_jacobian(
    state: torch.Tensor,
    matrix: torch.Tensor,
    cubic: torch.Tensor,
    eta: float,
    momentum: float,
) -> torch.Tensor:
    theta, _ = state.chunk(2)
    hessian = matrix + torch.diag(cubic * theta)
    dimension = theta.numel()
    identity = torch.eye(dimension, dtype=DTYPE)
    return torch.cat(
        (
            torch.cat((identity - eta * hessian, -momentum * identity), dim=1),
            torch.cat((eta * hessian, momentum * identity), dim=1),
        ),
        dim=0,
    )


def nonlinear_closure_tests() -> None:
    generator = torch.Generator().manual_seed(11939)
    issued = 0
    strict_improvements = 0
    for _ in range(500):
        dimension = 2
        horizon = 5
        eta = 0.02
        momentum = 0.75
        raw = torch.randn(dimension, dimension, generator=generator, dtype=DTYPE)
        matrix = 0.08 * (raw + raw.T)
        cubic = 0.05 + 0.05 * torch.rand(dimension, generator=generator, dtype=DTYPE)
        initial = 0.05 * torch.randn(2 * dimension, generator=generator, dtype=DTYPE)

        truth = [initial]
        for _step in range(horizon):
            truth.append(momentum_map(truth[-1], matrix, cubic, eta, momentum))

        reference = [initial]
        for step in range(1, horizon + 1):
            perturbation = 2e-4 * torch.randn(2 * dimension, generator=generator, dtype=DTYPE)
            reference.append(truth[step] + perturbation)

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
        response = (full @ defects.reshape(-1)).reshape(horizon, 2 * dimension)
        parameter_response = response[:, :dimension]

        jvps = [lambda value, matrix=row: matrix @ value for row in jacobians]
        vjps = [lambda value, matrix=row: matrix.T @ value for row in jacobians]
        apply, _ = make_structured_parameter_green_products(jvps, vjps, dimension, eta)
        explicit_structured_columns = []
        for index in range(horizon * dimension):
            basis = torch.zeros(horizon * dimension, dtype=DTYPE)
            basis[index] = 1.0
            explicit_structured_columns.append(apply(basis))
        structured = torch.stack(explicit_structured_columns, dim=1)

        gain = float(torch.linalg.matrix_norm(structured, ord=2))
        full_gain = float(torch.linalg.matrix_norm(full, ord=2))
        lipschitz = float(cubic.max())
        structured_root = structured_quadratic_root(
            float(torch.linalg.vector_norm(parameter_response)), gain, lipschitz
        )
        full_root = structured_quadratic_root(
            float(torch.linalg.vector_norm(response)),
            full_gain,
            math.sqrt(2.0) * eta * lipschitz,
        )
        if structured_root is None or full_root is None:
            continue
        issued += 1
        parameter_error = torch.stack(
            [(truth[step] - reference[step])[:dimension] for step in range(1, horizon + 1)]
        )
        assert float(torch.linalg.vector_norm(parameter_error)) <= structured_root * (1.0 + 2e-12)
        assert structured_root <= full_root * (1.0 + 2e-12)
        if structured_root < 0.999 * full_root:
            strict_improvements += 1

    assert issued == 500
    assert strict_improvements >= 490


def main() -> None:
    linear_product_tests()
    nonlinear_closure_tests()
    print(
        "PASS: structured parameter Green adjoints, batched products, exact "
        "dominance, and 500 nonlinear momentum closures"
    )


if __name__ == "__main__":
    main()

