#!/usr/bin/env python3
"""Property and nonlinear tests for the structured two-response theorem."""
from __future__ import annotations

import math

import torch

from structured_directional_two_response import (
    make_batched_parameter_channel_green_products,
    make_parameter_channel_green_products,
    split_scaled_momentum_channels,
    structured_scaled_momentum_taylor_sequence_error,
    structured_directional_response_bound,
)
from directional_two_response import scaled_momentum_taylor_sequence_error
from structured_parameter_green_v2 import profiled_quadratic_root
from test_structured_parameter_green import (
    DTYPE,
    explicit_causal_green,
    momentum_jacobian,
    momentum_map,
)
from test_structured_parameter_green_v2 import explicit_projection_injection


def explicit_complement_injection(
    horizon: int, dimension: int
) -> torch.Tensor:
    injection = torch.zeros(
        horizon * 2 * dimension, horizon * dimension, dtype=DTYPE
    )
    identity = torch.eye(dimension, dtype=DTYPE)
    for step in range(horizon):
        parameter = slice(step * dimension, (step + 1) * dimension)
        state = step * 2 * dimension
        injection[state : state + dimension, parameter] = identity
        injection[state + dimension : state + 2 * dimension, parameter] = identity
    return injection


def channel_product_tests() -> None:
    generator = torch.Generator().manual_seed(340159)
    for horizon in (1, 3, 6):
        for dimension in (1, 2, 4):
            eta = 0.017
            jacobians = []
            for _ in range(horizon):
                raw = torch.randn(
                    2 * dimension, 2 * dimension, generator=generator, dtype=DTYPE
                )
                jacobians.append(raw / (3.0 + float(torch.linalg.matrix_norm(raw))))
            jvps = [lambda value, matrix=row: matrix @ value for row in jacobians]
            vjps = [lambda value, matrix=row: matrix.T @ value for row in jacobians]
            full = explicit_causal_green(jacobians)
            projection, structured_injection = explicit_projection_injection(
                horizon, dimension, eta
            )
            complement_injection = explicit_complement_injection(horizon, dimension)

            for left, right, injection in (
                (-eta, eta, structured_injection),
                (1.0, 1.0, complement_injection),
            ):
                apply, transpose = make_parameter_channel_green_products(
                    jvps, vjps, dimension, left, right
                )
                explicit = projection @ full @ injection
                vector = torch.randn(
                    horizon * dimension, generator=generator, dtype=DTYPE
                )
                cotangent = torch.randn(
                    horizon * dimension, generator=generator, dtype=DTYPE
                )
                assert torch.allclose(
                    apply(vector), explicit @ vector, atol=3e-14, rtol=3e-14
                )
                assert torch.allclose(
                    transpose(cotangent), explicit.T @ cotangent, atol=3e-14, rtol=3e-14
                )
                assert math.isclose(
                    float(cotangent @ apply(vector)),
                    float(transpose(cotangent) @ vector),
                    rel_tol=3e-14,
                    abs_tol=3e-14,
                )

                batched_jvps = [
                    lambda rows, matrix=row: rows @ matrix.T for row in jacobians
                ]
                batched_vjps = [
                    lambda rows, matrix=row: rows @ matrix for row in jacobians
                ]
                batch_apply, batch_transpose = (
                    make_batched_parameter_channel_green_products(
                        batched_jvps, batched_vjps, dimension, left, right
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

            arbitrary = torch.randn(
                horizon, 2 * dimension, generator=generator, dtype=DTYPE
            )
            structured, complement = split_scaled_momentum_channels(
                arbitrary, parameter_dimension=dimension, learning_rate=eta
            )
            rebuilt = torch.cat(
                (-eta * structured + complement, eta * structured + complement),
                dim=1,
            )
            assert torch.allclose(rebuilt, arbitrary, atol=3e-14, rtol=3e-14)


def response_bound_tests() -> None:
    generator = torch.Generator().manual_seed(719221)
    strict = 0
    for _ in range(1000):
        horizon = 5
        dimension = 3
        eta = 0.02
        jacobians = []
        for _step in range(horizon):
            raw = torch.randn(
                2 * dimension, 2 * dimension, generator=generator, dtype=DTYPE
            )
            jacobians.append(0.35 * raw / max(float(torch.linalg.matrix_norm(raw)), 1.0))
        full = explicit_causal_green(jacobians)
        projection, structured_injection = explicit_projection_injection(
            horizon, dimension, eta
        )
        complement_injection = explicit_complement_injection(horizon, dimension)
        structured_operator = projection @ full @ structured_injection
        complement_operator = projection @ full @ complement_injection

        q_tilde = 0.02 * torch.randn(
            horizon, dimension, generator=generator, dtype=DTYPE
        )
        q_error = 1e-4 * torch.randn(
            horizon, dimension, generator=generator, dtype=DTYPE
        )
        complement_path = 1e-5 * torch.randn(
            horizon, dimension, generator=generator, dtype=DTYPE
        )
        exact_source = (
            structured_injection @ (q_tilde + q_error).reshape(-1)
            + complement_injection @ complement_path.reshape(-1)
        )
        exact_parameter_response = projection @ full @ exact_source

        approximate_state_response = (
            full @ (structured_injection @ q_tilde.reshape(-1))
        ).reshape(horizon, 2 * dimension)
        approximate_state_response = approximate_state_response + 1e-7 * torch.randn(
            horizon, 2 * dimension, generator=generator, dtype=DTYPE
        )
        residual = []
        previous = torch.zeros(2 * dimension, dtype=DTYPE)
        for step in range(horizon):
            injection = torch.cat((-eta * q_tilde[step], eta * q_tilde[step]))
            residual.append(
                approximate_state_response[step]
                - jacobians[step] @ previous
                - injection
            )
            previous = approximate_state_response[step]
        residual = torch.stack(residual)
        structured_residual, complement_residual = split_scaled_momentum_channels(
            residual, parameter_dimension=dimension, learning_rate=eta
        )

        bound = structured_directional_response_bound(
            approximate_parameter_response_norm=float(
                torch.linalg.vector_norm(
                    approximate_state_response[:, :dimension]
                )
            ),
            structured_green_bound=float(
                torch.linalg.matrix_norm(structured_operator, ord=2)
            ),
            forcing_approximation_error_bound=float(
                torch.linalg.vector_norm(q_error)
            ),
            structured_response_residual_bound=float(
                torch.linalg.vector_norm(structured_residual)
            ),
            complement_green_bound=float(
                torch.linalg.matrix_norm(complement_operator, ord=2)
            ),
            complement_path_defect_bound=float(
                torch.linalg.vector_norm(complement_path)
            ),
            complement_response_residual_bound=float(
                torch.linalg.vector_norm(complement_residual)
            ),
        )
        assert float(torch.linalg.vector_norm(exact_parameter_response)) <= bound * (
            1.0 + 4e-13
        ) + 1e-14

        exact_approximation = full @ (structured_injection @ q_tilde.reshape(-1))
        old_full_state = float(torch.linalg.vector_norm(exact_approximation))
        new_parameter = float(
            torch.linalg.vector_norm(
                (projection @ exact_approximation).reshape(-1)
            )
        )
        assert new_parameter <= old_full_state + 2e-14
        if new_parameter < 0.999 * old_full_state:
            strict += 1
    assert strict >= 990


def nonlinear_corrected_path_tests() -> None:
    generator = torch.Generator().manual_seed(991772)
    strict = 0
    for _ in range(300):
        dimension = 2
        horizon = 5
        eta = 0.02
        momentum = 0.72
        raw = torch.randn(dimension, dimension, generator=generator, dtype=DTYPE)
        matrix = 0.05 * (raw + raw.T)
        cubic = 0.03 + 0.04 * torch.rand(
            dimension, generator=generator, dtype=DTYPE
        )
        initial = 0.04 * torch.randn(
            2 * dimension, generator=generator, dtype=DTYPE
        )

        truth = [initial]
        for _step in range(horizon):
            truth.append(momentum_map(truth[-1], matrix, cubic, eta, momentum))
        original = [initial]
        for step in range(1, horizon + 1):
            original.append(
                truth[step]
                + 4e-3
                * torch.randn(
                    2 * dimension, generator=generator, dtype=DTYPE
                )
            )

        original_jacobians = [
            momentum_jacobian(original[step], matrix, cubic, eta, momentum)
            for step in range(horizon)
        ]
        original_defect = torch.stack(
            [
                momentum_map(original[step], matrix, cubic, eta, momentum)
                - original[step + 1]
                for step in range(horizon)
            ]
        )
        original_green = explicit_causal_green(original_jacobians)
        correction_output = (
            original_green @ original_defect.reshape(-1)
        ).reshape(horizon, 2 * dimension)
        correction = [torch.zeros(2 * dimension, dtype=DTYPE)] + list(
            correction_output
        )
        corrected = [original[step] + correction[step] for step in range(horizon + 1)]

        corrected_defect = torch.stack(
            [
                momentum_map(corrected[step], matrix, cubic, eta, momentum)
                - corrected[step + 1]
                for step in range(horizon)
            ]
        )
        q = torch.stack(
            [
                0.5 * cubic * correction[step][:dimension].square()
                for step in range(horizon)
            ]
        )
        expected_defect = torch.cat((-eta * q, eta * q), dim=1)
        assert torch.allclose(
            corrected_defect, expected_defect, atol=2e-14, rtol=2e-11
        )
        assert float(torch.linalg.vector_norm(q[0])) == 0.0

        corrected_jacobians = [
            momentum_jacobian(corrected[step], matrix, cubic, eta, momentum)
            for step in range(horizon)
        ]
        corrected_green = explicit_causal_green(corrected_jacobians)
        projection, structured_injection = explicit_projection_injection(
            horizon, dimension, eta
        )
        structured_operator = projection @ corrected_green @ structured_injection
        parameter_response = structured_operator @ q.reshape(-1)
        direct_y = float(torch.linalg.vector_norm(parameter_response))
        restricted = structured_operator[:, dimension:]
        gain = float(torch.linalg.matrix_norm(restricted, ord=2))
        lipschitz = float(cubic.max())
        direct_root = profiled_quadratic_root(direct_y, gain * lipschitz)
        norm_only_y = gain * float(torch.linalg.vector_norm(q[1:]))
        norm_only_root = profiled_quadratic_root(norm_only_y, gain * lipschitz)
        assert direct_root is not None and norm_only_root is not None
        assert direct_root <= norm_only_root * (1.0 + 3e-12)
        if direct_root < 0.995 * norm_only_root:
            strict += 1

        parameter_error = torch.stack(
            [
                (truth[step] - corrected[step])[:dimension]
                for step in range(1, horizon + 1)
            ]
        )
        assert float(torch.linalg.vector_norm(parameter_error)) <= direct_root * (
            1.0 + 4e-10
        ) + 2e-14
    assert strict >= 250


def edge_tests() -> None:
    zero = structured_directional_response_bound(
        approximate_parameter_response_norm=0.0,
        structured_green_bound=1.0,
        forcing_approximation_error_bound=0.0,
        structured_response_residual_bound=0.0,
        complement_green_bound=1.0,
        complement_path_defect_bound=0.0,
        complement_response_residual_bound=0.0,
    )
    assert zero == 0.0
    try:
        split_scaled_momentum_channels(
            torch.zeros(2), parameter_dimension=1, learning_rate=0.0
        )
    except ValueError:
        pass
    else:
        raise AssertionError("zero learning rate was accepted")

    fourth = [1.2, 0.0, 3.4, 0.8]
    directions = [0.3, 0.0, 0.1, 0.5]
    eta = 0.017
    structured = structured_scaled_momentum_taylor_sequence_error(
        fourth, directions
    )
    full_state = scaled_momentum_taylor_sequence_error(
        learning_rate=eta,
        objective_fourth_derivative_bounds=fourth,
        parameter_direction_norms=directions,
    )
    assert math.isclose(
        full_state,
        math.sqrt(2.0) * eta * structured,
        rel_tol=2e-15,
        abs_tol=1e-18,
    )


def main() -> None:
    channel_product_tests()
    response_bound_tests()
    nonlinear_corrected_path_tests()
    edge_tests()
    print(
        "PASS: structured/complement adjoints, 1,000 inexact two-response "
        "bounds, and 300 corrected-path nonlinear closures"
    )


if __name__ == "__main__":
    main()
