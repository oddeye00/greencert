#!/usr/bin/env python3
"""Finite-dimensional checks of the causal structured resolvent theorem."""
from __future__ import annotations

import math

import torch

from causal_structured_resolvent import (
    causal_block_majorant,
    causal_block_majorant_gain,
    causal_block_majorant_response_bound,
    causal_directional_affine_bounds,
    causal_forward_quadratic_envelope,
    causal_profiled_block_majorant_gain,
    finite_geometric_sum,
    make_batched_causal_structured_resolvent_products,
    make_batched_scalar_hessian_optimizer_products,
    make_causal_structured_resolvent_products,
    preconditioned_structured_gain_bound,
    scalar_hessian_parameter_green_matrix,
    scalar_hessian_structured_gain,
    truncated_neumann_response,
    truncated_structured_response_error_bound,
)
from test_structured_parameter_green import DTYPE, explicit_causal_green
from test_structured_parameter_green_v2 import explicit_projection_injection


def block_diagonal(rows: list[torch.Tensor]) -> torch.Tensor:
    return torch.block_diag(*rows)


def shift_matrix(horizon: int, dimension: int) -> torch.Tensor:
    shift = torch.zeros(
        horizon * dimension, horizon * dimension, dtype=DTYPE
    )
    identity = torch.eye(dimension, dtype=DTYPE)
    for step in range(1, horizon):
        target = slice(step * dimension, (step + 1) * dimension)
        source = slice((step - 1) * dimension, step * dimension)
        shift[target, source] = identity
    return shift


def matrix_products(matrix: torch.Tensor):
    return (
        lambda value, row=matrix: row @ value,
        lambda value, row=matrix: row.T @ value,
    )


def batched_matrix_products(matrix: torch.Tensor):
    return (
        lambda rows, row=matrix: rows @ row.T,
        lambda rows, row=matrix: rows @ row,
    )


def identity_and_adjoint_tests() -> None:
    generator = torch.Generator().manual_seed(660201)
    cases = 0
    strict_truncations = 0
    for horizon in (1, 2, 4, 6):
        for dimension in (1, 2, 3):
            for scale in (0.01, 0.08, 0.22):
                eta = 0.017
                approximate = []
                deltas = []
                exact = []
                state_dimension = 2 * dimension
                parameter_projection = torch.cat(
                    (
                        torch.eye(dimension, dtype=DTYPE),
                        torch.zeros(dimension, dimension, dtype=DTYPE),
                    ),
                    dim=1,
                )
                injection = torch.cat(
                    (
                        -eta * torch.eye(dimension, dtype=DTYPE),
                        eta * torch.eye(dimension, dtype=DTYPE),
                    ),
                    dim=0,
                )
                for _step in range(horizon):
                    raw = torch.randn(
                        state_dimension,
                        state_dimension,
                        generator=generator,
                        dtype=DTYPE,
                    )
                    approximate_jacobian = raw / (
                        2.5 + float(torch.linalg.matrix_norm(raw))
                    )
                    raw_delta = torch.randn(
                        dimension,
                        dimension,
                        generator=generator,
                        dtype=DTYPE,
                    )
                    delta = scale * raw_delta / max(
                        float(torch.linalg.matrix_norm(raw_delta)), 1.0
                    )
                    approximate.append(approximate_jacobian)
                    deltas.append(delta)
                    exact.append(
                        approximate_jacobian
                        + injection @ delta @ parameter_projection
                    )

                approximate_green = explicit_causal_green(approximate)
                exact_green = explicit_causal_green(exact)
                projection, structured_injection = explicit_projection_injection(
                    horizon, dimension, eta
                )
                t0_matrix = projection @ approximate_green @ structured_injection
                exact_matrix = projection @ exact_green @ structured_injection
                mismatch_matrix = (
                    block_diagonal(deltas)
                    @ shift_matrix(horizon, dimension)
                    @ t0_matrix
                )

                approximate_products = [matrix_products(row) for row in approximate]
                delta_products = [matrix_products(row) for row in deltas]
                t0, t0_t, mismatch, mismatch_t = (
                    make_causal_structured_resolvent_products(
                        [row[0] for row in approximate_products],
                        [row[1] for row in approximate_products],
                        [row[0] for row in delta_products],
                        [row[1] for row in delta_products],
                        dimension,
                        eta,
                    )
                )
                vector = torch.randn(
                    horizon * dimension, generator=generator, dtype=DTYPE
                )
                cotangent = torch.randn(
                    horizon * dimension, generator=generator, dtype=DTYPE
                )
                assert torch.allclose(
                    t0(vector), t0_matrix @ vector, atol=6e-14, rtol=6e-14
                )
                assert torch.allclose(
                    t0_t(cotangent),
                    t0_matrix.T @ cotangent,
                    atol=6e-14,
                    rtol=6e-14,
                )
                assert torch.allclose(
                    mismatch(vector),
                    mismatch_matrix @ vector,
                    atol=6e-14,
                    rtol=6e-14,
                )
                assert torch.allclose(
                    mismatch_t(cotangent),
                    mismatch_matrix.T @ cotangent,
                    atol=6e-14,
                    rtol=6e-14,
                )
                assert math.isclose(
                    float(cotangent @ mismatch(vector)),
                    float(mismatch_t(cotangent) @ vector),
                    rel_tol=8e-14,
                    abs_tol=8e-14,
                )

                powers = torch.eye(horizon * dimension, dtype=DTYPE)
                inverse = powers.clone()
                for _power in range(1, horizon):
                    powers = powers @ mismatch_matrix
                    inverse = inverse + powers
                nilpotent = powers @ mismatch_matrix
                assert float(torch.linalg.matrix_norm(nilpotent)) <= 2e-12
                reconstructed = t0_matrix @ inverse
                assert torch.allclose(
                    reconstructed, exact_matrix, atol=3e-12, rtol=3e-12
                )

                kappa0 = float(torch.linalg.matrix_norm(t0_matrix, ord=2))
                alpha = float(torch.linalg.matrix_norm(mismatch_matrix, ord=2))
                exact_gain = float(torch.linalg.matrix_norm(exact_matrix, ord=2))
                gain_bound = preconditioned_structured_gain_bound(
                    approximate_gain_bound=kappa0,
                    mismatch_gain_bound=alpha,
                    horizon=horizon,
                )
                assert exact_gain <= gain_bound * (1.0 + 2e-12) + 2e-14

                q_tilde = torch.randn(
                    horizon * dimension, generator=generator, dtype=DTYPE
                )
                q_error = 1e-4 * torch.randn(
                    horizon * dimension, generator=generator, dtype=DTYPE
                )
                exact_response = exact_matrix @ (q_tilde + q_error)
                for maximum_power in range(horizon):
                    computed = truncated_neumann_response(
                        q_tilde,
                        apply_approximate_green=t0,
                        apply_mismatch=mismatch,
                        maximum_power=maximum_power,
                    )
                    error_bound = truncated_structured_response_error_bound(
                        approximate_gain_bound=kappa0,
                        mismatch_gain_bound=alpha,
                        horizon=horizon,
                        maximum_neumann_power=maximum_power,
                        approximate_forcing_norm=float(
                            torch.linalg.vector_norm(q_tilde)
                        ),
                        forcing_approximation_error_bound=float(
                            torch.linalg.vector_norm(q_error)
                        ),
                    )
                    observed = float(
                        torch.linalg.vector_norm(exact_response - computed)
                    )
                    assert observed <= error_bound * (1.0 + 3e-12) + 3e-14
                    if maximum_power == horizon - 1:
                        forcing_only = gain_bound * float(
                            torch.linalg.vector_norm(q_error)
                        )
                        assert error_bound <= forcing_only * (1.0 + 3e-12) + 3e-14
                    elif maximum_power > 0:
                        previous_bound = truncated_structured_response_error_bound(
                            approximate_gain_bound=kappa0,
                            mismatch_gain_bound=alpha,
                            horizon=horizon,
                            maximum_neumann_power=maximum_power - 1,
                            approximate_forcing_norm=float(
                                torch.linalg.vector_norm(q_tilde)
                            ),
                            forcing_approximation_error_bound=float(
                                torch.linalg.vector_norm(q_error)
                            ),
                        )
                        if alpha < 1.0 and error_bound < previous_bound:
                            strict_truncations += 1

                batched_approximate = [
                    batched_matrix_products(row) for row in approximate
                ]
                batched_delta = [batched_matrix_products(row) for row in deltas]
                bt0, bt0_t, ba, ba_t = (
                    make_batched_causal_structured_resolvent_products(
                        [row[0] for row in batched_approximate],
                        [row[1] for row in batched_approximate],
                        [row[0] for row in batched_delta],
                        [row[1] for row in batched_delta],
                        dimension,
                        eta,
                    )
                )
                block = torch.randn(
                    5,
                    horizon * dimension,
                    generator=generator,
                    dtype=DTYPE,
                )
                assert torch.allclose(
                    bt0(block), torch.stack([t0(row) for row in block]),
                    atol=8e-14, rtol=8e-14
                )
                assert torch.allclose(
                    bt0_t(block), torch.stack([t0_t(row) for row in block]),
                    atol=8e-14, rtol=8e-14
                )
                assert torch.allclose(
                    ba(block), torch.stack([mismatch(row) for row in block]),
                    atol=8e-14, rtol=8e-14
                )
                assert torch.allclose(
                    ba_t(block), torch.stack([mismatch_t(row) for row in block]),
                    atol=8e-14, rtol=8e-14
                )

                delta_norms = [
                    float(torch.linalg.matrix_norm(row, ord=2)) for row in deltas
                ]
                weighted = (
                    block_diagonal(
                        [value * torch.eye(dimension, dtype=DTYPE) for value in delta_norms]
                    )
                    @ shift_matrix(horizon, dimension)
                    @ t0_matrix
                )
                assert alpha <= float(torch.linalg.matrix_norm(weighted, ord=2)) * (
                    1.0 + 2e-12
                ) + 2e-14
                cases += 1
    assert cases == 36
    assert strict_truncations > 20, strict_truncations


def scalar_and_edge_tests() -> None:
    assert finite_geometric_sum(0.0, horizon=7) == 1.0
    assert finite_geometric_sum(0.0, horizon=7, start_power=1) == 0.0
    assert finite_geometric_sum(1.0, horizon=7) == 7.0
    assert finite_geometric_sum(2.0, horizon=5) == 31.0
    assert finite_geometric_sum(2.0, horizon=5, start_power=3) == 24.0
    assert preconditioned_structured_gain_bound(
        approximate_gain_bound=3.0,
        mismatch_gain_bound=0.0,
        horizon=9,
    ) == 3.0
    try:
        finite_geometric_sum(-1.0, horizon=2)
    except ValueError:
        pass
    else:
        raise AssertionError("negative mismatch gain was accepted")

    eta = 0.03
    momentum = 0.81
    scalars = (0.2, -0.1, 0.4, 0.05)
    temporal = scalar_hessian_parameter_green_matrix(
        scalars, learning_rate=eta, momentum=momentum
    )
    jacobians = [
        torch.tensor(
            (
                (1.0 - eta * value, -momentum),
                (eta * value, momentum),
            ),
            dtype=DTYPE,
        )
        for value in scalars
    ]
    full = explicit_causal_green(jacobians)
    projection, injection = explicit_projection_injection(len(scalars), 1, eta)
    explicit = projection @ full @ injection
    assert torch.allclose(temporal, explicit, atol=2e-15, rtol=2e-15)
    assert math.isclose(
        scalar_hessian_structured_gain(
            scalars, learning_rate=eta, momentum=momentum
        ),
        float(torch.linalg.matrix_norm(explicit, ord=2)),
        rel_tol=2e-15,
        abs_tol=2e-15,
    )

    dimension = 5
    jvp, vjp = make_batched_scalar_hessian_optimizer_products(
        parameter_dimension=dimension,
        learning_rate=eta,
        momentum=momentum,
        hessian_scalar=scalars[0],
    )
    generator = torch.Generator().manual_seed(92177)
    rows = torch.randn(7, 2 * dimension, generator=generator, dtype=DTYPE)
    scalar_jacobian = torch.cat(
        (
            torch.cat(
                (
                    (1.0 - eta * scalars[0])
                    * torch.eye(dimension, dtype=DTYPE),
                    -momentum * torch.eye(dimension, dtype=DTYPE),
                ),
                dim=1,
            ),
            torch.cat(
                (
                    eta * scalars[0] * torch.eye(dimension, dtype=DTYPE),
                    momentum * torch.eye(dimension, dtype=DTYPE),
                ),
                dim=1,
            ),
        ),
        dim=0,
    )
    assert torch.allclose(jvp(rows), rows @ scalar_jacobian.T)
    assert torch.allclose(vjp(rows), rows @ scalar_jacobian)


def block_majorant_tests() -> None:
    generator = torch.Generator().manual_seed(733901)
    cases = 0
    for horizon in (1, 2, 4, 6):
        for dimension in (1, 2, 3):
            for _trial in range(12):
                eta = 0.023
                state_dimension = 2 * dimension
                parameter_projection = torch.cat(
                    (
                        torch.eye(dimension, dtype=DTYPE),
                        torch.zeros(dimension, dimension, dtype=DTYPE),
                    ),
                    dim=1,
                )
                injection = torch.cat(
                    (
                        -eta * torch.eye(dimension, dtype=DTYPE),
                        eta * torch.eye(dimension, dtype=DTYPE),
                    ),
                    dim=0,
                )
                approximate = []
                deltas = []
                exact = []
                for _step in range(horizon):
                    raw = torch.randn(
                        state_dimension,
                        state_dimension,
                        generator=generator,
                        dtype=DTYPE,
                    )
                    row = raw / (3.0 + float(torch.linalg.matrix_norm(raw)))
                    raw_delta = torch.randn(
                        dimension,
                        dimension,
                        generator=generator,
                        dtype=DTYPE,
                    )
                    delta = 0.08 * raw_delta / max(
                        float(torch.linalg.matrix_norm(raw_delta)), 1.0
                    )
                    approximate.append(row)
                    deltas.append(delta)
                    exact.append(row + injection @ delta @ parameter_projection)

                projection, structured_injection = explicit_projection_injection(
                    horizon, dimension, eta
                )
                t0 = (
                    projection
                    @ explicit_causal_green(approximate)
                    @ structured_injection
                )
                exact_operator = (
                    projection
                    @ explicit_causal_green(exact)
                    @ structured_injection
                )
                block_bounds = torch.zeros(horizon, horizon, dtype=DTYPE)
                for output_step in range(horizon):
                    for forcing_step in range(output_step + 1):
                        block = t0[
                            output_step * dimension : (output_step + 1) * dimension,
                            forcing_step * dimension : (forcing_step + 1) * dimension,
                        ]
                        block_bounds[output_step, forcing_step] = (
                            torch.linalg.matrix_norm(block, ord=2)
                        )
                mismatch_bounds = [
                    float(torch.linalg.matrix_norm(delta, ord=2))
                    for delta in deltas
                ]
                majorant, exact_majorant = causal_block_majorant(
                    block_bounds, mismatch_bounds
                )
                assert torch.allclose(
                    torch.triu(majorant), torch.zeros_like(majorant)
                )
                for output_step in range(horizon):
                    for forcing_step in range(output_step + 1):
                        block = exact_operator[
                            output_step * dimension : (output_step + 1) * dimension,
                            forcing_step * dimension : (forcing_step + 1) * dimension,
                        ]
                        observed = float(torch.linalg.matrix_norm(block, ord=2))
                        assert observed <= float(
                            exact_majorant[output_step, forcing_step]
                        ) * (1.0 + 4e-12) + 4e-14
                exact_gain = float(
                    torch.linalg.matrix_norm(exact_operator, ord=2)
                )
                majorant_gain = causal_block_majorant_gain(
                    block_bounds, mismatch_bounds
                )
                assert exact_gain <= majorant_gain * (1.0 + 4e-12) + 4e-14

                forcing = torch.randn(
                    horizon, dimension, generator=generator, dtype=DTYPE
                )
                forcing_norms = [
                    float(torch.linalg.vector_norm(row)) for row in forcing
                ]
                response = exact_operator @ forcing.reshape(-1)
                response_bound = causal_block_majorant_response_bound(
                    exact_majorant, forcing_norms
                )
                assert float(torch.linalg.vector_norm(response)) <= (
                    response_bound * (1.0 + 4e-12) + 4e-14
                )

                curvature = [
                    0.0,
                    *[
                        0.2
                        + float(
                            torch.rand((), generator=generator, dtype=DTYPE)
                        )
                        for _ in range(horizon - 1)
                    ],
                ]
                profile = torch.zeros(
                    horizon * dimension,
                    max(0, horizon - 1) * dimension,
                    dtype=DTYPE,
                )
                for step in range(1, horizon):
                    profile[
                        step * dimension : (step + 1) * dimension,
                        (step - 1) * dimension : step * dimension,
                    ] = curvature[step] * torch.eye(dimension, dtype=DTYPE)
                profiled_gain = causal_profiled_block_majorant_gain(
                    block_bounds, mismatch_bounds, curvature
                )
                observed_profiled = (
                    0.0
                    if horizon == 1
                    else float(
                        torch.linalg.matrix_norm(
                            exact_operator @ profile, ord=2
                        )
                    )
                )
                assert observed_profiled <= profiled_gain * (
                    1.0 + 4e-12
                ) + 4e-14
                cases += 1
    assert cases == 144


def forward_radius_tests() -> None:
    generator = torch.Generator().manual_seed(492031)
    cases = 0
    for horizon in (1, 2, 4, 7):
        for dimension in (1, 2, 3):
            for _trial in range(20):
                blocks: list[list[torch.Tensor | None]] = [
                    [None for _ in range(horizon)] for _ in range(horizon)
                ]
                majorant = torch.zeros(horizon, horizon, dtype=DTYPE)
                for output_step in range(horizon):
                    for forcing_step in range(output_step + 1):
                        raw = torch.randn(
                            dimension,
                            dimension,
                            generator=generator,
                            dtype=DTYPE,
                        )
                        block = 0.18 * raw / max(
                            float(torch.linalg.matrix_norm(raw, ord=2)), 1.0
                        )
                        blocks[output_step][forcing_step] = block
                        majorant[output_step, forcing_step] = (
                            torch.linalg.matrix_norm(block, ord=2)
                        )

                affine_vectors = 0.015 * torch.randn(
                    horizon,
                    dimension,
                    generator=generator,
                    dtype=DTYPE,
                )
                affine_bounds = torch.linalg.vector_norm(
                    affine_vectors, dim=1
                )
                curvature = [
                    0.2
                    + float(torch.rand((), generator=generator, dtype=DTYPE))
                    for _ in range(horizon)
                ]
                nonlinear_directions = []
                for _step in range(horizon):
                    raw = torch.randn(
                        dimension,
                        dimension,
                        generator=generator,
                        dtype=DTYPE,
                    )
                    nonlinear_directions.append(
                        raw
                        / max(
                            float(torch.linalg.matrix_norm(raw, ord=2)),
                            1.0,
                        )
                    )

                parameter_errors = [torch.zeros(dimension, dtype=DTYPE)]
                nonlinear_forcing = [torch.zeros(dimension, dtype=DTYPE)]
                for output_step in range(horizon):
                    if output_step > 0:
                        previous = parameter_errors[output_step]
                        nonlinear_forcing.append(
                            0.5
                            * curvature[output_step]
                            * torch.linalg.vector_norm(previous)
                            * (nonlinear_directions[output_step] @ previous)
                        )
                    value = affine_vectors[output_step].clone()
                    for forcing_step in range(output_step + 1):
                        block = blocks[output_step][forcing_step]
                        assert block is not None
                        value = value + block @ nonlinear_forcing[forcing_step]
                    parameter_errors.append(value)

                radii = causal_forward_quadratic_envelope(
                    affine_bounds, majorant, curvature
                )
                observed = torch.stack(
                    [
                        torch.linalg.vector_norm(parameter_errors[step])
                        for step in range(1, horizon + 1)
                    ]
                )
                assert bool(
                    (
                        observed
                        <= radii * (1.0 + 5e-13) + 5e-15
                    ).all()
                )

                structured_error = [
                    1e-4 * (index + 1) for index in range(horizon)
                ]
                complement = 0.4 * majorant
                complement_error = [
                    2e-4 * (index + 1) for index in range(horizon)
                ]
                directional = causal_directional_affine_bounds(
                    affine_bounds,
                    majorant,
                    structured_error,
                    complement_green_block_majorant=complement,
                    complement_forcing_error_bounds=complement_error,
                )
                expected = (
                    affine_bounds
                    + majorant
                    @ torch.tensor(structured_error, dtype=DTYPE)
                    + complement
                    @ torch.tensor(complement_error, dtype=DTYPE)
                )
                assert torch.allclose(
                    directional, expected, atol=2e-15, rtol=2e-15
                )
                cases += 1

    # A late, very large curvature bound makes the scalar sequence-norm
    # discriminant fail even though the state entering that update is tiny.
    # The triangular recursion retains the time-local information.
    witness_majorant = torch.eye(4, dtype=DTYPE)
    witness_affine = torch.tensor((1e-2, 1e-2, 1e-8, 1e-2), dtype=DTYPE)
    witness_curvature = (0.0, 0.0, 0.0, 1e8)
    witness_radii = causal_forward_quadratic_envelope(
        witness_affine, witness_majorant, witness_curvature
    )
    global_discriminant = 1.0 - 2.0 * 1e8 * float(
        torch.linalg.vector_norm(witness_affine)
    )
    assert global_discriminant < 0.0
    assert float(witness_radii.max()) < 0.010001
    assert cases == 240


def main() -> None:
    scalar_and_edge_tests()
    identity_and_adjoint_tests()
    block_majorant_tests()
    forward_radius_tests()
    print(
        "PASS: 36 exact causal resolvent identities and 144 causal "
        "block-majorant checks plus 240 forward nonlinear envelopes"
    )


if __name__ == "__main__":
    main()
