#!/usr/bin/env python3
"""Independent algebra, partition, and autodiff gates for the directional bound."""
from __future__ import annotations

import math
import random

import numpy as np
import torch

from transformer_directional_fourth_bound import (
    BLOCK_COUNT,
    BlockJet4,
    add,
    directional_objective_fourth_bound,
    evaluate_polynomial,
    homogeneous_gradient,
    parameter_block_radii,
    product,
    smooth_map,
)
from transformer_fourth_jet_bound import Jet4, product as scalar_product
from transformer_fourth_jet_bound import smooth_map as scalar_smooth_map
from transformer_hvp_grokking import (
    TransformerConfig,
    flat_spec,
    flatten_parameters,
    make_template,
    objective,
)


def collapsed(jet: BlockJet4) -> Jet4:
    radii = np.ones(BLOCK_COUNT, dtype=np.float64)
    return Jet4(
        jet.value,
        evaluate_polynomial(jet.p1, radii),
        evaluate_polynomial(jet.p2, radii),
        evaluate_polynomial(jet.p3, radii),
        evaluate_polynomial(jet.p4, radii),
    )


def close(left: float, right: float, label: str) -> None:
    if not math.isclose(left, right, rel_tol=5.0e-13, abs_tol=5.0e-13):
        raise AssertionError(f"{label}: {left} != {right}")


def algebra_gate() -> None:
    rng = random.Random(20260831)
    for case in range(60):
        left_values = [rng.uniform(0.01, 1.0) for _ in range(5)]
        right_values = [rng.uniform(0.01, 1.0) for _ in range(5)]
        left = BlockJet4(
            left_values[0],
            {(0,): left_values[1]},
            {(0, 0): left_values[2]},
            {(0, 0, 0): left_values[3]},
            {(0, 0, 0, 0): left_values[4]},
        )
        right = BlockJet4(
            right_values[0],
            {(1,): right_values[1]},
            {(1, 1): right_values[2]},
            {(1, 1, 1): right_values[3]},
            {(1, 1, 1, 1): right_values[4]},
        )
        observed = collapsed(product(left, right, 0.37))
        expected = scalar_product(Jet4(*left_values), Jet4(*right_values), scale=0.37)
        for order, (a, b) in enumerate(zip(observed.__dict__.values(), expected.__dict__.values())):
            close(float(a), float(b), f"product case {case} order {order}")

        constants = dict(first=0.7, second=1.1, third=1.9, fourth=2.3)
        observed_map = collapsed(
            smooth_map(left, value=0.41, **constants)
        )
        expected_map = scalar_smooth_map(
            Jet4(*left_values), value=0.41, **constants
        )
        for order, (a, b) in enumerate(
            zip(observed_map.__dict__.values(), expected_map.__dict__.values())
        ):
            close(float(a), float(b), f"chain case {case} order {order}")
    print("gate A: fourth-order block algebra collapses to scalar identities  OK")


def mixed_gradient_gate() -> None:
    polynomial = {
        (0, 0, 0, 0): 1.3,
        (0, 0, 1, 2): 0.7,
        (1, 1, 2, 2): 2.1,
    }
    radii = np.zeros(BLOCK_COUNT)
    radii[:3] = (0.2, 0.0, 0.4)
    gradient = homogeneous_gradient(polynomial, radii, BLOCK_COUNT)
    epsilon = 1.0e-6
    for block in range(3):
        plus = radii.copy()
        minus = radii.copy()
        plus[block] += epsilon
        minus[block] -= epsilon
        finite = (
            evaluate_polynomial(polynomial, plus)
            - evaluate_polynomial(polynomial, minus)
        ) / (2.0 * epsilon)
        if not math.isclose(
            float(gradient[block]), float(finite), rel_tol=2.0e-9, abs_tol=2.0e-12
        ):
            raise AssertionError(
                f"gradient block {block}: {gradient[block]} != {finite}"
            )
    value = evaluate_polynomial(polynomial, radii)
    close(float(np.dot(gradient, radii)), 4.0 * value, "Euler identity")
    print("gate B: mixed three-known/one-free contraction and zero radii  OK")


def mixed_fourth(function, point: torch.Tensor, directions: list[torch.Tensor]) -> float:
    variable = point.detach().requires_grad_(True)
    value = function(variable)
    for direction in directions:
        (gradient,) = torch.autograd.grad(value, variable, create_graph=True)
        value = torch.dot(gradient, direction)
    return abs(float(value.detach()))


def network_gate() -> None:
    torch.set_default_dtype(torch.float64)
    config = TransformerConfig(
        modulus=3,
        model_dim=4,
        hidden_dim=6,
        heads=1,
        depth=1,
        seed=119,
        dtype="float64",
        loss="cross_entropy",
        normalization="none",
    )
    template = make_template(config)
    spec = flat_spec(template)
    parameter = flatten_parameters(template)
    pairs = torch.cartesian_prod(
        torch.arange(config.modulus), torch.arange(config.modulus)
    ).long()
    labels = (pairs[:, 0] + pairs[:, 1]) % config.modulus
    generator = torch.Generator().manual_seed(20260831)
    cases = 0
    worst_ratio = 0.0
    for scale in (2.0e-5, 1.0e-4):
        direction = torch.randn(parameter.shape, generator=generator)
        direction *= scale / torch.linalg.vector_norm(direction)
        radii = parameter_block_radii(direction, spec, config)
        close(float(np.linalg.norm(radii)), float(torch.linalg.vector_norm(direction)), "partition")
        record = directional_objective_fourth_bound(
            parameter, direction, spec, config
        )
        upper = float(record["mixed_fourth_derivative_upper"])
        if not (math.isfinite(upper) and upper > 0.0):
            raise AssertionError("directional fourth bound is not finite and positive")
        polynomial_value = float(record["objective_polynomial_at_direction"])
        if polynomial_value > upper * float(torch.linalg.vector_norm(direction)) * (1.0 + 1.0e-12):
            raise AssertionError("mixed bound fails its diagonal Euler consequence")
        for fraction in (0.0, 0.43, 1.0):
            point = parameter + fraction * direction
            for _ in range(2):
                dual = torch.randn(parameter.shape, generator=generator)
                dual /= torch.linalg.vector_norm(dual)
                observed = mixed_fourth(
                    lambda theta: objective(
                        theta, pairs, labels, template, spec, config
                    ),
                    point,
                    [direction, direction, direction, dual],
                )
                ratio = observed / upper
                worst_ratio = max(worst_ratio, ratio)
                if observed > upper * (1.0 + 2.0e-11):
                    raise AssertionError(
                        f"mixed fourth derivative exceeds bound: {observed} > {upper}"
                    )
                cases += 1
    print(
        "gate C: segment mixed-derivative autodiff stress "
        f"{cases} cases, 0 violations, worst ratio {worst_ratio:.3e}  OK"
    )


def main() -> None:
    algebra_gate()
    mixed_gradient_gate()
    network_gate()
    print("PASS")


if __name__ == "__main__":
    main()
