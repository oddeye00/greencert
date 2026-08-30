#!/usr/bin/env python3
"""Exact algebra and real-model adjoint gates for the Green operator."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from transformer_green_operator import (
    make_causal_green_products,
    make_transformer_green_products,
)
from transformer_green_protocol import maximum_operator_count
from transformer_hvp_grokking import (
    TransformerConfig,
    flat_spec,
    make_disjoint_split,
    make_template,
)

ROOT = Path(__file__).resolve().parents[1]


def gate_explicit_linear_sequence() -> None:
    generator = torch.Generator().manual_seed(41)
    horizon, dimension = 4, 3
    matrices = [
        0.3 * torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
        for _ in range(horizon)
    ]
    apply, transpose = make_causal_green_products(
        [lambda value, matrix=matrix: matrix @ value for matrix in matrices],
        [lambda value, matrix=matrix: matrix.T @ value for matrix in matrices],
        dimension,
    )
    total = horizon * dimension
    basis = torch.eye(total, dtype=torch.float64)
    explicit = torch.stack([apply(row) for row in basis], dim=1)
    vector = torch.randn(total, generator=generator, dtype=torch.float64)
    cotangent = torch.randn(total, generator=generator, dtype=torch.float64)
    assert torch.allclose(apply(vector), explicit @ vector, atol=1e-12, rtol=1e-12)
    assert torch.allclose(
        transpose(cotangent), explicit.T @ cotangent, atol=1e-12, rtol=1e-12
    )
    print("gate G1: causal Green JVP/VJP equal the explicit block operator  OK")


def gate_real_transformer_adjoint() -> None:
    seed, anchor, horizon = 321, 1440, 3
    payload = json.loads(
        (ROOT / "results" / f"transformer_hvp_prospective_seed_{seed}.json").read_text()
    )
    config = TransformerConfig(**payload["config"])
    template = make_template(config)
    spec = flat_spec(template)
    data = make_disjoint_split(config)
    checkpoint = np.load(
        ROOT / "results" / f"transformer_hvp_prospective_seed_{seed}.checkpoints.npz"
    )
    parameter = torch.from_numpy(checkpoint[f"step_{anchor}"]).clone()
    path = parameter.repeat(horizon, 1)
    apply, transpose = make_transformer_green_products(
        path, data[0], data[1], template, spec, config
    )
    generator = torch.Generator().manual_seed(43)
    size = horizon * 2 * parameter.numel()
    vector = torch.randn(size, generator=generator, dtype=parameter.dtype)
    cotangent = torch.randn(size, generator=generator, dtype=parameter.dtype)
    left = torch.dot(apply(vector), cotangent)
    right = torch.dot(vector, transpose(cotangent))
    relative = float(torch.abs(left - right)) / max(float(torch.abs(left)), 1.0)
    assert relative < 1e-11, relative
    print(f"gate G2: real Transformer Green products are adjoints ({relative:.2e})  OK")


def gate_accounting() -> None:
    count = maximum_operator_count()
    assert count["maximum_probabilistic_operators"] == 7_248
    print("gate G3: 7,248 maximum operators (301 output + one Green)  OK")


def gate_signed_response_shadowing() -> None:
    """Exercise the signed-response theorem on a nonlinear causal recurrence."""
    matrices = [
        torch.tensor([[1.2]], dtype=torch.float64),
        torch.tensor([[-1.1]], dtype=torch.float64),
        torch.tensor([[0.8]], dtype=torch.float64),
        torch.tensor([[-0.7]], dtype=torch.float64),
        torch.tensor([[0.5]], dtype=torch.float64),
    ]
    apply, _ = make_causal_green_products(
        [lambda value, matrix=matrix: matrix @ value for matrix in matrices],
        [lambda value, matrix=matrix: matrix.T @ value for matrix in matrices],
        1,
    )
    basis = torch.eye(len(matrices), dtype=torch.float64)
    explicit = torch.stack([apply(row) for row in basis], dim=1)
    kappa = float(torch.linalg.matrix_norm(explicit, ord=2))
    defect = torch.tensor([1.0, -1.0, 0.5, -0.5, 0.25], dtype=torch.float64) * 1e-4
    response = apply(defect)
    z_norm = float(torch.linalg.vector_norm(response))
    radius = 2.0 * z_norm
    drift = 0.2
    closure = 2.0 * kappa * drift * z_norm
    assert closure <= 1.0

    state = torch.zeros(1, dtype=torch.float64)
    errors = []
    for matrix, injection in zip(matrices, defect):
        state = matrix @ state + injection.reshape(1) + 0.5 * drift * state.square()
        errors.append(state.clone())
    actual = float(torch.linalg.vector_norm(torch.stack(errors)))
    assert actual <= radius
    print(
        "gate G4: signed-response nonlinear closure contains the exact causal path "
        f"({actual:.3e} <= {radius:.3e})  OK"
    )


def main() -> None:
    gate_explicit_linear_sequence()
    gate_real_transformer_adjoint()
    gate_accounting()
    gate_signed_response_shadowing()
    print("PASS")


if __name__ == "__main__":
    main()
