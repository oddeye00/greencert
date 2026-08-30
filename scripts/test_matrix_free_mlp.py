#!/usr/bin/env python3
from __future__ import annotations

import unittest

import torch

from matrix_free_mlp import (
    block_apply,
    block_krylov_basis,
    gauss_newton_vp,
    hvp_affine_reference,
    objective_hvp,
    residual_curvature_vp,
)
from smooth_mlp_certificate import exact_objective_hessian
from smooth_mlp_modular_grokking import Config, initialize, make_split


class MatrixFreeMLPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config(
            modulus=5, width=4, train_fraction=0.6, learning_rate=0.4,
            weight_decay=1e-3, seed=3,
        )
        self.parameter = initialize(self.config)
        self.train_pairs, self.train_labels, _, _ = make_split(self.config)

    def test_hvp_matches_dense_hessian(self) -> None:
        torch.manual_seed(4)
        vector = torch.randn_like(self.parameter)
        hessian, gauss = exact_objective_hessian(
            self.parameter, self.train_pairs, self.train_labels, self.config
        )
        self.assertTrue(torch.allclose(
            objective_hvp(
                self.parameter, vector, self.train_pairs, self.train_labels, self.config
            ),
            hessian @ vector,
            rtol=1e-10,
            atol=1e-11,
        ))
        self.assertTrue(torch.allclose(
            gauss_newton_vp(self.parameter, vector, self.train_pairs, self.config),
            gauss @ vector,
            rtol=1e-10,
            atol=1e-11,
        ))
        residual = residual_curvature_vp(
            self.parameter, vector, self.train_pairs, self.train_labels, self.config
        )
        expected = (hessian - gauss - self.config.weight_decay * torch.eye(
            len(vector), dtype=vector.dtype
        )) @ vector
        self.assertTrue(torch.allclose(residual, expected, rtol=1e-10, atol=1e-11))

    def test_block_krylov_is_orthonormal(self) -> None:
        hessian, _ = exact_objective_hessian(
            self.parameter, self.train_pairs, self.train_labels, self.config
        )
        starts = torch.stack((
            torch.arange(1, len(self.parameter) + 1, dtype=torch.float64),
            torch.linspace(-1.0, 1.0, len(self.parameter), dtype=torch.float64),
        ), dim=1)
        basis, diagnostic = block_krylov_basis(
            lambda v: hessian @ v, starts, rank=8
        )
        self.assertEqual(basis.shape, (len(self.parameter), 8))
        self.assertLess(diagnostic["orthogonality_error"], 1e-12)
        projected = basis.T @ block_apply(lambda v: hessian @ v, basis)
        self.assertTrue(torch.allclose(projected, projected.T, atol=1e-11, rtol=0.0))

    def test_hvp_reference_equals_dense_affine_recurrence(self) -> None:
        hessian, _ = exact_objective_hessian(
            self.parameter, self.train_pairs, self.train_labels, self.config
        )
        from smooth_mlp_modular_grokking import analytic_gradient

        gradient = analytic_gradient(
            self.parameter, self.train_pairs, self.train_labels, self.config
        )
        horizon = 12
        reference = hvp_affine_reference(
            self.parameter,
            gradient,
            lambda vector: objective_hvp(
                self.parameter,
                vector,
                self.train_pairs,
                self.train_labels,
                self.config,
            ),
            learning_rate=self.config.learning_rate,
            horizon=horizon,
        )
        displacement = torch.zeros_like(self.parameter)
        expected = [self.parameter.clone()]
        for _ in range(horizon):
            displacement = displacement - self.config.learning_rate * (
                gradient + hessian @ displacement
            )
            expected.append(self.parameter + displacement)
        self.assertTrue(torch.allclose(
            reference, torch.stack(expected), rtol=2e-10, atol=2e-11
        ))


if __name__ == "__main__":
    unittest.main()
