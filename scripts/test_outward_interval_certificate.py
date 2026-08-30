import unittest

import numpy as np
import torch

from generate_smooth_mlp_seed import frozen_config
from outward_interval_certificate import (
    _gradient_interval,
    _interval_matmul,
    _network_intervals,
    _verified_beta,
)
from smooth_mlp_modular_grokking import analytic_gradient, initialize, make_split


class OutwardIntervalTests(unittest.TestCase):
    def test_interval_matrix_product_contains_sampled_products(self):
        rng = np.random.default_rng(7)
        left_center = rng.normal(size=(3, 4))
        right_center = rng.normal(size=(4, 2))
        left_radius = rng.uniform(0.0, 1e-5, size=left_center.shape)
        right_radius = rng.uniform(0.0, 1e-5, size=right_center.shape)
        lower, upper = _interval_matmul(
            left_center - left_radius,
            left_center + left_radius,
            right_center - right_radius,
            right_center + right_radius,
        )
        for _ in range(100):
            left = rng.uniform(left_center - left_radius, left_center + left_radius)
            right = rng.uniform(right_center - right_radius, right_center + right_radius)
            product = left @ right
            self.assertTrue(np.all(lower <= product))
            self.assertTrue(np.all(product <= upper))

    def test_arb_network_and_gradient_enclose_float_evaluation(self):
        config = frozen_config(0)
        parameter = initialize(config)
        train_pairs, train_labels, _, _ = make_split(config)
        array = parameter.numpy()
        network = _network_intervals(array, train_pairs, train_labels, config)
        lower, upper = _gradient_interval(
            array, train_pairs, train_labels, config, network
        )
        gradient = analytic_gradient(
            parameter, train_pairs, train_labels, config
        ).numpy()
        self.assertTrue(np.all(lower <= gradient))
        self.assertTrue(np.all(gradient <= upper))
        self.assertGreater(float(np.max(upper - lower)), 0.0)

    def test_verified_beta_dominates_float_spectral_value(self):
        matrix = np.asarray(
            [[0.3, -0.1, 0.02], [-0.1, -0.05, 0.04], [0.02, 0.04, 0.2]],
            dtype=np.float64,
        )
        learning_rate = 2.0
        beta, diagnostics = _verified_beta(matrix, 0.0, learning_rate)
        exact_float_matrix_beta = float(
            np.max(np.abs(1.0 - learning_rate * np.linalg.eigvalsh(matrix)))
        )
        self.assertGreaterEqual(beta, exact_float_matrix_beta)
        self.assertGreaterEqual(diagnostics["total_eigenvalue_radius"], 0.0)


if __name__ == "__main__":
    unittest.main()
