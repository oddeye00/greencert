#!/usr/bin/env python3
from __future__ import annotations

import unittest

import torch

from transformer_hvp_grokking import (
    TransformerConfig,
    flat_spec,
    flatten_parameters,
    gradient,
    logits,
    make_disjoint_split,
    make_template,
    objective_hvp,
)


class TransformerHVPGrokkingTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        self.config = TransformerConfig(
            modulus=7,
            model_dim=8,
            hidden_dim=16,
            heads=2,
            steps=2,
            dtype="float64",
        )
        self.template = make_template(self.config)
        self.spec = flat_spec(self.template)
        self.parameter = flatten_parameters(self.template)
        self.data = make_disjoint_split(self.config)

    def test_split_is_disjoint_and_exhaustive(self) -> None:
        groups = []
        for pairs in self.data[::2]:
            groups.append({tuple(row) for row in pairs.tolist()})
        self.assertEqual(sum(map(len, groups)), self.config.modulus**2)
        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])

    def test_flat_functional_logits(self) -> None:
        pairs = self.data[0][:5]
        direct = self.template(pairs)
        flat = logits(self.parameter, pairs, self.template, self.spec)
        torch.testing.assert_close(direct, flat)

    def test_hvp_matches_gradient_difference(self) -> None:
        pairs, labels = self.data[:2]
        vector = torch.randn_like(self.parameter)
        vector /= torch.linalg.vector_norm(vector)
        product = objective_hvp(
            self.parameter, vector, pairs, labels,
            self.template, self.spec, self.config,
        )
        eps = 2e-5
        plus = gradient(
            self.parameter + eps * vector, pairs, labels,
            self.template, self.spec, self.config,
        )
        minus = gradient(
            self.parameter - eps * vector, pairs, labels,
            self.template, self.spec, self.config,
        )
        torch.testing.assert_close(product, (plus - minus) / (2 * eps), rtol=2e-4, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
