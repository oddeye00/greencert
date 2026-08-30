#!/usr/bin/env python3
from __future__ import annotations

import unittest

import torch

from transformer_hvp_grokking import (
    TransformerConfig,
    flat_spec,
    flatten_parameters,
    make_disjoint_split,
    make_template,
)
from transformer_modal_forecast import optimizer_jvp, optimizer_map


class TransformerModalForecastTests(unittest.TestCase):
    def test_optimizer_jvp_matches_difference(self) -> None:
        torch.set_num_threads(1)
        config = TransformerConfig(
            modulus=7,
            model_dim=8,
            hidden_dim=16,
            heads=2,
            learning_rate=0.02,
            momentum=0.9,
            weight_decay=0.001,
        )
        template = make_template(config)
        spec = flat_spec(template)
        parameter = flatten_parameters(template)
        velocity = torch.randn_like(parameter) * 0.01
        state = torch.cat((parameter, velocity))
        direction = torch.randn_like(state)
        direction /= torch.linalg.vector_norm(direction)
        pairs, labels = make_disjoint_split(config)[:2]
        product = optimizer_jvp(
            state, direction, pairs, labels, template, spec, config
        )
        eps = 1e-5
        plus = optimizer_map(
            state + eps * direction, pairs, labels, template, spec, config
        )
        minus = optimizer_map(
            state - eps * direction, pairs, labels, template, spec, config
        )
        torch.testing.assert_close(product, (plus - minus) / (2 * eps), rtol=3e-4, atol=3e-6)


if __name__ == "__main__":
    unittest.main()
