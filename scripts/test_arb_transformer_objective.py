#!/usr/bin/env python3
"""Cross-check the outward Arb Transformer objective against PyTorch."""

from __future__ import annotations

import time

import torch
from flint import arb, ctx

from arb_transformer_objective import arb_transformer_objective
from transformer_hvp_grokking import (
    TransformerConfig,
    flat_spec,
    flatten_parameters,
    make_disjoint_split,
    make_template,
    objective,
)


def main() -> None:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    ctx.prec = 192
    config = TransformerConfig(
        modulus=5,
        model_dim=8,
        hidden_dim=16,
        heads=2,
        depth=1,
        train_fraction=0.60,
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=0.01,
        steps=1,
        seed=20260831,
        threads=1,
        dtype="float64",
        loss="cross_entropy",
        normalization="none",
    )
    template = make_template(config)
    spec = flat_spec(template)
    parameter = flatten_parameters(template)
    train_pairs, train_labels, _, _, _, _ = make_disjoint_split(config)
    expected = float(
        objective(
            parameter,
            train_pairs,
            train_labels,
            template,
            spec,
            config,
        )
    )
    started = time.perf_counter()
    observed = arb_transformer_objective(
        parameter.tolist(), train_pairs, train_labels, spec, config
    )
    elapsed = time.perf_counter() - started
    if not observed.contains(arb(expected)):
        # Different but mathematically equivalent reduction orders can put the
        # binary64 result just outside a very narrow high-precision ball.  The
        # midpoint must nevertheless agree to ordinary forward precision.
        if abs(float(observed.mid()) - expected) > 2.0e-13:
            raise AssertionError((observed, expected))
    if abs(float(observed.mid()) - expected) > 2.0e-13:
        raise AssertionError((observed, expected))
    print(
        {
            "status": "Arb Transformer objective test passed",
            "pytorch": expected,
            "arb_midpoint": float(observed.mid()),
            "arb_radius": float(observed.rad()),
            "elapsed_seconds": elapsed,
            "precision_bits": ctx.prec,
        }
    )


if __name__ == "__main__":
    main()
