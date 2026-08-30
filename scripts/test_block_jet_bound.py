#!/usr/bin/env python3
"""Validity gates for the block-aware ball-valid Transformer envelope.

Gate A -- monomial supremum is correct.
Gate B -- setting every s_b = 1 and using worst-case values reproduces the
          shipped scalar jet exactly (so the new bound is valid wherever the
          old one is).
Gate C -- the new bound never exceeds the shipped bound.
Gate D -- random-direction stress: the certified first/second derivative
          bounds are never violated by finite-difference directional
          derivatives of the real network, at radii spanning the certified ball.
Gate E -- the ball-valid value chain is a self-consistent fixed point.
"""
from __future__ import annotations

from math import sqrt

import numpy as np
import torch

from block_jet_bound import (
    BlockJet,
    add,
    block_linear,
    monomial_sphere_supremum,
    product,
    smooth_map,
)
from transformer_block_envelope import ball_valid_envelope, exact_stage_values
from transformer_hvp_grokking import (
    TransformerConfig,
    flat_spec,
    logits,
    make_template,
)
from transformer_jet_bound import transformer_output_jet_bound

SEED, ANCHOR = 321, 1440
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load():
    import json

    payload = json.loads(
        (ROOT / "results" / f"transformer_hvp_prospective_seed_{SEED}.json").read_text(
            encoding="utf-8"
        )
    )
    config = TransformerConfig(**payload["config"])
    template = make_template(config)
    spec = flat_spec(template)
    ckpt = np.load(
        ROOT / "results" / f"transformer_hvp_prospective_seed_{SEED}.checkpoints.npz"
    )
    parameter = torch.from_numpy(ckpt[f"step_{ANCHOR}"]).clone()
    return parameter, template, spec, config


def gate_a():
    assert abs(monomial_sphere_supremum((0,)) - 1.0) < 1e-12
    assert abs(monomial_sphere_supremum((0, 0)) - 1.0) < 1e-12
    assert abs(monomial_sphere_supremum((0, 1)) - 0.5) < 1e-12
    assert abs(monomial_sphere_supremum((0, 0, 0)) - 1.0) < 1e-12
    assert abs(monomial_sphere_supremum((0, 0, 1)) - (2 / 3) * sqrt(1 / 3)) < 1e-12
    assert abs(monomial_sphere_supremum((0, 1, 2)) - (1 / 3) ** 1.5) < 1e-12
    # brute-force check on the sphere
    rng = np.random.default_rng(7)
    for monomial in [(0, 1), (0, 0, 1), (0, 1, 2), (0, 0, 0)]:
        best = 0.0
        for _ in range(200_000):
            s = rng.normal(size=3)
            s /= np.linalg.norm(s)
            v = 1.0
            for b in monomial:
                v *= abs(s[b])
            best = max(best, v)
        assert best <= monomial_sphere_supremum(monomial) + 1e-6, (monomial, best)
    print("gate A: monomial sphere suprema correct  OK")


def gate_a2_mixed_terms():
    """Collapse arbitrary block radii and compare every mixed product/chain term."""
    rng = np.random.default_rng(81)
    for _ in range(100):
        radii = {index: float(value) for index, value in enumerate(rng.random(5))}
        left = add(
            block_linear(float(rng.random()), 0, float(rng.random())),
            product(
                block_linear(float(rng.random()), 1, float(rng.random())),
                block_linear(float(rng.random()), 2, float(rng.random())),
            ),
        )
        right = smooth_map(
            add(
                block_linear(float(rng.random()), 3, float(rng.random())),
                block_linear(float(rng.random()), 4, float(rng.random())),
            ),
            value=float(rng.random()),
            first=0.7,
            second=1.1,
            third=1.9,
        )
        got = product(left, right, scale=0.37)
        lv = [left.value] + [left.evaluate(order, radii) for order in (1, 2, 3)]
        rv = [right.value] + [right.evaluate(order, radii) for order in (1, 2, 3)]
        expected = (
            0.37 * (lv[1] * rv[0] + lv[0] * rv[1]),
            0.37 * (lv[2] * rv[0] + 2 * lv[1] * rv[1] + lv[0] * rv[2]),
            0.37 * (
                lv[3] * rv[0]
                + 3 * lv[2] * rv[1]
                + 3 * lv[1] * rv[2]
                + lv[0] * rv[3]
            ),
        )
        for order, reference in enumerate(expected, 1):
            actual = got.evaluate(order, radii)
            assert abs(actual - reference) <= 1e-11 * max(abs(reference), 1.0)
    print("gate A2: all mixed degree-1/2/3 product and chain terms agree  OK")


def gate_b_c(parameter, template, spec, config):
    generator = torch.Generator().manual_seed(550)
    for perturbation in (0.0, 1e-5):
        if perturbation:
            direction = torch.randn(
                parameter.shape, generator=generator, dtype=parameter.dtype
            )
            direction /= torch.linalg.vector_norm(direction)
            point = parameter + perturbation * direction
        else:
            point = parameter
        for radius in (0.0, 1e-8, 1e-4):
            shipped = transformer_output_jet_bound(
                point, template, spec, config, radius=radius
            )
            scalar = ball_valid_envelope(
                point,
                spec,
                config,
                epsilon=radius,
                exact_values=False,
                sphere=False,
            )
            for name, a, b in (
                ("value", shipped.value, scalar["value"]),
                ("first", shipped.first, scalar["first"]),
                ("second", shipped.second, scalar["second"]),
                ("third", shipped.third, scalar["third"]),
            ):
                rel = abs(a - b) / max(abs(a), 1e-30)
                assert rel < 1e-9, (name, perturbation, radius, a, b, rel)
    print(f"gate B: scalar reduction reproduces shipped jet   max rel dev < 1e-9  OK")

    shipped = transformer_output_jet_bound(parameter, template, spec, config, radius=0.0)
    scalar = ball_valid_envelope(
        parameter, spec, config, epsilon=0.0, exact_values=False, sphere=False
    )
    tight = ball_valid_envelope(
        parameter, spec, config, epsilon=5.37e-8, exact_values=True, sphere=True
    )
    for name in ("first", "second", "third"):
        assert tight[name] <= scalar[name] * (1 + 1e-12), (name, tight[name], scalar[name])
    print(
        "gate C: new <= shipped  "
        f"first {shipped.first:.4g} -> {tight['first']:.4g} "
        f"({shipped.first/tight['first']:.1f}x)  "
        f"second {shipped.second:.4g} -> {tight['second']:.4g} "
        f"({shipped.second/tight['second']:.1f}x)  OK"
    )
    return tight


@torch.no_grad()
def gate_d(parameter, template, spec, config, tight):
    """Directional finite differences must not exceed the certified bounds."""
    p = config.modulus
    pairs = torch.cartesian_prod(torch.arange(p), torch.arange(p)).long()
    generator = torch.Generator().manual_seed(31337)
    worst_first = 0.0
    worst_second = 0.0
    checked = 0
    for h in (1e-6, 1e-5, 1e-4):
        for _ in range(8):
            u = torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
            u = u / torch.linalg.vector_norm(u)
            base = logits(parameter, pairs, template, spec)
            plus = logits(parameter + h * u, pairs, template, spec)
            minus = logits(parameter - h * u, pairs, template, spec)
            d1 = float(torch.linalg.matrix_norm(plus - minus) / (2 * h))
            d2 = float(torch.linalg.matrix_norm(plus - 2 * base + minus) / (h * h))
            worst_first = max(worst_first, d1)
            worst_second = max(worst_second, d2)
            checked += 1
            assert d1 <= tight["first"] * (1 + 1e-6), (h, d1, tight["first"])
            assert d2 <= tight["second"] * (1 + 1e-6), (h, d2, tight["second"])
    print(
        f"gate D: {checked} random-direction stress tests, 0 violations   "
        f"max observed first {worst_first:.4g} vs bound {tight['first']:.4g} "
        f"(ratio {worst_first/tight['first']:.4f}); "
        f"second {worst_second:.4g} vs {tight['second']:.4g} "
        f"(ratio {worst_second/tight['second']:.2e})  OK"
    )


def gate_e(tight):
    assert tight["fixed_point_consistent"], tight["inflation"]
    for name, first in tight["stage_first"].items():
        assert first * 5.37e-8 <= tight["inflation"][name] * (1 + 1e-9) + 1e-18
    print(
        "gate E: ball value chain is a consistent fixed point   "
        f"max inflation {max(tight['inflation'].values()):.3e}  OK"
    )


def main():
    parameter, template, spec, config = load()
    gate_a()
    gate_a2_mixed_terms()
    tight = gate_b_c(parameter, template, spec, config)
    gate_d(parameter, template, spec, config, tight)
    gate_e(tight)
    print("PASS")


if __name__ == "__main__":
    main()
