#!/usr/bin/env python3
"""Validity gates for the probabilistic centre-Jacobian enclosure.

Gate P1 -- c_delta satisfies 2*Phi(c_delta) - 1 = delta^{1/m} exactly.
Gate P2 -- on an explicit small matrix where ||J|| is computable by SVD, the
           bound holds and the empirical failure rate over many independent
           probe draws is at most the nominal delta.
Gate P3 -- the Gram operator really is J^T J: compare A v against an explicitly
           formed Jacobian on the real Transformer, in double precision.
Gate P4 -- the bound upper-bounds the power-iteration lower estimate and the
           finite-difference directional derivatives on the real Transformer.
Gate P5 -- the operator-identity -> RNG map is deterministic and collision-free
           across the enumerated protocol coordinates.
"""
from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch

from probe_jacobian_bound import (
    ProbeConfig,
    c_delta,
    jacobian_norm_bound,
    make_gram_operator,
    permitted_operator_count,
    power_iteration_lower_estimate,
    probe_seed,
)
from transformer_hvp_grokking import (
    TransformerConfig,
    flat_spec,
    logits,
    make_template,
)

ROOT = Path(__file__).resolve().parents[1]
SEED, ANCHOR = 321, 1440


def load():
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


def gate_p1():
    nd = NormalDist()
    for delta in (1e-3, 1e-6, 1.39e-10):
        for m in (4, 16, 64):
            cd = c_delta(delta, m)
            assert abs((2 * nd.cdf(cd) - 1) - delta ** (1.0 / m)) < 1e-12
    print("gate P1: c_delta identity exact  OK")


def gate_p2():
    """Empirical failure rate on a matrix with known spectral norm."""
    torch.manual_seed(0)
    d, n = 40, 25
    J = torch.randn(n, d, dtype=torch.float64)
    truth = float(torch.linalg.matrix_norm(J, ord=2))
    A = J.T @ J

    m, q, delta = 8, 4, 0.05
    cd = c_delta(delta, m)
    trials, failures = 4000, 0
    generator = torch.Generator().manual_seed(2024)
    for _ in range(trials):
        best = 0.0
        for _ in range(m):
            g = torch.randn(d, generator=generator, dtype=torch.float64)
            w = g
            for _ in range(q):
                w = A @ w
            best = max(best, float(torch.linalg.vector_norm(w)))
        bound = (best / cd) ** (1.0 / (2 * q))
        if bound < truth * (1 - 1e-12):
            failures += 1
    rate = failures / trials
    print(
        f"gate P2: nominal delta={delta}, empirical failure rate={rate:.4f} "
        f"over {trials} trials (truth {truth:.4f})  "
        + ("OK" if rate <= delta else "FAIL")
    )
    assert rate <= delta, (rate, delta)


def gate_p3(parameter, template, spec, config):
    """A v must equal J^T (J v) for the explicitly formed J."""
    p = config.modulus
    pairs = torch.cartesian_prod(torch.arange(p), torch.arange(p)).long()[:12]

    def forward(theta):
        return logits(theta, pairs, template, spec).reshape(-1)

    J = torch.autograd.functional.jacobian(forward, parameter, vectorize=False)
    apply, products = make_gram_operator(parameter, pairs, template, spec)
    generator = torch.Generator().manual_seed(5)
    worst = 0.0
    for _ in range(3):
        v = torch.randn(parameter.numel(), generator=generator, dtype=parameter.dtype)
        reference = J.T @ (J @ v)
        got = apply(v)
        worst = max(
            worst,
            float(torch.linalg.vector_norm(got - reference))
            / max(float(torch.linalg.vector_norm(reference)), 1e-30),
        )
    v = torch.randn(parameter.numel(), generator=generator, dtype=parameter.dtype)
    w = torch.randn(J.shape[0], generator=generator, dtype=parameter.dtype)
    jv_relative = float(torch.linalg.vector_norm(products["jvp"](v) - J @ v)) / max(
        float(torch.linalg.vector_norm(J @ v)), 1e-30
    )
    vjp_relative = float(
        torch.linalg.vector_norm(products["vjp"](w) - J.T @ w)
    ) / max(float(torch.linalg.vector_norm(J.T @ w)), 1e-30)
    assert worst < 1e-10, worst
    assert jv_relative < 1e-10, jv_relative
    assert vjp_relative < 1e-10, vjp_relative
    explicit = float(torch.linalg.matrix_norm(J, ord=2))
    print(f"gate P3: Jv, J^T w, and Gram operator == J^T J   "
          f"rel dev {jv_relative:.1e}/{vjp_relative:.1e}/{worst:.1e}  "
          f"(explicit ||J||={explicit:.4f} on 12 inputs)  OK")
    return pairs, explicit


def gate_p4(parameter, template, spec, config):
    p = config.modulus
    pairs = torch.cartesian_prod(torch.arange(p), torch.arange(p)).long()
    cfg = ProbeConfig(probes=16, power=8, delta=1.39e-10)
    result = jacobian_norm_bound(parameter, pairs, template, spec, cfg, (SEED, ANCHOR, 0, 0))
    lower = power_iteration_lower_estimate(parameter, pairs, template, spec)

    generator = torch.Generator().manual_seed(99)
    h, worst_fd = 1e-6, 0.0
    for _ in range(8):
        u = torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
        u = u / torch.linalg.vector_norm(u)
        plus = logits(parameter + h * u, pairs, template, spec)
        minus = logits(parameter - h * u, pairs, template, spec)
        worst_fd = max(worst_fd, float(torch.linalg.matrix_norm(plus - minus) / (2 * h)))

    bound = result["jacobian_norm_upper_bound"]
    assert bound >= lower * (1 - 1e-9), (bound, lower)
    assert bound >= worst_fd * (1 - 1e-9), (bound, worst_fd)
    print(
        f"gate P4: bound {bound:.4f} >= power-iteration lower estimate {lower:.4f} "
        f"(ratio {bound/lower:.4f}) and >= max finite-difference {worst_fd:.4f}  OK"
    )
    return result, lower


def gate_p5():
    seen = {}
    for seed in range(321, 341):
        for gate in range(3):
            for step in range(300):
                key = probe_seed(seed, gate, step)
                identity = (seed, gate, step)
                assert key not in seen or seen[key] == identity, (key, identity, seen[key])
                seen[key] = identity
    assert probe_seed(1, 2, 3) == probe_seed(1, 2, 3)
    print(f"gate P5: {len(seen)} operator identities, deterministic, no collisions  OK")


def main():
    parameter, template, spec, config = load()
    gate_p1()
    gate_p2()
    gate_p3(parameter, template, spec, config)
    result, lower = gate_p4(parameter, template, spec, config)
    gate_p5()
    count = permitted_operator_count(
        seeds=8, gates_per_seed=3, candidates_per_gate=1,
        horizon_steps=300, operators_per_step=1,
        screen_anchors_per_gate=30, operators_per_screen_anchor=1,
    )
    print("\npermitted operator count:", json.dumps(count))
    print(f"  cost per operator: {result['gram_applications']} JVP + "
          f"{result['gram_applications']} VJP")
    print("PASS")


if __name__ == "__main__":
    main()
