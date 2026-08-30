#!/usr/bin/env python3
"""Independent protocol and scaled-optimizer operator gates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from probe_jacobian_bound import namespaced_probe_seed
from transformer_certificate_protocol import (
    FAMILY_FAILURE_PROBABILITY,
    HORIZON,
    OPTIMIZER_JACOBIAN,
    OUTPUT_JACOBIAN,
    SWEEPS,
    Candidate,
    candidate_universe,
    make_registry,
    maximum_operator_count,
    operator_identity,
    per_operator_failure_probability,
    scan_anchor_count,
)
from transformer_hvp_grokking import (
    TransformerConfig,
    flat_spec,
    make_disjoint_split,
    make_template,
)
from transformer_optimizer_probe import make_scaled_optimizer_jvp_vjp

ROOT = Path(__file__).resolve().parents[1]


def gate_count_and_endpoints() -> None:
    count = maximum_operator_count()
    assert scan_anchor_count() == 31
    assert count["probabilistic_screen_operators"] == 0
    assert count["output_jacobian_states"] == 301
    assert count["optimizer_jacobian_transitions"] == 300
    assert count["maximum_probabilistic_operators"] == 14_424
    delta = per_operator_failure_probability()
    assert delta * count["maximum_probabilistic_operators"] <= FAMILY_FAILURE_PROBABILITY
    print("gate C1: inclusive endpoints and 14,424-operator maximum  OK")


def gate_registry() -> None:
    candidates = (
        Candidate(321, 0.70, 1440),
        Candidate(322, 0.70, 2400),
        Candidate(322, 0.80, 2640),
    )
    universe = candidate_universe(candidates)
    assert len(universe) == len(candidates) * 601
    registry = make_registry(candidates, "independent-development-audit-v1")
    identity = operator_identity(candidates[0], 0, OUTPUT_JACOBIAN)
    seed = registry.claim(identity)
    assert seed == namespaced_probe_seed("independent-development-audit-v1", identity)
    try:
        registry.claim(identity)
    except RuntimeError:
        pass
    else:
        raise AssertionError("duplicate probabilistic query was not rejected")
    outside = (1, 999, 0, 0, 0, SWEEPS, OUTPUT_JACOBIAN)
    try:
        registry.claim(outside)
    except RuntimeError:
        pass
    else:
        raise AssertionError("out-of-universe query was not rejected")
    assert registry.summary()["all_queries_predeclared"]
    print("gate C2: namespaced streams and hard runtime universe  OK")


def gate_information_barrier() -> None:
    selector = (ROOT / "scripts" / "run_transformer_hvp_prospective_audit.py").read_text(
        encoding="utf-8"
    ).lower()
    forbidden = (
        "probe_jacobian_bound",
        "transformer_optimizer_probe",
        "jacobian_norm_bound(",
        "scaled_optimizer_norm_bound(",
    )
    assert not any(token in selector for token in forbidden)
    assert SWEEPS == 4
    print("gate C2b: candidate selector cannot call a probabilistic probe  OK")


def load_anchor():
    seed, anchor = 321, 1440
    payload = json.loads(
        (ROOT / "results" / f"transformer_hvp_prospective_seed_{seed}.json").read_text()
    )
    config = TransformerConfig(**payload["config"])
    template = make_template(config)
    spec = flat_spec(template)
    checkpoint = np.load(
        ROOT / "results" / f"transformer_hvp_prospective_seed_{seed}.checkpoints.npz"
    )
    parameter = torch.from_numpy(checkpoint[f"step_{anchor}"]).clone()
    train_pairs, train_labels = make_disjoint_split(config)[:2]
    return parameter, train_pairs, train_labels, template, spec, config


def gate_scaled_optimizer_products() -> None:
    parameter, pairs, labels, template, spec, config = load_anchor()
    jvp, vjp = make_scaled_optimizer_jvp_vjp(
        parameter, pairs, labels, template, spec, config
    )
    generator = torch.Generator().manual_seed(904)
    for _ in range(3):
        direction = torch.randn(
            2 * parameter.numel(), generator=generator, dtype=parameter.dtype
        )
        cotangent = torch.randn(
            2 * parameter.numel(), generator=generator, dtype=parameter.dtype
        )
        left = torch.dot(jvp(direction), cotangent)
        right = torch.dot(direction, vjp(cotangent))
        relative = float(torch.abs(left - right)) / max(float(torch.abs(left)), 1.0)
        assert relative < 1e-11, relative

        gram_direction = vjp(jvp(direction))
        quadratic = float(torch.dot(direction, gram_direction))
        squared = float(torch.dot(jvp(direction), jvp(direction)))
        assert quadratic >= -1e-8
        assert abs(quadratic - squared) / max(abs(squared), 1.0) < 1e-11
    print("gate C3: scaled momentum JVP/VJP are adjoints and J^T J is PSD  OK")


def main() -> None:
    gate_count_and_endpoints()
    gate_registry()
    gate_information_barrier()
    gate_scaled_optimizer_products()
    print("PASS")


if __name__ == "__main__":
    main()
