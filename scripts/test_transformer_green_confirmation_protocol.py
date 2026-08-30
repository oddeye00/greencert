#!/usr/bin/env python3
"""Mechanical gates for the frozen fresh signed-Green protocol."""
from __future__ import annotations

import inspect
import json
import tempfile
from pathlib import Path

import torch

from probe_jacobian_bound import ProbeRegistry
from run_transformer_green_confirmation import scan_seed
from transformer_certificate_protocol import Candidate
from transformer_green_confirmation_certificate import gate_slacks, safe_json
from transformer_green_confirmation_protocol import (
    FAMILY_FAILURE_PROBABILITY,
    HORIZON,
    MASTER_NONCE,
    PERSISTENCE,
    PROTOCOL_VERSION,
    SEEDS,
    THRESHOLDS,
    candidate_universe,
    maximum_operator_count,
    probe_config,
)
from transformer_green_operator import make_causal_green_products


def gate_family_accounting() -> None:
    count = maximum_operator_count()
    assert len(SEEDS) == 24
    assert count["maximum_candidates"] == 72
    assert count["operators_per_candidate"] == 302
    assert count["maximum_probabilistic_operators"] == 21_744
    assert probe_config().delta == FAMILY_FAILURE_PROBABILITY / 21_744
    print("gate F1: 24 seeds and 21,744-operator maximum  OK")


def gate_stream_universe() -> None:
    candidates = tuple(
        Candidate(seed, threshold, 0) for seed in SEEDS for threshold in THRESHOLDS
    )
    horizons = {candidate: HORIZON for candidate in candidates}
    universe = candidate_universe(candidates, horizons)
    assert len(universe) == 21_744
    assert all(identity[0] == PROTOCOL_VERSION for identity in universe)
    registry = ProbeRegistry(universe, MASTER_NONCE)
    assert registry.stream_count == len(universe)
    assert len(MASTER_NONCE) == 64
    print("gate F2: full fresh universe is predeclared and collision-free  OK")


def gate_screen_is_deterministic() -> None:
    source = inspect.getsource(scan_seed).lower()
    forbidden = ("jacobian_norm_bound", "green_norm_bound", "probe_config", ".claim(")
    assert not any(token in source for token in forbidden)
    print("gate F3: blind candidate scanner contains no probabilistic query  OK")


def gate_forbidden_reads() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        allowed = root / "blind.json"
        allowed.write_text(json.dumps({"ok": True}), encoding="utf-8")
        assert safe_json(allowed)["ok"]
        for name in ("seed.outcomes.json", "seed.sealed.log"):
            path = root / name
            path.write_text("{}", encoding="utf-8")
            try:
                safe_json(path)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"forbidden read did not fail: {name}")
    print("gate F4: outcome and sealed-log reads hard-fail  OK")


def gate_green_norm_floor() -> None:
    matrices = [torch.tensor([[0.2]], dtype=torch.float64) for _ in range(4)]
    apply, _ = make_causal_green_products(
        [lambda value, matrix=matrix: matrix @ value for matrix in matrices],
        [lambda value, matrix=matrix: matrix.T @ value for matrix in matrices],
        1,
    )
    basis = torch.eye(4, dtype=torch.float64)
    explicit = torch.stack([apply(row) for row in basis], dim=1)
    assert float(torch.linalg.matrix_norm(explicit, ord=2)) >= 1.0
    final_only = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    assert torch.equal(apply(final_only), final_only)
    print("gate F5: causal Green operator has ||K_H|| >= 1  OK")


def gate_logic_slacks() -> None:
    logits = torch.tensor(
        [[3.0, 1.0], [0.0, 2.0], [1.5, 0.0]], dtype=torch.float64
    )
    labels = torch.tensor([0, 1, 0])
    guarantee, exclusion = gate_slacks(logits, labels, 0.1, required=2)
    assert guarantee > 1.0
    assert exclusion < 0.0
    assert PERSISTENCE == 25
    print("gate F6: strict gate slack has the expected sign and order statistic  OK")


def main() -> None:
    gate_family_accounting()
    gate_stream_universe()
    gate_screen_is_deterministic()
    gate_forbidden_reads()
    gate_green_norm_floor()
    gate_logic_slacks()
    print("PASS")


if __name__ == "__main__":
    main()
