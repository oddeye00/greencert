#!/usr/bin/env python3
"""Pre-seal structural and algebraic tests for the WDBC confirmation."""
from __future__ import annotations

import ast
import inspect
import json
import math
import tempfile
from pathlib import Path
from unittest import mock

import run_real_dataset_confirmation as runner
from probe_jacobian_bound import ProbeRegistry
from real_dataset_greencert import minimal_admissible_radius
from real_dataset_mlp import RealMLPConfig, make_selection_split, make_split


def test_selection_loader_barrier() -> None:
    config = RealMLPConfig(seed=101)
    selection = make_selection_split(config)
    serialized = json.dumps(
        {
            key: value
            for key, value in selection.items()
            if not hasattr(value, "shape")
        }
    ).lower()
    assert all("certificate" not in key.lower() for key in selection)
    assert "certificate" not in serialized
    full = make_split(config)
    assert "certificate_x" in full and "certificate_y" in full


def test_training_call_graph_uses_selection_loader() -> None:
    tree = ast.parse(inspect.getsource(runner._train_one))
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "make_selection_split" in calls
    assert "make_split" not in calls


def test_selector_reads_only_blind_records() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "run"
        blind_dir = root / "blind"
        blind_dir.mkdir(parents=True)
        for seed in runner.FRESH_SEEDS:
            runner.write_json(
                blind_dir / f"seed_{seed}.json",
                {
                    "seed": seed,
                    "train_accuracy": [0.0] * 5 + [0.85] * 16,
                    "trigger_accuracy": [0.0] * 5 + [0.89] * 16,
                },
            )
        candidate_seal = Path(temporary) / "candidate_seal.json"
        opened: list[Path] = []
        original_read = runner.read_json

        def audited_read(path: Path) -> dict:
            opened.append(Path(path).resolve())
            return original_read(path)

        with (
            mock.patch.object(runner, "CANDIDATE_SEAL_PATH", candidate_seal),
            mock.patch.object(runner, "verify_method_seal", return_value={}),
            mock.patch.object(runner, "method_hash", return_value="TEST"),
            mock.patch.object(runner, "read_json", side_effect=audited_read),
        ):
            runner.phase_select(root)
        assert len(opened) == len(runner.FRESH_SEEDS)
        assert all(path.parent == blind_dir.resolve() for path in opened)
        manifest = original_read(root / "candidates_blind.json")
        assert len(manifest["rows"]) == 72


def test_identity_universe_and_streams() -> None:
    rows = [
        {"seed": seed, "gate_index": gate, "anchor": 5 * (gate + 1)}
        for seed in runner.FRESH_SEEDS
        for gate in range(len(runner.THRESHOLDS))
    ]
    identities = [runner.candidate_identity(row) for row in rows]
    assert len(identities) == 72
    assert len(set(identities)) == 72
    registry = ProbeRegistry(identities, "00" * 32)
    assert registry.summary()["collision_free_stream_count"] == 72


def test_minimal_root_and_feasibility() -> None:
    for z_norm in (0.0, 1e-12, 0.01, 0.4):
        for kappa in (1.0, 2.5, 10.0):
            for drift in (0.0, 0.01, 0.2, 3.0):
                statistic = 2.0 * kappa * drift * z_norm
                radius = minimal_admissible_radius(z_norm, kappa, drift)
                if statistic > 1.0:
                    assert radius is None
                    continue
                assert radius is not None
                residual = z_norm + 0.5 * kappa * drift * radius**2 - radius
                assert abs(residual) <= 2e-14 * max(1.0, radius)
                assert z_norm <= radius + 1e-15
                assert radius <= 2.0 * z_norm + 1e-15
    assert minimal_admissible_radius(0.1, 1.0, 5.0) is not None
    assert math.isclose(minimal_admissible_radius(0.1, 1.0, 5.0), 0.2)


def test_persistent_event_indexing() -> None:
    values = [0.0, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9]
    old = runner.PERSISTENCE
    try:
        runner.PERSISTENCE = 3
        assert runner._events(values, 0.9) == 4
    finally:
        runner.PERSISTENCE = old


def main() -> None:
    tests = [
        test_selection_loader_barrier,
        test_training_call_graph_uses_selection_loader,
        test_selector_reads_only_blind_records,
        test_identity_universe_and_streams,
        test_minimal_root_and_feasibility,
        test_persistent_event_indexing,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
