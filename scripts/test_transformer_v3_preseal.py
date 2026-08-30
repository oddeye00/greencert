#!/usr/bin/env python3
"""Pre-seal adversarial checks for the Transformer v3 execution path."""

from __future__ import annotations

import json
import math

from one_shot_recenter_closure import conservative_one_shot_closure
from run_transformer_v3_confirmation import (
    METHOD_FILES,
    ROOT,
    expected_config,
    fresh_targets,
)
from transformer_hvp_grokking import TransformerConfig, artifact_paths
from transformer_v3_certificate import _q_geometry
from transformer_v3_protocol import MAXIMUM_POWER, SEEDS


def main() -> None:
    assert all((ROOT / name).exists() for name in METHOD_FILES)
    assert len(METHOD_FILES) == len(set(METHOD_FILES))
    existing = [path for path in fresh_targets() if path.exists()]
    assert not existing, f"v3 targets already exist before freeze: {existing}"
    for seed in SEEDS:
        blind, checkpoints = artifact_paths(seed, development=False)
        outcomes = blind.with_name(blind.stem + ".outcomes.json")
        assert not blind.exists() and not checkpoints.exists() and not outcomes.exists()
        assert expected_config(seed).seed == seed

    # Replay the q-level deterministic arithmetic against a completed old
    # record. This avoids spending another six minutes while testing exactly
    # the geometry and closure functions used by v3.
    replay_path = ROOT / "results" / (
        "progressive_probe_replay_seed_333_gate_0_anchor_3000.json"
    )
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    blind_path, _ = artifact_paths(333, development=False)
    blind = json.loads(blind_path.read_text(encoding="utf-8"))
    config = TransformerConfig(**blind["config"])
    response_norm = float(replay["power_rows"][-1]["one_shot_closure"][
        "response_sequence_norm"
    ])
    response_max = float(replay["power_rows"][-1]["one_shot_closure"][
        "response_max_state_norm"
    ])
    domain = float(replay["power_rows"][-1]["one_shot_closure"]["domain_radius"])
    for power in range(1, MAXIMUM_POWER + 1):
        map_drift, _ = _q_geometry(
            power=power,
            output_rows=replay["output_rows"],
            config=config,
            domain_radius=domain,
        )
        expected = replay["power_rows"][power - 1]
        assert math.isclose(
            map_drift,
            float(expected["maximum_optimizer_derivative_drift_upper"]),
            rel_tol=2.0e-15,
        )
        closure = conservative_one_shot_closure(
            kappa=float(expected["green_operator_norm_upper"]),
            derivative_drift=map_drift,
            response_sequence_norm=response_norm,
            response_max_state_norm=response_max,
            domain_radius=domain,
        )
        assert closure.closure_passed == bool(
            expected["one_shot_closure"]["closure_passed"]
        )
        if closure.total_pointwise_radius is not None:
            assert math.isclose(
                closure.total_pointwise_radius,
                float(expected["one_shot_closure"]["total_pointwise_radius"]),
                rel_tol=2.0e-15,
            )
    print("PASS: v3 pre-seal targets, manifests, and q-level arithmetic are clean.")


if __name__ == "__main__":
    main()
