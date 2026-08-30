#!/usr/bin/env python3
"""Generate fresh-seed artifacts for the frozen smooth-MLP transfer."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from smooth_mlp_modular_grokking import Config, train


ROOT = Path(__file__).resolve().parents[1]


def frozen_config(seed: int) -> Config:
    return Config(
        modulus=11,
        width=24,
        train_fraction=0.70,
        learning_rate=2.0,
        weight_decay=1e-4,
        init_multiplier=1.0,
        steps=180_000,
        log_every=25,
        checkpoint_every=250,
        seed=seed,
    )


def artifact_paths(seed: int) -> tuple[Path, Path]:
    result = ROOT / "results" / f"smooth_mlp_modular_seed_{seed}.json"
    return result, result.with_suffix(".checkpoints.npz")


def run_seed(seed: int) -> dict:
    config = frozen_config(seed)
    trajectory, checkpoints, summary = train(config, keep_checkpoints=True)
    result_path, checkpoint_path = artifact_paths(seed)
    payload = {
        "model": "one-hidden-layer tanh MLP with biases",
        "optimizer": "literal full-batch gradient descent with coupled L2",
        "protocol": "SECOND_ARCHITECTURE_PROTOCOL.md",
        "config": asdict(config),
        "summary": summary,
        "trajectory_columns": [
            "step",
            "train_mse",
            "test_mse",
            "train_accuracy",
            "test_accuracy",
            "parameter_norm",
        ],
        "trajectory": trajectory.tolist(),
        "checkpoint_steps": sorted(checkpoints),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        checkpoint_path,
        **{f"step_{step}": value for step, value in checkpoints.items()},
    )
    return {
        "result": str(result_path),
        "checkpoints": str(checkpoint_path),
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(run_seed(args.seed), indent=2))


if __name__ == "__main__":
    main()

