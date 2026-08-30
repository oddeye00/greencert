#!/usr/bin/env python3
"""Every-iterate replay of frozen smooth-MLP accuracy events."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from generate_smooth_mlp_seed import artifact_paths, frozen_config
from smooth_mlp_modular_grokking import analytic_gradient, initialize, logits, make_split


ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = (0.60, 0.70, 0.80, 0.90, 0.95)


def required_counts(example_count: int) -> dict[float, int]:
    return {threshold: int(np.ceil(threshold * example_count)) for threshold in THRESHOLDS}


def crossing_output(seed: int) -> Path:
    return ROOT / "results" / f"smooth_mlp_modular_seed_{seed}.crossings.json"


@torch.no_grad()
def replay(seed: int) -> dict:
    torch.set_num_threads(1)
    config = frozen_config(seed)
    _, checkpoint_path = artifact_paths(seed)
    checkpoints = np.load(checkpoint_path)
    train_pairs, train_labels, test_pairs, test_labels = make_split(config)
    required = required_counts(len(test_pairs))
    crossings: dict[str, int | None] = {f"{threshold:.2f}": None for threshold in THRESHOLDS}
    parameter = initialize(config)
    maximum_checkpoint_error = 0.0
    maximum_accuracy = 0.0
    for step in range(config.steps + 1):
        if step % config.checkpoint_every == 0:
            key = f"step_{step}"
            maximum_checkpoint_error = max(
                maximum_checkpoint_error,
                float(torch.linalg.vector_norm(parameter - torch.from_numpy(checkpoints[key]))),
            )
        prediction = torch.argmax(logits(parameter, test_pairs, config), dim=1)
        correct = int(torch.sum(prediction == test_labels))
        maximum_accuracy = max(maximum_accuracy, correct / len(test_pairs))
        for threshold, count in required.items():
            key = f"{threshold:.2f}"
            if crossings[key] is None and correct >= count:
                crossings[key] = step
        if all(value is not None for value in crossings.values()):
            break
        gradient = analytic_gradient(parameter, train_pairs, train_labels, config)
        parameter.add_(gradient, alpha=-config.learning_rate)
    return {
        "seed": seed,
        "protocol": "SECOND_ARCHITECTURE_PROTOCOL.md",
        "thresholds": list(THRESHOLDS),
        "required_correct": {f"{key:.2f}": value for key, value in required.items()},
        "crossing_steps": crossings,
        "steps_replayed": step + 1,
        "maximum_accuracy": maximum_accuracy,
        "maximum_checkpoint_parameter_error": maximum_checkpoint_error,
        "all_checkpoint_replays_exact": maximum_checkpoint_error == 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    result = replay(args.seed)
    output = crossing_output(args.seed)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **result}, indent=2))


if __name__ == "__main__":
    main()

