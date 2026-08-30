#!/usr/bin/env python3
"""Development-only screen for a non-modular signed-Green stress test.

This script intentionally inspects all three split trajectories.  It is not a
confirmation and cannot create a paper claim.  Its only purpose is to decide
whether a separately sealed, outcome-blind digits study is worth running.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np
import torch
from sklearn.datasets import load_digits

from real_dataset_mlp import (
    ParameterSpec,
    RealMLPConfig,
    _stratified_indices,
    accuracy,
    gradient,
    objective,
)


THRESHOLDS = (0.80, 0.85, 0.90, 0.925, 0.95, 0.975)
PERSISTENCE = 10


def first_persistent(values: list[float], threshold: float) -> int | None:
    run = 0
    for step, value in enumerate(values):
        run = run + 1 if value >= threshold else 0
        if run >= PERSISTENCE:
            return step - PERSISTENCE + 1
    return None


def make_data(config: RealMLPConfig) -> tuple[dict, ParameterSpec]:
    digits = load_digits()
    features = np.asarray(digits.data, dtype=np.float64) / 16.0
    labels = (np.asarray(digits.target, dtype=np.int64) % 2).astype(np.int64)
    train_idx, trigger_idx, certificate_idx = _stratified_indices(labels, config.seed)
    mean = features[train_idx].mean(axis=0)
    scale = features[train_idx].std(axis=0)
    scale[scale < 1e-12] = 1.0
    features = (features - mean) / scale
    dtype = torch.float64 if config.dtype == "float64" else torch.float32

    def rows(indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.as_tensor(features[indices], dtype=dtype),
            torch.as_tensor(labels[indices], dtype=torch.long),
        )

    train_x, train_y = rows(train_idx)
    trigger_x, trigger_y = rows(trigger_idx)
    certificate_x, certificate_y = rows(certificate_idx)
    return {
        "train_x": train_x,
        "train_y": train_y,
        "trigger_x": trigger_x,
        "trigger_y": trigger_y,
        "certificate_x": certificate_x,
        "certificate_y": certificate_y,
    }, ParameterSpec(input_dim=features.shape[1], width=config.width, classes=2)


def initialize(spec: ParameterSpec, config: RealMLPConfig) -> torch.Tensor:
    generator = torch.Generator().manual_seed(config.seed)
    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    w1 = torch.randn(
        spec.width, spec.input_dim, generator=generator, dtype=dtype
    ) / np.sqrt(spec.input_dim)
    b1 = torch.zeros(spec.width, dtype=dtype)
    w2 = torch.randn(
        spec.classes, spec.width, generator=generator, dtype=dtype
    ) / np.sqrt(spec.width)
    b2 = torch.zeros(spec.classes, dtype=dtype)
    return torch.cat((w1.reshape(-1), b1, w2.reshape(-1), b2))


def run(config: RealMLPConfig) -> dict:
    torch.set_num_threads(config.threads)
    data, spec = make_data(config)
    parameter = initialize(spec, config)
    trajectories = {"train": [], "trigger": [], "certificate": []}
    losses: list[float] = []
    started = time.perf_counter()
    for step in range(config.steps + 1):
        for split in trajectories:
            trajectories[split].append(
                accuracy(parameter, data[f"{split}_x"], data[f"{split}_y"], spec)
            )
        losses.append(
            float(objective(parameter, data["train_x"], data["train_y"], spec, config))
        )
        if step < config.steps:
            parameter = parameter - config.learning_rate * gradient(
                parameter, data["train_x"], data["train_y"], spec, config
            )
    events = {
        f"{threshold:.3f}": {
            split: first_persistent(values, threshold)
            for split, values in trajectories.items()
        }
        for threshold in THRESHOLDS
    }
    return {
        "status": "DEVELOPMENT ONLY; ALL OUTCOMES INSPECTED",
        "config": asdict(config),
        "parameter_count": spec.size,
        "events": events,
        "final": {split: values[-1] for split, values in trajectories.items()},
        "final_train_loss": losses[-1],
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[0.002, 0.005, 0.01])
    parser.add_argument("--widths", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()
    rows = []
    for width in args.widths:
        for learning_rate in args.learning_rates:
            for seed in args.seeds:
                row = run(
                    RealMLPConfig(
                        width=width,
                        learning_rate=learning_rate,
                        weight_decay=1e-3,
                        steps=args.steps,
                        checkpoint_every=5,
                        seed=seed,
                        threads=1,
                        dtype="float64",
                    )
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
    print(json.dumps({"status": "DEVELOPMENT ONLY", "runs": rows}, indent=2))


if __name__ == "__main__":
    main()
