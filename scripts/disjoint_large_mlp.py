#!/usr/bin/env python3
"""Larger smooth modular MLP with disjoint trigger and certificate sets."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from smooth_mlp_modular_grokking import (
    Config,
    analytic_gradient,
    initialize,
    logits,
    parameter_count,
)


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DisjointConfig:
    modulus: int = 13
    width: int = 48
    train_fraction: float = 0.60
    learning_rate: float = 2.0
    weight_decay: float = 1e-4
    init_multiplier: float = 1.0
    steps: int = 220_000
    log_every: int = 50
    checkpoint_every: int = 250
    seed: int = 0
    train_accuracy_gate: float = 0.99
    trigger_accuracy_gate: float = 0.95
    certificate_accuracy_gate: float = 0.95

    def model_config(self) -> Config:
        return Config(
            modulus=self.modulus,
            width=self.width,
            train_fraction=self.train_fraction,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            init_multiplier=self.init_multiplier,
            steps=self.steps,
            log_every=self.log_every,
            checkpoint_every=self.checkpoint_every,
            seed=self.seed,
            train_accuracy_gate=self.train_accuracy_gate,
            test_accuracy_gate=self.certificate_accuracy_gate,
        )


def make_disjoint_split(
    config: DisjointConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    p = config.modulus
    pairs = torch.cartesian_prod(torch.arange(p), torch.arange(p))
    labels = (pairs[:, 0] + pairs[:, 1]) % p
    generator = torch.Generator().manual_seed(config.seed + 20_026)
    order = torch.randperm(len(pairs), generator=generator)
    n_train = int(round(config.train_fraction * len(pairs)))
    remaining = len(pairs) - n_train
    if remaining < 4 or remaining % 2:
        raise ValueError("the nontraining population must split evenly")
    n_trigger = remaining // 2
    train_index = order[:n_train]
    trigger_index = order[n_train : n_train + n_trigger]
    certificate_index = order[n_train + n_trigger :]
    return (
        pairs[train_index], labels[train_index],
        pairs[trigger_index], labels[trigger_index],
        pairs[certificate_index], labels[certificate_index],
    )


@torch.no_grad()
def split_metrics(
    parameter: torch.Tensor,
    data: tuple[torch.Tensor, ...],
    model_config: Config,
) -> tuple[float, float, float, float, float, float, float]:
    train_pairs, train_labels, trigger_pairs, trigger_labels, cert_pairs, cert_labels = data

    def loss_accuracy(pairs: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
        values = logits(parameter, pairs, model_config)
        target = F.one_hot(labels, num_classes=model_config.modulus).to(parameter.dtype)
        loss = float(0.5 * torch.mean((values - target).square()))
        accuracy = float(torch.mean((values.argmax(dim=1) == labels).to(torch.float64)))
        return loss, accuracy

    train_loss, train_accuracy = loss_accuracy(train_pairs, train_labels)
    trigger_loss, trigger_accuracy = loss_accuracy(trigger_pairs, trigger_labels)
    cert_loss, cert_accuracy = loss_accuracy(cert_pairs, cert_labels)
    return (
        train_loss, trigger_loss, cert_loss,
        train_accuracy, trigger_accuracy, cert_accuracy,
        float(torch.linalg.vector_norm(parameter)),
    )


def first_logged_step(trajectory: np.ndarray, column: int, gate: float) -> int | None:
    indices = np.flatnonzero(trajectory[:, column] >= gate)
    return None if len(indices) == 0 else int(trajectory[indices[0], 0])


def train_disjoint(
    config: DisjointConfig,
    *,
    keep_checkpoints: bool,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict]:
    torch.set_num_threads(1)
    model_config = config.model_config()
    data = make_disjoint_split(config)
    train_pairs, train_labels = data[:2]
    parameter = initialize(model_config)
    rows: list[list[float]] = []
    checkpoints: dict[int, np.ndarray] = {}
    started = time.perf_counter()

    for step in range(config.steps + 1):
        if step % config.log_every == 0 or step == config.steps:
            rows.append([float(step), *split_metrics(parameter, data, model_config)])
        if keep_checkpoints and step % config.checkpoint_every == 0:
            checkpoints[step] = parameter.numpy().copy()
        if step == config.steps:
            break
        parameter.add_(
            analytic_gradient(
                parameter, train_pairs, train_labels, model_config
            ),
            alpha=-config.learning_rate,
        )

    trajectory = np.asarray(rows, dtype=np.float64)
    fit = first_logged_step(trajectory, 4, config.train_accuracy_gate)
    trigger95 = first_logged_step(trajectory, 5, config.trigger_accuracy_gate)
    cert95 = first_logged_step(trajectory, 6, config.certificate_accuracy_gate)
    summary = {
        "fit_step": fit,
        "trigger_95_step": trigger95,
        "certificate_95_step": cert95,
        "certificate_delay_ratio": (
            None if fit is None or cert95 is None else cert95 / max(fit, 1)
        ),
        "final_train_accuracy": float(trajectory[-1, 4]),
        "final_trigger_accuracy": float(trajectory[-1, 5]),
        "final_certificate_accuracy": float(trajectory[-1, 6]),
        "parameter_count": parameter_count(model_config),
        "train_examples": len(data[0]),
        "trigger_examples": len(data[2]),
        "certificate_examples": len(data[4]),
        "elapsed_seconds": time.perf_counter() - started,
        "finite": bool(np.all(np.isfinite(trajectory))),
    }
    return trajectory, checkpoints, summary


def artifact_paths(seed: int, *, development: bool) -> tuple[Path, Path]:
    kind = "development" if development else "prospective"
    result = ROOT / "results" / f"disjoint_large_{kind}_seed_{seed}.json"
    return result, result.with_suffix(".checkpoints.npz")


def run(config: DisjointConfig, *, development: bool, overwrite: bool = False) -> dict:
    result_path, checkpoint_path = artifact_paths(config.seed, development=development)
    if not overwrite and (result_path.exists() or checkpoint_path.exists()):
        raise FileExistsError(f"refusing to overwrite {result_path} or {checkpoint_path}")
    trajectory, checkpoints, summary = train_disjoint(config, keep_checkpoints=True)
    payload = {
        "status": "development" if development else "prospective frozen run",
        "model": "one-hidden-layer tanh MLP with disjoint trigger/certificate sets",
        "config": asdict(config),
        "summary": summary,
        "trajectory_columns": [
            "step", "train_mse", "trigger_mse", "certificate_mse",
            "train_accuracy", "trigger_accuracy", "certificate_accuracy",
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
    return {"result": str(result_path), "checkpoints": str(checkpoint_path), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=220_000)
    parser.add_argument("--modulus", type=int, default=13)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--learning-rate", type=float, default=2.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--prospective", action="store_true")
    parser.add_argument("--overwrite-development", action="store_true")
    args = parser.parse_args()
    config = DisjointConfig(
        modulus=args.modulus,
        width=args.width,
        train_fraction=args.train_fraction,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        steps=args.steps,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
    )
    print(json.dumps(run(
        config,
        development=not args.prospective,
        overwrite=args.overwrite_development and not args.prospective,
    ), indent=2))


if __name__ == "__main__":
    main()
