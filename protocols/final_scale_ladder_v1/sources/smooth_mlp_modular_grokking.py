#!/usr/bin/env python3
"""CPU-scale modular-addition grokking in a smooth tanh MLP.

The architecture is deliberately different from the quadratic network used in
the primary certificate audit:

    f(a, b) = V tanh(W [e_a; e_b] + q) + c.

Inputs and targets are one-hot, the data loss is mean squared error, and the
optimizer is literal full-batch gradient descent with optional coupled L2.
The analytic gradient makes long exploratory runs inexpensive while ``loss``
is written in ordinary PyTorch operations for exact-Hessian audits.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SEARCH_OUT = ROOT / "results" / "smooth_mlp_modular_search.json"
RUN_OUT = ROOT / "results" / "smooth_mlp_modular_grokking.json"
CHECKPOINT_OUT = ROOT / "results" / "smooth_mlp_modular_grokking.checkpoints.npz"
FIGURE = ROOT / "figures" / "smooth_mlp_modular_grokking.png"


@dataclass(frozen=True)
class Config:
    modulus: int = 11
    width: int = 16
    train_fraction: float = 0.60
    learning_rate: float = 1.0
    weight_decay: float = 0.0
    init_multiplier: float = 1.0
    steps: int = 100_000
    log_every: int = 25
    checkpoint_every: int = 250
    seed: int = 0
    train_accuracy_gate: float = 0.99
    test_accuracy_gate: float = 0.95


def make_split(config: Config) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    p = config.modulus
    pairs = torch.cartesian_prod(torch.arange(p), torch.arange(p))
    labels = (pairs[:, 0] + pairs[:, 1]) % p
    generator = torch.Generator().manual_seed(config.seed + 10_003)
    order = torch.randperm(len(pairs), generator=generator)
    n_train = int(round(config.train_fraction * len(pairs)))
    train_index = order[:n_train]
    test_index = order[n_train:]
    return pairs[train_index], labels[train_index], pairs[test_index], labels[test_index]


def parameter_count(config: Config) -> int:
    p, h = config.modulus, config.width
    return h * (2 * p) + h + p * h + p


def unpack(
    parameter: torch.Tensor, config: Config
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    p, h = config.modulus, config.width
    w_end = h * 2 * p
    q_end = w_end + h
    v_end = q_end + p * h
    w = parameter[:w_end].reshape(h, 2 * p)
    q = parameter[w_end:q_end]
    v = parameter[q_end:v_end].reshape(p, h)
    c = parameter[v_end:]
    return w, q, v, c


def initialize(config: Config) -> torch.Tensor:
    """Match the defaults of two PyTorch ``Linear`` layers in float64."""
    p, h = config.modulus, config.width
    generator = torch.Generator().manual_seed(config.seed)
    w = torch.empty((h, 2 * p), dtype=torch.float64)
    q = torch.empty(h, dtype=torch.float64)
    v = torch.empty((p, h), dtype=torch.float64)
    c = torch.empty(p, dtype=torch.float64)
    w_bound = 1.0 / np.sqrt(2 * p)
    v_bound = 1.0 / np.sqrt(h)
    w.uniform_(-w_bound, w_bound, generator=generator)
    q.uniform_(-w_bound, w_bound, generator=generator)
    v.uniform_(-v_bound, v_bound, generator=generator)
    c.uniform_(-v_bound, v_bound, generator=generator)
    return config.init_multiplier * torch.cat((w.reshape(-1), q, v.reshape(-1), c))


def logits(parameter: torch.Tensor, pairs: torch.Tensor, config: Config) -> torch.Tensor:
    p = config.modulus
    w, q, v, c = unpack(parameter, config)
    preactivation = w[:, pairs[:, 0]].T + w[:, p + pairs[:, 1]].T + q
    return torch.tanh(preactivation) @ v.T + c


def data_loss(
    parameter: torch.Tensor,
    pairs: torch.Tensor,
    labels: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    target = F.one_hot(labels, num_classes=config.modulus).to(dtype=parameter.dtype)
    return 0.5 * torch.mean((logits(parameter, pairs, config) - target).square())


def objective(
    parameter: torch.Tensor,
    pairs: torch.Tensor,
    labels: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    return data_loss(parameter, pairs, labels, config) + 0.5 * config.weight_decay * torch.sum(
        parameter.square()
    )


@torch.no_grad()
def analytic_gradient(
    parameter: torch.Tensor,
    pairs: torch.Tensor,
    labels: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    p, h = config.modulus, config.width
    w, q, v, c = unpack(parameter, config)
    preactivation = w[:, pairs[:, 0]].T + w[:, p + pairs[:, 1]].T + q
    hidden = torch.tanh(preactivation)
    prediction = hidden @ v.T + c
    target = F.one_hot(labels, num_classes=p).to(dtype=parameter.dtype)
    output_gradient = (prediction - target) / prediction.numel()
    grad_v = output_gradient.T @ hidden
    grad_c = torch.sum(output_gradient, dim=0)
    grad_preactivation = (output_gradient @ v) * (1.0 - hidden.square())
    grad_w = torch.zeros_like(w)
    grad_w.index_add_(1, pairs[:, 0], grad_preactivation.T)
    grad_w.index_add_(1, p + pairs[:, 1], grad_preactivation.T)
    grad_q = torch.sum(grad_preactivation, dim=0)
    gradient = torch.cat((grad_w.reshape(h * 2 * p), grad_q, grad_v.reshape(p * h), grad_c))
    if config.weight_decay:
        gradient.add_(parameter, alpha=config.weight_decay)
    return gradient


@torch.no_grad()
def metrics(
    parameter: torch.Tensor,
    train_pairs: torch.Tensor,
    train_labels: torch.Tensor,
    test_pairs: torch.Tensor,
    test_labels: torch.Tensor,
    config: Config,
) -> tuple[float, float, float, float, float]:
    train_logits = logits(parameter, train_pairs, config)
    test_logits = logits(parameter, test_pairs, config)
    train_target = F.one_hot(train_labels, num_classes=config.modulus).to(dtype=parameter.dtype)
    test_target = F.one_hot(test_labels, num_classes=config.modulus).to(dtype=parameter.dtype)
    train_mse = float(0.5 * torch.mean((train_logits - train_target).square()))
    test_mse = float(0.5 * torch.mean((test_logits - test_target).square()))
    train_accuracy = float((train_logits.argmax(dim=1) == train_labels).double().mean())
    test_accuracy = float((test_logits.argmax(dim=1) == test_labels).double().mean())
    return train_mse, test_mse, train_accuracy, test_accuracy, float(torch.linalg.vector_norm(parameter))


def first_logged_step(trajectory: np.ndarray, column: int, threshold: float) -> int | None:
    indices = np.flatnonzero(trajectory[:, column] >= threshold)
    return int(trajectory[indices[0], 0]) if len(indices) else None


def train(
    config: Config, keep_checkpoints: bool = False
) -> tuple[np.ndarray, dict[int, np.ndarray], dict]:
    torch.set_num_threads(1)
    train_pairs, train_labels, test_pairs, test_labels = make_split(config)
    parameter = initialize(config)
    rows: list[list[float]] = []
    checkpoints: dict[int, np.ndarray] = {}

    for step in range(config.steps + 1):
        if step % config.log_every == 0 or step == config.steps:
            rows.append(
                [
                    float(step),
                    *metrics(
                        parameter,
                        train_pairs,
                        train_labels,
                        test_pairs,
                        test_labels,
                        config,
                    ),
                ]
            )
        if keep_checkpoints and step % config.checkpoint_every == 0:
            checkpoints[step] = parameter.numpy().copy()
        if step == config.steps:
            break
        gradient = analytic_gradient(parameter, train_pairs, train_labels, config)
        parameter.add_(gradient, alpha=-config.learning_rate)

    trajectory = np.asarray(rows, dtype=np.float64)
    fit_step = first_logged_step(trajectory, 3, config.train_accuracy_gate)
    generalization_step = first_logged_step(trajectory, 4, config.test_accuracy_gate)
    summary = {
        "fit_step": fit_step,
        "generalization_step": generalization_step,
        "delay_steps": None if fit_step is None or generalization_step is None else generalization_step - fit_step,
        "delay_ratio": None
        if fit_step is None or generalization_step is None
        else generalization_step / max(fit_step, 1),
        "final_train_accuracy": float(trajectory[-1, 3]),
        "final_test_accuracy": float(trajectory[-1, 4]),
        "minimum_test_mse": float(np.min(trajectory[:, 2])),
        "parameter_count": parameter_count(config),
        "finite": bool(np.all(np.isfinite(trajectory))),
    }
    return trajectory, checkpoints, summary


def exploratory_configs(base: Config) -> list[Config]:
    """Small, declared architecture search; seed 0 only, no certificate calls."""
    configs: list[Config] = []
    for modulus in (7, 11):
        for width in (8, 12, 16, 24):
            for train_fraction in (0.50, 0.60, 0.70):
                for learning_rate, weight_decay in (
                    (0.5, 0.0),
                    (1.0, 0.0),
                    (1.0, 1e-4),
                    (1.0, 1e-3),
                    (2.0, 1e-4),
                ):
                    configs.append(
                        replace(
                            base,
                            modulus=modulus,
                            width=width,
                            train_fraction=train_fraction,
                            learning_rate=learning_rate,
                            weight_decay=weight_decay,
                            seed=0,
                        )
                    )
    return configs


def search(base: Config) -> None:
    candidates: list[dict] = []
    for index, config in enumerate(exploratory_configs(base)):
        trajectory, _, summary = train(config)
        row = {
            "index": index,
            "config": asdict(config),
            "summary": summary,
            "trajectory_tail": trajectory[-5:].tolist(),
        }
        candidates.append(row)
        print(index, json.dumps({"config": row["config"], "summary": summary}), flush=True)
    SEARCH_OUT.parent.mkdir(parents=True, exist_ok=True)
    SEARCH_OUT.write_text(json.dumps({"search_status": "exploratory", "candidates": candidates}, indent=2) + "\n")


def render(trajectory: np.ndarray, summary: dict) -> None:
    steps = trajectory[:, 0]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
    axes[0].semilogx(steps + 1, trajectory[:, 1], label="train MSE", color="#374151")
    axes[0].semilogx(steps + 1, trajectory[:, 2], label="test MSE", color="#087e8b")
    axes[0].set(xlabel="full-batch GD step", ylabel="mean squared error", title="Smooth-MLP modular loss")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)
    axes[1].semilogx(steps + 1, trajectory[:, 3], label="train accuracy", color="#374151")
    axes[1].semilogx(steps + 1, trajectory[:, 4], label="test accuracy", color="#d1495b")
    if summary["fit_step"] is not None:
        axes[1].axvline(summary["fit_step"] + 1, color="#6b7280", linestyle="--", label="fit")
    if summary["generalization_step"] is not None:
        axes[1].axvline(
            summary["generalization_step"] + 1,
            color="#d1495b",
            linestyle="--",
            label="generalization",
        )
    axes[1].set(xlabel="full-batch GD step", ylabel="accuracy", title="Delayed generalization", ylim=(-0.03, 1.03))
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=220)
    plt.close(fig)


def run(config: Config) -> None:
    trajectory, checkpoints, summary = train(config, keep_checkpoints=True)
    render(trajectory, summary)
    payload = {
        "model": "one-hidden-layer tanh MLP with biases",
        "optimizer": "literal full-batch gradient descent with coupled L2",
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
    RUN_OUT.parent.mkdir(parents=True, exist_ok=True)
    RUN_OUT.write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(CHECKPOINT_OUT, **{f"step_{step}": value for step, value in checkpoints.items()})
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--modulus", type=int, default=11)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--init-multiplier", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        modulus=args.modulus,
        width=args.width,
        train_fraction=args.train_fraction,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        init_multiplier=args.init_multiplier,
        steps=args.steps,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
    )
    if args.search:
        search(config)
    else:
        run(config)


if __name__ == "__main__":
    main()
