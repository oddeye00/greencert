#!/usr/bin/env python3
"""Smooth full-batch modular Transformer for matrix-free trajectory audits.

The model is intentionally small enough for repeated Hessian-vector products
on CPU, but is architecturally distinct from the one-hidden-layer MLP used in
the primary certificate experiments.  Training is literal gradient descent so
the checkpoint map has the HVP Jacobian ``I - eta H`` assumed by the local
clock.  Trigger and certification examples are disjoint.
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.func import functional_call
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TransformerConfig:
    modulus: int = 31
    model_dim: int = 32
    hidden_dim: int = 128
    heads: int = 4
    depth: int = 1
    train_fraction: float = 0.50
    learning_rate: float = 1.0
    momentum: float = 0.0
    weight_decay: float = 1e-4
    steps: int = 20_000
    log_every: int = 50
    checkpoint_every: int = 100
    seed: int = 0
    threads: int = 4
    dtype: str = "float64"
    loss: str = "cross_entropy"
    normalization: str = "layernorm"


class SmoothTransformerBlock(nn.Module):
    """A one-layer smooth attention block without normalization."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        d = config.model_dim
        self.attention = nn.MultiheadAttention(
            d, config.heads, dropout=0.0, batch_first=True
        )
        self.linear1 = nn.Linear(d, config.hidden_dim)
        self.linear2 = nn.Linear(config.hidden_dim, d)

    def forward(self, hidden: Tensor, mask: Tensor) -> Tensor:
        attended, _ = self.attention(
            hidden, hidden, hidden, attn_mask=mask, need_weights=False
        )
        hidden = hidden + attended
        return hidden + self.linear2(F.gelu(self.linear1(hidden)))


class SmoothModularTransformer(nn.Module):
    """Pre-norm causal encoder block with a smooth GELU feed-forward map."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        p, d = config.modulus, config.model_dim
        self.modulus = p
        self.normalization = config.normalization
        self.token_embedding = nn.Embedding(p + 1, d)
        self.position_embedding = nn.Parameter(torch.empty(3, d))
        if config.normalization not in {"layernorm", "none"}:
            raise ValueError(f"unknown normalization: {config.normalization}")
        if config.normalization == "layernorm":
            self.blocks = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=d,
                        nhead=config.heads,
                        dim_feedforward=config.hidden_dim,
                        dropout=0.0,
                        activation=F.gelu,
                        batch_first=True,
                        norm_first=True,
                    )
                    for _ in range(config.depth)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [SmoothTransformerBlock(config) for _ in range(config.depth)]
            )
        self.readout = nn.Linear(d, p, bias=False)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, pairs: Tensor) -> Tensor:
        equals = torch.full(
            (len(pairs), 1), self.modulus, dtype=pairs.dtype, device=pairs.device
        )
        tokens = torch.cat((pairs, equals), dim=1)
        hidden = self.token_embedding(tokens) + self.position_embedding
        mask = torch.triu(
            torch.ones(3, 3, dtype=torch.bool, device=pairs.device), diagonal=1
        )
        # The math kernel supports the double backward required by exact HVPs;
        # PyTorch's CPU flash-attention backward currently does not.
        with sdpa_kernel(SDPBackend.MATH):
            for block in self.blocks:
                hidden = (
                    block(hidden, src_mask=mask)
                    if self.normalization == "layernorm"
                    else block(hidden, mask)
                )
        return self.readout(hidden[:, -1])


@dataclass(frozen=True)
class FlatSpec:
    names: tuple[str, ...]
    shapes: tuple[torch.Size, ...]
    sizes: tuple[int, ...]


def make_template(config: TransformerConfig) -> SmoothModularTransformer:
    torch.manual_seed(config.seed)
    model = SmoothModularTransformer(config)
    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    model = model.to(dtype=dtype)
    model.eval()
    return model


def flat_spec(model: nn.Module) -> FlatSpec:
    rows = tuple(model.named_parameters())
    return FlatSpec(
        names=tuple(name for name, _ in rows),
        shapes=tuple(value.shape for _, value in rows),
        sizes=tuple(value.numel() for _, value in rows),
    )


@torch.no_grad()
def flatten_parameters(model: nn.Module) -> Tensor:
    return torch.cat([value.reshape(-1) for value in model.parameters()]).clone()


def unflatten_parameters(parameter: Tensor, spec: FlatSpec) -> Mapping[str, Tensor]:
    if parameter.numel() != sum(spec.sizes):
        raise ValueError("flat parameter has the wrong dimension")
    rows: dict[str, Tensor] = {}
    offset = 0
    for name, shape, size in zip(spec.names, spec.shapes, spec.sizes):
        rows[name] = parameter[offset : offset + size].reshape(shape)
        offset += size
    return rows


def logits(
    parameter: Tensor,
    pairs: Tensor,
    template: SmoothModularTransformer,
    spec: FlatSpec,
) -> Tensor:
    return functional_call(
        template,
        unflatten_parameters(parameter, spec),
        (pairs,),
        strict=True,
    )


def objective(
    parameter: Tensor,
    pairs: Tensor,
    labels: Tensor,
    template: SmoothModularTransformer,
    spec: FlatSpec,
    config: TransformerConfig,
) -> Tensor:
    values = logits(parameter, pairs, template, spec)
    if config.loss == "cross_entropy":
        data = F.cross_entropy(values, labels)
    elif config.loss == "mse":
        target = F.one_hot(labels, num_classes=config.modulus).to(values.dtype)
        data = 0.5 * torch.mean((values - target).square())
    else:
        raise ValueError(f"unknown loss: {config.loss}")
    return data + 0.5 * config.weight_decay * torch.dot(parameter, parameter)


def gradient(
    parameter: Tensor,
    pairs: Tensor,
    labels: Tensor,
    template: SmoothModularTransformer,
    spec: FlatSpec,
    config: TransformerConfig,
    *,
    create_graph: bool = False,
) -> Tensor:
    with torch.enable_grad():
        point = parameter.detach().requires_grad_(True)
        value = objective(point, pairs, labels, template, spec, config)
        (result,) = torch.autograd.grad(value, point, create_graph=create_graph)
    return result if create_graph else result.detach()


def objective_hvp(
    parameter: Tensor,
    vector: Tensor,
    pairs: Tensor,
    labels: Tensor,
    template: SmoothModularTransformer,
    spec: FlatSpec,
    config: TransformerConfig,
) -> Tensor:
    with torch.enable_grad():
        point = parameter.detach().requires_grad_(True)
        value = objective(point, pairs, labels, template, spec, config)
        (grad,) = torch.autograd.grad(value, point, create_graph=True)
        (product,) = torch.autograd.grad(torch.dot(grad, vector.detach()), point)
    return product.detach()


def gradient_and_objective_hvp(
    parameter: Tensor,
    vector: Tensor,
    pairs: Tensor,
    labels: Tensor,
    template: SmoothModularTransformer,
    spec: FlatSpec,
    config: TransformerConfig,
) -> tuple[Tensor, Tensor]:
    """Evaluate a gradient and HVP from one shared reverse-mode graph.

    A variational recentering sweep needs both quantities at the same center.
    Calling :func:`gradient` and :func:`objective_hvp` separately rebuilds the
    forward and first reverse pass twice.  This fused primitive exposes the
    gradient already constructed by the HVP calculation and changes no
    mathematical operation.
    """

    with torch.enable_grad():
        point = parameter.detach().requires_grad_(True)
        value = objective(point, pairs, labels, template, spec, config)
        (result_gradient,) = torch.autograd.grad(value, point, create_graph=True)
        (product,) = torch.autograd.grad(
            torch.dot(result_gradient, vector.detach()), point
        )
    return result_gradient.detach(), product.detach()


def gradient_hvp_and_third_contraction(
    parameter: Tensor,
    vector: Tensor,
    pairs: Tensor,
    labels: Tensor,
    template: SmoothModularTransformer,
    spec: FlatSpec,
    config: TransformerConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Share one nested graph across gradient, HVP, and ``D^3F[v,v,.]``.

    The contracted third derivative is the cancellation-safe quadratic
    forcing used by the response-centered theorem.  Computing it separately
    from a map/JVP pair repeats both the neural forward pass and the first
    reverse pass.  This fused primitive returns all three exact autodiff
    quantities from the graph already required by the third contraction.
    """

    fixed_vector = vector.detach()
    with torch.enable_grad():
        point = parameter.detach().requires_grad_(True)
        value = objective(point, pairs, labels, template, spec, config)
        (result_gradient,) = torch.autograd.grad(value, point, create_graph=True)
        (hessian_direction,) = torch.autograd.grad(
            torch.dot(result_gradient, fixed_vector),
            point,
            create_graph=True,
        )
        if bool(torch.any(fixed_vector != 0.0)):
            (third_direction,) = torch.autograd.grad(
                torch.dot(hessian_direction, fixed_vector), point
            )
        else:
            third_direction = torch.zeros_like(point)
    return (
        result_gradient.detach(),
        hessian_direction.detach(),
        third_direction.detach(),
    )


def replayable_gradient_and_hvp(
    parameter: Tensor,
    pairs: Tensor,
    labels: Tensor,
    template: SmoothModularTransformer,
    spec: FlatSpec,
    config: TransformerConfig,
) -> tuple[Tensor, Callable[[Tensor], Tensor]]:
    """Build one gradient tape and return repeatable matrix-free HVPs.

    This is useful at an anchor whose Hessian is applied throughout an affine
    finite-window reference.  The autograd graph is retained, but no dense
    Hessian is formed.  Each call performs only the second reverse traversal.
    The closure must not outlive the tensors and should be released after the
    finite-window construction.
    """

    with torch.enable_grad():
        point = parameter.detach().requires_grad_(True)
        value = objective(point, pairs, labels, template, spec, config)
        (result_gradient,) = torch.autograd.grad(value, point, create_graph=True)

    def apply(vector: Tensor) -> Tensor:
        if vector.shape != parameter.shape:
            raise ValueError("HVP vector shape does not match parameter")
        with torch.enable_grad():
            (product,) = torch.autograd.grad(
                torch.dot(result_gradient, vector.detach()),
                point,
                retain_graph=True,
            )
        return product.detach()

    return result_gradient.detach(), apply


def make_disjoint_split(
    config: TransformerConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    p = config.modulus
    pairs = torch.cartesian_prod(torch.arange(p), torch.arange(p)).long()
    labels = (pairs[:, 0] + pairs[:, 1]).remainder(p).long()
    generator = torch.Generator().manual_seed(config.seed + 90_017)
    order = torch.randperm(len(pairs), generator=generator)
    n_train = int(round(config.train_fraction * len(pairs)))
    remaining = len(pairs) - n_train
    n_trigger = remaining // 2
    train = order[:n_train]
    trigger = order[n_train : n_train + n_trigger]
    certificate = order[n_train + n_trigger :]
    return (
        pairs[train], labels[train],
        pairs[trigger], labels[trigger],
        pairs[certificate], labels[certificate],
    )


@torch.no_grad()
def accuracy(
    parameter: Tensor,
    pairs: Tensor,
    labels: Tensor,
    template: SmoothModularTransformer,
    spec: FlatSpec,
) -> float:
    return float((logits(parameter, pairs, template, spec).argmax(1) == labels).double().mean())


def train(
    config: TransformerConfig,
    *,
    keep_checkpoints: bool = True,
) -> tuple[np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]], dict]:
    torch.set_num_threads(config.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    template = make_template(config)
    spec = flat_spec(template)
    parameter = flatten_parameters(template)
    velocity = torch.zeros_like(parameter)
    data = make_disjoint_split(config)
    train_pairs, train_labels, trigger_pairs, trigger_labels, cert_pairs, cert_labels = data
    rows: list[list[float]] = []
    checkpoints: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    started = time.perf_counter()

    for step in range(config.steps + 1):
        if step % config.log_every == 0 or step == config.steps:
            rows.append([
                float(step),
                accuracy(parameter, train_pairs, train_labels, template, spec),
                accuracy(parameter, trigger_pairs, trigger_labels, template, spec),
                accuracy(parameter, cert_pairs, cert_labels, template, spec),
                float(torch.linalg.vector_norm(parameter)),
            ])
        if keep_checkpoints and step % config.checkpoint_every == 0:
            checkpoints[step] = (
                parameter.detach().cpu().numpy().copy(),
                velocity.detach().cpu().numpy().copy(),
            )
        if step == config.steps:
            break
        velocity = config.momentum * velocity + gradient(
            parameter, train_pairs, train_labels, template, spec, config
        )
        parameter = parameter - config.learning_rate * velocity

    trajectory = np.asarray(rows, dtype=np.float64)

    def first(column: int, gate: float) -> int | None:
        hit = np.flatnonzero(trajectory[:, column] >= gate)
        return None if len(hit) == 0 else int(trajectory[hit[0], 0])

    summary = {
        "parameter_count": int(parameter.numel()),
        "train_examples": len(train_pairs),
        "trigger_examples": len(trigger_pairs),
        "certificate_examples": len(cert_pairs),
        "fit_step": first(1, 0.99),
        "trigger_60_step": first(2, 0.60),
        "certificate_60_step": first(3, 0.60),
        "certificate_95_step": first(3, 0.95),
        "final_train_accuracy": float(trajectory[-1, 1]),
        "final_trigger_accuracy": float(trajectory[-1, 2]),
        "final_certificate_accuracy": float(trajectory[-1, 3]),
        "elapsed_seconds": time.perf_counter() - started,
        "finite": bool(np.all(np.isfinite(trajectory))),
    }
    return trajectory, checkpoints, summary


def artifact_paths(seed: int, *, development: bool) -> tuple[Path, Path]:
    label = "development" if development else "prospective"
    result = ROOT / "results" / f"transformer_hvp_{label}_seed_{seed}.json"
    return result, result.with_suffix(".checkpoints.npz")


def run(config: TransformerConfig, *, development: bool, overwrite: bool = False) -> dict:
    result_path, checkpoint_path = artifact_paths(config.seed, development=development)
    outcome_path = result_path.with_name(result_path.stem + ".outcomes.json")
    protected = (result_path, checkpoint_path) if development else (result_path, outcome_path, checkpoint_path)
    if not overwrite and any(path.exists() for path in protected):
        raise FileExistsError(f"refusing to overwrite prospective/development artifacts for seed {config.seed}")
    trajectory, checkpoints, summary = train(config, keep_checkpoints=True)
    payload = {
        "status": "development" if development else "prospective frozen run",
        "model": "smooth one-layer causal Transformer; disjoint trigger/certification sets",
        "config": asdict(config),
        "summary": summary,
        "trajectory_columns": [
            "step", "train_accuracy", "trigger_accuracy",
            "certificate_accuracy", "parameter_norm",
        ],
        "trajectory": trajectory.tolist(),
        "checkpoint_steps": sorted(checkpoints),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if development:
        result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        # A prospective scanner reads only this blind artifact.  Certification
        # outcomes are written separately and joined after candidates freeze.
        blind_summary = {
            key: value
            for key, value in summary.items()
            if not key.startswith("certificate_") and not key.startswith("final_certificate")
        }
        blind = {
            "status": "prospective blind trigger artifact",
            "model": payload["model"],
            "config": payload["config"],
            "summary": blind_summary,
            "trajectory_columns": [
                "step", "train_accuracy", "trigger_accuracy", "parameter_norm"
            ],
            "trajectory": trajectory[:, (0, 1, 2, 4)].tolist(),
            "checkpoint_steps": payload["checkpoint_steps"],
        }
        outcome = {
            "status": "sealed prospective certification outcomes",
            "seed": config.seed,
            "trajectory_columns": ["step", "certificate_accuracy"],
            "trajectory": trajectory[:, (0, 3)].tolist(),
            "certificate_60_step": summary["certificate_60_step"],
            "certificate_95_step": summary["certificate_95_step"],
            "final_certificate_accuracy": summary["final_certificate_accuracy"],
        }
        result_path.write_text(json.dumps(blind, indent=2) + "\n", encoding="utf-8")
        outcome_path.write_text(json.dumps(outcome, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        checkpoint_path,
        **{
            name: value
            for step, (parameter, velocity) in checkpoints.items()
            for name, value in (
                (f"step_{step}", parameter),
                (f"velocity_{step}", velocity),
            )
        },
    )
    return {
        "result": str(result_path),
        "outcomes": None if development else str(outcome_path),
        "checkpoints": str(checkpoint_path),
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--modulus", type=int, default=31)
    parser.add_argument("--train-fraction", type=float, default=0.50)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--model-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--loss", choices=("cross_entropy", "mse"), default="cross_entropy")
    parser.add_argument("--normalization", choices=("layernorm", "none"), default="layernorm")
    parser.add_argument("--prospective", action="store_true")
    parser.add_argument("--overwrite-development", action="store_true")
    args = parser.parse_args()
    config = TransformerConfig(
        seed=args.seed,
        modulus=args.modulus,
        train_fraction=args.train_fraction,
        steps=args.steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        model_dim=args.model_dim,
        hidden_dim=args.hidden_dim,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        loss=args.loss,
        normalization=args.normalization,
    )
    print(json.dumps(run(
        config,
        development=not args.prospective,
        overwrite=args.overwrite_development and not args.prospective,
    ), indent=2))


if __name__ == "__main__":
    main()
