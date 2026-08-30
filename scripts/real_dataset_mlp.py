#!/usr/bin/env python3
"""Deterministic tanh MLP on the Wisconsin breast-cancer dataset.

This module is deliberately independent of the modular-arithmetic code.  It
provides a finite, stratified train/trigger/certification split and a smooth
full-batch optimizer map suitable for a GreenCert transfer experiment.
Candidate selection code is expected to use only the train and trigger rows;
the certification rows are returned separately so an information barrier can
be enforced by the calling protocol.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RealMLPConfig:
    width: int = 8
    learning_rate: float = 0.05
    weight_decay: float = 1e-3
    steps: int = 1_000
    checkpoint_every: int = 5
    seed: int = 0
    threads: int = 4
    dtype: str = "float64"


@dataclass(frozen=True)
class ParameterSpec:
    input_dim: int
    width: int
    classes: int

    @property
    def size(self) -> int:
        return (
            self.width * self.input_dim
            + self.width
            + self.classes * self.width
            + self.classes
        )


def _stratified_indices(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a deterministic 60/20/20 split, stratified by class."""
    rng = np.random.default_rng(seed + 90_017)
    train: list[int] = []
    trigger: list[int] = []
    certificate: list[int] = []
    for label in sorted(np.unique(labels).tolist()):
        rows = np.flatnonzero(labels == label)
        rows = rows[rng.permutation(len(rows))]
        n_train = int(np.floor(0.60 * len(rows)))
        n_trigger = int(np.floor(0.20 * len(rows)))
        train.extend(rows[:n_train].tolist())
        trigger.extend(rows[n_train : n_train + n_trigger].tolist())
        certificate.extend(rows[n_train + n_trigger :].tolist())
    return (
        np.asarray(sorted(train), dtype=np.int64),
        np.asarray(sorted(trigger), dtype=np.int64),
        np.asarray(sorted(certificate), dtype=np.int64),
    )


def _make_split(
    config: RealMLPConfig, *, include_certificate: bool
) -> dict[str, Tensor | dict]:
    """Load, split, and train-standardize the real dataset.

    The selection-only entry point deliberately does not materialize or return
    certification tensors. This gives the fresh runner a structural barrier:
    training and candidate selection cannot accidentally inspect certification
    labels, predictions, counts, or trajectories.
    """
    dataset_path = ROOT / "data" / "wdbc_breast_cancer.csv"
    rows = np.loadtxt(dataset_path, delimiter=",", skiprows=1, dtype=np.float64)
    features = rows[:, :-1]
    labels = rows[:, -1].astype(np.int64)
    train_idx, trigger_idx, certificate_idx = _stratified_indices(labels, config.seed)
    mean = features[train_idx].mean(axis=0)
    scale = features[train_idx].std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (features - mean) / scale
    dtype = torch.float64 if config.dtype == "float64" else torch.float32

    def rows(indices: np.ndarray) -> tuple[Tensor, Tensor]:
        return (
            torch.as_tensor(standardized[indices], dtype=dtype),
            torch.as_tensor(labels[indices], dtype=torch.long),
        )

    train_x, train_y = rows(train_idx)
    trigger_x, trigger_y = rows(trigger_idx)
    result: dict[str, Tensor | dict] = {
        "train_x": train_x,
        "train_y": train_y,
        "trigger_x": trigger_x,
        "trigger_y": trigger_y,
        "metadata": {
            "dataset": "Wisconsin Diagnostic Breast Cancer",
            "source": "vendored scikit-learn/UCI WDBC CSV",
            "data_file": "data/wdbc_breast_cancer.csv",
            "examples": int(len(features)),
            "features": int(features.shape[1]),
            "classes": int(len(np.unique(labels))),
            "train_examples": int(len(train_idx)),
            "trigger_examples": int(len(trigger_idx)),
            "split_seed": int(config.seed + 90_017),
            "standardization": "train mean and population standard deviation",
        },
    }
    if include_certificate:
        certificate_x, certificate_y = rows(certificate_idx)
        result["certificate_x"] = certificate_x
        result["certificate_y"] = certificate_y
        result["metadata"]["certificate_examples"] = int(len(certificate_idx))
    return result


def make_selection_split(config: RealMLPConfig) -> dict[str, Tensor | dict]:
    """Return only train/trigger tensors for outcome-blind selection phases."""
    return _make_split(config, include_certificate=False)


def make_split(config: RealMLPConfig) -> dict[str, Tensor | dict]:
    """Return train, trigger, and certification tensors after candidate seal."""
    return _make_split(config, include_certificate=True)


def parameter_spec(config: RealMLPConfig) -> ParameterSpec:
    return ParameterSpec(input_dim=30, width=config.width, classes=2)


def initialize(config: RealMLPConfig) -> Tensor:
    spec = parameter_spec(config)
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


def unpack(parameter: Tensor, spec: ParameterSpec) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if parameter.numel() != spec.size:
        raise ValueError("parameter has the wrong dimension")
    offset = 0
    n = spec.width * spec.input_dim
    w1 = parameter[offset : offset + n].reshape(spec.width, spec.input_dim)
    offset += n
    b1 = parameter[offset : offset + spec.width]
    offset += spec.width
    n = spec.classes * spec.width
    w2 = parameter[offset : offset + n].reshape(spec.classes, spec.width)
    offset += n
    b2 = parameter[offset : offset + spec.classes]
    return w1, b1, w2, b2


def logits(parameter: Tensor, features: Tensor, spec: ParameterSpec) -> Tensor:
    w1, b1, w2, b2 = unpack(parameter, spec)
    hidden = torch.tanh(features @ w1.T + b1)
    return hidden @ w2.T + b2


def objective(
    parameter: Tensor,
    features: Tensor,
    labels: Tensor,
    spec: ParameterSpec,
    config: RealMLPConfig,
) -> Tensor:
    return F.cross_entropy(logits(parameter, features, spec), labels) + 0.5 * config.weight_decay * torch.dot(parameter, parameter)


def gradient(
    parameter: Tensor,
    features: Tensor,
    labels: Tensor,
    spec: ParameterSpec,
    config: RealMLPConfig,
    *,
    create_graph: bool = False,
) -> Tensor:
    if create_graph:
        with torch.enable_grad():
            point = parameter.detach().requires_grad_(True)
            value = objective(point, features, labels, spec, config)
            (result,) = torch.autograd.grad(value, point, create_graph=True)
        return result
    return analytic_gradient(parameter, features, labels, spec, config)


def objective_hvp(
    parameter: Tensor,
    vector: Tensor,
    features: Tensor,
    labels: Tensor,
    spec: ParameterSpec,
    config: RealMLPConfig,
) -> Tensor:
    return analytic_objective_hvp(
        parameter, vector, features, labels, spec, config
    )


@torch.no_grad()
def analytic_gradient(
    parameter: Tensor,
    features: Tensor,
    labels: Tensor,
    spec: ParameterSpec,
    config: RealMLPConfig,
) -> Tensor:
    """Closed-form gradient of mean cross-entropy plus L2."""
    w1, b1, w2, b2 = unpack(parameter, spec)
    preactivation = features @ w1.T + b1
    hidden = torch.tanh(preactivation)
    probabilities = torch.softmax(hidden @ w2.T + b2, dim=1)
    target = F.one_hot(labels, num_classes=spec.classes).to(parameter.dtype)
    error = (probabilities - target) / len(features)
    back_hidden = error @ w2
    back_pre = back_hidden * (1.0 - hidden.square())
    g_w1 = back_pre.T @ features + config.weight_decay * w1
    g_b1 = back_pre.sum(dim=0) + config.weight_decay * b1
    g_w2 = error.T @ hidden + config.weight_decay * w2
    g_b2 = error.sum(dim=0) + config.weight_decay * b2
    return torch.cat((g_w1.reshape(-1), g_b1, g_w2.reshape(-1), g_b2))


@torch.no_grad()
def analytic_objective_hvp(
    parameter: Tensor,
    vector: Tensor,
    features: Tensor,
    labels: Tensor,
    spec: ParameterSpec,
    config: RealMLPConfig,
) -> Tensor:
    """Closed-form Hessian-vector product for the smooth real-data MLP."""
    w1, b1, w2, b2 = unpack(parameter, spec)
    d_w1, d_b1, d_w2, d_b2 = unpack(vector, spec)
    preactivation = features @ w1.T + b1
    hidden = torch.tanh(preactivation)
    first = 1.0 - hidden.square()
    values = hidden @ w2.T + b2
    probabilities = torch.softmax(values, dim=1)
    target = F.one_hot(labels, num_classes=spec.classes).to(parameter.dtype)
    error = (probabilities - target) / len(features)

    d_preactivation = features @ d_w1.T + d_b1
    d_hidden = first * d_preactivation
    d_values = d_hidden @ w2.T + hidden @ d_w2.T + d_b2
    d_probabilities = probabilities * (
        d_values - (probabilities * d_values).sum(dim=1, keepdim=True)
    )
    d_error = d_probabilities / len(features)

    back_hidden = error @ w2
    d_back_hidden = d_error @ w2 + error @ d_w2
    d_first = -2.0 * hidden * d_hidden
    d_back_pre = d_back_hidden * first + back_hidden * d_first

    h_w1 = d_back_pre.T @ features + config.weight_decay * d_w1
    h_b1 = d_back_pre.sum(dim=0) + config.weight_decay * d_b1
    h_w2 = (
        d_error.T @ hidden
        + error.T @ d_hidden
        + config.weight_decay * d_w2
    )
    h_b2 = d_error.sum(dim=0) + config.weight_decay * d_b2
    return torch.cat((h_w1.reshape(-1), h_b1, h_w2.reshape(-1), h_b2))


def autograd_objective_hvp(
    parameter: Tensor,
    vector: Tensor,
    features: Tensor,
    labels: Tensor,
    spec: ParameterSpec,
    config: RealMLPConfig,
) -> Tensor:
    """Reference implementation used only by validity tests."""
    with torch.enable_grad():
        point = parameter.detach().requires_grad_(True)
        value = objective(point, features, labels, spec, config)
        (grad,) = torch.autograd.grad(value, point, create_graph=True)
        (product,) = torch.autograd.grad(torch.dot(grad, vector.detach()), point)
    return product.detach()


@torch.no_grad()
def accuracy(parameter: Tensor, features: Tensor, labels: Tensor, spec: ParameterSpec) -> float:
    return float((logits(parameter, features, spec).argmax(dim=1) == labels).double().mean())


@torch.no_grad()
def count_correct(parameter: Tensor, features: Tensor, labels: Tensor, spec: ParameterSpec) -> int:
    return int((logits(parameter, features, spec).argmax(dim=1) == labels).sum())


def optimizer_map(
    parameter: Tensor,
    train_x: Tensor,
    train_y: Tensor,
    spec: ParameterSpec,
    config: RealMLPConfig,
) -> Tensor:
    return parameter - config.learning_rate * gradient(
        parameter, train_x, train_y, spec, config
    )


def optimizer_jvp_vjp(
    parameter: Tensor,
    train_x: Tensor,
    train_y: Tensor,
    spec: ParameterSpec,
    config: RealMLPConfig,
):
    """Return products with the symmetric full-batch GD Jacobian."""
    def product(vector: Tensor) -> Tensor:
        return vector - config.learning_rate * objective_hvp(
            parameter, vector, train_x, train_y, spec, config
        )

    return product, product


def config_dict(config: RealMLPConfig) -> dict:
    payload = asdict(config)
    payload["parameter_count"] = parameter_spec(config).size
    return payload
