#!/usr/bin/env python3
"""Outcome-barrier loader for the prospective handwritten-digits study."""
from __future__ import annotations

import hashlib

import numpy as np
import torch
from sklearn.datasets import load_digits

from real_dataset_mlp import ParameterSpec, RealMLPConfig


def raw_digits() -> tuple[np.ndarray, np.ndarray]:
    dataset = load_digits()
    features = np.asarray(dataset.data, dtype=np.float64)
    labels = np.asarray(dataset.target, dtype=np.int64)
    return features, labels


def raw_data_sha256() -> str:
    features, labels = raw_digits()
    digest = hashlib.sha256()
    digest.update(features.tobytes(order="C"))
    digest.update(labels.tobytes(order="C"))
    return digest.hexdigest().upper()


def stratified_indices(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a deterministic 60/20/20 split, stratified by parity label."""
    rng = np.random.default_rng(seed + 170_141)
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


def _make_split(config: RealMLPConfig, *, include_certificate: bool) -> dict:
    features, digits = raw_digits()
    labels = (digits % 2).astype(np.int64)
    train_idx, trigger_idx, certificate_idx = stratified_indices(labels, config.seed)
    mean = features[train_idx].mean(axis=0)
    scale = features[train_idx].std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (features - mean) / scale
    dtype = torch.float64 if config.dtype == "float64" else torch.float32

    def rows(indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.as_tensor(standardized[indices], dtype=dtype),
            torch.as_tensor(labels[indices], dtype=torch.long),
        )

    train_x, train_y = rows(train_idx)
    trigger_x, trigger_y = rows(trigger_idx)
    result = {
        "train_x": train_x,
        "train_y": train_y,
        "trigger_x": trigger_x,
        "trigger_y": trigger_y,
        "metadata": {
            "dataset": "scikit-learn digits test subset of UCI Optical Recognition of Handwritten Digits",
            "task": "binary parity of the digit label",
            "examples": int(len(features)),
            "features": int(features.shape[1]),
            "classes": 2,
            "train_examples": int(len(train_idx)),
            "trigger_examples": int(len(trigger_idx)),
            "split_seed": int(config.seed + 170_141),
            "standardization": "train mean and population standard deviation",
            "raw_data_sha256": raw_data_sha256(),
        },
    }
    if include_certificate:
        certificate_x, certificate_y = rows(certificate_idx)
        result["certificate_x"] = certificate_x
        result["certificate_y"] = certificate_y
        result["metadata"]["certificate_examples"] = int(len(certificate_idx))
    return result


def make_selection_split(config: RealMLPConfig) -> dict:
    """Materialize train/trigger tensors only; certification rows stay absent."""
    return _make_split(config, include_certificate=False)


def make_split(config: RealMLPConfig) -> dict:
    """Materialize all splits after the candidate list has been sealed."""
    return _make_split(config, include_certificate=True)


def parameter_spec(config: RealMLPConfig) -> ParameterSpec:
    return ParameterSpec(input_dim=64, width=config.width, classes=2)


def initialize(config: RealMLPConfig) -> torch.Tensor:
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
