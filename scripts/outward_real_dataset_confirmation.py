#!/usr/bin/env python3
"""Post-seal outward verification for the real-data GreenCert batch.

The four-sweep float64 centerline is treated as a sequence of exact binary
reference points. IEEE-754 gamma bounds enclose array arithmetic and 192-bit
Arb encloses tanh/logistic evaluations. A verified dense Hessian enclosure
then propagates a one-step exact-real state tube. This is deliberately
independent of the probabilistic Green radius: it is a numerically rigorous
cross-check of every issued first-passage event on the same exact-real map.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import flint
import numpy as np
import torch
from flint import arb, ctx

from outward_interval_certificate import (
    PRECISION_BITS,
    _arb_bounds,
    _center_radius,
    _down,
    _gamma,
    _interval_add,
    _interval_div_positive,
    _interval_matmul,
    _interval_mul,
    _interval_scale,
    _interval_sub,
    _interval_sum,
    _l2_upper,
    _max_row_sum_upper,
    _nonnegative_add_upper,
    _nonnegative_matmul_upper,
    _sum_nonnegative_upper,
    _up,
    _upper_add,
    _upper_div,
    _upper_mul,
    _upper_sqrt,
    _verified_beta,
)
from real_dataset_greencert import build_centerline, persistent_bracket
from real_dataset_mlp import (
    RealMLPConfig,
    make_split,
    parameter_spec,
    unpack,
)


ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "results" / "real_dataset_confirmation"
CACHE = ROOT / "results" / "real_dataset_outward_cache"
BLIND_SUMMARY = ROOT / "results" / "real_dataset_outward_blind.json"
JOINED_SUMMARY = ROOT / "results" / "real_dataset_outward_joined.json"
VERSION = "wdbc-arb-outward-v2-2026-08-24"
NUMERIC_CAP = 1e4


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _exact_arb(value: float) -> arb:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite endpoint")
    numerator, denominator = value.as_integer_ratio()
    return arb(numerator) / arb(denominator)


def _monotone_arb_bounds(lower: np.ndarray, upper: np.ndarray, function) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    result_lower = np.empty_like(lower)
    result_upper = np.empty_like(upper)
    old_precision = ctx.prec
    ctx.prec = PRECISION_BITS
    try:
        for index in np.ndindex(lower.shape):
            low_value = function(_exact_arb(float(lower[index])))
            high_value = function(_exact_arb(float(upper[index])))
            result_lower[index] = float(np.nextafter(float(low_value.lower()), -np.inf))
            result_upper[index] = float(np.nextafter(float(high_value.upper()), np.inf))
    finally:
        ctx.prec = old_precision
    return result_lower, result_upper


def _tanh_bounds(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _monotone_arb_bounds(lower, upper, lambda value: value.tanh())


def _sigmoid_bounds(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _monotone_arb_bounds(
        lower,
        upper,
        lambda value: arb(1) / (arb(1) + (-value).exp()),
    )


def _sqrt_integer_upper(value: int) -> float:
    old_precision = ctx.prec
    ctx.prec = PRECISION_BITS
    try:
        return float(np.nextafter(float(arb(value).sqrt().upper()), np.inf))
    finally:
        ctx.prec = old_precision


SQRT2 = _sqrt_integer_upper(2)
SQRT3 = _sqrt_integer_upper(3)


def _tanh_second_upper() -> float:
    old_precision = ctx.prec
    ctx.prec = PRECISION_BITS
    try:
        value = arb(4) / (arb(3) * arb(3).sqrt())
        return float(np.nextafter(float(value.upper()), np.inf))
    finally:
        ctx.prec = old_precision


TANH_SECOND = _tanh_second_upper()
TANH_THIRD = 2.0


def _parameter_arrays(parameter: np.ndarray, spec):
    tensor = torch.from_numpy(np.asarray(parameter, dtype=np.float64))
    return tuple(value.numpy() for value in unpack(tensor, spec))


def _network_intervals(parameter: np.ndarray, features: torch.Tensor, labels: torch.Tensor, spec):
    w1, b1, w2, b2 = _parameter_arrays(parameter, spec)
    x = features.numpy()
    pre_lower, pre_upper = _interval_matmul(x, x, w1.T, w1.T)
    pre_lower, pre_upper = _interval_add(
        pre_lower,
        pre_upper,
        np.broadcast_to(b1, pre_lower.shape),
        np.broadcast_to(b1, pre_upper.shape),
    )
    hidden_lower, hidden_upper = _tanh_bounds(pre_lower, pre_upper)
    hidden_square_lower, hidden_square_upper = _interval_mul(
        hidden_lower, hidden_upper, hidden_lower, hidden_upper
    )
    one = np.ones_like(hidden_lower)
    first_lower, first_upper = _interval_sub(
        one, one, hidden_square_lower, hidden_square_upper
    )
    hidden_first_lower, hidden_first_upper = _interval_mul(
        hidden_lower, hidden_upper, first_lower, first_upper
    )
    second_lower, second_upper = _interval_scale(
        hidden_first_lower, hidden_first_upper, -2.0
    )
    logits_lower, logits_upper = _interval_matmul(
        hidden_lower, hidden_upper, w2.T, w2.T
    )
    logits_lower, logits_upper = _interval_add(
        logits_lower,
        logits_upper,
        np.broadcast_to(b2, logits_lower.shape),
        np.broadcast_to(b2, logits_upper.shape),
    )
    difference_lower, difference_upper = _interval_sub(
        logits_lower[:, 0], logits_upper[:, 0], logits_lower[:, 1], logits_upper[:, 1]
    )
    p0_lower, p0_upper = _sigmoid_bounds(difference_lower, difference_upper)
    p1_lower, p1_upper = _interval_sub(
        np.ones_like(p0_lower),
        np.ones_like(p0_upper),
        p0_lower,
        p0_upper,
    )
    probability_lower = np.stack((p0_lower, p1_lower), axis=1)
    probability_upper = np.stack((p0_upper, p1_upper), axis=1)
    target = np.eye(spec.classes, dtype=np.float64)[labels.numpy()]
    error_lower, error_upper = _interval_sub(
        probability_lower, probability_upper, target, target
    )
    return {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "x": x,
        "hidden_lower": hidden_lower,
        "hidden_upper": hidden_upper,
        "first_lower": first_lower,
        "first_upper": first_upper,
        "second_lower": second_lower,
        "second_upper": second_upper,
        "logits_lower": logits_lower,
        "logits_upper": logits_upper,
        "probability_lower": probability_lower,
        "probability_upper": probability_upper,
        "error_lower": error_lower,
        "error_upper": error_upper,
    }


def _gradient_interval(parameter, features, labels, spec, config, network=None):
    network = (
        _network_intervals(parameter, features, labels, spec)
        if network is None
        else network
    )
    count = len(features)
    error_lower, error_upper = _interval_div_positive(
        network["error_lower"], network["error_upper"], count
    )
    back_lower, back_upper = _interval_matmul(
        error_lower, error_upper, network["w2"], network["w2"]
    )
    pre_lower, pre_upper = _interval_mul(
        back_lower,
        back_upper,
        network["first_lower"],
        network["first_upper"],
    )
    gw1_lower, gw1_upper = _interval_matmul(
        pre_lower.T, pre_upper.T, network["x"], network["x"]
    )
    gb1_lower, gb1_upper = _interval_sum(pre_lower, pre_upper, axis=0)
    gw2_lower, gw2_upper = _interval_matmul(
        error_lower.T,
        error_upper.T,
        network["hidden_lower"],
        network["hidden_upper"],
    )
    gb2_lower, gb2_upper = _interval_sum(error_lower, error_upper, axis=0)
    lower = np.concatenate(
        (gw1_lower.reshape(-1), gb1_lower, gw2_lower.reshape(-1), gb2_lower)
    )
    upper = np.concatenate(
        (gw1_upper.reshape(-1), gb1_upper, gw2_upper.reshape(-1), gb2_upper)
    )
    if config.weight_decay:
        reg_lower, reg_upper = _interval_scale(
            parameter, parameter, config.weight_decay
        )
        lower, upper = _interval_add(lower, upper, reg_lower, reg_upper)
    return lower, upper


def _dense_hessian_interval(parameter, features, labels, spec, config, network=None):
    """Enclose the full mean-CE Hessian by its binary-softmax decomposition."""
    network = (
        _network_intervals(parameter, features, labels, spec)
        if network is None
        else network
    )
    n = len(features)
    p = spec.size
    h = spec.width
    d = spec.input_dim
    w2 = network["w2"]
    x = network["x"]
    x_aug = np.concatenate((x, np.ones((n, 1), dtype=np.float64)), axis=1)
    j_lower = np.zeros((n, p), dtype=np.float64)
    j_upper = np.zeros_like(j_lower)

    difference_lower, difference_upper = _interval_sub(
        w2[0], w2[0], w2[1], w2[1]
    )
    gain_lower, gain_upper = _interval_mul(
        network["first_lower"],
        network["first_upper"],
        np.broadcast_to(difference_lower, (n, h)),
        np.broadcast_to(difference_upper, (n, h)),
    )
    for unit in range(h):
        product_lower, product_upper = _interval_mul(
            gain_lower[:, unit, None],
            gain_upper[:, unit, None],
            x,
            x,
        )
        start = unit * d
        j_lower[:, start : start + d] = product_lower
        j_upper[:, start : start + d] = product_upper
        j_lower[:, h * d + unit] = gain_lower[:, unit]
        j_upper[:, h * d + unit] = gain_upper[:, unit]

    w2_offset = h * d + h
    j_lower[:, w2_offset : w2_offset + h] = network["hidden_lower"]
    j_upper[:, w2_offset : w2_offset + h] = network["hidden_upper"]
    j_lower[:, w2_offset + h : w2_offset + 2 * h] = -network["hidden_upper"]
    j_upper[:, w2_offset + h : w2_offset + 2 * h] = -network["hidden_lower"]
    b2_offset = w2_offset + 2 * h
    j_lower[:, b2_offset] = 1.0
    j_upper[:, b2_offset] = 1.0
    j_lower[:, b2_offset + 1] = -1.0
    j_upper[:, b2_offset + 1] = -1.0

    weight_lower, weight_upper = _interval_mul(
        network["probability_lower"][:, 0],
        network["probability_upper"][:, 0],
        network["probability_lower"][:, 1],
        network["probability_upper"][:, 1],
    )
    weighted_lower, weighted_upper = _interval_mul(
        j_lower,
        j_upper,
        weight_lower[:, None],
        weight_upper[:, None],
    )
    gauss_lower, gauss_upper = _interval_matmul(
        j_lower.T, j_upper.T, weighted_lower, weighted_upper
    )
    gauss_lower, gauss_upper = _interval_div_positive(
        gauss_lower, gauss_upper, n
    )

    residual_lower = np.zeros((p, p), dtype=np.float64)
    residual_upper = np.zeros_like(residual_lower)
    for unit in range(h):
        weighted_error_lower, weighted_error_upper = _interval_mul(
            network["error_lower"],
            network["error_upper"],
            np.broadcast_to(w2[:, unit], (n, spec.classes)),
            np.broadcast_to(w2[:, unit], (n, spec.classes)),
        )
        summed_lower, summed_upper = _interval_sum(
            weighted_error_lower, weighted_error_upper, axis=1
        )
        coefficient_lower, coefficient_upper = _interval_mul(
            summed_lower,
            summed_upper,
            network["second_lower"][:, unit],
            network["second_upper"][:, unit],
        )
        weighted_x_lower, weighted_x_upper = _interval_mul(
            coefficient_lower[:, None],
            coefficient_upper[:, None],
            x_aug,
            x_aug,
        )
        block_lower, block_upper = _interval_matmul(
            x_aug.T, x_aug.T, weighted_x_lower, weighted_x_upper
        )
        block_lower, block_upper = _interval_div_positive(
            block_lower, block_upper, n
        )
        hidden_indices = list(range(unit * d, (unit + 1) * d)) + [h * d + unit]
        residual_lower[np.ix_(hidden_indices, hidden_indices)] = block_lower
        residual_upper[np.ix_(hidden_indices, hidden_indices)] = block_upper

        for output in range(spec.classes):
            cross_lower, cross_upper = _interval_mul(
                network["error_lower"][:, output],
                network["error_upper"][:, output],
                network["first_lower"][:, unit],
                network["first_upper"][:, unit],
            )
            cross_x_lower, cross_x_upper = _interval_mul(
                cross_lower[:, None], cross_upper[:, None], x_aug, x_aug
            )
            vector_lower, vector_upper = _interval_sum(
                cross_x_lower, cross_x_upper, axis=0
            )
            vector_lower, vector_upper = _interval_div_positive(
                vector_lower, vector_upper, n
            )
            output_index = w2_offset + output * h + unit
            residual_lower[hidden_indices, output_index] = vector_lower
            residual_upper[hidden_indices, output_index] = vector_upper
            residual_lower[output_index, hidden_indices] = vector_lower
            residual_upper[output_index, hidden_indices] = vector_upper

    lower, upper = _interval_add(
        gauss_lower, gauss_upper, residual_lower, residual_upper
    )
    if config.weight_decay:
        diagonal = np.eye(p, dtype=np.float64) * config.weight_decay
        lower, upper = _interval_add(lower, upper, diagonal, diagonal)
    return lower, upper


def _hessian_beta(parameter, features, labels, spec, config, network=None):
    center, evaluation_error = _dense_hessian_center_error(
        parameter, features, labels, spec, config, network
    )
    beta, diagnostics = _verified_beta(
        center, evaluation_error, config.learning_rate
    )
    return beta, {"hessian_interval_row_radius": evaluation_error, **diagnostics}


def _dense_hessian_center_error(parameter, features, labels, spec, config, network=None):
    """Fast rigorous center plus spectral error for the exact CE Hessian.

    The expensive Gauss--Newton block is evaluated once at interval midpoints.
    Frobenius/operator inequalities enclose all Jacobian and probability
    uncertainty, while a gamma bound encloses the central BLAS product. The
    sparse residual-logit Hessian is still evaluated entrywise by intervals.
    """
    network = (
        _network_intervals(parameter, features, labels, spec)
        if network is None
        else network
    )
    n = len(features)
    p = spec.size
    h = spec.width
    d = spec.input_dim
    w2 = network["w2"]
    x = network["x"]
    x_aug = np.concatenate((x, np.ones((n, 1), dtype=np.float64)), axis=1)
    j_lower = np.zeros((n, p), dtype=np.float64)
    j_upper = np.zeros_like(j_lower)
    difference_lower, difference_upper = _interval_sub(
        w2[0], w2[0], w2[1], w2[1]
    )
    gain_lower, gain_upper = _interval_mul(
        network["first_lower"],
        network["first_upper"],
        np.broadcast_to(difference_lower, (n, h)),
        np.broadcast_to(difference_upper, (n, h)),
    )
    for unit in range(h):
        product_lower, product_upper = _interval_mul(
            gain_lower[:, unit, None],
            gain_upper[:, unit, None],
            x,
            x,
        )
        start = unit * d
        j_lower[:, start : start + d] = product_lower
        j_upper[:, start : start + d] = product_upper
        j_lower[:, h * d + unit] = gain_lower[:, unit]
        j_upper[:, h * d + unit] = gain_upper[:, unit]
    w2_offset = h * d + h
    j_lower[:, w2_offset : w2_offset + h] = network["hidden_lower"]
    j_upper[:, w2_offset : w2_offset + h] = network["hidden_upper"]
    j_lower[:, w2_offset + h : w2_offset + 2 * h] = -network["hidden_upper"]
    j_upper[:, w2_offset + h : w2_offset + 2 * h] = -network["hidden_lower"]
    b2_offset = w2_offset + 2 * h
    j_lower[:, b2_offset] = j_upper[:, b2_offset] = 1.0
    j_lower[:, b2_offset + 1] = j_upper[:, b2_offset + 1] = -1.0
    weight_lower, weight_upper = _interval_mul(
        network["probability_lower"][:, 0],
        network["probability_upper"][:, 0],
        network["probability_lower"][:, 1],
        network["probability_upper"][:, 1],
    )
    j_center, j_radius = _center_radius(j_lower, j_upper)
    weight_center, weight_radius = _center_radius(weight_lower, weight_upper)
    weighted_center = weight_center[:, None] * j_center
    gauss_center = (j_center.T @ weighted_center) / n

    j_norm = _l2_upper(j_center)
    j_error = _l2_upper(j_radius)
    weight_norm = float(_up(np.max(np.abs(weight_center) + weight_radius)))
    weight_error = float(_up(np.max(weight_radius)))
    uncertainty = _upper_add(
        _upper_mul(weight_error, _upper_mul(j_norm, j_norm)),
        _upper_add(
            _upper_mul(2.0 * weight_norm, _upper_mul(j_norm, j_error)),
            _upper_mul(weight_norm, _upper_mul(j_error, j_error)),
        ),
    )
    uncertainty = _upper_div(uncertainty, n)
    absolute_weighted = _up(
        np.abs(weight_center)[:, None] * np.abs(j_center)
    )
    absolute_product = _nonnegative_matmul_upper(
        np.abs(j_center).T, absolute_weighted
    )
    central_roundoff = _upper_div(
        _upper_mul(_gamma(n + 2), _l2_upper(absolute_product)), n
    )
    gauss_error = _upper_add(uncertainty, central_roundoff)

    residual_lower = np.zeros((p, p), dtype=np.float64)
    residual_upper = np.zeros_like(residual_lower)
    for unit in range(h):
        weighted_error_lower, weighted_error_upper = _interval_mul(
            network["error_lower"],
            network["error_upper"],
            np.broadcast_to(w2[:, unit], (n, spec.classes)),
            np.broadcast_to(w2[:, unit], (n, spec.classes)),
        )
        summed_lower, summed_upper = _interval_sum(
            weighted_error_lower, weighted_error_upper, axis=1
        )
        coefficient_lower, coefficient_upper = _interval_mul(
            summed_lower,
            summed_upper,
            network["second_lower"][:, unit],
            network["second_upper"][:, unit],
        )
        weighted_x_lower, weighted_x_upper = _interval_mul(
            coefficient_lower[:, None],
            coefficient_upper[:, None],
            x_aug,
            x_aug,
        )
        block_lower, block_upper = _interval_matmul(
            x_aug.T, x_aug.T, weighted_x_lower, weighted_x_upper
        )
        block_lower, block_upper = _interval_div_positive(
            block_lower, block_upper, n
        )
        hidden_indices = list(range(unit * d, (unit + 1) * d)) + [h * d + unit]
        residual_lower[np.ix_(hidden_indices, hidden_indices)] = block_lower
        residual_upper[np.ix_(hidden_indices, hidden_indices)] = block_upper
        for output in range(spec.classes):
            cross_lower, cross_upper = _interval_mul(
                network["error_lower"][:, output],
                network["error_upper"][:, output],
                network["first_lower"][:, unit],
                network["first_upper"][:, unit],
            )
            cross_x_lower, cross_x_upper = _interval_mul(
                cross_lower[:, None], cross_upper[:, None], x_aug, x_aug
            )
            vector_lower, vector_upper = _interval_sum(
                cross_x_lower, cross_x_upper, axis=0
            )
            vector_lower, vector_upper = _interval_div_positive(
                vector_lower, vector_upper, n
            )
            output_index = w2_offset + output * h + unit
            residual_lower[hidden_indices, output_index] = vector_lower
            residual_upper[hidden_indices, output_index] = vector_upper
            residual_lower[output_index, hidden_indices] = vector_lower
            residual_upper[output_index, hidden_indices] = vector_upper
    residual_center, residual_radius = _center_radius(
        residual_lower, residual_upper
    )
    residual_error = _max_row_sum_upper(residual_radius)
    regularizer = np.eye(p, dtype=np.float64) * config.weight_decay
    hessian = gauss_center + residual_center + regularizer
    combine_magnitude = _nonnegative_add_upper(
        np.abs(gauss_center),
        np.abs(residual_center),
        np.abs(regularizer),
        np.abs(hessian),
    )
    combine_roundoff = _max_row_sum_upper(_up(3.0 * np.finfo(np.float64).eps * combine_magnitude))
    evaluation_error = _upper_add(
        gauss_error, _upper_add(residual_error, combine_roundoff)
    )
    return hessian, evaluation_error


def _defect_upper(parameter, next_parameter, features, labels, spec, config, network=None):
    lower, upper = _gradient_interval(
        parameter, features, labels, spec, config, network
    )
    scaled_lower, scaled_upper = _interval_scale(
        lower, upper, config.learning_rate
    )
    mapped_lower, mapped_upper = _interval_sub(
        parameter, parameter, scaled_lower, scaled_upper
    )
    defect_lower, defect_upper = _interval_sub(
        mapped_lower, mapped_upper, next_parameter, next_parameter
    )
    return _l2_upper(np.maximum(np.abs(defect_lower), np.abs(defect_upper)))


def _max_augmented_input_norm(features: torch.Tensor) -> float:
    x = features.numpy()
    augmented = np.concatenate((x, np.ones((len(x), 1), dtype=np.float64)), axis=1)
    return max(_l2_upper(row) for row in augmented)


def _hessian_lipschitz_upper(parameter, train_x, spec, radius: float) -> float:
    _, _, w2, _ = _parameter_arrays(parameter, spec)
    w_norm = _upper_add(_l2_upper(w2), radius)  # Frobenius dominates spectral norm.
    x_norm = _max_augmented_input_norm(train_x)
    hidden_norm = _sqrt_integer_upper(spec.width)
    hidden_bias = _upper_sqrt(_upper_add(_upper_mul(hidden_norm, hidden_norm), 1.0))
    wx = _upper_mul(w_norm, x_norm)
    first = _upper_sqrt(_upper_add(_upper_mul(hidden_bias, hidden_bias), _upper_mul(wx, wx)))
    second = _upper_add(
        _upper_mul(2.0, x_norm),
        _upper_mul(_upper_mul(w_norm, TANH_SECOND), _upper_mul(x_norm, x_norm)),
    )
    x2 = _upper_mul(x_norm, x_norm)
    x3 = _upper_mul(x2, x_norm)
    third = _upper_add(
        _upper_mul(_upper_mul(3.0, TANH_SECOND), x2),
        _upper_mul(_upper_mul(w_norm, TANH_THIRD), x3),
    )
    first_cubed = _upper_mul(_upper_mul(first, first), first)
    return _upper_add(
        _upper_add(
            _upper_mul(2.0, first_cubed),
            _upper_mul(1.5, _upper_mul(first, second)),
        ),
        _upper_mul(SQRT2, third),
    )


def _verified_geometry(parameter, next_parameter, data, spec, config):
    network = _network_intervals(parameter, data["train_x"], data["train_y"], spec)
    beta, diagnostics = _hessian_beta(
        parameter,
        data["train_x"],
        data["train_y"],
        spec,
        config,
        network,
    )
    defect = _defect_upper(
        parameter,
        next_parameter,
        data["train_x"],
        data["train_y"],
        spec,
        config,
        network,
    )
    return beta, defect, diagnostics


def verified_tube(reference: np.ndarray, data, spec, config, *, progress: str | None = None):
    radius = np.zeros(len(reference), dtype=np.float64)
    diagnostics = []
    reached = 0
    for step in range(len(reference) - 1):
        current = float(radius[step])
        if not np.isfinite(current) or current > NUMERIC_CAP:
            break
        if progress and (step == 0 or (step + 1) % 10 == 0):
            print(f"  {progress}: state {step + 1}/{len(reference) - 1}", flush=True)
        beta, defect, row = _verified_geometry(
            reference[step], reference[step + 1], data, spec, config
        )
        lipschitz = _upper_mul(
            config.learning_rate,
            _hessian_lipschitz_upper(reference[step], data["train_x"], spec, current),
        )
        nonlinear = _upper_mul(
            0.5, _upper_mul(lipschitz, _upper_mul(current, current))
        )
        next_radius = _upper_add(
            _upper_add(_upper_mul(beta, current), defect), nonlinear
        )
        diagnostics.append(
            {
                "step": step,
                "beta_upper": beta,
                "defect_norm_upper": defect,
                "optimizer_jacobian_lipschitz_upper": lipschitz,
                "next_radius": next_radius,
                **row,
            }
        )
        if not np.isfinite(next_radius) or next_radius > NUMERIC_CAP:
            break
        radius[step + 1] = next_radius
        reached = step + 1
    return radius[: reached + 1], reached, diagnostics


def _margin_gradient_upper(parameter, feature: np.ndarray, spec, radius: float) -> float:
    _, _, w2, _ = _parameter_arrays(parameter, spec)
    difference_lower, difference_upper = _interval_sub(
        w2[0], w2[0], w2[1], w2[1]
    )
    row_norm = _upper_add(
        _l2_upper(np.maximum(np.abs(difference_lower), np.abs(difference_upper))),
        _upper_mul(SQRT2, radius),
    )
    x_aug = _l2_upper(np.concatenate((feature, np.ones(1, dtype=np.float64))))
    hidden_part = _upper_mul(2.0, float(spec.width))
    parameter_part = _upper_mul(_upper_mul(row_norm, x_aug), _upper_mul(row_norm, x_aug))
    return _upper_sqrt(_upper_add(_upper_add(hidden_part, 2.0), parameter_part))


def certified_count_paths(reference: np.ndarray, radius: np.ndarray, data, spec):
    features = data["certificate_x"]
    labels = data["certificate_y"].numpy()
    guaranteed = []
    possible = []
    minimum_logic_slack = math.inf
    for step, center in enumerate(reference):
        network = _network_intervals(center, features, data["certificate_y"], spec)
        guaranteed_count = 0
        excluded_count = 0
        for sample, label in enumerate(labels):
            competitor = 1 - int(label)
            margin_lower = float(
                _down(
                    network["logits_lower"][sample, label]
                    - network["logits_upper"][sample, competitor]
                )
            )
            margin_upper = float(
                _up(
                    network["logits_upper"][sample, label]
                    - network["logits_lower"][sample, competitor]
                )
            )
            b1 = _margin_gradient_upper(
                center, features[sample].numpy(), spec, float(radius[step])
            )
            error = _upper_mul(b1, float(radius[step]))
            lower = float(_down(margin_lower - error))
            upper = float(_up(margin_upper + error))
            minimum_logic_slack = min(minimum_logic_slack, abs(lower), abs(upper))
            guaranteed_count += int(lower > 0.0)
            excluded_count += int(upper < 0.0)
        guaranteed.append(guaranteed_count)
        possible.append(len(labels) - excluded_count)
    return np.asarray(guaranteed), np.asarray(possible), minimum_logic_slack


def _config_from_seal(method: dict, seed: int) -> RealMLPConfig:
    payload = dict(method["config"])
    payload.pop("parameter_count", None)
    payload["seed"] = seed
    return RealMLPConfig(**payload)


def cache_path(seed: int, anchor: int) -> Path:
    return CACHE / f"seed_{seed}_anchor_{anchor}.json"


@torch.no_grad()
def verify_anchor(seed: int, anchor: int, candidates: list[dict], *, use_cache: bool = True):
    method_path = EXPORT / "REAL_DATA_GREENCERT_METHOD_SEAL.json"
    method = read_json(method_path)
    output = cache_path(seed, anchor)
    maximum_horizon = max(int(row["certificate_horizon"]) for row in candidates)
    candidate_hashes = sorted(row["certificate_sha256"] for row in candidates)
    if use_cache and output.exists():
        payload = read_json(output)
        if (
            payload.get("version") == VERSION
            and payload.get("method_seal_sha256") == sha256(method_path)
            and payload.get("candidate_sha256") == candidate_hashes
            and payload.get("requested_horizon") == maximum_horizon
        ):
            return payload
    config = _config_from_seal(method, seed)
    torch.set_num_threads(config.threads)
    data = make_split(config)
    spec = parameter_spec(config)
    checkpoints = np.load(EXPORT / "checkpoints" / f"seed_{seed}.checkpoints.npz")
    parameter = torch.from_numpy(checkpoints[f"step_{anchor}"]).clone()
    paths, _ = build_centerline(
        parameter,
        data,
        spec,
        config,
        horizon=int(method["horizon"]),
        sweeps=int(method["sweeps"]),
    )
    reference = paths[-1][: maximum_horizon + 1].numpy()
    started = time.perf_counter()
    radius, reached, diagnostics = verified_tube(
        reference,
        data,
        spec,
        config,
        progress=f"seed {seed} anchor {anchor}",
    )
    guaranteed, possible, minimum_logic_slack = certified_count_paths(
        reference[: len(radius)], radius, data, spec
    )
    events = {}
    for candidate in candidates:
        required = int(candidate["required_correct"])
        bracket = persistent_bracket(
            guaranteed, possible, required, int(method["persistence"])
        )
        events[f"{float(candidate['threshold']):.3f}"] = {
            "green_float_bracket": candidate["certified_bracket"],
            "outward_bracket": bracket,
        }
    payload = {
        "status": "post-seal outward exact-real map verification",
        "scope_note": (
            "Reference points and checkpoint are exact binary floats. The Arb tube encloses "
            "the exact-real optimizer map, independently of the Green radius."
        ),
        "version": VERSION,
        "method_seal_sha256": sha256(method_path),
        "candidate_sha256": candidate_hashes,
        "python_flint_version": flint.__version__,
        "arb_precision_bits": PRECISION_BITS,
        "seed": seed,
        "anchor": anchor,
        "requested_horizon": maximum_horizon,
        "reached_horizon": reached,
        "events": events,
        "maximum_radius": float(np.max(radius)),
        "minimum_logic_slack": minimum_logic_slack,
        "maximum_hessian_interval_row_radius": max(
            (row["hessian_interval_row_radius"] for row in diagnostics), default=0.0
        ),
        "maximum_eigen_numeric_error": max(
            (row["eigen_numeric_error"] for row in diagnostics), default=0.0
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "diagnostics": diagnostics,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(output.with_suffix(".npz"), radius=radius)
    return payload


def _issued_candidates() -> list[dict]:
    manifest = read_json(EXPORT / "certificate_manifest.json")
    issued = []
    for record in manifest["records"]:
        if not record["issued"]:
            continue
        certificate = read_json(EXPORT / "certificates" / record["path"])
        issued.append({**certificate, "certificate_sha256": record["sha256"]})
    return issued


def _verify_task(args):
    seed, anchor, rows, use_cache = args
    return seed, anchor, verify_anchor(seed, anchor, rows, use_cache=use_cache)


def verify_all(*, use_cache: bool = True, workers: int = 1) -> dict:
    issued = _issued_candidates()
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in issued:
        grouped[(int(row["seed"]), int(row["anchor"]))].append(row)
    tasks = [
        (seed, anchor, rows, use_cache)
        for (seed, anchor), rows in sorted(grouped.items())
    ]
    caches = {}
    if workers <= 1:
        for index, task in enumerate(tasks, start=1):
            seed, anchor, _, _ = task
            print(f"outward anchor {index}/{len(tasks)}: seed {seed}, anchor {anchor}", flush=True)
            _, _, payload = _verify_task(task)
            caches[(seed, anchor)] = payload
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_verify_task, task): task[:2] for task in tasks}
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                seed, anchor, payload = future.result()
                caches[(seed, anchor)] = payload
                completed += 1
                print(
                    f"outward anchor complete {completed}/{len(tasks)}: seed {seed}, anchor {anchor}",
                    flush=True,
                )
    blind_rows = []
    for row in issued:
        cache = caches[(int(row["seed"]), int(row["anchor"]))]
        event = cache["events"][f"{float(row['threshold']):.3f}"]
        blind_rows.append(
            {
                "seed": int(row["seed"]),
                "threshold": float(row["threshold"]),
                "anchor": int(row["anchor"]),
                "green_float_bracket": row["certified_bracket"],
                "outward_bracket": event["outward_bracket"],
                "outward_issued": event["outward_bracket"] is not None,
            }
        )
    blind = {
        "status": "post-seal outward verification before outcome join within this script",
        "version": VERSION,
        "issued_green_candidates": len(issued),
        "unique_seed_anchor_tubes": len(grouped),
        "outward_retained": sum(row["outward_issued"] for row in blind_rows),
        "rows": blind_rows,
    }
    BLIND_SUMMARY.write_text(json.dumps(blind, indent=2) + "\n", encoding="utf-8")
    return blind


def join_outcomes() -> dict:
    blind = read_json(BLIND_SUMMARY)
    final = read_json(EXPORT / "final_audit.json")
    actual = {
        (int(row["seed"]), int(row["gate_index"])): row["actual_event"]
        for row in final["rows"]
    }
    thresholds = read_json(EXPORT / "REAL_DATA_GREENCERT_METHOD_SEAL.json")["thresholds"]
    rows = []
    for row in blind["rows"]:
        gate = thresholds.index(row["threshold"])
        event = actual[(row["seed"], gate)]
        bracket = row["outward_bracket"]
        rows.append(
            {
                **row,
                "actual_event": event,
                "outward_covered": (
                    None
                    if bracket is None or event is None
                    else int(bracket[0]) <= int(event) <= int(bracket[1])
                ),
            }
        )
    issued = [row for row in rows if row["outward_issued"]]
    covered = [row for row in issued if row["outward_covered"]]
    summary = {
        "status": "post-seal outward verification joined to sealed outcomes",
        "green_issued": len(rows),
        "outward_issued": len(issued),
        "outward_covered": len(covered),
        "distinct_outward_issuing_seeds": len({row["seed"] for row in issued}),
        "outward_brackets_identical_to_green": sum(
            row["outward_bracket"] == row["green_float_bracket"] for row in issued
        ),
        "maximum_outward_bracket_width": max(
            (row["outward_bracket"][1] - row["outward_bracket"][0] for row in issued),
            default=None,
        ),
    }
    result = {"summary": summary, "rows": rows}
    JOINED_SUMMARY.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("one", "blind", "join", "all"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--anchor", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    if args.phase == "one":
        if args.seed is None or args.anchor is None:
            raise ValueError("one requires --seed and --anchor")
        rows = [
            row
            for row in _issued_candidates()
            if int(row["seed"]) == args.seed and int(row["anchor"]) == args.anchor
        ]
        if not rows:
            raise ValueError("no issued candidate for seed/anchor")
        print(json.dumps(verify_anchor(args.seed, args.anchor, rows, use_cache=not args.no_cache), indent=2))
    elif args.phase == "blind":
        verify_all(use_cache=not args.no_cache, workers=args.workers)
    elif args.phase == "join":
        join_outcomes()
    else:
        verify_all(use_cache=not args.no_cache, workers=args.workers)
        join_outcomes()


if __name__ == "__main__":
    main()
