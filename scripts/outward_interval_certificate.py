#!/usr/bin/env python3
"""Outward-enclosed re-evaluation of prospective v2 certificate candidates.

The recentered float64 path is treated as an arbitrary sequence of exact
binary reference points.  Arb encloses every tanh evaluation.  Array
operations use interval endpoints plus standard IEEE-754 gamma bounds for dot
products and reductions.  The state recursion is propagated upward, and
logit-margin tests are evaluated with outward endpoints.

The resulting tube certifies the exact-real gradient-descent map initialized
at the binary checkpoint.  It does not enclose roundoff accumulated by the
separately observed float64 training trajectory; that trajectory remains an
outcome audit.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import flint
from flint import arb, ctx

from generate_smooth_mlp_seed import frozen_config
from modular_accuracy_certificate import event_bracket
from prospective_v2_primary import (
    CACHE_DIR,
    HORIZON,
    METHOD_VERSION,
    OUT as PROSPECTIVE_RESULT,
    PIPELINE_VERSION,
    SCAN_OUT,
    cache_paths,
    clopper_pearson,
    manifest_sha256,
    protocol_sha256,
    verify_manifest,
)
from replay_smooth_mlp_thresholds import THRESHOLDS, required_counts
from smooth_mlp_certificate import exact_objective_hessian
from smooth_mlp_modular_grokking import logits, make_split, unpack


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "prospective_v2_interval_cache"
BLIND_SUMMARY = ROOT / "results" / "prospective_v2_interval_blind.json"
SUMMARY = ROOT / "results" / "prospective_v2_interval.json"
PRECISION_BITS = 192
EPS = np.finfo(np.float64).eps
TINY = np.nextafter(0.0, np.inf)
NUMERIC_CAP = 1e4
INTERVAL_VERSION = "arb-outward-v1-2026-08-21"


def _up(value):
    return np.nextafter(np.asarray(value, dtype=np.float64), np.inf)


def _down(value):
    return np.nextafter(np.asarray(value, dtype=np.float64), -np.inf)


def _gamma(count: int) -> float:
    if count <= 0:
        return 0.0
    product = count * EPS
    if product >= 1.0:
        raise ValueError("roundoff gamma is undefined")
    return float(_up(product / (1.0 - product)))


def _upper_add(left: float, right: float) -> float:
    return float(_up(float(left) + float(right)))


def _upper_mul(left: float, right: float) -> float:
    if left < 0.0 or right < 0.0:
        raise ValueError("upper multiplication expects nonnegative operands")
    return float(_up(float(left) * float(right)))


def _upper_div(value: float, divisor: float) -> float:
    if value < 0.0 or divisor <= 0.0:
        raise ValueError("upper division expects nonnegative value and positive divisor")
    return float(_up(float(value) / float(divisor)))


def _upper_sqrt(value: float) -> float:
    if value < 0.0:
        raise ValueError("square root requires a nonnegative value")
    return float(_up(math.sqrt(float(value))))


def _sum_nonnegative_upper(values: np.ndarray, axis=None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if np.any(array < 0.0):
        raise ValueError("nonnegative upper sum received a negative entry")
    count = array.size if axis is None else array.shape[axis]
    raw = np.sum(array, axis=axis)
    return _up(raw / (1.0 - _gamma(count)))


def _nonnegative_add_upper(*values: np.ndarray) -> np.ndarray:
    """Elementwise upper bound on a sum of nonnegative arrays."""
    return _sum_nonnegative_upper(np.stack(values, axis=0), axis=0)


def _l2_upper(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    magnitudes = np.abs(array)
    squares = _up(magnitudes * magnitudes)
    total = float(_sum_nonnegative_upper(squares))
    return _upper_sqrt(total)


def _max_row_sum_upper(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if np.any(array < 0.0):
        raise ValueError("row-sum upper bound received a negative entry")
    rows = _sum_nonnegative_upper(array, axis=1)
    return float(_up(np.max(rows)))


def _center_radius(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    center = 0.5 * lower + 0.5 * upper
    radius = _up(np.maximum(np.abs(center - lower), np.abs(upper - center)))
    return center, radius


def _interval_add(
    left_lower: np.ndarray,
    left_upper: np.ndarray,
    right_lower: np.ndarray,
    right_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return _down(left_lower + right_lower), _up(left_upper + right_upper)


def _interval_sub(
    left_lower: np.ndarray,
    left_upper: np.ndarray,
    right_lower: np.ndarray,
    right_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return _down(left_lower - right_upper), _up(left_upper - right_lower)


def _interval_mul(
    left_lower: np.ndarray,
    left_upper: np.ndarray,
    right_lower: np.ndarray,
    right_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.stack(
        (
            left_lower * right_lower,
            left_lower * right_upper,
            left_upper * right_lower,
            left_upper * right_upper,
        )
    )
    return _down(np.min(candidates, axis=0)), _up(np.max(candidates, axis=0))


def _interval_scale(
    lower: np.ndarray, upper: np.ndarray, scalar: float
) -> tuple[np.ndarray, np.ndarray]:
    scalar_array = np.asarray(scalar, dtype=np.float64)
    if scalar >= 0.0:
        return _down(lower * scalar_array), _up(upper * scalar_array)
    return _down(upper * scalar_array), _up(lower * scalar_array)


def _interval_div_positive(
    lower: np.ndarray, upper: np.ndarray, divisor: float
) -> tuple[np.ndarray, np.ndarray]:
    if divisor <= 0.0:
        raise ValueError("interval divisor must be positive")
    return _down(lower / divisor), _up(upper / divisor)


def _interval_sum(
    lower: np.ndarray, upper: np.ndarray, axis: int
) -> tuple[np.ndarray, np.ndarray]:
    count = lower.shape[axis]
    lower_raw = np.sum(lower, axis=axis)
    upper_raw = np.sum(upper, axis=axis)
    lower_magnitude = _sum_nonnegative_upper(np.abs(lower), axis=axis)
    upper_magnitude = _sum_nonnegative_upper(np.abs(upper), axis=axis)
    lower_error = _up(_gamma(count) * lower_magnitude)
    upper_error = _up(_gamma(count) * upper_magnitude)
    return _down(lower_raw - lower_error), _up(upper_raw + upper_error)


def _nonnegative_matmul_upper(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if np.any(left < 0.0) or np.any(right < 0.0):
        raise ValueError("nonnegative matrix product received a negative entry")
    count = left.shape[1]
    raw = left @ right
    return _up(raw / (1.0 - _gamma(count)))


def _interval_matmul(
    left_lower: np.ndarray,
    left_upper: np.ndarray,
    right_lower: np.ndarray,
    right_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left_center, left_radius = _center_radius(left_lower, left_upper)
    right_center, right_radius = _center_radius(right_lower, right_upper)
    product = left_center @ right_center
    center_abs_product = _nonnegative_matmul_upper(
        np.abs(left_center), np.abs(right_center)
    )
    roundoff = _up(_gamma(left_center.shape[1]) * center_abs_product)
    uncertainty = _nonnegative_matmul_upper(np.abs(left_center), right_radius)
    uncertainty = _nonnegative_add_upper(
        uncertainty,
        _nonnegative_matmul_upper(left_radius, np.abs(right_center)),
        _nonnegative_matmul_upper(left_radius, right_radius),
        roundoff,
    )
    return _down(product - uncertainty), _up(product + uncertainty)


def _arb_bounds(value: arb) -> tuple[float, float]:
    lower = float(value.lower())
    upper = float(value.upper())
    return float(np.nextafter(lower, -np.inf)), float(np.nextafter(upper, np.inf))


def _arb_positive_upper(value: arb) -> float:
    return float(np.nextafter(float(value.upper()), np.inf))


def _tanh_intervals(preactivation_terms: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray]:
    shape = preactivation_terms[0].shape
    lower = np.empty(shape, dtype=np.float64)
    upper = np.empty(shape, dtype=np.float64)
    old_precision = ctx.prec
    ctx.prec = PRECISION_BITS
    try:
        for index in np.ndindex(shape):
            total = arb(0)
            for term in preactivation_terms:
                total += arb(float(term[index]))
            lower[index], upper[index] = _arb_bounds(total.tanh())
    finally:
        ctx.prec = old_precision
    return lower, upper


def _analytic_constants() -> dict[str, float]:
    old_precision = ctx.prec
    ctx.prec = PRECISION_BITS
    try:
        sqrt2 = _arb_positive_upper(arb(2).sqrt())
        sqrt3 = _arb_positive_upper(arb(3).sqrt())
        d2 = _arb_positive_upper(arb(4) / (arb(3) * arb(3).sqrt()))
    finally:
        ctx.prec = old_precision
    return {"sqrt2": sqrt2, "sqrt3": sqrt3, "d2": d2, "d3": 2.0}


CONSTANTS = _analytic_constants()


def _network_intervals(parameter: np.ndarray, pairs: torch.Tensor, labels: torch.Tensor, config):
    p, h = config.modulus, config.width
    tensor = torch.from_numpy(np.asarray(parameter, dtype=np.float64))
    w_t, q_t, v_t, c_t = unpack(tensor, config)
    w = w_t.numpy()
    q = q_t.numpy()
    v = v_t.numpy()
    c = c_t.numpy()
    pair_array = pairs.numpy()
    first_term = w[:, pair_array[:, 0]].T
    second_term = w[:, p + pair_array[:, 1]].T
    q_term = np.broadcast_to(q, first_term.shape)
    hidden_lower, hidden_upper = _tanh_intervals((first_term, second_term, q_term))
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

    v_lower = v.T.copy()
    v_upper = v.T.copy()
    prediction_lower, prediction_upper = _interval_matmul(
        hidden_lower, hidden_upper, v_lower, v_upper
    )
    prediction_lower, prediction_upper = _interval_add(
        prediction_lower,
        prediction_upper,
        np.broadcast_to(c, prediction_lower.shape),
        np.broadcast_to(c, prediction_upper.shape),
    )
    target = np.eye(p, dtype=np.float64)[labels.numpy()]
    residual_lower, residual_upper = _interval_sub(
        prediction_lower, prediction_upper, target, target
    )
    return {
        "w": w,
        "q": q,
        "v": v,
        "c": c,
        "hidden_lower": hidden_lower,
        "hidden_upper": hidden_upper,
        "first_lower": first_lower,
        "first_upper": first_upper,
        "second_lower": second_lower,
        "second_upper": second_upper,
        "prediction_lower": prediction_lower,
        "prediction_upper": prediction_upper,
        "residual_lower": residual_lower,
        "residual_upper": residual_upper,
    }


def _gradient_interval(parameter: np.ndarray, pairs, labels, config, network=None):
    network = _network_intervals(parameter, pairs, labels, config) if network is None else network
    p, h = config.modulus, config.width
    n = len(pairs)
    output_lower, output_upper = _interval_div_positive(
        network["residual_lower"], network["residual_upper"], n * p
    )
    grad_v_lower, grad_v_upper = _interval_matmul(
        output_lower.T,
        output_upper.T,
        network["hidden_lower"],
        network["hidden_upper"],
    )
    grad_c_lower, grad_c_upper = _interval_sum(output_lower, output_upper, axis=0)
    v_exact = network["v"]
    back_lower, back_upper = _interval_matmul(
        output_lower,
        output_upper,
        v_exact,
        v_exact,
    )
    pre_lower, pre_upper = _interval_mul(
        back_lower,
        back_upper,
        network["first_lower"],
        network["first_upper"],
    )
    pair_array = pairs.numpy()
    grad_w_lower = np.zeros((h, 2 * p), dtype=np.float64)
    grad_w_upper = np.zeros_like(grad_w_lower)
    for value in range(p):
        mask_a = pair_array[:, 0] == value
        mask_b = pair_array[:, 1] == value
        grad_w_lower[:, value], grad_w_upper[:, value] = _interval_sum(
            pre_lower[mask_a], pre_upper[mask_a], axis=0
        )
        grad_w_lower[:, p + value], grad_w_upper[:, p + value] = _interval_sum(
            pre_lower[mask_b], pre_upper[mask_b], axis=0
        )
    grad_q_lower, grad_q_upper = _interval_sum(pre_lower, pre_upper, axis=0)
    lower = np.concatenate(
        (grad_w_lower.reshape(-1), grad_q_lower, grad_v_lower.reshape(-1), grad_c_lower)
    )
    upper = np.concatenate(
        (grad_w_upper.reshape(-1), grad_q_upper, grad_v_upper.reshape(-1), grad_c_upper)
    )
    if config.weight_decay:
        regularizer_lower, regularizer_upper = _interval_scale(
            parameter, parameter, config.weight_decay
        )
        lower, upper = _interval_add(lower, upper, regularizer_lower, regularizer_upper)
    return lower, upper


def _jacobian_interval_arrays(parameter, pairs, config, network):
    p, h = config.modulus, config.width
    n = len(pairs)
    total = len(parameter)
    pair_array = pairs.numpy()
    units = np.arange(h)
    samples = np.arange(n)
    outputs = np.arange(p)
    active = np.stack(
        (
            units[None, :] * (2 * p) + pair_array[:, 0, None],
            units[None, :] * (2 * p) + p + pair_array[:, 1, None],
            np.broadcast_to(h * (2 * p) + units, (n, h)),
        ),
        axis=2,
    )
    row_index = np.broadcast_to(
        samples[:, None, None] * p + outputs[None, :, None],
        (n, p, h),
    )
    v = network["v"]
    gain_lower, gain_upper = _interval_mul(
        network["first_lower"][:, None, :],
        network["first_upper"][:, None, :],
        v[None, :, :],
        v[None, :, :],
    )
    lower = np.zeros((n * p, total), dtype=np.float64)
    upper = np.zeros_like(lower)
    for slot in range(3):
        columns = np.broadcast_to(active[:, None, :, slot], (n, p, h))
        lower[row_index.reshape(-1), columns.reshape(-1)] = gain_lower.reshape(-1)
        upper[row_index.reshape(-1), columns.reshape(-1)] = gain_upper.reshape(-1)
    v_base = h * (2 * p) + h
    v_columns = np.broadcast_to(
        v_base + outputs[None, :, None] * h + units[None, None, :],
        (n, p, h),
    )
    lower[row_index.reshape(-1), v_columns.reshape(-1)] = np.broadcast_to(
        network["hidden_lower"][:, None, :], (n, p, h)
    ).reshape(-1)
    upper[row_index.reshape(-1), v_columns.reshape(-1)] = np.broadcast_to(
        network["hidden_upper"][:, None, :], (n, p, h)
    ).reshape(-1)
    c_base = v_base + p * h
    c_rows = np.broadcast_to(samples[:, None] * p + outputs[None, :], (n, p))
    c_columns = np.broadcast_to(c_base + outputs[None, :], (n, p))
    lower[c_rows.reshape(-1), c_columns.reshape(-1)] = 1.0
    upper[c_rows.reshape(-1), c_columns.reshape(-1)] = 1.0
    return lower, upper, active, row_index


def _symmetric_eigen_numeric_radius(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    matrix = np.asarray(matrix, dtype=np.float64)
    values, vectors = np.linalg.eigh(matrix)
    gram = vectors.T @ vectors
    gram_abs = _nonnegative_matmul_upper(np.abs(vectors.T), np.abs(vectors))
    gram_round = _up(_gamma(matrix.shape[0]) * gram_abs)
    gram_residual = _up(np.abs(gram - np.eye(len(values))))
    orthogonality = _max_row_sum_upper(
        _nonnegative_add_upper(gram_residual, gram_round)
    )
    if orthogonality >= 1.0:
        raise RuntimeError("eigenvector orthogonality enclosure is singular")

    scaled = vectors * values[None, :]
    reconstruction = scaled @ vectors.T
    product_abs = _nonnegative_matmul_upper(np.abs(scaled), np.abs(vectors.T))
    product_round = _up(_gamma(matrix.shape[0]) * product_abs)
    scaling_round = _up(EPS * np.abs(scaled))
    scaling_effect = _nonnegative_matmul_upper(scaling_round, np.abs(vectors.T))
    residual_round = _up(
        EPS * _nonnegative_add_upper(np.abs(matrix), np.abs(reconstruction))
    )
    reconstruction_error = _l2_upper(matrix - reconstruction)
    reconstruction_error = _upper_add(
        reconstruction_error,
        _max_row_sum_upper(
            _nonnegative_add_upper(product_round, scaling_effect, residual_round)
        ),
    )
    q_error = _upper_div(
        orthogonality,
        float(_down(1.0 - orthogonality)),
    )
    singular_upper_plus_one = _upper_add(
        _upper_sqrt(_upper_add(1.0, orthogonality)), 1.0
    )
    basis_change = _upper_mul(
        q_error,
        _upper_mul(
            float(_up(np.max(np.abs(values)))), singular_upper_plus_one
        ),
    )
    return values, _upper_add(reconstruction_error, basis_change)


def _hessian_evaluation_error(parameter, pairs, labels, config, hessian, gauss, network):
    p, h = config.modulus, config.width
    n = len(pairs)
    lower_a, upper_a, _, _ = _jacobian_interval_arrays(
        parameter, pairs, config, network
    )

    tensor = torch.from_numpy(parameter)
    w_t, q_t, v_t, c_t = unpack(tensor, config)
    pair_array_t = pairs
    pre = w_t[:, pair_array_t[:, 0]].T + w_t[:, p + pair_array_t[:, 1]].T + q_t
    hidden = torch.tanh(pre)
    first = 1.0 - hidden.square()
    second = -2.0 * hidden * first
    prediction = hidden @ v_t.T + c_t
    target = torch.nn.functional.one_hot(labels, num_classes=p).to(torch.float64)
    residual = prediction - target

    total = len(parameter)
    a0 = np.zeros((n * p, total), dtype=np.float64)
    units = np.arange(h)
    samples = np.arange(n)
    outputs = np.arange(p)
    pair_array = pairs.numpy()
    active = np.stack(
        (
            units[None, :] * (2 * p) + pair_array[:, 0, None],
            units[None, :] * (2 * p) + p + pair_array[:, 1, None],
            np.broadcast_to(h * (2 * p) + units, (n, h)),
        ),
        axis=2,
    )
    row_index = np.broadcast_to(
        samples[:, None, None] * p + outputs[None, :, None], (n, p, h)
    )
    gain0 = (first[:, None, :] * v_t[None, :, :]).numpy()
    for slot in range(3):
        columns = np.broadcast_to(active[:, None, :, slot], (n, p, h))
        a0[row_index.reshape(-1), columns.reshape(-1)] = gain0.reshape(-1)
    v_base = h * (2 * p) + h
    v_columns = np.broadcast_to(
        v_base + outputs[None, :, None] * h + units[None, None, :], (n, p, h)
    )
    a0[row_index.reshape(-1), v_columns.reshape(-1)] = np.broadcast_to(
        hidden.numpy()[:, None, :], (n, p, h)
    ).reshape(-1)
    c_base = v_base + p * h
    c_rows = np.broadcast_to(samples[:, None] * p + outputs[None, :], (n, p))
    c_columns = np.broadcast_to(c_base + outputs[None, :], (n, p))
    a0[c_rows.reshape(-1), c_columns.reshape(-1)] = 1.0

    delta_a = _up(
        np.maximum(
            _up(np.abs(lower_a - a0)),
            _up(np.abs(upper_a - a0)),
        )
    )
    delta_a_f = _l2_upper(delta_a)
    a0_f = _l2_upper(a0)
    gn_sensitivity = _upper_div(
        _upper_add(_upper_mul(2.0 * a0_f, delta_a_f), _upper_mul(delta_a_f, delta_a_f)),
        n * p,
    )
    abs_product = _nonnegative_matmul_upper(np.abs(a0.T), np.abs(a0))
    gn_product_round = _up(_gamma(n * p) * abs_product)
    gn_product_round = _up(gn_product_round / (n * p))
    gn_storage_round = _up(EPS * np.abs(gauss))
    gn_round_matrix = _nonnegative_add_upper(
        gn_product_round, gn_storage_round
    )
    gn_round = _max_row_sum_upper(gn_round_matrix)

    residual_v_lower, residual_v_upper = _interval_matmul(
        network["residual_lower"],
        network["residual_upper"],
        network["v"],
        network["v"],
    )
    aa_lower, aa_upper = _interval_mul(
        residual_v_lower,
        residual_v_upper,
        network["second_lower"],
        network["second_upper"],
    )
    aa_lower, aa_upper = _interval_div_positive(aa_lower, aa_upper, n * p)
    aa0 = (((residual @ v_t) * second) / (n * p)).numpy()
    aa_error = _up(
        np.maximum(
            _up(np.abs(aa_lower - aa0)),
            _up(np.abs(aa_upper - aa0)),
        )
    )

    cross_lower, cross_upper = _interval_mul(
        network["residual_lower"][:, :, None],
        network["residual_upper"][:, :, None],
        network["first_lower"][:, None, :],
        network["first_upper"][:, None, :],
    )
    cross_lower, cross_upper = _interval_div_positive(
        cross_lower, cross_upper, n * p
    )
    cross0 = (residual[:, :, None] * first[:, None, :] / (n * p)).numpy()
    cross_error = _up(
        np.maximum(
            _up(np.abs(cross_lower - cross0)),
            _up(np.abs(cross_upper - cross0)),
        )
    )
    coefficient_error = _upper_add(
        float(_sum_nonnegative_upper(_up(9.0 * aa_error))),
        float(_sum_nonnegative_upper(_up(6.0 * cross_error))),
    )
    contribution_abs = _upper_add(
        float(_sum_nonnegative_upper(_up(9.0 * np.abs(aa0)))),
        float(_sum_nonnegative_upper(_up(6.0 * np.abs(cross0)))),
    )
    scatter_round = _upper_mul(_gamma(3 * n + 3), contribution_abs)
    residual_part = hessian - gauss
    combine_magnitude = _nonnegative_add_upper(
        np.abs(gauss), np.abs(residual_part), np.abs(hessian)
    )
    combine_round = _max_row_sum_upper(_up(EPS * combine_magnitude))
    return _upper_add(
        _upper_add(gn_sensitivity, gn_round),
        _upper_add(coefficient_error, _upper_add(scatter_round, combine_round)),
    )


def _verified_beta(hessian: np.ndarray, evaluation_error: float, learning_rate: float):
    symmetrized = 0.5 * (hessian + hessian.T)
    symmetry_residual = _up(np.abs(hessian - symmetrized))
    symmetry_round = _up(
        EPS
        * _nonnegative_add_upper(
            np.abs(hessian), np.abs(hessian.T), np.abs(symmetrized)
        )
    )
    asymmetry = _max_row_sum_upper(
        _nonnegative_add_upper(symmetry_residual, symmetry_round)
    )
    values, eigen_numeric = _symmetric_eigen_numeric_radius(symmetrized)
    scaled_lower = _down(learning_rate * values)
    scaled_upper = _up(learning_rate * values)
    affine_lower = _down(1.0 - scaled_upper)
    affine_upper = _up(1.0 - scaled_lower)
    central = float(
        _up(np.max(np.maximum(np.abs(affine_lower), np.abs(affine_upper))))
    )
    total_radius = _upper_add(evaluation_error, _upper_add(asymmetry, eigen_numeric))
    beta = _upper_add(central, _upper_mul(abs(learning_rate), total_radius))
    return beta, {
        "hessian_evaluation_error": evaluation_error,
        "hessian_asymmetry_error": asymmetry,
        "eigen_numeric_error": eigen_numeric,
        "total_eigenvalue_radius": total_radius,
    }


def _verified_matrix_norm(parameter: np.ndarray, config) -> float:
    _, _, v_t, _ = unpack(torch.from_numpy(parameter), config)
    v = v_t.numpy()
    product = v @ v.T
    abs_product = _nonnegative_matmul_upper(np.abs(v), np.abs(v.T))
    product_error = _max_row_sum_upper(_up(_gamma(v.shape[1]) * abs_product))
    symmetrized = 0.5 * (product + product.T)
    symmetry_residual = _up(np.abs(product - symmetrized))
    symmetry_round = _up(
        EPS
        * _nonnegative_add_upper(
            np.abs(product), np.abs(product.T), np.abs(symmetrized)
        )
    )
    product_to_symmetric = _max_row_sum_upper(
        _nonnegative_add_upper(symmetry_residual, symmetry_round)
    )
    values, numeric = _symmetric_eigen_numeric_radius(symmetrized)
    upper_eigenvalue = _upper_add(
        float(_up(np.max(values))),
        _upper_add(product_error, _upper_add(product_to_symmetric, numeric)),
    )
    return _upper_sqrt(max(0.0, upper_eigenvalue))


def _objective_hessian_lipschitz_upper(parameter: np.ndarray, config, radius: float) -> float:
    _, _, _, c_t = unpack(torch.from_numpy(parameter), config)
    v_norm = _upper_add(_verified_matrix_norm(parameter, config), radius)
    c_norm = _upper_add(_l2_upper(c_t.numpy()), radius)
    x_norm = CONSTANTS["sqrt3"]
    hidden_norm = _upper_sqrt(float(config.width))
    b0 = _upper_add(_upper_add(_upper_mul(v_norm, hidden_norm), c_norm), 1.0)
    vx = _upper_mul(v_norm, x_norm)
    b1 = _upper_sqrt(
        _upper_add(_upper_add(_upper_mul(vx, vx), float(config.width)), 1.0)
    )
    aa = _upper_mul(
        _upper_mul(v_norm, CONSTANTS["d2"]), _upper_mul(x_norm, x_norm)
    )
    ab = x_norm
    matrix = np.asarray([[aa, ab], [ab, 0.0]], dtype=np.float64)
    values, numeric = _symmetric_eigen_numeric_radius(matrix)
    b2 = _upper_add(float(_up(np.max(np.abs(values)))), numeric)
    x2 = _upper_mul(x_norm, x_norm)
    x3 = _upper_mul(x2, x_norm)
    b3 = _upper_add(
        _upper_mul(3.0 * CONSTANTS["d2"], x2),
        _upper_mul(_upper_mul(v_norm, CONSTANTS["d3"]), x3),
    )
    numerator = _upper_add(
        _upper_mul(3.0, _upper_mul(b1, b2)),
        _upper_mul(b0, b3),
    )
    return _upper_div(numerator, config.modulus)


def _margin_b1_upper(parameter: np.ndarray, config, label: int, competitor: int, radius: float):
    _, _, v_t, _ = unpack(torch.from_numpy(parameter), config)
    difference_lower, difference_upper = _interval_sub(
        v_t[label].numpy(),
        v_t[label].numpy(),
        v_t[competitor].numpy(),
        v_t[competitor].numpy(),
    )
    row_norm = _l2_upper(np.maximum(np.abs(difference_lower), np.abs(difference_upper)))
    row_norm = _upper_add(row_norm, _upper_mul(CONSTANTS["sqrt2"], radius))
    x_product = _upper_mul(row_norm, CONSTANTS["sqrt3"])
    return _upper_sqrt(
        _upper_add(
            _upper_mul(x_product, x_product),
            2.0 * config.width + 2.0,
        )
    )


def _defect_upper(parameter, next_parameter, pairs, labels, config, network):
    gradient_lower, gradient_upper = _gradient_interval(
        parameter, pairs, labels, config, network
    )
    scaled_lower, scaled_upper = _interval_scale(
        gradient_lower, gradient_upper, config.learning_rate
    )
    mapped_lower, mapped_upper = _interval_sub(
        parameter, parameter, scaled_lower, scaled_upper
    )
    defect_lower, defect_upper = _interval_sub(
        mapped_lower, mapped_upper, next_parameter, next_parameter
    )
    maximum = np.maximum(np.abs(defect_lower), np.abs(defect_upper))
    return _l2_upper(maximum)


@torch.no_grad()
def _verified_geometry(parameter, next_parameter, pairs, labels, config):
    network = _network_intervals(parameter, pairs, labels, config)
    hessian_t, gauss_t = exact_objective_hessian(
        torch.from_numpy(parameter), pairs, labels, config
    )
    hessian = hessian_t.numpy()
    gauss = gauss_t.numpy()
    evaluation_error = _hessian_evaluation_error(
        parameter, pairs, labels, config, hessian, gauss, network
    )
    beta, diagnostics = _verified_beta(
        hessian, evaluation_error, config.learning_rate
    )
    defect = _defect_upper(
        parameter, next_parameter, pairs, labels, config, network
    )
    return beta, defect, diagnostics


def _certified_counts(reference: np.ndarray, radius: np.ndarray, test_pairs, test_labels, config):
    sample_count = len(test_pairs)
    label_array = test_labels.numpy()
    guaranteed = np.zeros(len(reference), dtype=np.int64)
    possible = np.zeros(len(reference), dtype=np.int64)
    for step, center in enumerate(reference):
        network = _network_intervals(center, test_pairs, test_labels, config)
        lower_logits = network["prediction_lower"]
        upper_logits = network["prediction_upper"]
        guaranteed_examples = 0
        excluded_examples = 0
        for sample in range(sample_count):
            label = int(label_array[sample])
            all_positive = True
            one_negative = False
            for competitor in range(config.modulus):
                if competitor == label:
                    continue
                margin_lower = float(
                    _down(lower_logits[sample, label] - upper_logits[sample, competitor])
                )
                margin_upper = float(
                    _up(upper_logits[sample, label] - lower_logits[sample, competitor])
                )
                b1 = _margin_b1_upper(
                    center, config, label, competitor, float(radius[step])
                )
                error = _upper_mul(b1, float(radius[step]))
                lower_with_error = float(_down(margin_lower - error))
                upper_with_error = float(_up(margin_upper + error))
                if lower_with_error <= 0.0:
                    all_positive = False
                if upper_with_error < 0.0:
                    one_negative = True
            guaranteed_examples += int(all_positive)
            excluded_examples += int(one_negative)
        guaranteed[step] = guaranteed_examples
        possible[step] = sample_count - excluded_examples
    return guaranteed, possible


def interval_cache_path(seed: int, anchor: int) -> Path:
    return OUT_DIR / f"seed_{seed}_anchor_{anchor}_h{HORIZON}.json"


@torch.no_grad()
def verified_tube(
    reference: np.ndarray,
    train_pairs: torch.Tensor,
    train_labels: torch.Tensor,
    config,
    *,
    progress_prefix: str | None = None,
) -> tuple[np.ndarray, int, list[dict]]:
    """Propagate the outward state recursion around an explicit reference."""
    radius = np.zeros(len(reference), dtype=np.float64)
    reached = 0
    diagnostic_rows: list[dict] = []
    for step in range(len(reference) - 1):
        current = float(radius[step])
        if not np.isfinite(current) or current > NUMERIC_CAP:
            break
        if progress_prefix is not None:
            print(
                f"  {progress_prefix}: state {step + 1}/{len(reference) - 1}",
                flush=True,
            )
        beta, defect, diagnostics = _verified_geometry(
            reference[step],
            reference[step + 1],
            train_pairs,
            train_labels,
            config,
        )
        lipschitz = _upper_mul(
            config.learning_rate,
            _objective_hessian_lipschitz_upper(reference[step], config, current),
        )
        square = _upper_mul(current, current)
        nonlinear = _upper_mul(0.5, _upper_mul(lipschitz, square))
        next_radius = _upper_add(
            _upper_add(_upper_mul(beta, current), defect), nonlinear
        )
        diagnostic_rows.append(
            {
                "step": step,
                "beta_upper": beta,
                "defect_norm_upper": defect,
                "lipschitz_upper": lipschitz,
                "next_radius": next_radius,
                **diagnostics,
            }
        )
        if not np.isfinite(next_radius) or next_radius > NUMERIC_CAP:
            break
        radius[step + 1] = next_radius
        reached = step + 1
    return radius[: reached + 1], reached, diagnostic_rows


@torch.no_grad()
def verify_anchor(seed: int, anchor: int, *, use_cache: bool = True) -> dict:
    verify_manifest()
    output = interval_cache_path(seed, anchor)
    if use_cache and output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if (
            payload.get("interval_version") == INTERVAL_VERSION
            and payload.get("protocol_sha256") == protocol_sha256()
            and payload.get("manifest_sha256") == manifest_sha256()
        ):
            return payload
    certificate_path, array_path = cache_paths(seed, anchor)
    candidate = json.loads(certificate_path.read_text(encoding="utf-8"))
    if candidate["method_version"] != METHOD_VERSION:
        raise RuntimeError("float candidate method version changed")
    if candidate["manifest_sha256"] != manifest_sha256():
        raise RuntimeError("float candidate implementation changed")
    arrays = np.load(array_path)
    candidate_uppers = [
        int(event["v2_bracket"][1])
        for event in candidate["events"].values()
        if event["v2_bracket"] is not None
    ]
    if not candidate_uppers:
        raise ValueError("anchor has no float candidate bracket to verify")
    candidate_horizon = max(candidate_uppers)
    reference = arrays["corrected_reference"][: candidate_horizon + 1]
    config = frozen_config(seed)
    train_pairs, train_labels, test_pairs, test_labels = make_split(config)
    torch.set_num_threads(4)

    radius, reached, diagnostic_rows = verified_tube(
        reference,
        train_pairs,
        train_labels,
        config,
        progress_prefix=f"interval seed {seed} anchor {anchor}",
    )
    keep = len(radius)
    guaranteed, possible = _certified_counts(
        reference[:keep], radius, test_pairs, test_labels, config
    )
    required = required_counts(len(test_pairs))
    events = {}
    for threshold in THRESHOLDS:
        key = f"{threshold:.2f}"
        events[key] = {
            "float_candidate_bracket": candidate["events"][key]["v2_bracket"],
            "outward_bracket": event_bracket(
                guaranteed, possible, required[threshold]
            ),
        }
    payload = {
        "status": "outward-enclosed exact-real map verification",
        "scope_note": (
            "Reference points are exact binary floats; observed-training roundoff is audited, "
            "not enclosed by this exact-real map tube."
        ),
        "protocol_sha256": protocol_sha256(),
        "manifest_sha256": manifest_sha256(),
        "pipeline_version": PIPELINE_VERSION,
        "method_version": METHOD_VERSION,
        "interval_version": INTERVAL_VERSION,
        "python_flint_version": flint.__version__,
        "arb_precision_bits": PRECISION_BITS,
        "seed": seed,
        "anchor": anchor,
        "requested_horizon": candidate_horizon,
        "reached_horizon": reached,
        "events": events,
        "maximum_radius": float(np.max(radius)),
        "maximum_hessian_evaluation_error": max(
            (row["hessian_evaluation_error"] for row in diagnostic_rows), default=0.0
        ),
        "maximum_eigen_numeric_error": max(
            (row["eigen_numeric_error"] for row in diagnostic_rows), default=0.0
        ),
        "diagnostic_rows": diagnostic_rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(output.with_suffix(".npz"), radius=radius)
    return payload


def verify_all_issued() -> dict:
    """Outward-check all blind float candidates without reading outcomes."""
    verify_manifest()
    scan = json.loads(SCAN_OUT.read_text(encoding="utf-8"))
    if scan["protocol_sha256"] != protocol_sha256():
        raise RuntimeError("protocol changed after prospective scan")
    if scan["manifest_sha256"] != manifest_sha256():
        raise RuntimeError("implementation changed after prospective scan")
    candidate_rows = []
    for seed_row in scan["seed_rows"]:
        seed = int(seed_row["seed"])
        for threshold in THRESHOLDS:
            key = f"{threshold:.2f}"
            anchor = seed_row["triggers"][key]
            if anchor is None:
                continue
            certificate_path, _ = cache_paths(seed, int(anchor))
            candidate = json.loads(certificate_path.read_text(encoding="utf-8"))
            if candidate["protocol_sha256"] != protocol_sha256():
                raise RuntimeError("blind candidate protocol changed")
            if candidate["manifest_sha256"] != manifest_sha256():
                raise RuntimeError("blind candidate implementation changed")
            bracket = candidate["events"][key]["v2_bracket"]
            if bracket is not None:
                candidate_rows.append(
                    {
                        "seed": seed,
                        "threshold": threshold,
                        "anchor": int(anchor),
                        "float_candidate_bracket": bracket,
                    }
                )
    anchors = sorted({(row["seed"], row["anchor"]) for row in candidate_rows})
    caches = {}
    for index, (seed, anchor) in enumerate(anchors, start=1):
        print(f"outward verification {index}/{len(anchors)}: seed {seed}, anchor {anchor}", flush=True)
        caches[(seed, anchor)] = verify_anchor(seed, anchor)
    rows = []
    for candidate in candidate_rows:
        seed = candidate["seed"]
        anchor = candidate["anchor"]
        key = f"{float(candidate['threshold']):.2f}"
        interval_event = caches[(seed, anchor)]["events"][key]
        bracket = interval_event["outward_bracket"]
        rows.append(
            {
                "seed": seed,
                "threshold": candidate["threshold"],
                "anchor": anchor,
                "float_candidate_bracket": candidate["float_candidate_bracket"],
                "outward_bracket": bracket,
                "outward_issued": bracket is not None,
            }
        )
    issued = [row for row in rows if row["outward_issued"]]
    summary = {
        "float_candidates": len(rows),
        "outward_certificates_issued": len(issued),
        "all_float_candidates_retained": len(issued) == len(rows),
        "distinct_outward_issuing_seeds": sorted({row["seed"] for row in issued}),
    }
    payload = {
        "status": "blind outward-enclosed candidate verification; no outcomes read",
        "scope_note": (
            "The state and output inequalities enclose exact-real gradient descent from the "
            "binary anchor. The observed float64 optimizer path is a separate outcome audit."
        ),
        "protocol_sha256": protocol_sha256(),
        "manifest_sha256": manifest_sha256(),
        "interval_version": INTERVAL_VERSION,
        "summary": summary,
        "rows": rows,
    }
    BLIND_SUMMARY.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return payload


def audit_outcomes() -> dict:
    """Join sealed first-passage outcomes only after interval verification."""
    verify_manifest()
    blind = json.loads(BLIND_SUMMARY.read_text(encoding="utf-8"))
    result = json.loads(PROSPECTIVE_RESULT.read_text(encoding="utf-8"))
    for payload, label in ((blind, "blind interval result"), (result, "outcome result")):
        if payload["protocol_sha256"] != protocol_sha256():
            raise RuntimeError(f"protocol changed after {label}")
        if payload["manifest_sha256"] != manifest_sha256():
            raise RuntimeError(f"implementation changed after {label}")
    outcome_rows = {
        (int(row["seed"]), f"{float(row['threshold']):.2f}", int(row["anchor"])): row
        for row in result["rows"]
        if row["anchor"] is not None and row["v2_certificate_issued"]
    }
    rows = []
    for row in blind["rows"]:
        key = (int(row["seed"]), f"{float(row['threshold']):.2f}", int(row["anchor"]))
        observed = outcome_rows.pop(key)
        if observed["v2_bracket"] != row["float_candidate_bracket"]:
            raise RuntimeError("float candidate changed before outcome join")
        bracket = row["outward_bracket"]
        lead = observed["actual_lead"]
        rows.append(
            {
                **row,
                "actual_lead": lead,
                "outward_covered": (
                    None
                    if bracket is None or lead is None
                    else bool(bracket[0] <= lead <= bracket[1])
                ),
            }
        )
    if outcome_rows:
        raise RuntimeError("outcome aggregation contains an unmatched float candidate")
    issued = [row for row in rows if row["outward_issued"]]
    covered = sum(row["outward_covered"] is True for row in issued)
    issuing_seeds = sorted({row["seed"] for row in issued})
    all_covered_seeds = [
        seed
        for seed in issuing_seeds
        if all(row["outward_covered"] is True for row in issued if row["seed"] == seed)
    ]
    spans = [
        int(row["outward_bracket"][1] - row["outward_bracket"][0])
        for row in issued
    ]
    leads = [
        int(row["actual_lead"])
        for row in issued
        if row["actual_lead"] is not None
    ]
    summary = {
        **blind["summary"],
        "outward_coverage_count": [covered, len(issued)],
        "outward_false_issued": len(issued) - covered,
        "outward_coverage_exact_95_interval_iid_event_working_model": clopper_pearson(
            covered, len(issued)
        ),
        "outward_all_covered_issuing_seeds": all_covered_seeds,
        "outward_seed_level_all_covered_count": [
            len(all_covered_seeds),
            len(issuing_seeds),
        ],
        "outward_seed_level_exact_95_interval_iid_seed_working_model": clopper_pearson(
            len(all_covered_seeds), len(issuing_seeds)
        ),
        "outward_median_bracket_span": None if not spans else float(np.median(spans)),
        "outward_maximum_bracket_span": None if not spans else int(max(spans)),
        "outward_median_lead": None if not leads else float(np.median(leads)),
        "outward_minimum_lead": None if not leads else int(min(leads)),
        "outward_maximum_lead": None if not leads else int(max(leads)),
    }
    payload = {
        **blind,
        "status": "outward-enclosed candidates joined to sealed outcomes",
        "summary": summary,
        "rows": rows,
    }
    SUMMARY.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--all-issued", action="store_true")
    action.add_argument("--audit-outcomes", action="store_true")
    action.add_argument("--anchor", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--step", type=int)
    args = parser.parse_args()
    if args.all_issued:
        verify_all_issued()
    elif args.audit_outcomes:
        audit_outcomes()
    else:
        if args.seed is None or args.step is None:
            parser.error("--anchor requires --seed and --step")
        print(json.dumps(verify_anchor(args.seed, args.step, use_cache=False), indent=2))


if __name__ == "__main__":
    main()
