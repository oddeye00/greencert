#!/usr/bin/env python3
"""Frozen moving-window event certificates for the smooth tanh MLP.

This module implements the transfer protocol in
``SECOND_ARCHITECTURE_PROTOCOL.md``.  It uses an exact analytic Hessian at each
anchor, literal frozen-Hessian modal dynamics, and global tanh derivative
bounds.  No observed future parameter error enters a certificate.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from generate_smooth_mlp_seed import artifact_paths, frozen_config
from modular_accuracy_certificate import event_bracket
from replay_smooth_mlp_thresholds import THRESHOLDS, crossing_output, required_counts
from smooth_mlp_modular_grokking import (
    Config,
    analytic_gradient,
    logits,
    make_split,
    objective,
    unpack,
)


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "smooth_mlp_certificate_cache"
LEGACY_PRIMARY_OUT = ROOT / "results" / "smooth_mlp_certificate_audit.json"
HORIZON = 5_000
RAW_STRIDE = 25
CERTIFICATE_BUFFER = 250
TUBE_NUMERIC_CAP = 1e4
TANH_D1 = 1.0
TANH_D2 = 4.0 / (3.0 * np.sqrt(3.0))
TANH_D3 = 2.0


def native_indices(config: Config, hidden: int, pair: torch.Tensor) -> tuple[list[int], int]:
    """Native parameter indices for the three active affine inputs."""
    p, h = config.modulus, config.width
    a, b = int(pair[0]), int(pair[1])
    w_a = hidden * (2 * p) + a
    w_b = hidden * (2 * p) + p + b
    q = h * (2 * p) + hidden
    return [w_a, w_b, q], h * (2 * p) + h


@torch.no_grad()
def exact_objective_hessian(
    parameter: torch.Tensor,
    pairs: torch.Tensor,
    labels: torch.Tensor,
    config: Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact Hessian and Gauss--Newton part of half-mean-squared error plus L2."""
    p, h = config.modulus, config.width
    w, q, v, c = unpack(parameter, config)
    preactivation = w[:, pairs[:, 0]].T + w[:, p + pairs[:, 1]].T + q
    hidden = torch.tanh(preactivation)
    first = 1.0 - hidden.square()
    second = -2.0 * hidden * first
    prediction = hidden @ v.T + c
    target = F.one_hot(labels, num_classes=p).to(parameter.dtype)
    residual = prediction - target
    n = len(pairs)
    parameter_total = parameter.numel()
    v_base = h * (2 * p) + h
    c_base = v_base + p * h

    # Each hidden preactivation depends on exactly three native parameters:
    # W[r,a], W[r,p+b], and q[r].  Build all Jacobian and residual-Hessian
    # blocks with flattened scatter-adds.  This is algebraically identical to
    # the scalar construction but makes path-Hessian evaluation practical.
    units = torch.arange(h, dtype=torch.long)
    samples = torch.arange(n, dtype=torch.long)
    outputs = torch.arange(p, dtype=torch.long)
    active = torch.stack(
        (
            units[None, :] * (2 * p) + pairs[:, 0, None],
            units[None, :] * (2 * p) + p + pairs[:, 1, None],
            (h * (2 * p) + units)[None, :].expand(n, h),
        ),
        dim=2,
    )
    row_index = (samples[:, None, None] * p + outputs[None, :, None]).expand(n, p, h)
    gain = first[:, None, :] * v[None, :, :]
    flat_jacobian = torch.zeros((n * p, parameter_total), dtype=parameter.dtype)
    flat_jacobian_storage = flat_jacobian.reshape(-1)
    for slot in range(3):
        columns = active[:, None, :, slot].expand(n, p, h)
        flat_jacobian_storage.index_add_(
            0,
            (row_index * parameter_total + columns).reshape(-1),
            gain.reshape(-1),
        )
    v_columns = (
        v_base + outputs[None, :, None] * h + units[None, None, :]
    ).expand(n, p, h)
    flat_jacobian_storage.index_add_(
        0,
        (row_index * parameter_total + v_columns).reshape(-1),
        hidden[:, None, :].expand(n, p, h).reshape(-1),
    )
    c_rows = (samples[:, None] * p + outputs[None, :]).expand(n, p)
    c_columns = (c_base + outputs[None, :]).expand(n, p)
    flat_jacobian_storage.index_add_(
        0,
        (c_rows * parameter_total + c_columns).reshape(-1),
        torch.ones(n * p, dtype=parameter.dtype),
    )
    gauss_newton = flat_jacobian.T @ flat_jacobian / (n * p)
    residual_hessian = torch.zeros_like(gauss_newton)
    residual_storage = residual_hessian.reshape(-1)
    aa_coefficient = ((residual @ v) * second / (n * p)).reshape(-1)
    for left_slot in range(3):
        left = active[:, :, left_slot]
        for right_slot in range(3):
            right = active[:, :, right_slot]
            residual_storage.index_add_(
                0,
                (left * parameter_total + right).reshape(-1),
                aa_coefficient,
            )
    cross_coefficient = residual[:, :, None] * first[:, None, :] / (n * p)
    v_index = v_columns
    for slot in range(3):
        affine_index = active[:, None, :, slot].expand(n, p, h)
        residual_storage.index_add_(
            0,
            (affine_index * parameter_total + v_index).reshape(-1),
            cross_coefficient.reshape(-1),
        )
        residual_storage.index_add_(
            0,
            (v_index * parameter_total + affine_index).reshape(-1),
            cross_coefficient.reshape(-1),
        )

    hessian = gauss_newton + residual_hessian
    if config.weight_decay:
        hessian.diagonal().add_(config.weight_decay)
    return hessian, gauss_newton


@torch.no_grad()
def output_taylor_paths(
    parameter: torch.Tensor,
    displacement: torch.Tensor,
    pairs: torch.Tensor,
    config: Config,
) -> dict[str, torch.Tensor]:
    """Jacobian-only, second-order, and exact centerline logits."""
    p = config.modulus
    w, q, v, c = unpack(parameter, config)
    preactivation = w[:, pairs[:, 0]].T + w[:, p + pairs[:, 1]].T + q
    hidden = torch.tanh(preactivation)
    first = 1.0 - hidden.square()
    second = -2.0 * hidden * first
    anchor_logits = hidden @ v.T + c
    linear_rows: list[torch.Tensor] = []
    quadratic_rows: list[torch.Tensor] = []
    exact_rows: list[torch.Tensor] = []
    for delta in displacement:
        dw, dq, dv, dc = unpack(delta, config)
        delta_activation = dw[:, pairs[:, 0]].T + dw[:, p + pairs[:, 1]].T + dq
        first_hidden = first * delta_activation
        linear_change = first_hidden @ v.T + hidden @ dv.T + dc
        second_directional = (second * delta_activation.square()) @ v.T + 2.0 * first_hidden @ dv.T
        linear_rows.append(anchor_logits + linear_change)
        quadratic_rows.append(anchor_logits + linear_change + 0.5 * second_directional)
        exact_rows.append(logits(parameter + delta, pairs, config))
    return {
        "linear": torch.stack(linear_rows),
        "quadratic": torch.stack(quadratic_rows),
        "exact": torch.stack(exact_rows),
    }


def _symmetric_two_group_norm(aa: float, ab: float) -> float:
    matrix = np.asarray([[aa, ab], [ab, 0.0]], dtype=np.float64)
    return float(np.linalg.norm(matrix, ord=2))


@torch.no_grad()
def objective_hessian_lipschitz(
    parameter: torch.Tensor,
    config: Config,
    radius: float,
) -> float:
    """Global upper bound on the objective Hessian Lipschitz constant in a ball."""
    _, _, v, c = unpack(parameter, config)
    x_norm = np.sqrt(3.0)
    hidden_norm = np.sqrt(config.width)
    v_norm = float(torch.linalg.matrix_norm(v, ord=2)) + radius
    c_norm = float(torch.linalg.vector_norm(c)) + radius
    b0 = v_norm * hidden_norm + c_norm + 1.0
    b1 = np.sqrt((v_norm * x_norm) ** 2 + hidden_norm**2 + 1.0)
    b2 = _symmetric_two_group_norm(v_norm * TANH_D2 * x_norm**2, x_norm)
    b3 = 3.0 * TANH_D2 * x_norm**2 + v_norm * TANH_D3 * x_norm**3
    return float((3.0 * b1 * b2 + b0 * b3) / config.modulus)


@torch.no_grad()
def margin_derivative_bounds(
    parameter: torch.Tensor,
    config: Config,
    label: int,
    competitor: int,
    radius: float,
) -> tuple[float, float, float]:
    """Bounds on the first three derivatives of one logit margin in a ball."""
    _, _, v, _ = unpack(parameter, config)
    x_norm = np.sqrt(3.0)
    hidden_norm = np.sqrt(config.width)
    row_difference = float(torch.linalg.vector_norm(v[label] - v[competitor])) + np.sqrt(2.0) * radius
    b1 = np.sqrt((row_difference * x_norm) ** 2 + 2.0 * hidden_norm**2 + 2.0)
    b2 = _symmetric_two_group_norm(
        row_difference * TANH_D2 * x_norm**2,
        np.sqrt(2.0) * x_norm,
    )
    b3 = 3.0 * np.sqrt(2.0) * TANH_D2 * x_norm**2 + row_difference * TANH_D3 * x_norm**3
    return float(b1), float(b2), float(b3)


def modal_path(
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    gradient: torch.Tensor,
    learning_rate: float,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    factors = 1.0 - learning_rate * eigenvalues
    modal_gradient = eigenvectors.T @ gradient
    modal = torch.zeros_like(eigenvalues)
    rows = [modal.clone()]
    for _ in range(horizon):
        modal = factors * modal - learning_rate * modal_gradient
        rows.append(modal.clone())
        if not bool(torch.all(torch.isfinite(modal))):
            break
    return torch.stack(rows), factors


def first_count_crossing(counts: np.ndarray, required: int, steps: np.ndarray) -> int | None:
    indices = np.flatnonzero(counts >= required)
    return None if len(indices) == 0 else int(steps[indices[0]])


@torch.no_grad()
def raw_forecasts(
    parameter: torch.Tensor,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    gradient: torch.Tensor,
    test_pairs: torch.Tensor,
    test_labels: torch.Tensor,
    config: Config,
) -> tuple[dict[str, dict[str, int | None]], torch.Tensor]:
    modal, factors = modal_path(
        eigenvalues, eigenvectors, gradient, config.learning_rate, HORIZON
    )
    available_horizon = len(modal) - 1
    coarse_steps = torch.arange(0, available_horizon + 1, RAW_STRIDE, dtype=torch.long)
    if int(coarse_steps[-1]) != available_horizon:
        coarse_steps = torch.cat((coarse_steps, torch.tensor([available_horizon])))
    coarse_displacement = modal[coarse_steps] @ eigenvectors.T
    coarse_paths = output_taylor_paths(parameter, coarse_displacement, test_pairs, config)
    required = required_counts(len(test_pairs))
    result: dict[str, dict[str, int | None]] = {}
    for threshold in THRESHOLDS:
        key = f"{threshold:.2f}"
        result[key] = {}
        for name, path_key in (("full_quadratic", "quadratic"), ("jacobian_only", "linear")):
            count = (
                torch.argmax(coarse_paths[path_key], dim=2) == test_labels[None, :]
            ).sum(dim=1).numpy()
            coarse_crossing = first_count_crossing(count, required[threshold], coarse_steps.numpy())
            if coarse_crossing is None or coarse_crossing == 0:
                result[key][name] = coarse_crossing
                continue
            start = max(0, coarse_crossing - RAW_STRIDE)
            fine_steps = torch.arange(start, coarse_crossing + 1, dtype=torch.long)
            fine_displacement = modal[fine_steps] @ eigenvectors.T
            fine_path = output_taylor_paths(parameter, fine_displacement, test_pairs, config)[path_key]
            fine_count = (torch.argmax(fine_path, dim=2) == test_labels[None, :]).sum(dim=1).numpy()
            result[key][name] = first_count_crossing(
                fine_count, required[threshold], fine_steps.numpy()
            )
    return result, factors


@torch.no_grad()
def exact_path_tube_until(
    parameter: torch.Tensor,
    hessian: torch.Tensor,
    gradient: torch.Tensor,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    train_pairs: torch.Tensor,
    train_labels: torch.Tensor,
    config: Config,
    requested_horizon: int,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    modal, factors = modal_path(
        eigenvalues, eigenvectors, gradient, config.learning_rate, requested_horizon
    )
    displacement = modal @ eigenvectors.T
    amplification = float(torch.max(torch.abs(factors)))
    epsilon: list[float] = [0.0]
    defects: list[float] = []
    injections: list[float] = []
    for step in range(min(requested_horizon, len(displacement) - 1)):
        current_epsilon = epsilon[-1]
        if not np.isfinite(current_epsilon) or current_epsilon > TUBE_NUMERIC_CAP:
            break
        moved = parameter + displacement[step]
        exact_gradient = analytic_gradient(moved, train_pairs, train_labels, config)
        linear_gradient = gradient + hessian @ displacement[step]
        path_defect = float(torch.linalg.vector_norm(exact_gradient - linear_gradient))
        path_norm = float(torch.linalg.vector_norm(displacement[step]))
        lipschitz = objective_hessian_lipschitz(
            parameter, config, path_norm + current_epsilon
        )
        injection = path_defect + lipschitz * (
            path_norm + 0.5 * current_epsilon
        ) * current_epsilon
        next_epsilon = amplification * current_epsilon + config.learning_rate * injection
        defects.append(path_defect)
        injections.append(injection)
        if not np.isfinite(next_epsilon) or next_epsilon > TUBE_NUMERIC_CAP:
            break
        epsilon.append(next_epsilon)
    reached = len(epsilon)
    return (
        displacement[:reached],
        np.asarray(epsilon, dtype=np.float64),
        np.asarray(defects[: max(0, reached - 1)], dtype=np.float64),
        np.asarray(injections[: max(0, reached - 1)], dtype=np.float64),
    )


@torch.no_grad()
def certified_counts(
    parameter: torch.Tensor,
    displacement: torch.Tensor,
    epsilon: np.ndarray,
    test_pairs: torch.Tensor,
    test_labels: torch.Tensor,
    config: Config,
) -> dict[str, dict[str, np.ndarray]]:
    paths = output_taylor_paths(parameter, displacement, test_pairs, config)
    path_norm = torch.linalg.vector_norm(displacement, dim=1).numpy()
    horizon, sample_count, class_count = paths["linear"].shape
    result: dict[str, dict[str, np.ndarray]] = {}
    for name, path_key in (("full_quadratic", "quadratic"), ("jacobian_only", "linear")):
        values = paths[path_key]
        true_values = torch.gather(
            values,
            2,
            test_labels[None, :, None].expand(horizon, sample_count, 1),
        )
        margins = (true_values - values).numpy()
        errors = np.zeros_like(margins)
        for step in range(horizon):
            for sample in range(sample_count):
                label = int(test_labels[sample])
                for competitor in range(class_count):
                    if competitor == label:
                        continue
                    state_b1, _, _ = margin_derivative_bounds(
                        parameter,
                        config,
                        label,
                        competitor,
                        float(path_norm[step] + epsilon[step]),
                    )
                    _, b2, b3 = margin_derivative_bounds(
                        parameter,
                        config,
                        label,
                        competitor,
                        float(path_norm[step]),
                    )
                    taylor = (
                        b3 * path_norm[step] ** 3 / 6.0
                        if name == "full_quadratic"
                        else b2 * path_norm[step] ** 2 / 2.0
                    )
                    errors[step, sample, competitor] = taylor + state_b1 * epsilon[step]
        lower = margins - errors
        upper = margins + errors
        rows = np.arange(sample_count)
        lower[:, rows, test_labels.numpy()] = np.inf
        upper[:, rows, test_labels.numpy()] = np.inf
        guaranteed = np.all(lower > 0.0, axis=2).sum(axis=1)
        possible = sample_count - np.any(upper < 0.0, axis=2).sum(axis=1)
        result[name] = {
            "guaranteed": guaranteed,
            "possible": possible,
            "maximum_margin_error": np.max(errors, axis=(1, 2)),
        }
    return result


@torch.no_grad()
def audit_state_path(
    parameter: torch.Tensor,
    displacement: torch.Tensor,
    epsilon: np.ndarray,
    train_pairs: torch.Tensor,
    train_labels: torch.Tensor,
    config: Config,
    horizon: int,
) -> dict:
    actual = parameter.clone()
    violations = 0
    maximum_ratio = 0.0
    for step in range(horizon + 1):
        observed = float(torch.linalg.vector_norm((actual - parameter) - displacement[step]))
        bound = float(epsilon[step])
        if observed > bound * (1.0 + 1e-8) + 1e-10:
            violations += 1
        if bound > 1e-14:
            maximum_ratio = max(maximum_ratio, observed / bound)
        if step < horizon:
            actual.add_(
                analytic_gradient(actual, train_pairs, train_labels, config),
                alpha=-config.learning_rate,
            )
    return {
        "checked_steps": horizon + 1,
        "violations": violations,
        "maximum_error_to_bound_ratio": maximum_ratio,
    }


@torch.no_grad()
def selected_anchors(seed: int) -> list[int]:
    """Frozen coarse plus past-only record-improvement checkpoint policy."""
    config = frozen_config(seed)
    result_path, checkpoint_path = artifact_paths(seed)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoints = np.load(checkpoint_path)
    _, _, test_pairs, test_labels = make_split(config)
    required_95 = required_counts(len(test_pairs))[0.95]
    anchors = set(range(5_000, config.steps + 1, 5_000))
    record = -1
    for step in payload["checkpoint_steps"]:
        parameter = torch.from_numpy(checkpoints[f"step_{step}"])
        correct = int(torch.sum(torch.argmax(logits(parameter, test_pairs, config), dim=1) == test_labels))
        if correct > record:
            if step > 0 and correct < required_95:
                anchors.add(int(step))
            record = correct
    return sorted(anchors)


def cache_path(seed: int, anchor: int) -> Path:
    return CACHE / f"seed_{seed}_anchor_{anchor}.json"


def audit_output(seed: int) -> Path:
    return ROOT / "results" / f"smooth_mlp_certificate_seed_{seed}.json"


def audit_figure(seed: int) -> Path:
    return ROOT / "figures" / f"smooth_mlp_certificate_seed_{seed}.png"


@torch.no_grad()
def audit_anchor(seed: int, anchor: int, crossings: dict[str, int | None]) -> dict:
    path = cache_path(seed, anchor)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    config = frozen_config(seed)
    _, checkpoint_path = artifact_paths(seed)
    checkpoints = np.load(checkpoint_path)
    parameter = torch.from_numpy(checkpoints[f"step_{anchor}"]).clone()
    train_pairs, train_labels, test_pairs, test_labels = make_split(config)
    current_prediction = torch.argmax(logits(parameter, test_pairs, config), dim=1)
    current_correct = int(torch.sum(current_prediction == test_labels))
    required = required_counts(len(test_pairs))
    if crossings["0.95"] is not None and anchor >= crossings["0.95"]:
        row = {
            "seed": seed,
            "anchor": anchor,
            "current_correct": current_correct,
            "current_accuracy": current_correct / len(test_pairs),
            "skipped_all_events_already_reached": True,
            "raw_forecasts": {},
            "certificate_horizon_requested": 0,
            "certificate_horizon_reached": 0,
            "certificates": {},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        return row

    hessian, gauss_newton = exact_objective_hessian(
        parameter, train_pairs, train_labels, config
    )
    gradient = analytic_gradient(parameter, train_pairs, train_labels, config)
    eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
    raw, factors = raw_forecasts(
        parameter,
        eigenvalues,
        eigenvectors,
        gradient,
        test_pairs,
        test_labels,
        config,
    )
    pending_raw = [
        int(raw[f"{threshold:.2f}"]["full_quadratic"])
        for threshold in THRESHOLDS
        if current_correct < required[threshold]
        and crossings[f"{threshold:.2f}"] is not None
        and anchor < int(crossings[f"{threshold:.2f}"])
        and raw[f"{threshold:.2f}"]["full_quadratic"] not in (None, 0)
    ]
    requested = min(HORIZON, max(pending_raw) + CERTIFICATE_BUFFER) if pending_raw else 0
    certificates: dict[str, dict] = {}
    reached = 0
    tube_summary: dict = {}
    if requested:
        displacement, epsilon, defect, injected = exact_path_tube_until(
            parameter,
            hessian,
            gradient,
            eigenvalues,
            eigenvectors,
            train_pairs,
            train_labels,
            config,
            requested,
        )
        reached = len(epsilon) - 1
        variants = certified_counts(
            parameter,
            displacement,
            epsilon,
            test_pairs,
            test_labels,
            config,
        )
        for threshold in THRESHOLDS:
            key = f"{threshold:.2f}"
            if (
                current_correct >= required[threshold]
                or crossings[key] is None
                or anchor >= int(crossings[key])
                or raw[key]["full_quadratic"] is None
            ):
                continue
            full_bracket = event_bracket(
                variants["full_quadratic"]["guaranteed"],
                variants["full_quadratic"]["possible"],
                required[threshold],
            )
            linear_bracket = event_bracket(
                variants["jacobian_only"]["guaranteed"],
                variants["jacobian_only"]["possible"],
                required[threshold],
            )
            state_audit = None
            if full_bracket is not None:
                state_audit = audit_state_path(
                    parameter,
                    displacement,
                    epsilon,
                    train_pairs,
                    train_labels,
                    config,
                    full_bracket[1],
                )
            certificates[key] = {
                "full_quadratic_bracket": full_bracket,
                "jacobian_only_bracket": linear_bracket,
                "full_certificate_issued": full_bracket is not None,
                "jacobian_only_certificate_issued": linear_bracket is not None,
                "state_audit": state_audit,
            }
        tube_summary = {
            "maximum_deterministic_path_defect": float(np.max(defect)) if len(defect) else 0.0,
            "maximum_injected_defect_bound": float(np.max(injected)) if len(injected) else 0.0,
            "state_error_bound_at_reached_horizon": float(epsilon[-1]),
        }
    row = {
        "seed": seed,
        "anchor": anchor,
        "current_correct": current_correct,
        "current_accuracy": current_correct / len(test_pairs),
        "skipped_all_events_already_reached": False,
        "minimum_train_hessian_eigenvalue": float(eigenvalues[0]),
        "maximum_train_hessian_eigenvalue": float(eigenvalues[-1]),
        "gauss_newton_frobenius_fraction": float(
            torch.linalg.matrix_norm(gauss_newton) / torch.linalg.matrix_norm(hessian)
        ),
        "frozen_propagator_norm": float(torch.max(torch.abs(factors))),
        "raw_forecasts": raw,
        "certificate_horizon_requested": requested,
        "certificate_horizon_reached": reached,
        "certificates": certificates,
        "tube_summary": tube_summary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    return row


def attach_outcomes(rows: list[dict], crossings: dict[str, int | None]) -> list[dict]:
    evaluated: list[dict] = []
    for row in rows:
        anchor = row["anchor"]
        for key, raw in row.get("raw_forecasts", {}).items():
            crossing = crossings.get(key)
            if crossing is None or crossing <= anchor:
                continue
            certificate = row.get("certificates", {}).get(key, {})
            bracket = certificate.get("full_quadratic_bracket")
            raw_offset = raw.get("full_quadratic")
            actual_offset = int(crossing - anchor)
            evaluated.append(
                {
                    "seed": row["seed"],
                    "anchor": anchor,
                    "threshold": float(key),
                    "actual_crossing": int(crossing),
                    "actual_offset": actual_offset,
                    "raw_full_quadratic_offset": raw_offset,
                    "raw_jacobian_offset": raw.get("jacobian_only"),
                    "raw_candidate": raw_offset not in (None, 0),
                    "certificate_issued": bracket is not None,
                    "bracket": bracket,
                    "covered": (
                        None
                        if bracket is None
                        else bool(bracket[0] <= actual_offset <= bracket[1])
                    ),
                    "bracket_width": None if bracket is None else int(bracket[1] - bracket[0]),
                    "lead_time": None if bracket is None else actual_offset,
                    "raw_absolute_error": (
                        None if raw_offset in (None, 0) else abs(int(raw_offset) - actual_offset)
                    ),
                    "raw_fractional_error": (
                        None
                        if raw_offset in (None, 0)
                        else abs(int(raw_offset) - actual_offset) / actual_offset
                    ),
                    "certificate_horizon_reached": row.get("certificate_horizon_reached", 0),
                    "state_audit": certificate.get("state_audit"),
                }
            )
    return evaluated


def summarize(evaluated: list[dict]) -> dict:
    raw_candidates = [row for row in evaluated if row["raw_candidate"]]
    issued = [row for row in evaluated if row["certificate_issued"]]
    raw_errors = [row["raw_absolute_error"] for row in raw_candidates]
    return {
        "opportunities": len(evaluated),
        "raw_candidates": len(raw_candidates),
        "certificates_issued": len(issued),
        "coverage_when_issued": (
            None if not issued else sum(bool(row["covered"]) for row in issued) / len(issued)
        ),
        "all_anchor_abstention_rate": (
            None if not evaluated else 1.0 - len(issued) / len(evaluated)
        ),
        "raw_candidate_abstention_rate": (
            None if not raw_candidates else 1.0 - len(issued) / len(raw_candidates)
        ),
        "median_bracket_width": (
            None if not issued else float(np.median([row["bracket_width"] for row in issued]))
        ),
        "median_lead_time": (
            None if not issued else float(np.median([row["lead_time"] for row in issued]))
        ),
        "maximum_lead_time": None if not issued else max(row["lead_time"] for row in issued),
        "first_issued_checkpoint": None if not issued else min(row["anchor"] for row in issued),
        "median_raw_absolute_error": None if not raw_errors else float(np.median(raw_errors)),
        "median_raw_fractional_error": (
            None
            if not raw_candidates
            else float(np.median([row["raw_fractional_error"] for row in raw_candidates]))
        ),
        "state_tube_violations": sum(
            (row["state_audit"] or {}).get("violations", 0) for row in issued
        ),
    }


def render(evaluated: list[dict], crossings: dict[str, int | None], seed: int) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    for threshold in THRESHOLDS:
        key = f"{threshold:.2f}"
        crossing = crossings.get(key)
        if crossing is not None:
            ax.scatter(crossing, threshold, marker="*", s=180, color="#d1495b", zorder=4)
    for row in evaluated:
        if row["raw_candidate"]:
            ax.scatter(
                row["anchor"] + row["raw_full_quadratic_offset"],
                row["threshold"],
                s=25,
                facecolors="none",
                edgecolors="#087e8b",
                alpha=0.7,
            )
        if row["certificate_issued"]:
            left = row["anchor"] + row["bracket"][0]
            right = row["anchor"] + row["bracket"][1]
            ax.hlines(row["threshold"], left, right, color="#1f2937", linewidth=4, zorder=3)
    ax.set(
        xlabel="absolute full-batch GD step",
        ylabel="accuracy threshold",
        title=f"Smooth-MLP transfer audit, fresh seed {seed}",
        yticks=list(THRESHOLDS),
        yticklabels=[f"{int(100 * value)}%" for value in THRESHOLDS],
    )
    ax.grid(alpha=0.2)
    figure = audit_figure(seed)
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=220)
    plt.close(fig)


def run(seed: int) -> dict:
    torch.set_num_threads(1)
    crossing_payload = json.loads(crossing_output(seed).read_text(encoding="utf-8"))
    crossings = crossing_payload["crossing_steps"]
    anchors = selected_anchors(seed)
    rows: list[dict] = []
    for index, anchor in enumerate(anchors):
        print(f"anchor {index + 1}/{len(anchors)}: {anchor}", flush=True)
        rows.append(audit_anchor(seed, anchor, crossings))
    evaluated = attach_outcomes(rows, crossings)
    summary = summarize(evaluated)
    payload = {
        "protocol": "SECOND_ARCHITECTURE_PROTOCOL.md",
        "seed": seed,
        "config": asdict(frozen_config(seed)),
        "anchors": anchors,
        "crossings": crossings,
        "rows": rows,
        "evaluated_events": evaluated,
        "summary": summary,
    }
    output = audit_output(seed)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if seed == 2:
        LEGACY_PRIMARY_OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    render(evaluated, crossings, seed)
    print(json.dumps(summary, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--anchor", type=int)
    args = parser.parse_args()
    if args.anchor is None:
        run(args.seed)
    else:
        crossings = json.loads(crossing_output(args.seed).read_text(encoding="utf-8"))[
            "crossing_steps"
        ]
        print(json.dumps(audit_anchor(args.seed, args.anchor, crossings), indent=2))


if __name__ == "__main__":
    main()
