#!/usr/bin/env python3
"""Defect-corrected variational certificates for the smooth modular MLP.

This is a post-audit theorem-development implementation.  It never overwrites
the frozen results from ``SECOND_ARCHITECTURE_PROTOCOL.md``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from generate_smooth_mlp_seed import artifact_paths, frozen_config
from modular_accuracy_certificate import event_bracket
from replay_smooth_mlp_thresholds import THRESHOLDS, crossing_output, required_counts
from smooth_mlp_certificate import (
    exact_objective_hessian,
    margin_derivative_bounds,
    modal_path,
    objective_hessian_lipschitz,
)
from smooth_mlp_modular_grokking import analytic_gradient, logits, make_split
from variational_shadowing import StepLinearization, defect_corrected_variational_tube


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "variational_theorem_development"


@torch.no_grad()
def build_variational_tube(
    seed: int,
    anchor: int,
    horizon: int,
) -> tuple[torch.Tensor, object, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], object]:
    config = frozen_config(seed)
    _, checkpoint_path = artifact_paths(seed)
    checkpoints = np.load(checkpoint_path)
    parameter = torch.from_numpy(checkpoints[f"step_{anchor}"]).clone()
    train_pairs, train_labels, test_pairs, test_labels = make_split(config)
    anchor_hessian, _ = exact_objective_hessian(
        parameter, train_pairs, train_labels, config
    )
    anchor_gradient = analytic_gradient(parameter, train_pairs, train_labels, config)
    eigenvalues, eigenvectors = torch.linalg.eigh(anchor_hessian)
    modal, _ = modal_path(
        eigenvalues,
        eigenvectors,
        anchor_gradient,
        config.learning_rate,
        horizon,
    )
    reference = parameter[None, :] + modal @ eigenvectors.T

    def linearize(center: torch.Tensor) -> StepLinearization:
        gradient = analytic_gradient(center, train_pairs, train_labels, config)
        hessian, _ = exact_objective_hessian(center, train_pairs, train_labels, config)
        hessian_eigenvalues = torch.linalg.eigvalsh(hessian)
        jacobian = torch.eye(
            parameter.numel(), dtype=parameter.dtype
        ) - config.learning_rate * hessian
        beta = float(
            torch.max(torch.abs(1.0 - config.learning_rate * hessian_eigenvalues))
        )
        return StepLinearization(
            mapped_center=center - config.learning_rate * gradient,
            jacobian=jacobian,
            jacobian_operator_norm=beta,
            jacobian_lipschitz=lambda radius, center=center: (
                config.learning_rate
                * objective_hessian_lipschitz(center, config, radius)
            ),
        )

    tube = defect_corrected_variational_tube(reference, linearize, numeric_cap=1e4)
    return parameter, tube, (train_pairs, train_labels, test_pairs, test_labels), config


@torch.no_grad()
def certified_accuracy_counts(
    corrected_reference: torch.Tensor,
    radius: np.ndarray,
    test_pairs: torch.Tensor,
    test_labels: torch.Tensor,
    config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accuracy counts using exact reference logits and a state-only remainder."""
    step_count = len(corrected_reference)
    sample_count = len(test_pairs)
    class_count = config.modulus
    guaranteed = np.zeros(step_count, dtype=np.int64)
    possible = np.zeros(step_count, dtype=np.int64)
    center_correct = np.zeros(step_count, dtype=np.int64)
    label_array = test_labels.numpy()
    rows = np.arange(sample_count)
    for step in range(step_count):
        center = corrected_reference[step]
        values = logits(center, test_pairs, config)
        center_correct[step] = int(torch.sum(torch.argmax(values, dim=1) == test_labels))
        true_values = values[rows, test_labels]
        margins = (true_values[:, None] - values).numpy()
        error = np.zeros_like(margins)
        for label in range(class_count):
            for competitor in range(class_count):
                if label == competitor:
                    continue
                b1, _, _ = margin_derivative_bounds(
                    center,
                    config,
                    label,
                    competitor,
                    float(radius[step]),
                )
                mask = label_array == label
                error[mask, competitor] = b1 * float(radius[step])
        lower = margins - error
        upper = margins + error
        lower[rows, label_array] = np.inf
        upper[rows, label_array] = np.inf
        guaranteed[step] = int(np.sum(np.all(lower > 0.0, axis=1)))
        possible[step] = int(sample_count - np.sum(np.any(upper < 0.0, axis=1)))
    return guaranteed, possible, center_correct


@torch.no_grad()
def audit_actual_path(
    anchor_parameter: torch.Tensor,
    corrected_reference: torch.Tensor,
    radius: np.ndarray,
    train_pairs: torch.Tensor,
    train_labels: torch.Tensor,
    config,
) -> dict:
    actual = anchor_parameter.clone()
    violations = 0
    maximum_ratio = 0.0
    observed_errors: list[float] = []
    for step in range(len(corrected_reference)):
        observed = float(torch.linalg.vector_norm(actual - corrected_reference[step]))
        bound = float(radius[step])
        observed_errors.append(observed)
        if observed > bound * (1.0 + 1e-8) + 1e-10:
            violations += 1
        if bound > 1e-14:
            maximum_ratio = max(maximum_ratio, observed / bound)
        if step + 1 < len(corrected_reference):
            actual.add_(
                analytic_gradient(actual, train_pairs, train_labels, config),
                alpha=-config.learning_rate,
            )
    return {
        "checked_steps": len(corrected_reference),
        "violations": violations,
        "maximum_error_to_bound_ratio": maximum_ratio,
        "maximum_observed_error": max(observed_errors),
    }


def run(
    seed: int,
    anchor: int,
    horizon: int,
    *,
    use_cache: bool = False,
    print_output: bool = True,
) -> dict:
    output = OUT_DIR / f"seed_{seed}_anchor_{anchor}_h{horizon}.json"
    if use_cache and output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("method_version") == "variational-v1-frozen-2026-08-21":
            return payload
    torch.set_num_threads(4)
    parameter, tube, data, config = build_variational_tube(seed, anchor, horizon)
    train_pairs, train_labels, test_pairs, test_labels = data
    guaranteed, possible, center_correct = certified_accuracy_counts(
        tube.corrected_reference,
        tube.remainder_radius,
        test_pairs,
        test_labels,
        config,
    )
    required = required_counts(len(test_pairs))
    crossing_payload = json.loads(crossing_output(seed).read_text(encoding="utf-8"))
    actual_crossings = crossing_payload["crossing_steps"]
    events: dict[str, dict] = {}
    for threshold in THRESHOLDS:
        key = f"{threshold:.2f}"
        bracket = event_bracket(guaranteed, possible, required[threshold])
        center_indices = np.flatnonzero(center_correct >= required[threshold])
        center_crossing = None if len(center_indices) == 0 else int(center_indices[0])
        absolute_actual = actual_crossings[key]
        actual_offset = (
            None
            if absolute_actual is None or int(absolute_actual) <= anchor
            else int(absolute_actual) - anchor
        )
        events[key] = {
            "corrected_reference_crossing": center_crossing,
            "certified_bracket": bracket,
            "actual_offset": actual_offset,
            "covered": (
                None
                if bracket is None or actual_offset is None
                else bool(bracket[0] <= actual_offset <= bracket[1])
            ),
        }
    actual_audit = audit_actual_path(
        parameter,
        tube.corrected_reference,
        tube.remainder_radius,
        train_pairs,
        train_labels,
        config,
    )
    payload = {
        "status": "post-audit theorem development; not part of frozen replication",
        "method": "defect-corrected nonautonomous variational shadowing",
        "method_version": "variational-v1-frozen-2026-08-21",
        "seed": seed,
        "anchor": anchor,
        "requested_horizon": horizon,
        "reached_horizon": tube.reached_horizon,
        "events": events,
        "tube": {
            "maximum_reference_defect_norm": (
                float(np.max(tube.reference_defect_norm))
                if len(tube.reference_defect_norm)
                else 0.0
            ),
            "maximum_correction_norm": float(np.max(tube.correction_norm)),
            "maximum_remainder_radius": float(np.max(tube.remainder_radius)),
            "maximum_total_error_radius": float(np.max(tube.total_error_radius)),
            "maximum_jacobian_norm": (
                float(np.max(tube.jacobian_norm)) if len(tube.jacobian_norm) else 0.0
            ),
        },
        "actual_path_audit": actual_audit,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if print_output:
        print(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--anchor", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    args = parser.parse_args()
    run(args.seed, args.anchor, args.horizon)


if __name__ == "__main__":
    main()
