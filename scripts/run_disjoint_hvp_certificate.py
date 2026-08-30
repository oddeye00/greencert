#!/usr/bin/env python3
"""Run the projected HVP certificate on a disjoint-set model checkpoint."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from disjoint_large_mlp import DisjointConfig, artifact_paths, make_disjoint_split
from hvp_projected_mlp_certificate import (
    brackets_for_thresholds,
    build_hvp_projected_certificate,
    projected_certified_counts,
)
from replay_smooth_mlp_thresholds import THRESHOLDS, required_counts
from smooth_mlp_modular_grokking import analytic_gradient, logits


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "disjoint_hvp_development"


@torch.no_grad()
def audit_actual_path(
    parameter: torch.Tensor,
    result,
    data: tuple[torch.Tensor, ...],
    config,
) -> dict:
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = data
    actual = parameter.clone()
    active_violations = 0
    complement_violations = 0
    maximum_active_ratio = 0.0
    maximum_complement_ratio = 0.0
    actual_counts: list[int] = []
    for step in range(len(result.tube.reference)):
        error = actual - result.tube.reference[step]
        active_error = float(torch.linalg.vector_norm(result.basis.T @ error))
        complement_error = float(torch.linalg.vector_norm(
            error - result.basis @ (result.basis.T @ error)
        ))
        active_bound = float(result.tube.active_radius[step])
        complement_bound = float(result.tube.complement_radius[step])
        if active_error > active_bound * (1.0 + 1e-8) + 1e-10:
            active_violations += 1
        if complement_error > complement_bound * (1.0 + 1e-8) + 1e-10:
            complement_violations += 1
        if active_bound > 1e-14:
            maximum_active_ratio = max(maximum_active_ratio, active_error / active_bound)
        if complement_bound > 1e-14:
            maximum_complement_ratio = max(
                maximum_complement_ratio, complement_error / complement_bound
            )
        actual_counts.append(int(torch.sum(
            torch.argmax(logits(actual, cert_pairs, config), dim=1) == cert_labels
        )))
        if step + 1 < len(result.tube.reference):
            actual.add_(
                analytic_gradient(actual, train_pairs, train_labels, config),
                alpha=-config.learning_rate,
            )
    return {
        "checked_steps": len(result.tube.reference),
        "active_violations": active_violations,
        "complement_violations": complement_violations,
        "maximum_active_error_to_bound_ratio": maximum_active_ratio,
        "maximum_complement_error_to_bound_ratio": maximum_complement_ratio,
        "actual_certificate_counts": actual_counts,
    }


def run(
    seed: int,
    anchor: int,
    horizon: int,
    *,
    development: bool,
    rank: int,
    margin_starts: int,
    geometry_stride: int,
    power: int,
    probes: int,
    failure_probability: float,
    recenter_sweeps: int,
    persistence: int,
    use_cache: bool,
) -> dict:
    output_dir = OUT_DIR if development else ROOT / "results" / "disjoint_hvp_prospective"
    output = output_dir / (
        f"seed_{seed}_anchor_{anchor}_h{horizon}"
        f"_r{rank}_m{margin_starts}_g{geometry_stride}"
        f"_q{power}_p{probes}_s{recenter_sweeps}_k{persistence}.json"
    )
    if use_cache and output.exists():
        return json.loads(output.read_text(encoding="utf-8"))

    result_path, checkpoint_path = artifact_paths(seed, development=development)
    model_payload = json.loads(result_path.read_text(encoding="utf-8"))
    config = DisjointConfig(**model_payload["config"])
    if config.seed != seed:
        raise ValueError("artifact seed does not match requested seed")
    model_config = config.model_config()
    checkpoints = np.load(checkpoint_path)
    data = make_disjoint_split(config)
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = data
    saved_steps = sorted(int(key.removeprefix("step_")) for key in checkpoints.files)
    candidates = [step for step in saved_steps if step <= anchor]
    if not candidates:
        raise ValueError(f"no saved checkpoint precedes anchor {anchor}")
    loaded_step = max(candidates)
    parameter = torch.from_numpy(checkpoints[f"step_{loaded_step}"]).clone()
    for _ in range(loaded_step, anchor):
        parameter.add_(
            analytic_gradient(parameter, train_pairs, train_labels, model_config),
            alpha=-model_config.learning_rate,
        )
    started = time.perf_counter()
    result = build_hvp_projected_certificate(
        parameter,
        train_pairs,
        train_labels,
        cert_pairs,
        cert_labels,
        model_config,
        horizon=horizon,
        rank=rank,
        margin_starts=margin_starts,
        geometry_stride=geometry_stride,
        power=power,
        probes=probes,
        total_failure_probability=failure_probability,
        random_seed=90_001 + seed * 10_007 + anchor,
        recenter_sweeps=recenter_sweeps,
    )
    guaranteed, possible, center_correct, max_error = projected_certified_counts(
        result, cert_pairs, cert_labels, model_config
    )
    required = required_counts(len(cert_pairs))
    brackets = brackets_for_thresholds(
        guaranteed, possible, required, persistence=persistence
    )
    audit = audit_actual_path(parameter, result, data, model_config)
    actual_counts = np.asarray(audit.pop("actual_certificate_counts"), dtype=np.int64)

    events = {}
    for threshold in THRESHOLDS:
        key = f"{threshold:.2f}"
        actual_index = np.asarray([
            np.all(actual_counts[j : j + persistence] >= required[threshold])
            for j in range(max(len(actual_counts) - persistence + 1, 0))
        ])
        center_index = np.asarray([
            np.all(center_correct[j : j + persistence] >= required[threshold])
            for j in range(max(len(center_correct) - persistence + 1, 0))
        ])
        actual_candidates = np.flatnonzero(actual_index)
        center_candidates = np.flatnonzero(center_index)
        actual = None if len(actual_candidates) == 0 else int(actual_candidates[0])
        center = None if len(center_candidates) == 0 else int(center_candidates[0])
        bracket = brackets[key]
        events[key] = {
            "required": int(required[threshold]),
            "center_crossing": center,
            "local_actual_crossing": actual,
            "certified_bracket": bracket,
            "covered_local_crossing": (
                None if bracket is None or actual is None
                else bool(bracket[0] <= actual <= bracket[1])
            ),
        }

    payload = {
        "status": "development" if development else "prospective blind candidate",
        "method": "matrix-free projected HVP variational certificate",
        "seed": seed,
        "anchor": anchor,
        "loaded_checkpoint_step": loaded_step,
        "requested_horizon": horizon,
        "reached_horizon": result.tube.reached_horizon,
        "parameter_count": parameter.numel(),
        "active_rank": result.basis.shape[1],
        "event_persistence": persistence,
        "events": events,
        "construction": result.construction_diagnostics,
        "probe_diagnostics": list(result.probe_diagnostics),
        "tube": {
            "maximum_active_radius": float(np.max(result.tube.active_radius)),
            "maximum_complement_radius": float(np.max(result.tube.complement_radius)),
            "maximum_total_radius": float(np.max(result.tube.total_radius)),
            "maximum_margin_error": float(np.max(max_error)),
            "maximum_active_defect": float(np.max(result.tube.active_defect_norm, initial=0.0)),
            "maximum_complement_defect": float(
                np.max(result.tube.complement_defect_norm, initial=0.0)
            ),
            "trace": {
                "active_radius": result.tube.active_radius.tolist(),
                "complement_radius": result.tube.complement_radius.tolist(),
                "total_radius": result.tube.total_radius.tolist(),
                "active_defect_norm": result.tube.active_defect_norm.tolist(),
                "complement_defect_norm": result.tube.complement_defect_norm.tolist(),
                "active_block_norm": result.tube.active_block_norm.tolist(),
                "cross_block_norm": result.tube.cross_block_norm.tolist(),
                "complement_block_norm": result.tube.complement_block_norm.tolist(),
                "nonlinear_injection": result.tube.nonlinear_injection.tolist(),
            },
        },
        "actual_path_audit": audit,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--anchor", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--rank", type=int, default=24)
    parser.add_argument("--margin-starts", type=int, default=4)
    parser.add_argument("--geometry-stride", type=int, default=25)
    parser.add_argument("--power", type=int, default=24)
    parser.add_argument("--probes", type=int, default=8)
    parser.add_argument("--failure-probability", type=float, default=1e-6)
    parser.add_argument("--recenter-sweeps", type=int, default=1)
    parser.add_argument("--persistence", type=int, default=1)
    parser.add_argument("--prospective", action="store_true")
    parser.add_argument("--use-cache", action="store_true")
    args = parser.parse_args()
    payload = run(
        args.seed,
        args.anchor,
        args.horizon,
        development=not args.prospective,
        rank=args.rank,
        margin_starts=args.margin_starts,
        geometry_stride=args.geometry_stride,
        power=args.power,
        probes=args.probes,
        failure_probability=args.failure_probability,
        recenter_sweeps=args.recenter_sweeps,
        persistence=args.persistence,
        use_cache=args.use_cache,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
