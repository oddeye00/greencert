#!/usr/bin/env python3
"""Development-only signed/unsigned GreenCert screen on digits parity.

All outcomes for these development seeds have already been inspected.  Results
from this script are diagnostic only and must never be described as fresh.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from develop_digits_signed_screen import initialize, make_data
from probe_jacobian_bound import ProbeConfig, ProbeRegistry
from real_dataset_greencert import certify_candidate, trigger_only_anchor
from real_dataset_mlp import RealMLPConfig, accuracy, gradient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--sweeps", type=int, default=4)
    parser.add_argument("--probes", type=int, default=4)
    parser.add_argument("--power", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = RealMLPConfig(
        width=args.width,
        learning_rate=args.learning_rate,
        weight_decay=1e-3,
        steps=args.steps,
        checkpoint_every=5,
        seed=args.seed,
        threads=1,
        dtype="float64",
    )
    torch.set_num_threads(config.threads)
    data, spec = make_data(config)
    parameter = initialize(spec, config)
    train_accuracy: list[float] = []
    trigger_accuracy: list[float] = []
    checkpoints: dict[int, torch.Tensor] = {}
    for step in range(config.steps + 1):
        train_accuracy.append(accuracy(parameter, data["train_x"], data["train_y"], spec))
        trigger_accuracy.append(accuracy(parameter, data["trigger_x"], data["trigger_y"], spec))
        if step % config.checkpoint_every == 0:
            checkpoints[step] = parameter.detach().clone()
        if step < config.steps:
            parameter = parameter - config.learning_rate * gradient(
                parameter, data["train_x"], data["train_y"], spec, config
            )

    anchor = trigger_only_anchor(
        train_accuracy,
        trigger_accuracy,
        threshold=args.threshold,
        checkpoint_every=config.checkpoint_every,
        minimum_train_accuracy=0.80,
        trigger_band=0.10,
    )
    if anchor is None:
        raise RuntimeError("development trigger produced no anchor")
    identity = (371, args.seed, int(round(args.threshold * 1000)), anchor, args.sweeps, args.horizon)
    probe = ProbeConfig(probes=args.probes, power=args.power, delta=1e-6)
    registry = ProbeRegistry([identity], "digits-development-only")
    result = certify_candidate(
        checkpoints[anchor],
        data,
        spec,
        config,
        seed=args.seed,
        gate_index=0,
        threshold=args.threshold,
        anchor=anchor,
        horizon=args.horizon,
        persistence=10,
        sweeps=args.sweeps,
        probe=probe,
        registry=registry,
        identity=identity,
    )
    payload = {
        "status_tier": "DEVELOPMENT ONLY; OUTCOMES PREVIOUSLY INSPECTED",
        "dataset": "scikit-learn digits, binary parity target",
        "config": asdict(config),
        "parameter_count": spec.size,
        "anchor": anchor,
        "threshold": args.threshold,
        "horizon": args.horizon,
        "sweeps": args.sweeps,
        "probe": {"probes": args.probes, "power": args.power, "delta": 1e-6},
        "certificate": result,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = {
        "anchor": anchor,
        "predicted_event": result.get("predicted_event"),
        "certificate_horizon": result.get("certificate_horizon"),
        "status": result.get("status"),
        "issued": result.get("certificate_issued"),
        "unsigned_issued": result.get("unsigned_right_inverse_certificate_issued"),
        "directional_gain_ratio": result.get("directional_gain_ratio"),
        "signed_radius": result.get("minimal_admissible_radius"),
        "unsigned_radius": result.get("unsigned_right_inverse_minimal_radius"),
        "elapsed_seconds": result.get("elapsed_seconds"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
