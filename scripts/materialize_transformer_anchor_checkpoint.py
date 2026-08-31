#!/usr/bin/env python3
"""Materialize a minimal exact checkpoint and audit a regenerated-state bridge."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value).tobytes(order="C")
    ).hexdigest().upper()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--anchor", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prefix = ROOT / "data" / (
        f"transformer_seed_{args.seed}_anchor_{args.anchor}"
    )
    parameter_path = prefix.with_name(prefix.name + "_parameter.npy")
    velocity_path = prefix.with_name(prefix.name + "_velocity.npy")
    metadata_path = prefix.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    exact_parameter = np.load(parameter_path, allow_pickle=False)
    exact_velocity = np.load(velocity_path, allow_pickle=False)
    if file_sha256(parameter_path) != metadata["parameter_sha256"]:
        raise RuntimeError("exact anchor parameter hash mismatch")
    if file_sha256(velocity_path) != metadata["velocity_sha256"]:
        raise RuntimeError("exact anchor velocity hash mismatch")

    checkpoint = (
        ROOT
        / "results"
        / f"transformer_hvp_prospective_seed_{args.seed}.checkpoints.npz"
    )
    bridge = {
        "status": "exact-anchor materialization without regeneration comparison",
        "seed": args.seed,
        "anchor": args.anchor,
        "exact_parameter_array_sha256": array_sha256(exact_parameter),
        "exact_velocity_array_sha256": array_sha256(exact_velocity),
        "exact_parameter_file_sha256": file_sha256(parameter_path),
        "exact_velocity_file_sha256": file_sha256(velocity_path),
        "regenerated_checkpoint_present": checkpoint.is_file(),
        "outcome_files_read": 0,
    }
    if checkpoint.is_file():
        with np.load(checkpoint) as archive:
            regenerated_parameter = np.asarray(
                archive[f"step_{args.anchor}"], dtype=np.float64
            )
            regenerated_velocity = np.asarray(
                archive[f"velocity_{args.anchor}"], dtype=np.float64
            )
        if regenerated_parameter.shape != exact_parameter.shape:
            raise RuntimeError("regenerated parameter shape mismatch")
        if regenerated_velocity.shape != exact_velocity.shape:
            raise RuntimeError("regenerated velocity shape mismatch")
        parameter_difference = regenerated_parameter - exact_parameter
        velocity_difference = regenerated_velocity - exact_velocity
        bridge.update(
            {
                "status": "regenerated-to-exact anchor bridge audited",
                "regenerated_parameter_array_sha256": array_sha256(
                    regenerated_parameter
                ),
                "regenerated_velocity_array_sha256": array_sha256(
                    regenerated_velocity
                ),
                "parameter_bitwise_equal": bool(
                    np.array_equal(regenerated_parameter, exact_parameter)
                ),
                "velocity_bitwise_equal": bool(
                    np.array_equal(regenerated_velocity, exact_velocity)
                ),
                "parameter_maximum_absolute_difference": float(
                    np.abs(parameter_difference).max(initial=0.0)
                ),
                "velocity_maximum_absolute_difference": float(
                    np.abs(velocity_difference).max(initial=0.0)
                ),
                "parameter_l2_difference": float(
                    np.linalg.vector_norm(parameter_difference)
                ),
                "velocity_l2_difference": float(
                    np.linalg.vector_norm(velocity_difference)
                ),
            }
        )
    if checkpoint.exists() and not args.force:
        raise FileExistsError(
            "checkpoint exists; pass --force after auditing the regenerated bridge"
        )
    np.savez_compressed(
        checkpoint,
        **{
            f"step_{args.anchor}": exact_parameter,
            f"velocity_{args.anchor}": exact_velocity,
        },
    )
    bridge["materialized_checkpoint_sha256"] = file_sha256(checkpoint)
    bridge["materialized_checkpoint_bytes"] = checkpoint.stat().st_size
    output = (
        ROOT
        / "results"
        / f"transformer_seed_{args.seed}_anchor_{args.anchor}_regeneration_bridge.json"
    )
    output.write_text(
        json.dumps(bridge, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(bridge, indent=2))


if __name__ == "__main__":
    main()
