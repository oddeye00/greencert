#!/usr/bin/env python3
"""Regenerate one omitted Transformer checkpoint archive from a blind record."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from transformer_hvp_grokking import (
    TransformerConfig,
    artifact_paths,
    train,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    blind_path, checkpoint_path = artifact_paths(
        args.seed, development=False
    )
    if not blind_path.is_file():
        raise FileNotFoundError(blind_path)
    if checkpoint_path.exists() and not args.force:
        raise FileExistsError(
            f"refusing to overwrite existing checkpoint archive: {checkpoint_path}"
        )
    blind = json.loads(blind_path.read_text(encoding="utf-8"))
    if blind.get("status") != "prospective blind trigger artifact":
        raise RuntimeError("source is not a prospective blind trigger record")
    if "certificate_accuracy" in blind.get("trajectory_columns", []):
        raise RuntimeError("blind source unexpectedly contains certification outcomes")

    config = TransformerConfig(**blind["config"])
    if config.seed != args.seed:
        raise RuntimeError("blind configuration seed mismatch")
    trajectory, checkpoints, summary = train(config, keep_checkpoints=True)
    visible = trajectory[:, (0, 1, 2, 4)]
    expected = np.asarray(blind["trajectory"], dtype=np.float64)
    if visible.shape != expected.shape:
        raise RuntimeError("regenerated trigger-visible trajectory shape changed")
    difference = np.abs(visible - expected)
    maximum_absolute_difference = float(difference.max(initial=0.0))
    if not np.allclose(visible, expected, rtol=3.0e-12, atol=3.0e-14):
        raise RuntimeError(
            "regenerated trigger-visible trajectory differs from the blind record; "
            f"maximum absolute difference {maximum_absolute_difference:.3e}"
        )
    if sorted(checkpoints) != [int(value) for value in blind["checkpoint_steps"]]:
        raise RuntimeError("regenerated checkpoint grid changed")
    for key, expected_value in blind["summary"].items():
        if key == "elapsed_seconds":
            continue
        observed = summary[key]
        if isinstance(expected_value, float):
            if not np.isclose(
                float(observed), expected_value, rtol=3.0e-12, atol=3.0e-14
            ):
                raise RuntimeError(f"regenerated summary field changed: {key}")
        elif observed != expected_value:
            raise RuntimeError(f"regenerated summary field changed: {key}")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
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
    print(
        json.dumps(
            {
                "status": "Transformer checkpoint regeneration passed",
                "seed": args.seed,
                "blind_source": blind_path.relative_to(ROOT).as_posix(),
                "blind_source_sha256": sha256(blind_path),
                "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
                "checkpoint_sha256": sha256(checkpoint_path),
                "checkpoint_bytes": checkpoint_path.stat().st_size,
                "maximum_trigger_visible_difference": maximum_absolute_difference,
                "outcome_files_read": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
