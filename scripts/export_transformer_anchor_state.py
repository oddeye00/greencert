#!/usr/bin/env python3
"""Export an exact candidate anchor from a regenerated full checkpoint archive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--anchor", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    source = (
        ROOT
        / "results"
        / f"transformer_hvp_prospective_seed_{args.seed}.checkpoints.npz"
    )
    if not source.is_file():
        raise FileNotFoundError(source)
    prefix = ROOT / "data" / (
        f"transformer_seed_{args.seed}_anchor_{args.anchor}"
    )
    parameter_path = prefix.with_name(prefix.name + "_parameter.npy")
    velocity_path = prefix.with_name(prefix.name + "_velocity.npy")
    metadata_path = prefix.with_suffix(".json")
    outputs = (parameter_path, velocity_path, metadata_path)
    if not args.force and any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite an existing anchor export")
    with np.load(source) as archive:
        parameter = np.asarray(archive[f"step_{args.anchor}"], dtype=np.float64)
        velocity = np.asarray(
            archive[f"velocity_{args.anchor}"], dtype=np.float64
        )
    if parameter.ndim != 1 or velocity.shape != parameter.shape:
        raise RuntimeError("anchor arrays have unexpected shapes")
    parameter_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(parameter_path, parameter, allow_pickle=False)
    np.save(velocity_path, velocity, allow_pickle=False)
    metadata = {
        "status": "exact sealed Transformer anchor export",
        "seed": args.seed,
        "anchor": args.anchor,
        "dtype": str(parameter.dtype),
        "parameter_count": int(parameter.size),
        "parameter_file": parameter_path.relative_to(ROOT).as_posix(),
        "parameter_sha256": sha256(parameter_path),
        "velocity_file": velocity_path.relative_to(ROOT).as_posix(),
        "velocity_sha256": sha256(velocity_path),
        "source_archive_sha256": sha256(source),
        "outcome_files_read": 0,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
