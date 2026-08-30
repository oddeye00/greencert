#!/usr/bin/env python3
"""One-shot generator for the frozen prospective-v2 seed population."""
from __future__ import annotations

import argparse
import json

from generate_smooth_mlp_seed import artifact_paths, run_seed
from prospective_v2_primary import SEEDS, verify_manifest


def generate(seed: int) -> dict:
    verify_manifest()
    if seed not in SEEDS:
        raise ValueError(f"seed {seed} is outside the frozen population {SEEDS}")
    result_path, checkpoint_path = artifact_paths(seed)
    existing = [path for path in (result_path, checkpoint_path) if path.exists()]
    if existing:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite prospective artifacts: {names}")
    generated = run_seed(seed)
    return {
        "status": "prospective artifacts written; training outcomes sealed",
        "seed": seed,
        "result": generated["result"],
        "checkpoints": generated["checkpoints"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.seed), indent=2))


if __name__ == "__main__":
    main()
