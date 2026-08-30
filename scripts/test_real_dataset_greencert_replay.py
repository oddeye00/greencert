#!/usr/bin/env python3
"""Replay one burned development certificate through the reusable fresh code."""
from __future__ import annotations

import json

import numpy as np
import torch

import run_real_dataset_confirmation as runner
from probe_jacobian_bound import ProbeRegistry
from real_dataset_greencert import certify_candidate
from real_dataset_mlp import make_split, parameter_spec


def main() -> None:
    seed = 0
    gate_index = 1
    threshold = 0.925
    anchor = 40
    identity = (92, seed, gate_index, anchor, runner.SWEEPS, runner.HORIZON)
    config = runner.RealMLPConfig(**{**runner.asdict(runner.BASE_CONFIG), "seed": seed})
    torch.set_num_threads(config.threads)
    data = make_split(config)
    spec = parameter_spec(config)
    checkpoints = np.load(
        runner.ROOT / "results" / "real_dataset_development" / "seed_0.checkpoints.npz"
    )
    parameter = torch.from_numpy(checkpoints[f"step_{anchor}"]).clone()
    registry = ProbeRegistry([identity], "11" * 32)
    row = certify_candidate(
        parameter,
        data,
        spec,
        config,
        seed=seed,
        gate_index=gate_index,
        threshold=threshold,
        anchor=anchor,
        horizon=runner.HORIZON,
        persistence=runner.PERSISTENCE,
        sweeps=runner.SWEEPS,
        probe=runner.PROBE,
        registry=registry,
        identity=identity,
    )
    assert row["predicted_event"] == 22
    assert row["certificate_issued"]
    assert row["certified_bracket"] == [22, 22]
    assert row["closure_statistic"] < 1.0
    assert row["unsigned_right_inverse_response_upper"] >= row["signed_response_sequence_norm"]
    print(
        json.dumps(
            {
                "status": "PASS reusable GreenCert replay",
                "predicted_event": row["predicted_event"],
                "certified_bracket": row["certified_bracket"],
                "closure_statistic": row["closure_statistic"],
                "elapsed_seconds": row["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
