#!/usr/bin/env python3
"""Alternating-order timing replay for 4- versus 16-probe Green blocks."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch

from probe_jacobian_bound import ProbeConfig, ProbeRegistry, gram_norm_bound
from transformer_certificate_protocol import Candidate
from transformer_four_sweep_development_audit import to_scaled
from transformer_green_development_audit import build_frozen_centerline
from transformer_green_operator import make_transformer_green_products
from transformer_v3_certificate import load_candidate


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "transformer_v3_relinearized_probe_block_benchmark.json"
CANDIDATE = Candidate(366, 0.70, 1040)
HORIZON = 52
DELTA = 4.59896983075791e-11
RUNS = 3


def from_scaled(path: torch.Tensor, dimension: int, eta: float) -> torch.Tensor:
    return torch.cat((path[..., :dimension], path[..., dimension:] / eta), dim=-1)


def main() -> None:
    config, template, spec, data, parameter, velocity = load_candidate(CANDIDATE)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    train_pairs, train_labels, _, _, _, _ = data
    dimension = int(parameter.numel())
    setup_started = time.perf_counter()
    path = build_frozen_centerline(
        config, template, spec, train_pairs, train_labels, parameter, velocity
    )
    center = path["center"][: HORIZON + 1]
    scaled = path["scaled_center"][: HORIZON + 1]
    mapped = [path["map_step"](center[j]) for j in range(HORIZON)]
    defect = torch.stack(
        [
            to_scaled(mapped[j], dimension, config.learning_rate) - scaled[j + 1]
            for j in range(HORIZON)
        ]
    )
    old_apply, _ = make_transformer_green_products(
        center[:HORIZON, :dimension],
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )
    response_rows = old_apply(defect.reshape(-1)).reshape(HORIZON, -1)
    response = torch.cat((torch.zeros_like(response_rows[:1]), response_rows), dim=0)
    corrected = from_scaled(scaled + response, dimension, config.learning_rate)
    apply_new, transpose_new = make_transformer_green_products(
        corrected[:HORIZON, :dimension],
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )
    setup_seconds = time.perf_counter() - setup_started

    def gram(vector: torch.Tensor) -> torch.Tensor:
        return transpose_new(apply_new(vector))

    rows = []
    orders = [(4, 16), (16, 4), (4, 16)]
    for repeat, order in enumerate(orders):
        for probes in order:
            identity = (94, repeat, probes, CANDIDATE.seed, CANDIDATE.anchor, HORIZON)
            registry = ProbeRegistry(
                [identity],
                f"relinearized-green-timing-v1-{repeat}-{probes}",
            )
            started = time.perf_counter()
            result = gram_norm_bound(
                gram,
                dimension=HORIZON * 2 * dimension,
                dtype=corrected.dtype,
                device=corrected.device,
                config=ProbeConfig(probes=probes, power=1, delta=DELTA),
                identity=identity,
                registry=registry,
            )
            rows.append(
                {
                    "repeat": repeat,
                    "order": list(order),
                    "probes": probes,
                    "seconds": time.perf_counter() - started,
                    "Y": float(result["Y"]),
                    "gram_applications": int(result["gram_applications"]),
                }
            )
            print(json.dumps(rows[-1]), flush=True)
    four = [row["seconds"] for row in rows if row["probes"] == 4]
    sixteen = [row["seconds"] for row in rows if row["probes"] == 16]
    paired = []
    for repeat in range(RUNS):
        small = next(row["seconds"] for row in rows if row["repeat"] == repeat and row["probes"] == 4)
        large = next(row["seconds"] for row in rows if row["repeat"] == repeat and row["probes"] == 16)
        paired.append(large / small)
    payload = {
        "status": "RELINEARIZED GREEN PROBE-BLOCK BENCHMARK COMPLETED",
        "evidence_boundary": (
            "Post-seal timing replay on one fixed corrected operator. No outcome "
            "is read and no prospective count changes."
        ),
        "candidate": CANDIDATE.__dict__,
        "horizon": HORIZON,
        "setup_seconds_excluded": setup_seconds,
        "rows": rows,
        "median_four_probe_seconds": statistics.median(four),
        "median_sixteen_probe_seconds": statistics.median(sixteen),
        "median_paired_speedup": statistics.median(paired),
        "minimum_paired_speedup": min(paired),
        "maximum_paired_speedup": max(paired),
        "logical_gram_application_reduction": 4.0,
        "outcome_files_read": 0,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

