#!/usr/bin/env python3
"""Paired incremental-cost benchmark for the v3 directional second response.

The comparison starts after the centerline, first signed response, and first
Green Gram power are available.  It times the two choices faced by the frozen
adaptive policy:

1. evaluate one more 16-probe Green Gram power; or
2. construct the cancellation-safe quadratic forcing and propagate one scalar
   causal response.

No future outcome artifact is opened.  This is post-seal systems evidence, not
prospective issuance evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from batched_green_operator import make_batched_transformer_green_products
from online_progressive_gram import OnlineGramState
from transformer_certificate_protocol import Candidate
from transformer_four_sweep_development_audit import to_scaled
from transformer_green_development_audit import build_frozen_centerline
from transformer_green_operator import make_transformer_green_products
from transformer_two_response import optimizer_center_quadratic_defect
from transformer_v3_certificate import (
    METHOD_SEAL,
    frozen_candidates,
    load_candidate,
    output_path,
    safe_json,
)
from transformer_v3_protocol import green_identity, make_registry, probe_config


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOURCE = RESULTS / "transformer_v3_two_response_postseal_audit.json"
OUTPUT = RESULTS / "transformer_v3_two_response_paired_benchmark.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def candidate_key(row: dict) -> tuple[int, float, int]:
    value = row["candidate"]
    return int(value["seed"]), float(value["threshold"]), int(value["anchor"])


def relative_error(left: float, right: float) -> float:
    return abs(left - right) / max(abs(right), torch.finfo(torch.float64).tiny)


def run(candidate: Candidate, repeats: int) -> dict:
    if repeats < 3:
        raise ValueError("at least three paired repeats are required")
    candidates, horizons, _ = frozen_candidates()
    if candidate not in horizons:
        raise ValueError("candidate is outside the frozen v3 cohort")
    horizon = int(horizons[candidate])
    certificate_path = output_path(candidate)
    certificate = safe_json(certificate_path)
    if certificate.get("green_trace") is None:
        raise ValueError("candidate has no sealed Green trace")
    source = safe_json(SOURCE)
    source_row = next(
        row
        for row in source["rows"]
        if candidate_key(row)
        == (candidate.seed, candidate.threshold, candidate.anchor)
    )
    if not source_row.get("evaluable"):
        raise ValueError("candidate is not Green-evaluable")

    config, template, spec, data, parameter, velocity = load_candidate(candidate)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    train_pairs, train_labels, _, _, _, _ = data
    dimension = int(parameter.numel())
    path = build_frozen_centerline(
        config,
        template,
        spec,
        train_pairs,
        train_labels,
        parameter,
        velocity,
    )
    if path["centerline_sha256"] != certificate["centerline_sha256"]:
        raise RuntimeError("centerline hash mismatch")
    center = path["center"][: horizon + 1]
    scaled_center = path["scaled_center"][: horizon + 1]
    residual = torch.stack(
        [
            to_scaled(path["map_step"](center[step]), dimension, config.learning_rate)
            - scaled_center[step + 1]
            for step in range(horizon)
        ]
    )
    scalar_green, _ = make_transformer_green_products(
        center[:horizon, :dimension],
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )
    signed = scalar_green(residual.reshape(-1)).reshape(horizon, -1)
    if relative_error(
        float(torch.linalg.vector_norm(signed)),
        float(source_row["response_sequence_norm"]),
    ) > 2.0e-12:
        raise RuntimeError("signed response differs from the audited source")

    batch_green, batch_green_t = make_batched_transformer_green_products(
        center[:horizon, :dimension],
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )

    def green_gram(rows: torch.Tensor) -> torch.Tensor:
        return batch_green_t(batch_green(rows))

    method = safe_json(METHOD_SEAL)
    registry = make_registry(candidates, horizons, str(method["master_nonce"]))
    state = OnlineGramState.initialize(
        dimension=horizon * 2 * dimension,
        dtype=parameter.dtype,
        device=parameter.device,
        config=probe_config(),
        seed=registry.claim(green_identity(candidate, horizon)),
    )
    state.step(green_gram)
    power_one_vectors = state.vectors.detach().clone()

    def directional_branch() -> tuple[float, float, float]:
        started = time.perf_counter()
        zero = torch.zeros_like(signed[0])
        rows = [zero]
        for step in range(1, horizon):
            rows.append(
                optimizer_center_quadratic_defect(
                    center[step, :dimension],
                    signed[step - 1],
                    train_pairs,
                    train_labels,
                    template,
                    spec,
                    config,
                )
            )
        quadratic = torch.stack(rows)
        response = scalar_green(quadratic.reshape(-1)).reshape(horizon, -1)
        elapsed = time.perf_counter() - started
        return (
            elapsed,
            float(torch.linalg.vector_norm(quadratic)),
            float(torch.linalg.vector_norm(response)),
        )

    def another_power() -> tuple[float, float]:
        started = time.perf_counter()
        rows = green_gram(power_one_vectors.clone())
        elapsed = time.perf_counter() - started
        return elapsed, float(torch.linalg.vector_norm(rows))

    # Warm both branches once before alternating their order.
    warm_directional = directional_branch()
    warm_power = another_power()
    directional_times: list[float] = []
    power_times: list[float] = []
    records = []
    reference_q = None
    reference_y = None
    reference_power = None
    for repeat in range(repeats):
        order = ("directional", "power") if repeat % 2 == 0 else ("power", "directional")
        observed: dict[str, tuple] = {}
        for branch in order:
            observed[branch] = directional_branch() if branch == "directional" else another_power()
        directional_seconds, q_norm, y_norm = observed["directional"]
        power_seconds, power_norm = observed["power"]
        reference_q = q_norm if reference_q is None else reference_q
        reference_y = y_norm if reference_y is None else reference_y
        reference_power = power_norm if reference_power is None else reference_power
        if relative_error(q_norm, reference_q) > 2.0e-15:
            raise RuntimeError("quadratic forcing changed across paired repeats")
        if relative_error(y_norm, reference_y) > 2.0e-15:
            raise RuntimeError("second response changed across paired repeats")
        if relative_error(power_norm, reference_power) > 2.0e-15:
            raise RuntimeError("Gram-power result changed across paired repeats")
        directional_times.append(directional_seconds)
        power_times.append(power_seconds)
        records.append(
            {
                "repeat": repeat + 1,
                "order": list(order),
                "directional_seconds": directional_seconds,
                "additional_gram_power_seconds": power_seconds,
                "paired_speedup": power_seconds / directional_seconds,
            }
        )

    if relative_error(reference_q, float(source_row["quadratic_surrogate_injection_norm"])) > 2.0e-12:
        raise RuntimeError("quadratic forcing differs from the audited source")
    if relative_error(reference_y, float(source_row["quadratic_surrogate_second_response_norm"])) > 2.0e-12:
        raise RuntimeError("second response differs from the audited source")
    payload = {
        "status": "PAIRED TWO-RESPONSE INCREMENTAL BENCHMARK PASSED",
        "evidence_boundary": (
            "Post-seal outcome-blind systems benchmark. It compares incremental "
            "operator choices and does not alter prospective issuance or coverage."
        ),
        "candidate": candidate.__dict__,
        "horizon": horizon,
        "repeats": repeats,
        "warmup_directional_seconds": warm_directional[0],
        "warmup_additional_gram_power_seconds": warm_power[0],
        "median_directional_seconds": statistics.median(directional_times),
        "median_additional_gram_power_seconds": statistics.median(power_times),
        "median_paired_speedup": statistics.median(
            row["paired_speedup"] for row in records
        ),
        "minimum_paired_speedup": min(row["paired_speedup"] for row in records),
        "maximum_paired_speedup": max(row["paired_speedup"] for row in records),
        "quadratic_forcing_norm": reference_q,
        "second_response_norm": reference_y,
        "additional_power_output_norm": reference_power,
        "certificate_sha256": sha256(certificate_path),
        "two_response_source_sha256": sha256(SOURCE),
        "outcome_files_read": 0,
        "records": records,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=366)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--anchor", type=int, default=1040)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    payload = run(
        Candidate(args.seed, args.threshold, args.anchor),
        repeats=args.repeats,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
