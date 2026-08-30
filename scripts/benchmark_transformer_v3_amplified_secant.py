#!/usr/bin/env python3
"""Paired benchmark for amplified secant, third-product, and Gram refinement."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch

from batched_green_operator import make_batched_transformer_green_products
from online_progressive_gram import OnlineGramState
from transformer_certificate_protocol import Candidate
from transformer_four_sweep_development_audit import to_scaled
from transformer_fourth_jet_bound import objective_fourth_derivative_bound
from transformer_green_development_audit import build_frozen_centerline
from transformer_green_operator import make_transformer_green_products
from transformer_hvp_grokking import gradient, objective_hvp
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
FULL_AUDIT = RESULTS / "transformer_v3_amplified_secant_full_audit.json"
OUTPUT = RESULTS / "transformer_v3_amplified_secant_paired_benchmark.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def relative_error(left: float, right: float) -> float:
    return abs(left - right) / max(abs(right), torch.finfo(torch.float64).tiny)


def run(candidate: Candidate, amplification: float, repeats: int) -> dict:
    if repeats < 3:
        raise ValueError("at least three paired repeats are required")
    candidates, horizons, _ = frozen_candidates()
    horizon = int(horizons[candidate])
    certificate_path = output_path(candidate)
    certificate = safe_json(certificate_path)
    source = safe_json(SOURCE)
    source_row = next(
        row
        for row in source["rows"]
        if row.get("candidate") == candidate.__dict__
    )
    full = safe_json(FULL_AUDIT)
    full_row = next(
        row
        for row in full["rows"]
        if float(row["amplification"]) == float(amplification)
    )
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
    mapped = [path["map_step"](center[step]) for step in range(horizon)]
    residual = torch.stack(
        [
            to_scaled(mapped[step], dimension, config.learning_rate)
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

    def secant_branch() -> tuple[float, float, float, float]:
        started = time.perf_counter()
        rows = [torch.zeros_like(signed[0])]
        terms = []
        for step in range(1, horizon):
            direction = signed[step - 1]
            a = direction[:dimension]
            point = center[step, :dimension]
            base_gradient = (
                mapped[step][dimension:]
                - config.momentum * center[step, dimension:]
            )
            hessian_direction = objective_hvp(
                point, a, train_pairs, train_labels, template, spec, config
            )
            shifted_gradient = gradient(
                point + amplification * a,
                train_pairs,
                train_labels,
                template,
                spec,
                config,
            )
            remainder = (
                shifted_gradient - base_gradient - amplification * hessian_direction
            ) / (amplification * amplification)
            scaled = config.learning_rate * remainder
            rows.append(torch.cat((-scaled, scaled)))
            norm = float(torch.linalg.vector_norm(a))
            fourth = objective_fourth_derivative_bound(
                point,
                template,
                spec,
                config,
                radius=max(1.0, amplification) * norm,
            )
            terms.append(
                abs(amplification - 1.0)
                * math.sqrt(2.0)
                * config.learning_rate
                * fourth
                * norm**3
                / 6.0
            )
        q = torch.stack(rows)
        y = scalar_green(q.reshape(-1)).reshape(horizon, -1)
        return (
            time.perf_counter() - started,
            float(torch.linalg.vector_norm(q)),
            float(torch.linalg.vector_norm(y)),
            math.sqrt(sum(term * term for term in terms)),
        )

    def third_branch() -> tuple[float, float, float, float]:
        started = time.perf_counter()
        rows = [torch.zeros_like(signed[0])]
        terms = []
        for step in range(1, horizon):
            direction = signed[step - 1]
            a = direction[:dimension]
            point = center[step, :dimension]
            rows.append(
                optimizer_center_quadratic_defect(
                    point,
                    direction,
                    train_pairs,
                    train_labels,
                    template,
                    spec,
                    config,
                )
            )
            norm = float(torch.linalg.vector_norm(a))
            fourth = objective_fourth_derivative_bound(
                point, template, spec, config, radius=norm
            )
            terms.append(
                math.sqrt(2.0)
                * config.learning_rate
                * fourth
                * norm**3
                / 6.0
            )
        q = torch.stack(rows)
        y = scalar_green(q.reshape(-1)).reshape(horizon, -1)
        return (
            time.perf_counter() - started,
            float(torch.linalg.vector_norm(q)),
            float(torch.linalg.vector_norm(y)),
            math.sqrt(sum(term * term for term in terms)),
        )

    def power_branch() -> tuple[float, float]:
        started = time.perf_counter()
        rows = green_gram(power_one_vectors.clone())
        return time.perf_counter() - started, float(torch.linalg.vector_norm(rows))

    warm = {
        "secant": secant_branch(),
        "third": third_branch(),
        "power": power_branch(),
    }
    records = []
    references: dict[str, tuple[float, ...]] = {}
    orders = [
        ("secant", "third", "power"),
        ("power", "secant", "third"),
        ("third", "power", "secant"),
    ]
    for repeat in range(repeats):
        order = orders[repeat % len(orders)]
        values: dict[str, tuple] = {}
        for name in order:
            values[name] = (
                secant_branch()
                if name == "secant"
                else third_branch()
                if name == "third"
                else power_branch()
            )
        for name, value in values.items():
            checksum = tuple(float(x) for x in value[1:])
            if name not in references:
                references[name] = checksum
            elif any(
                relative_error(left, right) > 2.0e-14
                for left, right in zip(checksum, references[name])
            ):
                raise RuntimeError(f"{name} checksum changed")
        secant_seconds = float(values["secant"][0])
        third_seconds = float(values["third"][0])
        power_seconds = float(values["power"][0])
        records.append(
            {
                "repeat": repeat + 1,
                "order": list(order),
                "secant_seconds": secant_seconds,
                "third_product_seconds": third_seconds,
                "additional_gram_power_seconds": power_seconds,
                "third_over_secant_speedup": third_seconds / secant_seconds,
                "power_over_secant_speedup": power_seconds / secant_seconds,
            }
        )
    secant_times = [row["secant_seconds"] for row in records]
    third_times = [row["third_product_seconds"] for row in records]
    power_times = [row["additional_gram_power_seconds"] for row in records]
    if relative_error(references["secant"][0], float(full_row["secant_injection_norm"])) > 2.0e-12:
        raise RuntimeError("secant forcing differs from full audit")
    if relative_error(references["secant"][1], float(full_row["secant_response_norm"])) > 2.0e-12:
        raise RuntimeError("secant response differs from full audit")
    if relative_error(references["third"][0], float(source_row["quadratic_surrogate_injection_norm"])) > 2.0e-12:
        raise RuntimeError("third-product forcing differs from source audit")
    payload = {
        "status": "PAIRED AMPLIFIED-SECANT BENCHMARK PASSED",
        "evidence_boundary": (
            "Post-seal outcome-blind systems benchmark. All three branches include "
            "their incremental work after the shared centerline, first response, "
            "and first Gram power; it does not alter prospective evidence."
        ),
        "candidate": candidate.__dict__,
        "horizon": horizon,
        "amplification": amplification,
        "repeats": repeats,
        "warmup_seconds": {name: float(value[0]) for name, value in warm.items()},
        "median_secant_seconds": statistics.median(secant_times),
        "median_third_product_seconds": statistics.median(third_times),
        "median_additional_gram_power_seconds": statistics.median(power_times),
        "median_paired_third_over_secant_speedup": statistics.median(
            row["third_over_secant_speedup"] for row in records
        ),
        "median_paired_power_over_secant_speedup": statistics.median(
            row["power_over_secant_speedup"] for row in records
        ),
        "minimum_power_over_secant_speedup": min(
            row["power_over_secant_speedup"] for row in records
        ),
        "maximum_power_over_secant_speedup": max(
            row["power_over_secant_speedup"] for row in records
        ),
        "checksums": {name: list(value) for name, value in references.items()},
        "certificate_sha256": sha256(certificate_path),
        "full_audit_sha256": sha256(FULL_AUDIT),
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
    parser.add_argument("--amplification", type=float, default=4096.0)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    payload = run(
        Candidate(args.seed, args.threshold, args.anchor),
        amplification=float(args.amplification),
        repeats=int(args.repeats),
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
