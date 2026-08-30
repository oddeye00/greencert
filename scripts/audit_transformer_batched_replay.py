#!/usr/bin/env python3
"""Post-seal end-to-end replay with block-batched random probes.

The replay rebuilds the frozen centerline and every analytic envelope.  It then
evaluates the original 16-by-8 Gaussian output and Green probes as batched
reverse products, using the exact committed seeds.  Existing certificates and
caches are read for comparison and are never modified.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from batched_green_operator import (
    batched_gram_norm_bound,
    make_batched_output_gram_operator,
    make_batched_transformer_green_products,
)
from probe_jacobian_bound import namespaced_probe_seed
from transformer_block_envelope import ball_valid_envelope, objective_hessian_lipschitz
from transformer_certificate_protocol import Candidate
from transformer_four_sweep_development_audit import (
    count_envelope,
    first_persistent,
    persistent_bracket,
    to_scaled,
)
from transformer_green_confirmation_certificate import (
    NUMERIC_RADIUS_CAP,
    ROOT,
    build_frozen_centerline,
    frozen_candidates,
    gate_slacks,
    load_candidate,
    output_path,
    persistent_certificate_slack,
    safe_json,
    sha256,
    verify_method_seal,
)
from transformer_green_confirmation_protocol import (
    MASTER_NONCE,
    PERSISTENCE,
    green_identity,
    output_identity,
    probe_config,
)
from transformer_green_operator import make_transformer_green_products
from transformer_hvp_grokking import logits


RESULTS = ROOT / "results"


def relative_scalar(left: float, right: float) -> float:
    return abs(left - right) / max(abs(right), 1e-300)


def destination(candidate: Candidate) -> Path:
    return RESULTS / (
        f"transformer_batched_replay_seed_{candidate.seed}_"
        f"gate_{candidate.gate_index}_anchor_{candidate.anchor}.json"
    )


def audit(candidate: Candidate) -> dict:
    method_seal = verify_method_seal()
    candidates, horizons, candidate_seal = frozen_candidates()
    if candidate not in horizons:
        raise ValueError(f"candidate is outside the sealed population: {candidate}")
    original_path = output_path(candidate)
    original = safe_json(original_path)
    horizon = horizons[candidate]
    probe = probe_config()
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()

    config, template, spec, data, parameter, velocity = load_candidate(candidate)
    torch.set_num_threads(config.threads)
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = data
    center_started = time.perf_counter()
    path = build_frozen_centerline(
        config,
        template,
        spec,
        train_pairs,
        train_labels,
        parameter,
        velocity,
    )
    center_seconds = time.perf_counter() - center_started
    center = path["center"]
    scaled_center = path["scaled_center"]
    dimension = parameter.numel()
    center_counts = np.asarray(
        [
            int(
                (
                    logits(state[:dimension], cert_pairs, template, spec).argmax(1)
                    == cert_labels
                ).sum()
            )
            for state in center
        ],
        dtype=np.int64,
    )
    required = int(math.ceil(candidate.threshold * len(cert_pairs)))
    predicted_event = first_persistent(center_counts, required)
    if predicted_event is None or predicted_event + PERSISTENCE - 1 != horizon:
        raise RuntimeError("reconstructed centerline does not match frozen event")

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
    signed_response = scalar_green(residual.reshape(-1)).reshape(horizon, -1)
    response_norm = float(torch.linalg.vector_norm(signed_response))
    radius = 2.0 * response_norm
    if not math.isfinite(radius) or radius > NUMERIC_RADIUS_CAP:
        raise RuntimeError("replayed signed radius is unusable")

    all_pairs = torch.cartesian_prod(
        torch.arange(config.modulus), torch.arange(config.modulus)
    ).long()
    logits_zero = logits(center[0, :dimension], cert_pairs, template, spec)
    zero_guarantee, zero_exclusion = gate_slacks(
        logits_zero, cert_labels, 0.0, required
    )
    guaranteed = [int(center_counts[0])]
    possible = [int(center_counts[0])]
    guarantee_slacks = [zero_guarantee]
    exclusion_slacks = [zero_exclusion]
    maximum_map_drift = 0.0
    fixed_points_consistent = True
    output_rows = []
    output_seconds = 0.0
    original_geometry = {int(row["step"]): row for row in original["geometry"]}

    for step in range(1, horizon + 1):
        theta = center[step, :dimension]
        output_apply = make_batched_output_gram_operator(theta, all_pairs, template, spec)
        probe_started = time.perf_counter()
        output_probe = batched_gram_norm_bound(
            output_apply,
            dimension=dimension,
            dtype=theta.dtype,
            device=theta.device,
            config=probe,
            seed=namespaced_probe_seed(MASTER_NONCE, output_identity(candidate, step)),
        )
        output_seconds += time.perf_counter() - probe_started
        block = ball_valid_envelope(
            theta,
            spec,
            config,
            epsilon=radius,
            exact_values=True,
            sphere=True,
        )
        fixed_points_consistent &= bool(block["fixed_point_consistent"])
        output_upper = float(output_probe["operator_norm_upper_bound"])
        first_ball = output_upper + block["second"] * radius
        objective_lipschitz = objective_hessian_lipschitz(
            first_ball, block["second"], block["third"]
        )
        map_drift = math.sqrt(2.0) * config.learning_rate * objective_lipschitz
        if step < horizon:
            maximum_map_drift = max(maximum_map_drift, map_drift)
        margin_radius = math.sqrt(2.0) * (
            output_upper * radius + 0.5 * block["second"] * radius * radius
        )
        center_logits = logits(theta, cert_pairs, template, spec)
        lower_count, upper_count = count_envelope(
            center_logits, cert_labels, margin_radius
        )
        guarantee_slack, exclusion_slack = gate_slacks(
            center_logits, cert_labels, margin_radius, required
        )
        guaranteed.append(lower_count)
        possible.append(upper_count)
        guarantee_slacks.append(guarantee_slack)
        exclusion_slacks.append(exclusion_slack)
        old_probe = original_geometry[step]["output_probe"]
        output_rows.append({
            "step": step,
            "batched_upper": output_upper,
            "scalar_upper": float(old_probe["jacobian_norm_upper_bound"]),
            "upper_relative_error": relative_scalar(
                output_upper, float(old_probe["jacobian_norm_upper_bound"])
            ),
            "Y_relative_error": relative_scalar(
                float(output_probe["Y"]), float(old_probe["Y"])
            ),
            "margin_radius": margin_radius,
        })

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

    green_started = time.perf_counter()
    green_probe = batched_gram_norm_bound(
        green_gram,
        dimension=horizon * 2 * dimension,
        dtype=parameter.dtype,
        device=parameter.device,
        config=probe,
        seed=namespaced_probe_seed(MASTER_NONCE, green_identity(candidate, horizon)),
    )
    green_seconds = time.perf_counter() - green_started
    kappa = float(green_probe["operator_norm_upper_bound"])
    closure_lhs = 2.0 * kappa * maximum_map_drift * response_norm
    closure_passed = fixed_points_consistent and closure_lhs <= 1.0
    raw_bracket = persistent_bracket(
        np.asarray(guaranteed, dtype=np.int64),
        np.asarray(possible, dtype=np.int64),
        required,
    )
    bracket = raw_bracket if closure_passed else None
    certificate_slack = persistent_certificate_slack(
        bracket, guarantee_slacks, exclusion_slacks
    )
    elapsed = time.perf_counter() - started
    old_green = original["green_probe"]
    result = {
        "status": "post-seal exact-probe block-batched replay",
        "candidate": candidate.__dict__,
        "horizon": horizon,
        "method_seal_sha256": sha256(ROOT / "TRANSFORMER_GREEN_CONFIRMATION_METHOD_SEAL.json"),
        "candidate_seal_sha256": sha256(ROOT / "TRANSFORMER_GREEN_CONFIRMATION_CANDIDATE_SEAL.json"),
        "original_certificate": str(original_path.relative_to(ROOT)),
        "original_certificate_sha256": sha256(original_path),
        "probe_config": probe.__dict__,
        "same_committed_random_streams": True,
        "same_probability_budget": True,
        "same_centerline_sha256": path["centerline_sha256"] == original["centerline_sha256"],
        "predicted_event": predicted_event,
        "signed_radius": radius,
        "green": {
            "batched_upper": kappa,
            "scalar_upper": float(old_green["green_operator_norm_upper_bound"]),
            "upper_relative_error": relative_scalar(
                kappa, float(old_green["green_operator_norm_upper_bound"])
            ),
            "Y_relative_error": relative_scalar(
                float(green_probe["Y"]), float(old_green["Y"])
            ),
        },
        "maximum_output_upper_relative_error": max(
            row["upper_relative_error"] for row in output_rows
        ),
        "maximum_output_Y_relative_error": max(
            row["Y_relative_error"] for row in output_rows
        ),
        "closure_lhs": closure_lhs,
        "original_closure_lhs": original["closure_lhs_2_kappa_M_Z"],
        "closure_relative_error": relative_scalar(
            closure_lhs, float(original["closure_lhs_2_kappa_M_Z"])
        ),
        "certified_bracket": bracket,
        "original_certified_bracket": original["certified_bracket"],
        "certificate_output_logic_slack": certificate_slack,
        "exact_disposition_match": bracket == original["certified_bracket"],
        "timings_seconds": {
            "centerline_and_sweeps": center_seconds,
            "batched_output_probes": output_seconds,
            "batched_green_probe": green_seconds,
            "total": elapsed,
            "original_serial_total": float(original["elapsed_seconds"]),
            "end_to_end_speedup": float(original["elapsed_seconds"]) / elapsed,
        },
        "output_rows": output_rows,
        "method_seal_status": method_seal["status"],
        "candidate_seal_status": candidate_seal["status"],
    }
    if not result["same_centerline_sha256"]:
        raise RuntimeError("centerline hash changed")
    if not result["exact_disposition_match"]:
        raise RuntimeError("batched replay changed the certificate disposition")
    if max(
        result["green"]["upper_relative_error"],
        result["maximum_output_upper_relative_error"],
        result["closure_relative_error"],
    ) >= 5e-11:
        raise RuntimeError("batched replay differs numerically from scalar certificate")
    out = destination(candidate)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite batched replay: {out}")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["output"] = str(out)
    result["sha256"] = sha256(out)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--anchor", type=int, required=True)
    args = parser.parse_args()
    result = audit(Candidate(args.seed, args.threshold, args.anchor))
    print(json.dumps({
        "candidate": result["candidate"],
        "horizon": result["horizon"],
        "bracket": result["certified_bracket"],
        "maximum_relative_error": max(
            result["green"]["upper_relative_error"],
            result["maximum_output_upper_relative_error"],
            result["closure_relative_error"],
        ),
        "timings_seconds": result["timings_seconds"],
        "output": result["output"],
        "sha256": result["sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
