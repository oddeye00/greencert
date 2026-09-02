#!/usr/bin/env python3
"""Outcome-blind development audit of causal row-Green closure.

The shortest disclosed development Transformer case is replayed with either a
two- or three-sweep reference. The ordinary Gaussian direct-image batch is
reused to obtain every output-time row bound; one deterministic quadratic
forcing row is appended to the same batch. No future training outcome is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
import time

import torch

from analytic_jet_release import logit_margin_radius, scaled_momentum_jacobian_drift
from audit_transformer_adaptive_sweep_cohort import first_persistent, raw_slacks, scaled, unscaled
from batched_green_operator import make_batched_transformer_green_products
from causal_row_green import (
    causal_row_quadratic_envelope,
    rowwise_signed_affine_bounds,
    simultaneous_row_direct_image_bounds,
)
from corrected_path_closure import exact_corrected_path_closure
from transformer_block_envelope import ball_valid_envelope
from transformer_certificate_protocol import Candidate
from transformer_hvp_grokking import logits
from transformer_mixed_directional_jet_v2 import mixed_directional_objective_bounds
from transformer_modal_forecast import optimizer_map
from transformer_optimizer_probe import make_scaled_optimizer_jvp_vjp
from streaming_variational_centerline import build_streaming_transformer_centerline
from transformer_two_response import optimizer_center_quadratic_defect
from transformer_v3_certificate import load_candidate, output_path, safe_json


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PARENT = RESULTS / "transformer_fully_recentered_three_sweep_audit.json"
RELEASED_STRUCTURED = (
    RESULTS / "anchor_fixed_structured_parameter_green_transformer_audit.json"
)
DEVELOPMENT_CANDIDATE = Candidate(366, 0.8, 1120)
PERSISTENCE = 25
FAMILY_DELTA = 1.0e-6 / 15.0
NONCE = "greencert/causal-row-development-v1/6cf1373b"


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes(order="C")
    ).hexdigest().upper()


def probe_seed(candidate: Candidate, sweeps: int, probes: int) -> int:
    payload = (
        f"{NONCE}|{candidate.seed}|{candidate.threshold}|{candidate.anchor}|"
        f"{sweeps}|{probes}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def parent_row(candidate: Candidate) -> dict:
    parent = safe_json(PARENT)
    for row in parent["rows"]:
        coordinates = row["candidate"]
        if (
            int(coordinates["seed"]) == candidate.seed
            and float(coordinates["threshold"]) == candidate.threshold
            and int(coordinates["anchor"]) == candidate.anchor
        ):
            return row
    raise RuntimeError(f"candidate is absent from the parent: {candidate}")


def released_structured_row(candidate: Candidate) -> dict:
    payload = safe_json(RELEASED_STRUCTURED)
    for row in payload["rows"]:
        if row["candidate"] == candidate.__dict__:
            return row
    raise RuntimeError(f"candidate is absent from the structured audit: {candidate}")


def persistent_bracket(
    guarantee: list[float], exclusion: list[float]
) -> tuple[list[int] | None, float | None]:
    lower = first_persistent([value <= 0.0 for value in exclusion])
    upper = first_persistent([value > 0.0 for value in guarantee])
    if lower is None or upper is None or lower > upper:
        return None, None
    prior = [max(exclusion[start : start + PERSISTENCE]) for start in range(lower)]
    lower_slack = math.inf if not prior else min(prior)
    upper_slack = min(guarantee[upper : upper + PERSISTENCE])
    return [lower, upper], min(lower_slack, upper_slack)


def run(
    candidate: Candidate,
    sweeps: int,
    probes: int,
    defect_route: str,
    *,
    family_delta: float = FAMILY_DELTA,
) -> dict:
    if defect_route not in {"quadratic", "response_free"}:
        raise ValueError("unknown defect route")
    started = time.perf_counter()
    timing: dict[str, float] = {}
    source = parent_row(candidate)
    certificate = safe_json(output_path(candidate))
    if bool(certificate.get("outcome_joined", False)):
        raise RuntimeError("certificate artifact unexpectedly contains joined outcomes")
    config, template, spec, data, parameter, velocity = load_candidate(candidate)
    torch.set_num_threads(config.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = data
    horizon = int(source["horizon"])
    dimension = int(parameter.numel())
    eta = float(config.learning_rate)
    domain = float(source["domain_radius_about_corrected_path"])

    phase = time.perf_counter()
    path = build_streaming_transformer_centerline(
        config,
        template,
        spec,
        train_pairs,
        train_labels,
        parameter,
        velocity,
        maximum_horizon=horizon,
        sweeps=sweeps,
    )
    center = path["center"]
    scaled_center = path["scaled_center"]
    pipeline = path["diagnostics"]
    timing["streaming_centerline"] = time.perf_counter() - phase

    phase = time.perf_counter()
    mapped = [
        optimizer_map(row, train_pairs, train_labels, template, spec, config)
        for row in center[:-1]
    ]
    residual = torch.stack(
        [
            scaled(mapped[step], dimension, eta) - scaled_center[step + 1]
            for step in range(horizon)
        ]
    )
    products = [
        make_scaled_optimizer_jvp_vjp(
            center[step, :dimension],
            train_pairs,
            train_labels,
            template,
            spec,
            config,
        )
        for step in range(horizon)
    ]
    corrections = []
    recurrence_errors = []
    prior = torch.zeros_like(residual[0])
    for step in range(horizon):
        linear = products[step][0](prior)
        current = linear + residual[step]
        corrections.append(current)
        recurrence_errors.append(current - linear - residual[step])
        prior = current
    correction_rows = torch.stack(corrections)
    recurrence_rows = torch.stack(recurrence_errors)
    correction = torch.cat((torch.zeros_like(correction_rows[:1]), correction_rows))
    corrected_scaled = scaled_center + correction
    corrected = unscaled(corrected_scaled, dimension, eta)
    timing["signed_first_response"] = time.perf_counter() - phase

    phase = time.perf_counter()
    quadratic_rows = [torch.zeros_like(correction_rows[0])]
    forcing_errors = [float(torch.linalg.vector_norm(recurrence_rows[0]))]
    mixed_rows = []
    quadratic_seconds = 0.0
    mixed_seconds = 0.0
    for step in range(1, horizon):
        if defect_route == "quadratic":
            local_started = time.perf_counter()
            quadratic_rows.append(
                optimizer_center_quadratic_defect(
                    center[step, :dimension],
                    correction[step],
                    train_pairs,
                    train_labels,
                    template,
                    spec,
                    config,
                )
            )
            quadratic_seconds += time.perf_counter() - local_started
        else:
            quadratic_rows.append(torch.zeros_like(correction_rows[0]))
        local_started = time.perf_counter()
        mixed = mixed_directional_objective_bounds(
            center[step, :dimension], correction[step, :dimension], spec, config
        )
        mixed_seconds += time.perf_counter() - local_started
        if defect_route == "quadratic":
            local_error = float(mixed["gradient_taylor_remainder_upper"])
        else:
            local_error = float(mixed["gradient_nonlinear_remainder_upper"])
        directional_error = math.sqrt(2.0) * eta * local_error
        recurrence = float(torch.linalg.vector_norm(recurrence_rows[step]))
        forcing_errors.append(directional_error + recurrence)
        mixed_rows.append(
            {
                "step": step,
                "forcing_error_upper": directional_error + recurrence,
                "direction_norm": float(
                    torch.linalg.vector_norm(correction[step, :dimension])
                ),
                "mixed_third_derivative_upper": float(
                    mixed["mixed_third_derivative_upper"]
                ),
                "mixed_fourth_derivative_upper": float(
                    mixed["mixed_fourth_derivative_upper"]
                ),
            }
        )
    quadratic = torch.stack(quadratic_rows)
    timing["quadratic_contractions"] = quadratic_seconds
    timing["mixed_directional_jets"] = mixed_seconds
    timing["defect_construction_total"] = time.perf_counter() - phase

    phase = time.perf_counter()
    blocks = [
        ball_valid_envelope(
            corrected[step, :dimension],
            spec,
            config,
            epsilon=domain,
            exact_values=True,
            sphere=True,
        )
        for step in range(1, horizon + 1)
    ]
    domain_geometry = all(bool(block["fixed_point_consistent"]) for block in blocks)
    curvature = [0.0] + [
        scaled_momentum_jacobian_drift(
            first=float(blocks[step - 1]["first"]),
            second=float(blocks[step - 1]["second"]),
            third=float(blocks[step - 1]["third"]),
            learning_rate=eta,
        )
        for step in range(1, horizon)
    ]
    timing["corrected_path_neural_jets"] = time.perf_counter() - phase

    phase = time.perf_counter()
    batch_apply, _ = make_batched_transformer_green_products(
        corrected[:horizon, :dimension],
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )
    generator = torch.Generator(device=center.device).manual_seed(
        probe_seed(candidate, sweeps, probes)
    )
    gaussian = torch.randn(
        probes,
        horizon * 2 * dimension,
        generator=generator,
        dtype=center.dtype,
        device=center.device,
    )
    if defect_route == "quadratic":
        combined = torch.cat((gaussian, quadratic.reshape(1, -1)), dim=0)
    else:
        combined = gaussian
    combined_images = batch_apply(combined).reshape(
        probes + (1 if defect_route == "quadratic" else 0),
        horizon,
        2 * dimension,
    )
    probe_images = combined_images[:probes]
    signed_second_response = (
        combined_images[probes]
        if defect_route == "quadratic"
        else torch.zeros_like(combined_images[0])
    )
    row_gains, row_audit = simultaneous_row_direct_image_bounds(
        probe_images, family_delta=family_delta
    )
    global_calibration = NormalDist().inv_cdf(
        0.5 * (1.0 + family_delta ** (1.0 / probes))
    )
    global_image_norms = torch.linalg.vector_norm(
        probe_images.reshape(probes, -1), dim=1
    )
    global_gain = float(global_image_norms.max()) / global_calibration
    timing["one_batched_green_pass"] = time.perf_counter() - phase

    phase = time.perf_counter()
    affine = rowwise_signed_affine_bounds(
        signed_second_response, row_gains, forcing_errors
    )
    radii = causal_row_quadratic_envelope(affine, row_gains, curvature)
    row_domain_passed = domain_geometry and bool((radii <= domain).all())

    quadratic_norm = float(torch.linalg.vector_norm(quadratic))
    forcing_error_norm = math.sqrt(math.fsum(value * value for value in forcing_errors))
    old_global_response = global_gain * (quadratic_norm + forcing_error_norm)
    signed_global_response = float(
        torch.linalg.vector_norm(signed_second_response)
    ) + global_gain * forcing_error_norm
    maximum_curvature = max(curvature, default=0.0)
    old_global_closure = exact_corrected_path_closure(
        kappa=global_gain,
        derivative_drift=maximum_curvature,
        defect_response_bound=old_global_response,
        domain_radius=domain,
    )
    signed_global_closure = exact_corrected_path_closure(
        kappa=global_gain,
        derivative_drift=maximum_curvature,
        defect_response_bound=signed_global_response,
        domain_radius=domain,
    )

    bracket = None
    logic_slack = None
    margins = [0.0]
    if row_domain_passed:
        required = int(certificate["required_correct"])
        raw = [
            raw_slacks(
                logits(corrected[step, :dimension], cert_pairs, template, spec),
                cert_labels,
                required,
            )
            for step in range(horizon + 1)
        ]
        margins.extend(
            logit_margin_radius(
                first=float(blocks[step - 1]["first"]),
                state_radius=float(radii[step - 1]),
            )
            for step in range(1, horizon + 1)
        )
        guarantee = [float(pair[0]) - margin for pair, margin in zip(raw, margins)]
        exclusion = [float(pair[1]) - margin for pair, margin in zip(raw, margins)]
        bracket, logic_slack = persistent_bracket(guarantee, exclusion)
    issued = bracket is not None and logic_slack is not None and logic_slack > 0.0
    timing["closures_and_event"] = time.perf_counter() - phase
    timing["end_to_end"] = time.perf_counter() - started

    released_row = released_structured_row(candidate) if sweeps == 4 else None
    return {
        "status": "outcome-blind causal row-Green candidate audit complete",
        "evidence_boundary": (
            "No future trajectory or event outcome read. "
            "Gaussian claims use the same ideal-PRNG model as the Transformer paper."
        ),
        "candidate": candidate.__dict__,
        "sweeps": sweeps,
        "defect_route": defect_route,
        "horizon": horizon,
        "probes": probes,
        "probe_seed": probe_seed(candidate, sweeps, probes),
        "family_delta": family_delta,
        "parameter_count": dimension,
        "centerline_sha256": tensor_sha256(scaled_center),
        "corrected_path_sha256": tensor_sha256(corrected_scaled),
        "parent_corrected_path_match": (
            tensor_sha256(corrected_scaled) == source["corrected_path_sha256"]
            if sweeps == 3
            else None
        ),
        "released_corrected_path_match": (
            tensor_sha256(corrected_scaled) == released_row["corrected_path_sha256"]
            if released_row is not None
            else None
        ),
        "pipeline_diagnostics": pipeline,
        "quadratic_forcing_norm": quadratic_norm,
        "forcing_error_sequence_norm": forcing_error_norm,
        "signed_second_response_norm": float(
            torch.linalg.vector_norm(signed_second_response)
        ),
        "global_gain_upper": global_gain,
        "row_gain_minimum": float(row_gains.min()),
        "row_gain_median": float(row_gains.median()),
        "row_gain_maximum": float(row_gains.max()),
        "row_calibration": row_audit,
        "old_global_response_upper": old_global_response,
        "signed_global_response_upper": signed_global_response,
        "old_global_closure": old_global_closure.as_dict(),
        "signed_global_closure": signed_global_closure.as_dict(),
        "row_affine_bounds": affine.tolist(),
        "row_radii": radii.tolist(),
        "maximum_row_radius": float(radii.max()),
        "domain_radius": domain,
        "row_domain_passed": row_domain_passed,
        "issued": issued,
        "bracket": bracket,
        "logic_slack": logic_slack,
        "sealed_four_sweep_bracket": source["sealed_four_sweep_bracket"],
        "retains_sealed_bracket": issued and bracket == source["sealed_four_sweep_bracket"],
        "maximum_margin_radius": max(margins),
        "mixed_forcing_rows": mixed_rows,
        "timings_seconds": timing,
        "additional_green_jvp_passes_vs_direct_image": 0,
        "additional_batched_rows_vs_direct_image": (
            1 if defect_route == "quadratic" else 0
        ),
        "sequential_hvp_depth": sweeps + 3,
        "outcome_files_read": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_CANDIDATE.seed)
    parser.add_argument(
        "--threshold", type=float, default=DEVELOPMENT_CANDIDATE.threshold
    )
    parser.add_argument("--anchor", type=int, default=DEVELOPMENT_CANDIDATE.anchor)
    parser.add_argument("--sweeps", type=int, choices=(2, 3, 4), default=3)
    parser.add_argument("--probes", type=int, default=4)
    parser.add_argument(
        "--defect-route",
        choices=("quadratic", "response_free"),
        default="quadratic",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.probes < 1:
        raise ValueError("probes must be positive")
    candidate = Candidate(args.seed, args.threshold, args.anchor)
    result = run(candidate, args.sweeps, args.probes, args.defect_route)
    destination = args.output or RESULTS / (
        f"transformer_causal_row_green_seed_{candidate.seed}_"
        f"gate_{candidate.gate_index}_anchor_{candidate.anchor}_"
        f"{args.defect_route}_"
        f"s{args.sweeps}_"
        f"m{args.probes}.json"
    )
    destination.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    summary = {
        key: result[key]
        for key in (
            "status",
            "candidate",
            "sweeps",
            "defect_route",
            "probes",
            "old_global_closure",
            "signed_global_closure",
            "maximum_row_radius",
            "domain_radius",
            "row_domain_passed",
            "issued",
            "bracket",
            "retains_sealed_bracket",
            "timings_seconds",
            "outcome_files_read",
        )
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
