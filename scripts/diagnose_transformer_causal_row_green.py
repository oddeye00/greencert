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
from batched_green_operator import (
    make_batched_causal_green_products,
    make_batched_scaled_optimizer_products,
    make_batched_transformer_green_products,
)
from causal_row_green import (
    causal_row_quadratic_envelope,
    rowwise_signed_affine_bounds,
    simultaneous_row_direct_image_bounds,
)
from corrected_path_closure import exact_corrected_path_closure
from transformer_block_envelope import ball_valid_envelope, objective_hessian_lipschitz
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
    closure_channel: str = "full_state",
    probe_chunk_size: int | None = None,
    probe_stream_size: int | None = None,
    probe_offset: int = 0,
) -> dict:
    if defect_route not in {"quadratic", "response_free"}:
        raise ValueError("unknown defect route")
    if closure_channel not in {"full_state", "structured_parameter"}:
        raise ValueError("unknown closure channel")
    if closure_channel == "structured_parameter" and defect_route != "quadratic":
        raise ValueError("structured parameter closure requires the quadratic route")
    if probe_chunk_size is not None and probe_chunk_size < 1:
        raise ValueError("probe_chunk_size must be positive")
    stream_size = probes if probe_stream_size is None else int(probe_stream_size)
    if stream_size < probes or probe_offset < 0 or probe_offset + probes > stream_size:
        raise ValueError("probe block lies outside the declared probe stream")
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
    gradient_forcing_errors = [0.0]
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
        gradient_forcing_errors.append(local_error)
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
    parameter_curvature = [0.0] + [
        objective_hessian_lipschitz(
            first=float(blocks[step - 1]["first"]),
            second=float(blocks[step - 1]["second"]),
            third=float(blocks[step - 1]["third"]),
        )
        for step in range(1, horizon)
    ]
    timing["corrected_path_neural_jets"] = time.perf_counter() - phase

    phase = time.perf_counter()
    generator = torch.Generator(device=center.device).manual_seed(
        probe_seed(candidate, sweeps, stream_size)
    )
    chunk_size = probes if probe_chunk_size is None else min(probes, probe_chunk_size)
    green_batches = 0

    def evaluate_probe_chunks(
        apply,
        probe_rows: torch.Tensor,
        known_row: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        nonlocal green_batches
        image_blocks = []
        known_image = None
        for start in range(0, probes, chunk_size):
            stop = min(probes, start + chunk_size)
            rows = probe_rows[start:stop]
            append_known = start == 0 and known_row is not None
            if append_known:
                rows = torch.cat((rows, known_row), dim=0)
            images = apply(rows).reshape(rows.shape[0], horizon, 2 * dimension)
            image_blocks.append(images[: stop - start])
            if append_known:
                known_image = images[-1]
            green_batches += 1
        return torch.cat(image_blocks, dim=0), known_image

    if closure_channel == "full_state":
        batch_apply, _ = make_batched_transformer_green_products(
            corrected[:horizon, :dimension],
            train_pairs,
            train_labels,
            template,
            spec,
            config,
        )
        if probe_offset:
            torch.randn(
                probe_offset,
                horizon * 2 * dimension,
                generator=generator,
                dtype=center.dtype,
                device=center.device,
            )
        gaussian = torch.randn(
            probes,
            horizon * 2 * dimension,
            generator=generator,
            dtype=center.dtype,
            device=center.device,
        )
        combined_images, known_image = evaluate_probe_chunks(
            batch_apply,
            gaussian,
            quadratic.reshape(1, -1) if defect_route == "quadratic" else None,
        )
        probe_images = combined_images
        signed_second_response = (
            known_image
            if defect_route == "quadratic"
            else torch.zeros_like(probe_images[0])
        )
        active_forcing_errors = forcing_errors
        active_curvature = curvature
        quadratic_norm = float(torch.linalg.vector_norm(quadratic))
    else:
        channel_products = [
            make_batched_scaled_optimizer_products(
                corrected[step, :dimension],
                train_pairs,
                train_labels,
                template,
                spec,
                config,
            )
            for step in range(horizon)
        ]
        channel_apply, _ = make_batched_causal_green_products(
            [row[0] for row in channel_products],
            [row[1] for row in channel_products],
            2 * dimension,
        )
        if probe_offset:
            torch.randn(
                probe_offset,
                horizon,
                dimension,
                generator=generator,
                dtype=center.dtype,
                device=center.device,
            )
        gaussian_parameter = torch.randn(
            probes,
            horizon,
            dimension,
            generator=generator,
            dtype=center.dtype,
            device=center.device,
        )
        gaussian_state = torch.cat(
            (-eta * gaussian_parameter, eta * gaussian_parameter), dim=2
        ).reshape(probes, -1)
        # The corrected-path defect is N(z)-d.  Retain both the directional
        # quadratic term and the signed recurrence residual; only the mixed
        # fourth-order remainder is scalarized.
        known_forcing = quadratic - recurrence_rows
        state_images, known_image = evaluate_probe_chunks(
            channel_apply,
            gaussian_state,
            known_forcing.reshape(1, -1),
        )
        if known_image is None:
            raise RuntimeError("structured signed forcing response was not evaluated")
        probe_images = state_images[:, :, :dimension]
        signed_second_response = known_image[:, :dimension]
        active_forcing_errors = gradient_forcing_errors
        active_curvature = parameter_curvature
        quadratic_norm = float(
            torch.linalg.vector_norm(quadratic[:, dimension:] / eta)
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
        signed_second_response, row_gains, active_forcing_errors
    )
    radii = causal_row_quadratic_envelope(affine, row_gains, active_curvature)
    row_domain_passed = domain_geometry and bool((radii <= domain).all())

    forcing_error_norm = math.sqrt(
        math.fsum(value * value for value in active_forcing_errors)
    )
    old_global_response = global_gain * (quadratic_norm + forcing_error_norm)
    signed_global_response = float(
        torch.linalg.vector_norm(signed_second_response)
    ) + global_gain * forcing_error_norm
    maximum_curvature = max(active_curvature, default=0.0)
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
    required = int(certificate["required_correct"])
    raw = [
        raw_slacks(
            logits(corrected[step, :dimension], cert_pairs, template, spec),
            cert_labels,
            required,
        )
        for step in range(horizon + 1)
    ]
    output_first_bounds = [0.0] + [
        float(blocks[step - 1]["first"]) for step in range(1, horizon + 1)
    ]
    if row_domain_passed:
        margins.extend(
            logit_margin_radius(
                first=output_first_bounds[step],
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
        "closure_channel": closure_channel,
        "horizon": horizon,
        "probes": probes,
        "probe_stream_size": stream_size,
        "probe_offset": probe_offset,
        "probe_chunk_size": chunk_size,
        "sequential_green_batches": green_batches,
        "probe_seed": probe_seed(candidate, sweeps, stream_size),
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
        "row_gain_bounds": row_gains.tolist(),
        "row_image_maxima": torch.linalg.vector_norm(
            probe_images, dim=2
        ).max(dim=0).values.tolist(),
        "row_calibration": row_audit,
        "old_global_response_upper": old_global_response,
        "signed_global_response_upper": signed_global_response,
        "old_global_closure": old_global_closure.as_dict(),
        "signed_global_closure": signed_global_closure.as_dict(),
        "row_affine_bounds": affine.tolist(),
        "row_radii": radii.tolist(),
        "active_curvature_bounds": list(active_curvature),
        "active_forcing_error_bounds": list(active_forcing_errors),
        "signed_response_row_norms": torch.linalg.vector_norm(
            signed_second_response, dim=1
        ).tolist(),
        "maximum_row_radius": float(radii.max()),
        "domain_radius": domain,
        "row_domain_passed": row_domain_passed,
        "issued": issued,
        "bracket": bracket,
        "logic_slack": logic_slack,
        "sealed_four_sweep_bracket": source["sealed_four_sweep_bracket"],
        "retains_sealed_bracket": issued and bracket == source["sealed_four_sweep_bracket"],
        "maximum_margin_radius": max(margins),
        "raw_event_slacks": [[float(left), float(right)] for left, right in raw],
        "output_first_derivative_bounds": output_first_bounds,
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
    parser.add_argument("--probe-chunk-size", type=int)
    parser.add_argument("--probe-stream-size", type=int)
    parser.add_argument("--probe-offset", type=int, default=0)
    parser.add_argument("--family-delta", type=float, default=FAMILY_DELTA)
    parser.add_argument(
        "--defect-route",
        choices=("quadratic", "response_free"),
        default="quadratic",
    )
    parser.add_argument(
        "--closure-channel",
        choices=("full_state", "structured_parameter"),
        default="full_state",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.probes < 1:
        raise ValueError("probes must be positive")
    candidate = Candidate(args.seed, args.threshold, args.anchor)
    result = run(
        candidate,
        args.sweeps,
        args.probes,
        args.defect_route,
        family_delta=args.family_delta,
        closure_channel=args.closure_channel,
        probe_chunk_size=args.probe_chunk_size,
        probe_stream_size=args.probe_stream_size,
        probe_offset=args.probe_offset,
    )
    destination = args.output or RESULTS / (
        f"transformer_causal_row_green_seed_{candidate.seed}_"
        f"gate_{candidate.gate_index}_anchor_{candidate.anchor}_"
        f"{args.defect_route}_{args.closure_channel}_"
        f"s{args.sweeps}_"
        f"m{args.probes}_offset{args.probe_offset}.json"
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
            "closure_channel",
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
