#!/usr/bin/env python3
"""Outcome-blind low-rank union-subspace causal-resolvent diagnostic.

This development audit computes the approximate Green operator exactly in the
union of the segment sketch spaces, retaining signed momentum cancellation.
Exact checkpoint Hessians are used only to construct low-rank sketches and to
probe the preconditioned mismatch with progressive Gaussian Gram powers.
Revealed outcomes are never read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from audit_transformer_direct_image_green_panel import tensor_sha256
from audit_transformer_relinearized_prefix_panel import (
    _gate_raw_slacks,
    _logic_slack,
    _persistent_bracket,
)
from batched_green_operator import objective_hvp_batch
from causal_structured_resolvent import (
    finite_geometric_sum,
    invariant_subspace_parameter_green_block_norms,
)
from diagnose_transformer_segmented_resolvent import (
    CANDIDATE,
    rebuild_corrected_path,
    sha256,
)
from direct_image_green_bound import direct_image_rows
from prefix_gram_enclosure import prefix_gram_rows
from structured_parameter_green import structured_quadratic_root
from transformer_hvp_grokking import logits


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "transformer_low_rank_resolvent_diagnostic.json"
MASTER_NONCE = "low-rank-skeleton-resolvent-v1-21f3a770"
CONFIGURATIONS = ((0, 26), (4, 26), (8, 26), (4, 7))
PROBES = 4
FAMILY_FAILURE = 1.0e-6
NUMERICAL_SPECTRAL_INFLATION = 1.0e-10
MISMATCH_GRAM_POWERS = (1, 2, 4, 8)
PATH_BRIDGE_MAXIMUM_ABSOLUTE_TOLERANCE = 1.0e-12
PATH_BRIDGE_L2_TOLERANCE = 1.0e-10


def bridge_sealed_parameter_path(
    *,
    row: dict,
    corrected: torch.Tensor,
    correction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Audit platform drift, then replay the exact sealed parameter path."""

    prefix = ROOT / "data" / (
        f"transformer_seed_{CANDIDATE.seed}_anchor_{CANDIDATE.anchor}"
    )
    metadata_path = prefix.with_name(prefix.name + "_corrected_path.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    parameter_path = ROOT / metadata["corrected_parameter_file"]
    correction_path = ROOT / metadata["correction_parameter_file"]
    if sha256(parameter_path) != metadata["corrected_parameter_file_sha256"]:
        raise RuntimeError("sealed corrected-parameter file hash mismatch")
    if sha256(correction_path) != metadata["correction_parameter_file_sha256"]:
        raise RuntimeError("sealed correction-parameter file hash mismatch")
    if metadata["source_full_corrected_path_sha256"] != row["corrected_path_sha256"]:
        raise RuntimeError("sealed corrected-path provenance hash mismatch")
    if metadata["candidate"] != CANDIDATE.__dict__:
        raise RuntimeError("sealed corrected-path candidate mismatch")

    dimension = corrected.shape[1] // 2
    exact_parameter = torch.from_numpy(
        np.load(parameter_path, allow_pickle=False)
    ).to(dtype=corrected.dtype, device=corrected.device)
    exact_correction = torch.from_numpy(
        np.load(correction_path, allow_pickle=False)
    ).to(dtype=correction.dtype, device=correction.device)
    if exact_parameter.shape != corrected[:, :dimension].shape:
        raise RuntimeError("sealed corrected-parameter shape mismatch")
    if exact_correction.shape != correction[:, :dimension].shape:
        raise RuntimeError("sealed correction-parameter shape mismatch")
    if tensor_sha256(exact_parameter) != metadata["corrected_parameter_tensor_sha256"]:
        raise RuntimeError("sealed corrected-parameter tensor hash mismatch")
    if tensor_sha256(exact_correction) != metadata["correction_parameter_tensor_sha256"]:
        raise RuntimeError("sealed correction-parameter tensor hash mismatch")

    parameter_difference = corrected[:, :dimension] - exact_parameter
    correction_difference = correction[:, :dimension] - exact_correction
    parameter_maximum = float(parameter_difference.abs().max())
    correction_maximum = float(correction_difference.abs().max())
    parameter_l2 = float(torch.linalg.vector_norm(parameter_difference))
    correction_l2 = float(torch.linalg.vector_norm(correction_difference))
    if max(parameter_maximum, correction_maximum) > PATH_BRIDGE_MAXIMUM_ABSOLUTE_TOLERANCE:
        raise RuntimeError("recomputed corrected path exceeds the absolute bridge tolerance")
    if max(parameter_l2, correction_l2) > PATH_BRIDGE_L2_TOLERANCE:
        raise RuntimeError("recomputed corrected path exceeds the L2 bridge tolerance")

    bridged_corrected = corrected.clone()
    bridged_correction = correction.clone()
    bridged_corrected[:, :dimension] = exact_parameter
    bridged_correction[:, :dimension] = exact_correction
    bridge = {
        "status": "recomputed-to-sealed corrected-parameter bridge passed",
        "maximum_absolute_tolerance": PATH_BRIDGE_MAXIMUM_ABSOLUTE_TOLERANCE,
        "l2_tolerance": PATH_BRIDGE_L2_TOLERANCE,
        "recomputed_unscaled_corrected_state_sha256": tensor_sha256(
            corrected
        ),
        "source_full_corrected_path_sha256": row["corrected_path_sha256"],
        "corrected_parameter_maximum_absolute_difference": parameter_maximum,
        "correction_parameter_maximum_absolute_difference": correction_maximum,
        "corrected_parameter_l2_difference": parameter_l2,
        "correction_parameter_l2_difference": correction_l2,
        "effective_corrected_parameter_tensor_sha256": tensor_sha256(
            bridged_corrected[:, :dimension]
        ),
        "effective_correction_parameter_tensor_sha256": tensor_sha256(
            bridged_correction[:, :dimension]
        ),
        "outcome_files_read": 0,
    }
    return bridged_corrected, bridged_correction, bridge


def seed_for(label: str, rank: int, block_size: int) -> int:
    payload = (
        f"{MASTER_NONCE}|{CANDIDATE.seed}|{CANDIDATE.threshold}|"
        f"{CANDIDATE.anchor}|{rank}|{block_size}|{label}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (
        2**63 - 1
    )


def apply_low_rank(rows: torch.Tensor, sketch: dict) -> torch.Tensor:
    if sketch["rank"] == 0:
        return torch.zeros_like(rows)
    basis = sketch["basis"]
    core = sketch["core"]
    return ((rows @ basis) @ core) @ basis.T


def union_reduced_green(
    *,
    sketches: dict,
    anchors: tuple[int, ...],
    anchor_for_step: tuple[int, ...],
    dimension: int,
    config,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, float, dict]:
    """Compute exact approximate-Green norms in the sketch union space."""

    active_bases = [
        sketches[anchor]["basis"]
        for anchor in anchors
        if sketches[anchor]["rank"] > 0
    ]
    if active_bases:
        union_basis, _ = torch.linalg.qr(
            torch.cat(active_bases, dim=1), mode="reduced"
        )
    else:
        union_basis = torch.empty(
            dimension, 0, dtype=dtype, device=device
        )
    reduced_hessians = []
    projection_residuals = []
    for anchor in anchor_for_step:
        sketch = sketches[anchor]
        if sketch["rank"] == 0:
            reduced = torch.zeros(
                union_basis.shape[1],
                union_basis.shape[1],
                dtype=dtype,
                device=device,
            )
            projection_residuals.append(0.0)
        else:
            coordinates = union_basis.T @ sketch["basis"]
            reduced = coordinates @ sketch["core"] @ coordinates.T
            projection_residuals.append(
                float(
                    torch.linalg.matrix_norm(
                        sketch["basis"] - union_basis @ coordinates,
                        ord=2,
                    )
                )
            )
        reduced_hessians.append(reduced)
    maximum_projection_residual = max(projection_residuals, default=0.0)
    if maximum_projection_residual > 1.0e-10:
        raise RuntimeError("sketch union space does not contain a sketch basis")
    block_norms, raw_gain = invariant_subspace_parameter_green_block_norms(
        reduced_hessians,
        learning_rate=float(config.learning_rate),
        momentum=float(config.momentum),
    )
    inflation = NUMERICAL_SPECTRAL_INFLATION * max(1.0, raw_gain)
    return block_norms + inflation, raw_gain + inflation, {
        "union_dimension": int(union_basis.shape[1]),
        "maximum_basis_projection_residual": maximum_projection_residual,
        "raw_approximate_structured_gain": raw_gain,
        "numerical_gain_inflation": inflation,
    }


def apply_approximate_t0(
    rows: torch.Tensor,
    *,
    sketches: dict,
    anchor_for_step: tuple[int, ...],
    dimension: int,
    config,
) -> torch.Tensor:
    """Apply the low-rank approximate parameter Green operator."""

    horizon = len(anchor_for_step)
    forcing = rows.reshape(rows.shape[0], horizon, dimension)
    parameter = torch.zeros_like(forcing[:, 0])
    velocity = torch.zeros_like(parameter)
    output = []
    eta = float(config.learning_rate)
    mu = float(config.momentum)
    for step, anchor in enumerate(anchor_for_step):
        hessian_parameter = apply_low_rank(parameter, sketches[anchor])
        next_velocity = mu * velocity + eta * hessian_parameter
        parameter = parameter - next_velocity - eta * forcing[:, step]
        velocity = next_velocity + eta * forcing[:, step]
        output.append(parameter)
    return torch.stack(output, dim=1).reshape(rows.shape[0], -1)


def apply_approximate_t0_transpose(
    rows: torch.Tensor,
    *,
    sketches: dict,
    anchor_for_step: tuple[int, ...],
    dimension: int,
    config,
) -> torch.Tensor:
    """Apply the transpose of the low-rank approximate Green operator."""

    horizon = len(anchor_for_step)
    outputs = rows.reshape(rows.shape[0], horizon, dimension)
    adjoint_parameter = torch.zeros_like(outputs[:, 0])
    adjoint_velocity = torch.zeros_like(adjoint_parameter)
    forcing_adjoint = torch.empty_like(outputs)
    eta = float(config.learning_rate)
    mu = float(config.momentum)
    for step in range(horizon - 1, -1, -1):
        adjoint_parameter = adjoint_parameter + outputs[:, step]
        forcing_adjoint[:, step] = eta * (
            adjoint_velocity - adjoint_parameter
        )
        difference = adjoint_velocity - adjoint_parameter
        previous_parameter = adjoint_parameter + eta * apply_low_rank(
            difference, sketches[anchor_for_step[step]]
        )
        previous_velocity = mu * difference
        adjoint_parameter = previous_parameter
        adjoint_velocity = previous_velocity
    return forcing_adjoint.reshape(rows.shape[0], -1)


def build_sketch(
    *,
    rank: int,
    anchor: int,
    generator: torch.Generator,
    corrected: torch.Tensor,
    train_pairs: torch.Tensor,
    train_labels: torch.Tensor,
    template,
    spec,
    config,
) -> dict:
    dimension = corrected.shape[1] // 2
    if rank == 0:
        return {
            "rank": 0,
            "anchor": anchor,
            "basis": None,
            "core": None,
            "eigenvalues": [],
            "hessian_norm_bound": 0.0,
            "identity_step_norm_bound": 1.0,
            "probe_sha256": None,
            "basis_sha256": None,
            "core_sha256": None,
            "batched_hvp_calls": 0,
            "logical_vector_hvp_calls": 0,
        }
    probes = torch.randn(
        rank,
        dimension,
        generator=generator,
        dtype=corrected.dtype,
        device=corrected.device,
    )
    image = objective_hvp_batch(
        corrected[anchor, :dimension],
        probes,
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )
    basis, _ = torch.linalg.qr(image.T, mode="reduced")
    hessian_basis_rows = objective_hvp_batch(
        corrected[anchor, :dimension],
        basis.T,
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )
    core = basis.T @ hessian_basis_rows.T
    core = 0.5 * (core + core.T)
    eigenvalues = torch.linalg.eigvalsh(core)
    core_norm = float(eigenvalues.abs().max())
    inflation = NUMERICAL_SPECTRAL_INFLATION * max(1.0, core_norm)
    eta = float(config.learning_rate)
    return {
        "rank": rank,
        "anchor": anchor,
        "basis": basis,
        "core": core,
        "eigenvalues": [float(value) for value in eigenvalues],
        "hessian_norm_bound": core_norm + inflation,
        "identity_step_norm_bound": max(
            1.0,
            max(abs(1.0 - eta * float(value)) for value in eigenvalues),
        )
        + eta * inflation,
        "numerical_spectral_inflation": inflation,
        "probe_sha256": tensor_sha256(probes),
        "basis_sha256": tensor_sha256(basis),
        "core_sha256": tensor_sha256(core),
        "batched_hvp_calls": 2,
        "logical_vector_hvp_calls": 2 * rank,
    }


def profiled_output_bracket(
    *,
    certificate: dict,
    corrected: torch.Tensor,
    correction: torch.Tensor,
    radii: torch.Tensor,
    dimension: int,
    cert_pairs: torch.Tensor,
    cert_labels: torch.Tensor,
    template,
    spec,
) -> dict:
    required = int(certificate["required_correct"])
    raw = [
        _gate_raw_slacks(
            logits(corrected[step, :dimension], cert_pairs, template, spec),
            cert_labels,
            required,
        )
        for step in range(len(corrected))
    ]
    maximum_power = min(
        len(row["trace"]["rows"]) for row in certificate["output_rows"]
    )
    for output_power in range(1, maximum_power + 1):
        guarantee_slacks = []
        exclusion_slacks = []
        margins = []
        for step, pair in enumerate(raw):
            radius = 0.0 if step == 0 else float(radii[step - 1])
            if step == 0:
                margin = 0.0
            else:
                output = certificate["output_rows"][step - 1]
                output_upper = float(
                    output["trace"]["rows"][output_power - 1][
                        "operator_norm_upper_bound"
                    ]
                )
                second = float(output["block_second"])
                shift = float(
                    torch.linalg.vector_norm(correction[step, :dimension])
                )
                margin = math.sqrt(2.0) * (
                    (output_upper + second * shift) * radius
                    + 0.5 * second * radius * radius
                )
            margins.append(margin)
            guarantee_slacks.append(pair[0] - margin)
            exclusion_slacks.append(pair[1] - margin)
        bracket = _persistent_bracket(guarantee_slacks, exclusion_slacks)
        if bracket is not None:
            return {
                "bracket": bracket,
                "output_power": output_power,
                "logic_slack": _logic_slack(
                    bracket, guarantee_slacks, exclusion_slacks
                ),
                "maximum_margin_radius": max(margins),
            }
    return {
        "bracket": None,
        "output_power": None,
        "logic_slack": None,
        "maximum_margin_radius": None,
    }


def run_configuration(
    rank: int,
    block_size: int,
    *,
    row: dict,
    certificate: dict,
    config,
    template,
    spec,
    train_pairs: torch.Tensor,
    train_labels: torch.Tensor,
    corrected: torch.Tensor,
    correction: torch.Tensor,
    cert_pairs: torch.Tensor,
    cert_labels: torch.Tensor,
) -> dict:
    started = time.perf_counter()
    horizon = int(row["horizon"])
    dimension = corrected.shape[1] // 2
    anchors = tuple(range(0, horizon, block_size))
    anchor_for_step = tuple(
        (step // block_size) * block_size for step in range(horizon)
    )
    sketch_generator = torch.Generator(device=corrected.device).manual_seed(
        seed_for("sketch", rank, block_size)
    )
    sketches = {
        anchor: build_sketch(
            rank=rank,
            anchor=anchor,
            generator=sketch_generator,
            corrected=corrected,
            train_pairs=train_pairs,
            train_labels=train_labels,
            template=template,
            spec=spec,
            config=config,
        )
        for anchor in anchors
    }

    approximate_blocks, approximate_gain, union_summary = union_reduced_green(
        sketches=sketches,
        anchors=anchors,
        anchor_for_step=anchor_for_step,
        dimension=dimension,
        config=config,
        dtype=corrected.dtype,
        device=corrected.device,
    )

    residual_generator = torch.Generator(device=corrected.device).manual_seed(
        seed_for("residual", rank, block_size)
    )
    probes = torch.randn(
        PROBES,
        horizon * dimension,
        generator=residual_generator,
        dtype=corrected.dtype,
        device=corrected.device,
    )

    def apply_mismatch(rows: torch.Tensor) -> torch.Tensor:
        response = apply_approximate_t0(
            rows,
            sketches=sketches,
            anchor_for_step=anchor_for_step,
            dimension=dimension,
            config=config,
        ).reshape(rows.shape[0], horizon, dimension)
        shifted = torch.zeros_like(response)
        shifted[:, 1:] = response[:, :-1]
        images = torch.zeros_like(response)
        for step in range(1, horizon):
            exact = objective_hvp_batch(
                corrected[step, :dimension],
                shifted[:, step],
                train_pairs,
                train_labels,
                template,
                spec,
                config,
            )
            images[:, step] = exact - apply_low_rank(
                shifted[:, step], sketches[anchor_for_step[step]]
            )
        return images.reshape(rows.shape[0], -1)

    def apply_mismatch_transpose(rows: torch.Tensor) -> torch.Tensor:
        values = rows.reshape(rows.shape[0], horizon, dimension)
        residual = torch.zeros_like(values)
        for step in range(1, horizon):
            exact = objective_hvp_batch(
                corrected[step, :dimension],
                values[:, step],
                train_pairs,
                train_labels,
                template,
                spec,
                config,
            )
            residual[:, step] = exact - apply_low_rank(
                values[:, step], sketches[anchor_for_step[step]]
            )
        shifted_transpose = torch.zeros_like(residual)
        shifted_transpose[:, :-1] = residual[:, 1:]
        return apply_approximate_t0_transpose(
            shifted_transpose.reshape(rows.shape[0], -1),
            sketches=sketches,
            anchor_for_step=anchor_for_step,
            dimension=dimension,
            config=config,
        )

    initial_norms = [
        float(value) for value in torch.linalg.vector_norm(probes, dim=1)
    ]
    mismatch_images = apply_mismatch(probes)
    image_norms = [
        float(value)
        for value in torch.linalg.vector_norm(mismatch_images, dim=1)
    ]
    stage_delta = FAMILY_FAILURE / len(CONFIGURATIONS)
    direct_bound = direct_image_rows(
        image_norms=image_norms,
        initial_norms=initial_norms,
        prefixes=(PROBES,),
        stage_delta=stage_delta,
    )[0]
    gram_rows = []
    gram_iterate = apply_mismatch_transpose(mismatch_images)
    adjoint_left = float(torch.dot(mismatch_images[0], mismatch_images[1]))
    adjoint_right = float(torch.dot(probes[0], gram_iterate[1]))
    adjoint_relative_residual = abs(adjoint_left - adjoint_right) / max(
        1.0, abs(adjoint_left), abs(adjoint_right)
    )
    if adjoint_relative_residual > 1.0e-9:
        raise RuntimeError("low-rank mismatch adjoint check failed")
    for power in range(1, max(MISMATCH_GRAM_POWERS) + 1):
        if power in MISMATCH_GRAM_POWERS:
            norms = [
                float(value)
                for value in torch.linalg.vector_norm(gram_iterate, dim=1)
            ]
            gram_rows.append(
                prefix_gram_rows(
                    final_norms=norms,
                    initial_norms=initial_norms,
                    prefixes=(PROBES,),
                    power=power,
                    stage_delta=stage_delta,
                )[0]
            )
        if power < max(MISMATCH_GRAM_POWERS):
            gram_iterate = apply_mismatch_transpose(
                apply_mismatch(gram_iterate)
            )
    candidates = [
        ("direct_image", float(direct_bound["operator_norm_upper_bound"]))
    ] + [
        (
            f"gram_q{int(bound['power'])}",
            float(bound["operator_norm_upper_bound"]),
        )
        for bound in gram_rows
    ]
    mismatch_route, mismatch_gain = min(candidates, key=lambda item: item[1])
    mismatch_lower = max(
        float(direct_bound["operator_norm_lower_estimate"]),
        *(float(bound["operator_norm_lower_estimate"]) for bound in gram_rows),
    )
    resolvent_multiplier = finite_geometric_sum(
        mismatch_gain, horizon=horizon
    )
    global_gain = approximate_gain * resolvent_multiplier
    selected = (
        row["stages"][-1]["direct"]
        if row["route"] == "direct_image"
        else row["stages"][-1]["gram"]
    )
    forcing = float(selected["parameter_forcing_upper"])
    curvature = float(selected["objective_hessian_lipschitz_upper"])
    event = {
        "bracket": None,
        "output_power": None,
        "logic_slack": None,
        "maximum_margin_radius": None,
    }
    global_radius = structured_quadratic_root(
        global_gain * forcing,
        global_gain,
        curvature,
    )
    domain_passed = bool(
        global_radius is not None
        and float(selected["correction_max_parameter_norm"])
        + float(global_radius)
        <= float(selected["domain_radius"])
    )
    if domain_passed:
        event = profiled_output_bracket(
            certificate=certificate,
            corrected=corrected,
            correction=correction,
            radii=torch.full(
                (horizon,),
                float(global_radius),
                dtype=corrected.dtype,
                device=corrected.device,
            ),
            dimension=dimension,
            cert_pairs=cert_pairs,
            cert_labels=cert_labels,
            template=template,
            spec=spec,
        )

    serialized_sketches = []
    for anchor in anchors:
        sketch = sketches[anchor]
        serialized_sketches.append(
            {
                key: value
                for key, value in sketch.items()
                if key not in {"basis", "core"}
            }
        )
    sketch_batched = sum(
        item["batched_hvp_calls"] for item in serialized_sketches
    )
    sketch_logical = sum(
        item["logical_vector_hvp_calls"] for item in serialized_sketches
    )
    finite_global_gain = math.isfinite(global_gain)
    mismatch_batched = (
        2 * max(MISMATCH_GRAM_POWERS) * (horizon - 1)
    )
    mismatch_logical = mismatch_batched * PROBES
    return {
        "rank": rank,
        "block_size": block_size,
        "segments": len(anchors),
        "anchors": list(anchors),
        "sketch_seed": seed_for("sketch", rank, block_size),
        "residual_seed": seed_for("residual", rank, block_size),
        "probes_per_residual": PROBES,
        "mismatch_operator_stage_delta": stage_delta,
        "sketches": serialized_sketches,
        "mismatch_probe_sha256": [
            tensor_sha256(probe) for probe in probes
        ],
        "initial_probe_norms": initial_norms,
        "mismatch_image_norms": image_norms,
        "mismatch_gram_rows": gram_rows,
        "mismatch_gram_powers": list(MISMATCH_GRAM_POWERS),
        "mismatch_adjoint_relative_residual": adjoint_relative_residual,
        "mismatch_route": mismatch_route,
        "mismatch_direct_gain_upper": float(
            direct_bound["operator_norm_upper_bound"]
        ),
        "minimum_mismatch_gram_gain_upper": min(
            float(bound["operator_norm_upper_bound"])
            for bound in gram_rows
        ),
        "mismatch_gain_upper": mismatch_gain,
        "mismatch_gain_lower_estimate": mismatch_lower,
        "finite_resolvent_multiplier_upper": resolvent_multiplier,
        "approximate_green_block_norm_maximum": float(
            approximate_blocks.max()
        ),
        "approximate_structured_gain_upper": approximate_gain,
        "union_reduction": union_summary,
        "structured_gain_upper": global_gain if finite_global_gain else None,
        "released_structured_gain_upper": float(
            selected["structured_gain_upper"]
        ),
        "gain_ratio_to_released": (
            global_gain / float(selected["structured_gain_upper"])
            if finite_global_gain
            else None
        ),
        "global_parameter_remainder_radius": global_radius,
        "domain_passed": domain_passed,
        **event,
        "issued": domain_passed and event["bracket"] is not None,
        "sketch_batched_hvp_calls": sketch_batched,
        "sketch_logical_vector_hvp_calls": sketch_logical,
        "mismatch_batched_hvp_calls": mismatch_batched,
        "mismatch_logical_vector_hvp_calls": mismatch_logical,
        "ideal_time_batched_mismatch_hvp_depth": 2
        * max(MISMATCH_GRAM_POWERS),
        "operator_batched_hvp_calls": sketch_batched + mismatch_batched,
        "operator_logical_vector_hvp_calls": sketch_logical + mismatch_logical,
        "released_direct_logical_vector_hvp_calls": horizon * PROBES,
        "outcome_files_read": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configurations",
        default="all",
        help="all or comma-separated rank:block_size pairs",
    )
    args = parser.parse_args()
    if args.configurations == "all":
        requested = CONFIGURATIONS
    else:
        requested = tuple(
            tuple(int(value) for value in item.split(":"))
            for item in args.configurations.split(",")
        )
        if any(item not in CONFIGURATIONS for item in requested):
            raise ValueError("configuration was not predeclared")

    rebuild_started = time.perf_counter()
    (
        row,
        certificate,
        config,
        template,
        spec,
        train_pairs,
        train_labels,
        corrected,
        correction,
        cert_pairs,
        cert_labels,
    ) = rebuild_corrected_path(require_corrected_path_hash=False)
    corrected, correction, corrected_path_bridge = bridge_sealed_parameter_path(
        row=row,
        corrected=corrected,
        correction=correction,
    )
    rebuild_seconds = time.perf_counter() - rebuild_started
    rows = []
    for rank, block_size in requested:
        result = run_configuration(
            rank,
            block_size,
            row=row,
            certificate=certificate,
            config=config,
            template=template,
            spec=spec,
            train_pairs=train_pairs,
            train_labels=train_labels,
            corrected=corrected,
            correction=correction,
            cert_pairs=cert_pairs,
            cert_labels=cert_labels,
        )
        rows.append(result)
        payload = {
            "status": "low-rank union-subspace causal-resolvent diagnostic in progress",
            "evidence_boundary": (
                "Post-release outcome-blind development audit. Mismatch "
                "spectral bounds use progressive ideal-Gaussian Gram powers; "
                "neural HVPs, sketches, and margins remain float64."
            ),
            "candidate": CANDIDATE.__dict__,
            "horizon": int(row["horizon"]),
            "parameter_count": int(corrected.shape[1] // 2),
            "configurations_predeclared": [
                list(item) for item in CONFIGURATIONS
            ],
            "configurations_executed": [list(item) for item in requested],
            "family_failure_upper": FAMILY_FAILURE,
            "source_sha256": sha256(Path(__file__)),
            "source_full_corrected_path_sha256": row[
                "corrected_path_sha256"
            ],
            "effective_corrected_parameter_path_sha256": tensor_sha256(
                corrected[..., : corrected.shape[1] // 2]
            ),
            "corrected_path_bridge": corrected_path_bridge,
            "segmented_source": (
                "results/transformer_segmented_resolvent_diagnostic.json"
            ),
            "segmented_source_sha256": sha256(
                ROOT
                / "results"
                / "transformer_segmented_resolvent_diagnostic.json"
            ),
            "common_corrected_path_rebuild_seconds": rebuild_seconds,
            "outcome_files_read": 0,
            "rows": rows,
        }
        OUTPUT.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(json.dumps(result, indent=2), flush=True)
    payload["status"] = "low-rank union-subspace causal-resolvent diagnostic complete"
    OUTPUT.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "rows"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
