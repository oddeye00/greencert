#!/usr/bin/env python3
"""Outcome-blind low-rank/skeleton causal-resolvent diagnostic.

This development audit replaces the approximate Green query by deterministic
2-by-2 optimizer-skeleton bounds.  Exact checkpoint Hessians are used only to
construct low-rank segment sketches and to probe the residual operators.
Revealed outcomes are never read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from audit_transformer_direct_image_green_panel import tensor_sha256
from audit_transformer_relinearized_prefix_panel import (
    _gate_raw_slacks,
    _logic_slack,
    _persistent_bracket,
)
from batched_green_operator import objective_hvp_batch
from causal_structured_resolvent import (
    causal_block_majorant,
    causal_forward_quadratic_envelope,
    skeleton_parameter_green_block_majorant,
)
from diagnose_transformer_segmented_resolvent import (
    CANDIDATE,
    rebuild_corrected_path,
    sha256,
)
from direct_image_green_bound import direct_image_rows
from structured_parameter_green import structured_quadratic_root
from transformer_hvp_grokking import logits


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "transformer_low_rank_resolvent_diagnostic.json"
MASTER_NONCE = "low-rank-skeleton-resolvent-v1-21f3a770"
CONFIGURATIONS = ((0, 26), (4, 26), (8, 26), (4, 7))
PROBES = 4
FAMILY_FAILURE = 1.0e-6
NUMERICAL_SPECTRAL_INFLATION = 1.0e-10


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

    hessian_bounds = [
        sketches[anchor_for_step[step]]["hessian_norm_bound"]
        for step in range(horizon)
    ]
    identity_bounds = [
        sketches[anchor_for_step[step]]["identity_step_norm_bound"]
        for step in range(horizon)
    ]
    approximate_blocks = skeleton_parameter_green_block_majorant(
        hessian_bounds,
        identity_bounds,
        learning_rate=float(config.learning_rate),
        momentum=float(config.momentum),
        dtype=corrected.dtype,
    )

    residual_generator = torch.Generator(device=corrected.device).manual_seed(
        seed_for("residual", rank, block_size)
    )
    stage_delta = FAMILY_FAILURE / (
        len(CONFIGURATIONS) * horizon
    )
    mismatch_bounds = []
    mismatch_lower = []
    residual_probe_hashes = []
    residual_image_maxima = []
    for step in range(horizon):
        probes = torch.randn(
            PROBES,
            dimension,
            generator=residual_generator,
            dtype=corrected.dtype,
            device=corrected.device,
        )
        exact_images = objective_hvp_batch(
            corrected[step, :dimension],
            probes,
            train_pairs,
            train_labels,
            template,
            spec,
            config,
        )
        sketch = sketches[anchor_for_step[step]]
        residual_images = exact_images - apply_low_rank(probes, sketch)
        initial_norms = [
            float(value) for value in torch.linalg.vector_norm(probes, dim=1)
        ]
        image_norms = [
            float(value)
            for value in torch.linalg.vector_norm(residual_images, dim=1)
        ]
        bound = direct_image_rows(
            image_norms=image_norms,
            initial_norms=initial_norms,
            prefixes=(PROBES,),
            stage_delta=stage_delta,
        )[0]
        mismatch_bounds.append(float(bound["operator_norm_upper_bound"]))
        mismatch_lower.append(float(bound["operator_norm_lower_estimate"]))
        residual_image_maxima.append(max(image_norms))
        residual_probe_hashes.append(tensor_sha256(probes))

    _, exact_majorant = causal_block_majorant(
        approximate_blocks, mismatch_bounds
    )
    finite_majorant = bool(torch.isfinite(exact_majorant).all())
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
    global_gain = math.inf
    global_radius = None
    causal_radii = None
    domain_passed = False
    if finite_majorant:
        global_gain = float(torch.linalg.matrix_norm(exact_majorant, ord=2))
        global_radius = structured_quadratic_root(
            global_gain * forcing,
            global_gain,
            curvature,
        )
        affine_bounds = (
            torch.linalg.vector_norm(exact_majorant, dim=1) * forcing
        )
        causal_radii = causal_forward_quadratic_envelope(
            affine_bounds,
            exact_majorant,
            [curvature] * horizon,
        )
        correction_norms = torch.linalg.vector_norm(
            correction[:horizon, :dimension], dim=1
        )
        derivative_domain_checks = correction_norms.clone()
        if horizon > 1:
            derivative_domain_checks[1:] += causal_radii[:-1]
        domain_passed = bool(
            torch.isfinite(causal_radii).all()
            and (
                derivative_domain_checks
                <= float(selected["domain_radius"])
            ).all()
        )
        if domain_passed:
            event = profiled_output_bracket(
                certificate=certificate,
                corrected=corrected,
                correction=correction,
                radii=causal_radii,
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
    return {
        "rank": rank,
        "block_size": block_size,
        "segments": len(anchors),
        "anchors": list(anchors),
        "sketch_seed": seed_for("sketch", rank, block_size),
        "residual_seed": seed_for("residual", rank, block_size),
        "probes_per_residual": PROBES,
        "residual_operator_stage_delta": stage_delta,
        "sketches": serialized_sketches,
        "residual_probe_sha256": residual_probe_hashes,
        "maximum_residual_image_norm": max(residual_image_maxima),
        "maximum_mismatch_norm_lower_estimate": max(mismatch_lower),
        "maximum_mismatch_norm_upper_bound": max(mismatch_bounds),
        "minimum_mismatch_norm_upper_bound": min(mismatch_bounds),
        "approximate_block_majorant_maximum": float(
            approximate_blocks.max()
        ),
        "exact_block_majorant_finite": finite_majorant,
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
        "causal_maximum_parameter_radius": (
            None
            if causal_radii is None
            else float(torch.max(causal_radii))
        ),
        "causal_parameter_radii": (
            None
            if causal_radii is None
            else [float(value) for value in causal_radii]
        ),
        "domain_passed": domain_passed,
        **event,
        "issued": domain_passed and event["bracket"] is not None,
        "sketch_batched_hvp_calls": sketch_batched,
        "sketch_logical_vector_hvp_calls": sketch_logical,
        "residual_batched_hvp_calls": horizon,
        "residual_logical_vector_hvp_calls": horizon * PROBES,
        "operator_batched_hvp_calls": sketch_batched + horizon,
        "operator_logical_vector_hvp_calls": sketch_logical + horizon * PROBES,
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
    ) = rebuild_corrected_path()
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
            "status": "low-rank skeleton causal-resolvent diagnostic in progress",
            "evidence_boundary": (
                "Post-release outcome-blind development audit. Residual "
                "spectral bounds use ideal-Gaussian direct images; neural "
                "HVPs, sketches, and margins remain float64."
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
            "corrected_path_sha256": tensor_sha256(
                torch.cat(
                    (
                        corrected[..., : corrected.shape[1] // 2],
                        float(config.learning_rate)
                        * corrected[..., corrected.shape[1] // 2 :],
                    ),
                    dim=-1,
                )
            ),
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
    payload["status"] = "low-rank skeleton causal-resolvent diagnostic complete"
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
