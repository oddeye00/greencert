#!/usr/bin/env python3
"""Outcome-blind segmented-resolvent diagnostic on the shortest sealed window.

The experiment replaces each corrected-path Hessian by the Hessian at the
preceding segment anchor and estimates the causal mismatch operator

    A = D_(H-H_anchor) S P K_tilde B.

It reads no revealed outcome.  This is a development diagnostic, not a
prospective certificate or a low-rank runtime benchmark: anchor-Hessian
products are still evaluated by autograd here.
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
    from_scaled,
    output_bracket,
)
from batched_green_operator import (
    make_batched_scaled_optimizer_products,
    objective_hvp_batch,
)
from causal_structured_resolvent import (
    finite_geometric_sum,
    make_batched_causal_structured_resolvent_products,
)
from direct_image_green_bound import direct_image_rows
from prefix_gram_enclosure import prefix_gram_rows
from streaming_variational_centerline import build_streaming_transformer_centerline
from transformer_certificate_protocol import Candidate
from transformer_green_operator import make_causal_green_products
from transformer_optimizer_probe import make_scaled_optimizer_jvp_vjp
from structured_parameter_green import structured_quadratic_root
from transformer_v3_certificate import load_candidate, output_path, safe_json


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "transformer_segmented_resolvent_diagnostic.json"
STRUCTURED = ROOT / "results" / "structured_parameter_green_transformer_audit.json"
CANDIDATE = Candidate(366, 0.8, 1120)
DEFAULT_BLOCK_SIZES = (26, 13, 7, 4, 2, 1)
PROBE_NONCE = "segmented-resolvent-diagnostic-v1-7edcfc46"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def source_row() -> dict:
    payload = safe_json(STRUCTURED)
    for row in payload["rows"]:
        candidate = row["candidate"]
        if (
            int(candidate["seed"]) == CANDIDATE.seed
            and float(candidate["threshold"]) == CANDIDATE.threshold
            and int(candidate["anchor"]) == CANDIDATE.anchor
        ):
            return row
    raise RuntimeError("structured source row is absent")


def probe_seed(block_size: int) -> int:
    payload = (
        f"{PROBE_NONCE}|{CANDIDATE.seed}|{CANDIDATE.threshold}|"
        f"{CANDIDATE.anchor}|{block_size}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (
        2**63 - 1
    )


def rebuild_corrected_path():
    row = source_row()
    certificate_path = output_path(CANDIDATE)
    if sha256(certificate_path) != row["certificate_sha256"]:
        raise RuntimeError("certificate hash mismatch")
    certificate = safe_json(certificate_path)
    config, template, spec, data, parameter, velocity = load_candidate(CANDIDATE)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = data
    horizon = int(row["horizon"])
    dimension = int(parameter.numel())
    path = build_streaming_transformer_centerline(
        config,
        template,
        spec,
        train_pairs,
        train_labels,
        parameter,
        velocity,
        maximum_horizon=horizon,
    )
    center = path["center"]
    scaled_center = path["scaled_center"]
    mapped = [path["map_step"](center[step]) for step in range(horizon)]
    residual = torch.stack(
        [
            torch.cat(
                (
                    mapped[step][:dimension],
                    config.learning_rate * mapped[step][dimension:],
                )
            )
            - scaled_center[step + 1]
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
    apply_green, _ = make_causal_green_products(
        [item[0] for item in products],
        [item[1] for item in products],
        2 * dimension,
    )
    correction_rows = apply_green(residual.reshape(-1)).reshape(horizon, -1)
    correction = torch.cat(
        (torch.zeros_like(correction_rows[:1]), correction_rows), dim=0
    )
    corrected_scaled = scaled_center + correction
    if tensor_sha256(corrected_scaled) != row["corrected_path_sha256"]:
        raise RuntimeError("corrected path hash mismatch")
    corrected = from_scaled(corrected_scaled, dimension, config.learning_rate)
    return (
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
    )


def audit_block_size(
    block_size: int,
    *,
    probes: int,
    stage_delta: float,
    row: dict,
    config,
    template,
    spec,
    train_pairs: torch.Tensor,
    train_labels: torch.Tensor,
    corrected: torch.Tensor,
    correction: torch.Tensor,
    certificate: dict,
    cert_pairs: torch.Tensor,
    cert_labels: torch.Tensor,
) -> dict:
    started = time.perf_counter()
    horizon = int(row["horizon"])
    dimension = corrected.shape[1] // 2
    anchors = tuple(range(0, horizon, block_size))
    anchor_for_step = tuple((step // block_size) * block_size for step in range(horizon))
    if block_size == 1:
        selected = (
            row["stages"][-1]["direct"]
            if row["route"] == "direct_image"
            else row["stages"][-1]["gram"]
        )
        return {
            "block_size": 1,
            "segments": horizon,
            "anchors": list(anchors),
            "mismatch_identically_zero": True,
            "mismatch_gain_upper": 0.0,
            "mismatch_gain_lower_estimate": 0.0,
            "finite_resolvent_multiplier_upper": 1.0,
            "preconditioned_structured_gain_upper": float(
                selected["structured_gain_upper"]
            ),
            "parameter_remainder_radius": float(
                selected["parameter_remainder_radius"]
            ),
            "bracket": selected["bracket"],
            "issued": bool(selected["issued"]),
            "inherited_exact_operator_bound": True,
            "batched_hvp_calls": 0,
            "logical_vector_hvp_calls": 0,
            "elapsed_seconds": time.perf_counter() - started,
        }

    anchor_products = {
        anchor: make_batched_scaled_optimizer_products(
            corrected[anchor, :dimension],
            train_pairs,
            train_labels,
            template,
            spec,
            config,
        )
        for anchor in anchors
    }
    approximate_jvps = [anchor_products[anchor_for_step[step]][0] for step in range(horizon)]
    approximate_vjps = [anchor_products[anchor_for_step[step]][1] for step in range(horizon)]

    delta_products = []
    for step in range(horizon):
        anchor = anchor_for_step[step]
        if anchor == step:
            delta_products.append(lambda rows: torch.zeros_like(rows))
            continue

        def delta(rows, *, current=step, reference=anchor):
            return objective_hvp_batch(
                corrected[current, :dimension],
                rows,
                train_pairs,
                train_labels,
                template,
                spec,
                config,
            ) - objective_hvp_batch(
                corrected[reference, :dimension],
                rows,
                train_pairs,
                train_labels,
                template,
                spec,
                config,
            )

        delta_products.append(delta)

    t0, _, mismatch, mismatch_transpose = (
        make_batched_causal_structured_resolvent_products(
            approximate_jvps,
            approximate_vjps,
            delta_products,
            delta_products,
            dimension,
            config.learning_rate,
        )
    )
    generator = torch.Generator(device=corrected.device).manual_seed(
        probe_seed(block_size)
    )
    vectors = torch.stack(
        [
            torch.randn(
                horizon * dimension,
                generator=generator,
                dtype=corrected.dtype,
                device=corrected.device,
            )
            for _ in range(probes)
        ]
    )
    initial_norms = [
        float(value) for value in torch.linalg.vector_norm(vectors, dim=1)
    ]
    approximate_images = t0(vectors)
    approximate_image_norms = [
        float(value)
        for value in torch.linalg.vector_norm(approximate_images, dim=1)
    ]
    images = mismatch(vectors)
    gram_images = mismatch_transpose(images)
    final_norms = [
        float(value) for value in torch.linalg.vector_norm(gram_images, dim=1)
    ]
    bound = prefix_gram_rows(
        final_norms=final_norms,
        initial_norms=initial_norms,
        prefixes=(probes,),
        power=1,
        stage_delta=stage_delta,
    )[0]
    alpha = float(bound["operator_norm_upper_bound"])
    approximate_bound = direct_image_rows(
        image_norms=approximate_image_norms,
        initial_norms=initial_norms,
        prefixes=(probes,),
        stage_delta=stage_delta,
    )[0]
    kappa0 = float(approximate_bound["operator_norm_upper_bound"])
    multiplier = finite_geometric_sum(alpha, horizon=horizon)
    preconditioned_gain = kappa0 * multiplier
    selected = (
        row["stages"][-1]["direct"]
        if row["route"] == "direct_image"
        else row["stages"][-1]["gram"]
    )
    forcing = float(selected["parameter_forcing_upper"])
    lipschitz = float(selected["objective_hessian_lipschitz_upper"])
    radius = structured_quadratic_root(
        preconditioned_gain * forcing,
        preconditioned_gain,
        lipschitz,
    )
    domain_passed = (
        radius is not None
        and float(selected["correction_max_parameter_norm"]) + float(radius)
        <= float(selected["domain_radius"])
    )
    event = {
        "bracket": None,
        "output_power": None,
        "logic_slack": None,
        "maximum_margin_radius": None,
    }
    if domain_passed:
        event = output_bracket(
            certificate=certificate,
            corrected=corrected,
            correction=correction,
            dimension=dimension,
            cert_pairs=cert_pairs,
            cert_labels=cert_labels,
            template=template,
            spec=spec,
            radius=float(radius),
        )
    active_delta_steps = horizon - len(anchors)
    # One A and one A^T product: two approximate Green sweeps and four
    # Hessian products per non-anchor delta block.  Each call is row-batched.
    batched_hvp_calls = 3 * horizon + 4 * active_delta_steps
    return {
        "block_size": int(block_size),
        "segments": len(anchors),
        "anchors": list(anchors),
        "probe_seed": probe_seed(block_size),
        "probes": probes,
        "stage_delta": stage_delta,
        "initial_probe_norms": initial_norms,
        "approximate_green_image_norms": approximate_image_norms,
        "gram_image_norms": final_norms,
        "approximate_structured_gain_upper": kappa0,
        "mismatch_gain_upper": alpha,
        "mismatch_gain_lower_estimate": float(
            bound["operator_norm_lower_estimate"]
        ),
        "finite_resolvent_multiplier_upper": multiplier,
        "preconditioned_structured_gain_upper": preconditioned_gain,
        "released_structured_gain_upper": float(
            selected["structured_gain_upper"]
        ),
        "preconditioned_to_released_gain_ratio": (
            preconditioned_gain / float(selected["structured_gain_upper"])
        ),
        "parameter_forcing_upper": forcing,
        "objective_hessian_lipschitz_upper": lipschitz,
        "parameter_remainder_radius": radius,
        "domain_passed": domain_passed,
        **event,
        "issued": domain_passed and event["bracket"] is not None,
        "batched_hvp_calls": batched_hvp_calls,
        "logical_vector_hvp_calls": probes * batched_hvp_calls,
        "active_delta_steps": active_delta_steps,
        "elapsed_seconds": time.perf_counter() - started,
        "outcome_files_read": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes", type=int, default=4)
    parser.add_argument(
        "--block-sizes",
        default=",".join(str(value) for value in DEFAULT_BLOCK_SIZES),
    )
    args = parser.parse_args()
    block_sizes = tuple(int(value) for value in args.block_sizes.split(","))
    if args.probes < 1 or any(value < 1 for value in block_sizes):
        raise ValueError("probes and block sizes must be positive")
    rebuilt = rebuild_corrected_path()
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
    ) = rebuilt
    probabilistic_rows = sum(value != 1 for value in block_sizes)
    stage_delta = 1.0e-6 / max(1, 2 * probabilistic_rows)
    rows = []
    for block_size in block_sizes:
        result = audit_block_size(
            block_size,
            probes=args.probes,
            stage_delta=stage_delta,
            row=row,
            config=config,
            template=template,
            spec=spec,
            train_pairs=train_pairs,
            train_labels=train_labels,
            corrected=corrected,
            correction=correction,
            certificate=certificate,
            cert_pairs=cert_pairs,
            cert_labels=cert_labels,
        )
        rows.append(result)
        print(json.dumps(result, indent=2))
        partial = {
            "status": "segmented causal-resolvent diagnostic in progress",
            "evidence_boundary": (
                "Outcome-blind post-release diagnostic; anchor Hessians are still "
                "autograd HVPs, not low-rank surrogates or a runtime claim."
            ),
            "candidate": CANDIDATE.__dict__,
            "horizon": int(row["horizon"]),
            "parameter_count": int(corrected.shape[1] // 2),
            "probes": args.probes,
            "family_failure_upper": 1.0e-6,
            "stage_delta": stage_delta,
            "source": STRUCTURED.relative_to(ROOT).as_posix(),
            "source_sha256": sha256(STRUCTURED),
            "certificate_sha256": row["certificate_sha256"],
            "corrected_path_sha256": row["corrected_path_sha256"],
            "outcome_files_read": 0,
            "rows": rows,
        }
        OUTPUT.write_text(
            json.dumps(partial, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    payload = safe_json(OUTPUT)
    payload["status"] = "segmented causal-resolvent diagnostic complete"
    payload["all_rows_finite"] = all(
        math.isfinite(float(item["mismatch_gain_upper"])) for item in rows
    )
    OUTPUT.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
