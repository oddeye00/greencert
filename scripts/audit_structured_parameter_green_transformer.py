#!/usr/bin/env python3
"""Post-release structured parameter-Green audit on 15 sealed operators.

The audit never reads revealed outcomes.  It reuses the frozen corrected paths
and inherited output enclosures, but replaces the full-state Green operator by
``T=P_theta K B`` for the exact scaled-momentum nonlinear forcing subspace.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch

from audit_transformer_direct_image_green_panel import (
    close,
    panel_index,
    tensor_sha256,
)
from audit_transformer_relinearized_prefix_panel import (
    CASE_ROWS,
    FAMILY_FAILURE,
    PREFIXES,
    from_scaled,
    output_bracket,
)
from batched_green_operator import make_batched_scaled_optimizer_products
from direct_image_green_bound import direct_image_rows
from prefix_gram_enclosure import equal_family_stage_delta, prefix_gram_rows
from streaming_variational_centerline import build_streaming_transformer_centerline
from structured_parameter_green import (
    make_batched_structured_parameter_green_products,
    structured_quadratic_root,
)
from transformer_certificate_protocol import Candidate
from transformer_green_operator import make_causal_green_products
from transformer_optimizer_probe import make_scaled_optimizer_jvp_vjp
from transformer_v3_certificate import load_candidate, output_path, safe_json


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "structured_parameter_green_transformer_audit.json"
CACHE = ROOT / "results" / "structured_parameter_green_transformer_cache"
PROTOCOL = ROOT / "STRUCTURED_PARAMETER_GREEN_AUDIT_PROTOCOL_V2.md"
THEOREM = ROOT / "STRUCTURED_PARAMETER_GREEN_THEOREM.md"
PANEL = ROOT / "results" / "transformer_v3_relinearized_prefix_panel_audit.json"
DIRECT_PANEL = ROOT / "results" / "transformer_direct_image_green_panel_audit.json"
VERSION = 2
MASTER_NONCE = "a2c8e7e5be93ab71ece3ae64ff00c0d685e5b49a6c9e91117b597ff9f7f5c829"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def case_set_sha256() -> str:
    payload = json.dumps(CASE_ROWS, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest().upper()


def dependency_paths() -> tuple[Path, ...]:
    return (
        Path(__file__),
        ROOT / "scripts" / "structured_parameter_green.py",
        ROOT / "scripts" / "test_structured_parameter_green.py",
        ROOT / "scripts" / "audit_transformer_direct_image_green_panel.py",
        ROOT / "scripts" / "audit_transformer_relinearized_prefix_panel.py",
        ROOT / "scripts" / "batched_green_operator.py",
        ROOT / "scripts" / "direct_image_green_bound.py",
        ROOT / "scripts" / "prefix_gram_enclosure.py",
        ROOT / "scripts" / "streaming_variational_centerline.py",
        ROOT / "scripts" / "transformer_green_operator.py",
        ROOT / "scripts" / "transformer_optimizer_probe.py",
        ROOT / "scripts" / "transformer_v3_certificate.py",
        THEOREM,
        ROOT / "STRUCTURED_PARAMETER_GREEN_AUDIT_PROTOCOL.md",
        ROOT / "STRUCTURED_PARAMETER_GREEN_AUDIT_ABORTED_V1.md",
        PANEL,
        DIRECT_PANEL,
    )


def dependency_hashes() -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in dependency_paths()
    }


def assert_protocol_frozen() -> None:
    if not PROTOCOL.is_file():
        raise RuntimeError("structured-parameter protocol is absent; refusing probes")
    text = PROTOCOL.read_text(encoding="utf-8").upper()
    required = {
        "MASTER_NONCE": MASTER_NONCE.upper(),
        "CASE_SET_SHA256": case_set_sha256(),
        **{f"DEPENDENCY:{name}": value for name, value in dependency_hashes().items()},
    }
    missing = [name for name, value in required.items() if value not in text]
    if missing:
        raise RuntimeError("structured-parameter protocol mismatch: " + ", ".join(missing))


def probe_seed(candidate: Candidate, horizon: int) -> int:
    payload = (
        f"{MASTER_NONCE}|{candidate.seed}|{candidate.gate_index}|"
        f"{candidate.anchor}|{horizon}|structured-parameter-v{VERSION}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def cache_path(candidate: Candidate) -> Path:
    return CACHE / (
        f"seed_{candidate.seed}_gate_{candidate.gate_index}_"
        f"anchor_{candidate.anchor}_v{VERSION}.json"
    )


def evaluate_attempt(
    *,
    kappa: float,
    panel_row: dict,
    certificate: dict,
    corrected: torch.Tensor,
    correction: torch.Tensor,
    dimension: int,
    learning_rate: float,
    cert_pairs: torch.Tensor,
    cert_labels: torch.Tensor,
    template,
    spec,
) -> dict:
    eta = float(learning_rate)
    if eta <= 0.0:
        raise ValueError("learning rate must be positive")
    recurrence = float(panel_row["measured_response_recurrence_residual_norm"])
    if recurrence != 0.0:
        raise RuntimeError("audit requires the frozen zero-residual response interface")
    state_to_parameter_forcing = math.sqrt(2.0) * eta
    forcing_upper = float(panel_row["total_corrected_injection_upper"]) / state_to_parameter_forcing
    hessian_lipschitz = float(panel_row["derivative_drift_upper"]) / state_to_parameter_forcing
    response_upper = float(kappa) * forcing_upper
    radius = structured_quadratic_root(response_upper, kappa, hessian_lipschitz)
    correction_max_parameter = float(panel_row["maximum_parameter_direction_norm"])
    domain = float(panel_row["domain_radius"])
    domain_passed = radius is not None and correction_max_parameter + radius <= domain
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
    discriminant = 1.0 - 2.0 * float(kappa) * hessian_lipschitz * response_upper
    return {
        "structured_gain_upper": float(kappa),
        "parameter_forcing_upper": forcing_upper,
        "parameter_response_upper": response_upper,
        "objective_hessian_lipschitz_upper": hessian_lipschitz,
        "discriminant": discriminant,
        "parameter_remainder_radius": radius,
        "correction_max_parameter_norm": correction_max_parameter,
        "domain_radius": domain,
        "domain_passed": domain_passed,
        **event,
        "issued": domain_passed and event["bracket"] is not None,
    }


def audit_case(case: tuple[int, float, int, int, str]) -> dict:
    assert_protocol_frozen()
    started = time.perf_counter()
    seed, threshold, anchor, horizon, certificate_sha = case
    candidate = Candidate(seed, threshold, anchor)
    source_row = panel_index()[(seed, threshold, anchor)]
    direct_panel = safe_json(DIRECT_PANEL)
    direct_index = {
        (
            int(row["candidate"]["seed"]),
            float(row["candidate"]["threshold"]),
            int(row["candidate"]["anchor"]),
        ): row
        for row in direct_panel["rows"]
    }
    direct_row = direct_index[(seed, threshold, anchor)]
    certificate_path = output_path(candidate)
    if sha256(certificate_path) != certificate_sha:
        raise RuntimeError(f"certificate hash mismatch for {candidate}")
    certificate = safe_json(certificate_path)
    config, template, spec, data, parameter, velocity = load_candidate(candidate)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = data
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
    old_products = [
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
    old_apply, _ = make_causal_green_products(
        [row[0] for row in old_products],
        [row[1] for row in old_products],
        2 * dimension,
    )
    correction_rows = old_apply(residual.reshape(-1)).reshape(horizon, -1)
    correction = torch.cat((torch.zeros_like(correction_rows[:1]), correction_rows), dim=0)
    corrected_scaled = scaled_center + correction
    if tensor_sha256(corrected_scaled) != source_row["corrected_path_sha256"]:
        raise RuntimeError(f"corrected path mismatch for {candidate}")
    corrected = from_scaled(corrected_scaled, dimension, config.learning_rate)

    products = [
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
    apply, transpose = make_batched_structured_parameter_green_products(
        [row[0] for row in products],
        [row[1] for row in products],
        dimension,
        config.learning_rate,
    )

    generator = torch.Generator(device=corrected.device).manual_seed(
        probe_seed(candidate, horizon)
    )
    stage_delta = equal_family_stage_delta(
        family_failure=FAMILY_FAILURE,
        operators=len(CASE_ROWS),
        prefixes=PREFIXES,
    )
    initial_norms: list[float] = []
    image_norms: list[float] = []
    gram_norms: list[float] = []
    probe_hashes: list[str] = []
    stages = []
    forward = 0
    transposed = 0
    route = None
    final_attempt = None
    for prefix in PREFIXES:
        count = prefix - len(initial_norms)
        vectors = torch.stack(
            [
                torch.randn(
                    horizon * dimension,
                    generator=generator,
                    dtype=corrected.dtype,
                    device=corrected.device,
                )
                for _ in range(count)
            ]
        )
        probe_hashes.extend(
            hashlib.sha256(row.detach().cpu().numpy().tobytes(order="C")).hexdigest().upper()
            for row in vectors
        )
        initial_norms.extend(
            float(value) for value in torch.linalg.vector_norm(vectors, dim=1)
        )
        images = apply(vectors)
        forward += count
        image_norms.extend(
            float(value) for value in torch.linalg.vector_norm(images, dim=1)
        )
        direct = direct_image_rows(
            image_norms=image_norms,
            initial_norms=initial_norms,
            prefixes=(prefix,),
            stage_delta=stage_delta,
        )[0]
        direct_attempt = evaluate_attempt(
            kappa=float(direct["operator_norm_upper_bound"]),
            panel_row=source_row,
            certificate=certificate,
            corrected=corrected,
            correction=correction,
            dimension=dimension,
            learning_rate=config.learning_rate,
            cert_pairs=cert_pairs,
            cert_labels=cert_labels,
            template=template,
            spec=spec,
        )
        stage = {"prefix": prefix, "direct": {**direct, **direct_attempt}, "gram": None}
        if direct_attempt["issued"]:
            route = "direct_image"
            final_attempt = direct_attempt
            stages.append(stage)
            break

        gram_block = transpose(images)
        transposed += count
        gram_norms.extend(
            float(value) for value in torch.linalg.vector_norm(gram_block, dim=1)
        )
        gram = prefix_gram_rows(
            final_norms=gram_norms,
            initial_norms=initial_norms,
            prefixes=(prefix,),
            power=1,
            stage_delta=stage_delta,
        )[0]
        gram_attempt = evaluate_attempt(
            kappa=float(gram["operator_norm_upper_bound"]),
            panel_row=source_row,
            certificate=certificate,
            corrected=corrected,
            correction=correction,
            dimension=dimension,
            learning_rate=config.learning_rate,
            cert_pairs=cert_pairs,
            cert_labels=cert_labels,
            template=template,
            spec=spec,
        )
        stage["gram"] = {**gram, **gram_attempt}
        stages.append(stage)
        if gram_attempt["issued"]:
            route = "gram_fallback"
            final_attempt = gram_attempt
            break

    full_sweeps = int(direct_row["logical_total_green_sweeps"])
    return {
        "version": VERSION,
        "evidence_boundary": "post-v1.0.1 outcome-blind method audit",
        "candidate": candidate.__dict__,
        "horizon": horizon,
        "route": route,
        "issued": final_attempt is not None and final_attempt["issued"],
        "bracket": None if final_attempt is None else final_attempt["bracket"],
        "inherited_bracket": source_row["bracket"],
        "bracket_preserved": final_attempt is not None and final_attempt["bracket"] == source_row["bracket"],
        "prefix": len(initial_norms),
        "logical_forward_green_sweeps": forward,
        "logical_transpose_green_sweeps": transposed,
        "logical_total_green_sweeps": forward + transposed,
        "full_state_staged_green_sweeps": full_sweeps,
        "sweep_reduction": full_sweeps / (forward + transposed),
        "probe_seed": probe_seed(candidate, horizon),
        "probe_hashes": probe_hashes,
        "initial_probe_norms": initial_norms,
        "direct_image_norms": image_norms,
        "gram_norms": gram_norms,
        "stages": stages,
        "corrected_path_sha256": tensor_sha256(corrected_scaled),
        "certificate_sha256": certificate_sha,
        "outcome_files_read": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }


def save_case(row: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    candidate = Candidate(**row["candidate"])
    cache_path(candidate).write_text(
        json.dumps(row, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    assert_protocol_frozen()
    cases = list(CASE_ROWS)
    if args.case_index is not None:
        cases = [cases[args.case_index]]

    rows = []
    if args.workers == 1 or len(cases) == 1:
        for case in cases:
            row = audit_case(case)
            save_case(row)
            rows.append(row)
            print(json.dumps({key: row[key] for key in ("candidate", "route", "bracket", "logical_total_green_sweeps", "sweep_reduction")}, indent=2))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(audit_case, case): case for case in cases}
            for future in as_completed(futures):
                row = future.result()
                save_case(row)
                rows.append(row)
                print(json.dumps({key: row[key] for key in ("candidate", "route", "bracket", "logical_total_green_sweeps", "sweep_reduction")}, indent=2))
    rows.sort(key=lambda row: (row["candidate"]["seed"], row["candidate"]["threshold"], row["candidate"]["anchor"]))
    if len(rows) == len(CASE_ROWS):
        full_total = sum(row["full_state_staged_green_sweeps"] for row in rows)
        structured_total = sum(row["logical_total_green_sweeps"] for row in rows)
        payload = {
            "status": "structured parameter Green Transformer audit complete",
            "version": VERSION,
            "evidence_boundary": "post-v1.0.1 outcome-blind method audit",
            "cases": len(rows),
            "issued": sum(row["issued"] for row in rows),
            "brackets_preserved": sum(row["bracket_preserved"] for row in rows),
            "route_distribution": dict(Counter(row["route"] for row in rows)),
            "full_state_staged_green_sweeps": full_total,
            "structured_parameter_green_sweeps": structured_total,
            "aggregate_sweep_reduction": full_total / structured_total,
            "median_pairwise_sweep_reduction": statistics.median(row["sweep_reduction"] for row in rows),
            "minimum_pairwise_sweep_reduction": min(row["sweep_reduction"] for row in rows),
            "maximum_pairwise_sweep_reduction": max(row["sweep_reduction"] for row in rows),
            "combined_family_failure_upper": FAMILY_FAILURE + 1.0e-6,
            "outcome_files_read": 0,
            "case_set_sha256": case_set_sha256(),
            "protocol_sha256": sha256(PROTOCOL),
            "dependency_sha256": dependency_hashes(),
            "rows": rows,
        }
        OUTPUT.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        print(json.dumps({key: payload[key] for key in payload if key != "rows" and key != "dependency_sha256"}, indent=2))


if __name__ == "__main__":
    main()
