#!/usr/bin/env python3
"""Outcome-blind staged direct-image/Gram audit on the 15 corrected operators."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from batched_green_operator import make_batched_transformer_green_products
from direct_image_green_bound import direct_image_rows
from prefix_gram_enclosure import prefix_gram_rows
from relinearized_green_closure import exact_relinearized_closure
from streaming_variational_centerline import build_streaming_transformer_centerline
from transformer_certificate_protocol import Candidate
from transformer_green_operator import make_causal_green_products
from transformer_optimizer_probe import make_scaled_optimizer_jvp_vjp
from transformer_v3_certificate import load_candidate, output_path, safe_json
from audit_transformer_relinearized_prefix_panel import (
    CASE_ROWS,
    FAMILY_FAILURE,
    MASTER_NONCE,
    PREFIXES,
    TWO_RESPONSE,
    from_scaled,
    identity,
    output_bracket,
)


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "results" / "transformer_v3_relinearized_prefix_panel_audit.json"
OUTPUT = ROOT / "results" / "transformer_direct_image_green_panel_audit.json"
CACHE = ROOT / "results" / "transformer_direct_image_green_panel_cache"
PROTOCOL = ROOT / "DIRECT_IMAGE_GREEN_PANEL_PROTOCOL.md"
VERSION = 1
EXPECTED_PANEL_SHA256 = "08E501B51FEAC3D96FFE02BE0B5D84E0E682C2E73CB906C083ED0FEF7E75E12B"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes(order="C")
    ).hexdigest().upper()


def close(left: float, right: float, tolerance: float = 3.0e-12) -> bool:
    return math.isclose(
        float(left), float(right), rel_tol=tolerance, abs_tol=1.0e-300
    )


def dependency_paths() -> tuple[Path, ...]:
    return (
        Path(__file__),
        ROOT / "scripts" / "direct_image_green_bound.py",
        ROOT / "scripts" / "streaming_variational_centerline.py",
        ROOT / "scripts" / "batched_green_operator.py",
        ROOT / "scripts" / "audit_transformer_relinearized_prefix_panel.py",
        ROOT / "scripts" / "relinearized_green_closure.py",
        ROOT / "scripts" / "transformer_green_operator.py",
        ROOT / "scripts" / "transformer_optimizer_probe.py",
        ROOT / "scripts" / "transformer_v3_certificate.py",
        ROOT / "DIRECT_IMAGE_GREEN_THEOREM.md",
        ROOT / "CAUSAL_PREFIX_RECENTERING_THEOREM.md",
        PANEL,
        TWO_RESPONSE,
    )


def dependency_hashes() -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in dependency_paths()
    }


def assert_protocol_frozen() -> None:
    if not PROTOCOL.exists():
        raise RuntimeError("direct-image protocol is absent; refusing operator queries")
    text = PROTOCOL.read_text(encoding="utf-8").upper()
    required = {
        "PANEL": EXPECTED_PANEL_SHA256,
        "NONCE": MASTER_NONCE.upper(),
        **{f"DEPENDENCY:{name}": value for name, value in dependency_hashes().items()},
    }
    missing = [name for name, value in required.items() if value not in text]
    if missing:
        raise RuntimeError("direct-image protocol seal mismatch: " + ", ".join(missing))


def panel_index() -> dict[tuple[int, float, int], dict]:
    payload = safe_json(PANEL)
    if sha256(PANEL) != EXPECTED_PANEL_SHA256:
        raise RuntimeError("corrected-path panel result changed")
    return {
        (
            int(row["candidate"]["seed"]),
            float(row["candidate"]["threshold"]),
            int(row["candidate"]["anchor"]),
        ): row
        for row in payload["rows"]
    }


def cache_path(candidate: Candidate) -> Path:
    return CACHE / (
        f"seed_{candidate.seed}_gate_{candidate.gate_index}_"
        f"anchor_{candidate.anchor}_v{VERSION}.json"
    )


def evaluate(
    *,
    kappa: float,
    panel_row: dict,
    certificate: dict,
    corrected: torch.Tensor,
    correction: torch.Tensor,
    dimension: int,
    cert_pairs: torch.Tensor,
    cert_labels: torch.Tensor,
    template,
    spec,
) -> dict:
    forcing = kappa * float(panel_row["total_corrected_injection_upper"])
    closure = exact_relinearized_closure(
        kappa=kappa,
        derivative_drift=float(panel_row["derivative_drift_upper"]),
        corrected_defect_response_bound=forcing,
        correction_max_state_norm=float(panel_row["correction_max_state_norm"]),
        domain_radius=float(panel_row["domain_radius"]),
    )
    event = {
        "bracket": None,
        "output_power": None,
        "logic_slack": None,
        "maximum_margin_radius": None,
    }
    if closure.closure_passed:
        event = output_bracket(
            certificate=certificate,
            corrected=corrected,
            correction=correction,
            dimension=dimension,
            cert_pairs=cert_pairs,
            cert_labels=cert_labels,
            template=template,
            spec=spec,
            radius=float(closure.remainder_radius),
        )
    return {
        "forcing_response_upper": forcing,
        "closure": closure.as_dict(),
        **event,
        "issued": event["bracket"] is not None,
    }


def audit_case(case: tuple[int, float, int, int, str]) -> dict:
    assert_protocol_frozen()
    started = time.perf_counter()
    seed, threshold, anchor, horizon, certificate_sha = case
    candidate = Candidate(seed, threshold, anchor)
    panel_row = panel_index()[(seed, threshold, anchor)]
    certificate_path = output_path(candidate)
    if sha256(certificate_path) != certificate_sha:
        raise RuntimeError(f"certificate hash mismatch for {candidate}")
    certificate = safe_json(certificate_path)
    config, template, spec, data, parameter, velocity = load_candidate(candidate)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = data
    dimension = int(parameter.numel())
    timings = {}

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
    )
    timings["streaming_centerline"] = time.perf_counter() - phase
    center = path["center"]
    scaled_center = path["scaled_center"]

    phase = time.perf_counter()
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
    correction = torch.cat(
        (torch.zeros_like(correction_rows[:1]), correction_rows), dim=0
    )
    corrected_scaled = scaled_center + correction
    corrected = from_scaled(corrected_scaled, dimension, config.learning_rate)
    timings["corrected_path"] = time.perf_counter() - phase
    if tensor_sha256(corrected_scaled) != panel_row["corrected_path_sha256"]:
        raise RuntimeError(f"corrected path mismatch for {candidate}")
    if not close(
        torch.linalg.vector_norm(correction_rows),
        panel_row["correction_sequence_norm"],
    ):
        raise RuntimeError(f"correction norm mismatch for {candidate}")

    phase = time.perf_counter()
    batch_apply, batch_transpose = make_batched_transformer_green_products(
        corrected[:horizon, :dimension],
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )
    timings["operator_setup"] = time.perf_counter() - phase

    generator = torch.Generator(device=corrected.device).manual_seed(
        int(panel_row["probe_seed"])
    )
    initial_norms = []
    image_norms = []
    gram_norms = []
    probe_hashes = []
    stage_rows = []
    forward_seconds = 0.0
    transpose_seconds = 0.0
    transpose_applications = 0
    route = None
    bracket = None
    stage_delta = float(panel_row["stage_delta"])

    for stage_index, prefix in enumerate(PREFIXES):
        count = prefix - len(initial_norms)
        vectors = []
        for _ in range(count):
            vector = torch.randn(
                horizon * 2 * dimension,
                generator=generator,
                dtype=corrected.dtype,
                device=corrected.device,
            )
            digest = hashlib.sha256(
                vector.detach().cpu().numpy().tobytes(order="C")
            ).hexdigest().upper()
            expected = panel_row["probe_hashes"][len(initial_norms)]
            if digest != expected:
                raise RuntimeError(f"probe hash mismatch for {candidate}")
            probe_hashes.append(digest)
            initial_norms.append(float(torch.linalg.vector_norm(vector)))
            vectors.append(vector)
        block = torch.stack(vectors)
        phase = time.perf_counter()
        images = batch_apply(block)
        forward_seconds += time.perf_counter() - phase
        image_norms.extend(
            float(value) for value in torch.linalg.vector_norm(images, dim=1)
        )
        direct = direct_image_rows(
            image_norms=image_norms,
            initial_norms=initial_norms,
            prefixes=(prefix,),
            stage_delta=stage_delta,
        )[0]
        direct_attempt = evaluate(
            kappa=float(direct["operator_norm_upper_bound"]),
            panel_row=panel_row,
            certificate=certificate,
            corrected=corrected,
            correction=correction,
            dimension=dimension,
            cert_pairs=cert_pairs,
            cert_labels=cert_labels,
            template=template,
            spec=spec,
        )
        stage = {"prefix": prefix, "direct": {**direct, **direct_attempt}, "gram": None}
        if direct_attempt["issued"]:
            route = "direct_image"
            bracket = direct_attempt["bracket"]
            stage_rows.append(stage)
            break

        phase = time.perf_counter()
        gram_block = batch_transpose(images)
        transpose_seconds += time.perf_counter() - phase
        transpose_applications += count
        new_gram_norms = [
            float(value) for value in torch.linalg.vector_norm(gram_block, dim=1)
        ]
        start = len(gram_norms)
        for offset, observed in enumerate(new_gram_norms):
            expected = float(panel_row["final_probe_norms"][start + offset])
            if not close(observed, expected, 2.0e-11):
                raise RuntimeError(f"Gram norm mismatch for {candidate}")
        gram_norms.extend(new_gram_norms)
        gram = prefix_gram_rows(
            final_norms=gram_norms,
            initial_norms=initial_norms,
            prefixes=(prefix,),
            power=1,
            stage_delta=stage_delta,
        )[0]
        gram_attempt = evaluate(
            kappa=float(gram["operator_norm_upper_bound"]),
            panel_row=panel_row,
            certificate=certificate,
            corrected=corrected,
            correction=correction,
            dimension=dimension,
            cert_pairs=cert_pairs,
            cert_labels=cert_labels,
            template=template,
            spec=spec,
        )
        stage["gram"] = {**gram, **gram_attempt}
        stage_rows.append(stage)
        if gram_attempt["issued"]:
            route = "gram_fallback"
            bracket = gram_attempt["bracket"]
            break

    if route is None or bracket is None:
        raise RuntimeError(f"staged direct/Gram audit abstained for {candidate}")
    if bracket != panel_row["bracket"]:
        raise RuntimeError(f"staged bracket mismatch for {candidate}")
    logical_forward = len(initial_norms)
    logical_total = logical_forward + transpose_applications
    old_probe_sweeps = 2 * int(panel_row["relinearized_green_gram_applications"])
    return {
        "version": VERSION,
        "candidate": candidate.__dict__,
        "horizon": horizon,
        "route": route,
        "bracket": bracket,
        "panel_bracket": panel_row["bracket"],
        "prefix": len(initial_norms),
        "stage_rows": stage_rows,
        "probe_hashes": probe_hashes,
        "initial_probe_norms": initial_norms,
        "direct_image_norms": image_norms,
        "gram_norms_computed": gram_norms,
        "logical_forward_green_applications": logical_forward,
        "logical_transpose_green_applications": transpose_applications,
        "logical_total_green_sweeps": logical_total,
        "panel_logical_green_sweeps": old_probe_sweeps,
        "probe_sweep_reduction": old_probe_sweeps / logical_total,
        "transpose_sweeps_avoided": logical_forward - transpose_applications,
        "timings_seconds": {
            **timings,
            "direct_forward_queries": forward_seconds,
            "transpose_fallback_queries": transpose_seconds,
            "staged_green_queries": forward_seconds + transpose_seconds,
            "total": time.perf_counter() - started,
        },
        "corrected_path_sha256": panel_row["corrected_path_sha256"],
        "panel_result_sha256": EXPECTED_PANEL_SHA256,
        "certificate_sha256": certificate_sha,
        "protocol_sha256": sha256(PROTOCOL),
        "source_sha256": sha256(Path(__file__)),
        "dependency_sha256": dependency_hashes(),
        "combined_family_failure_upper": 2.0e-6,
        "outcome_files_read": 0,
    }


def valid_cache(case, row: dict) -> bool:
    candidate = Candidate(case[0], case[1], case[2])
    return (
        int(row.get("version", -1)) == VERSION
        and row.get("candidate") == candidate.__dict__
        and row.get("panel_result_sha256") == EXPECTED_PANEL_SHA256
        and row.get("protocol_sha256") == sha256(PROTOCOL)
        and row.get("source_sha256") == sha256(Path(__file__))
        and row.get("dependency_sha256") == dependency_hashes()
        and int(row.get("outcome_files_read", -1)) == 0
    )


def write_cache(candidate: Candidate, row: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    destination = cache_path(candidate)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def aggregate(rows: list[dict], elapsed: float) -> dict:
    rows.sort(
        key=lambda row: (
            row["candidate"]["seed"],
            row["candidate"]["threshold"],
            row["candidate"]["anchor"],
        )
    )
    old = sum(row["panel_logical_green_sweeps"] for row in rows)
    new = sum(row["logical_total_green_sweeps"] for row in rows)
    return {
        "status": "STAGED DIRECT-IMAGE/GRAM PANEL COMPLETED",
        "evidence_boundary": (
            "Post-seal, outcome-blind implementation audit reusing the frozen "
            "64-vector family. Direct and Gram releases share one Gaussian "
            "event. Neural products and margins remain float64/high-confidence."
        ),
        "cases": len(rows),
        "issued": sum(row["bracket"] is not None for row in rows),
        "route_distribution": {
            route: sum(row["route"] == route for row in rows)
            for route in ("direct_image", "gram_fallback")
        },
        "prefix_distribution": {
            str(prefix): sum(row["prefix"] == prefix for row in rows)
            for prefix in PREFIXES
        },
        "panel_green_probe_sweeps": old,
        "staged_green_probe_sweeps": new,
        "aggregate_probe_sweep_reduction": old / new,
        "transpose_sweeps_avoided": sum(
            row["transpose_sweeps_avoided"] for row in rows
        ),
        "median_pairwise_probe_sweep_reduction": statistics.median(
            row["probe_sweep_reduction"] for row in rows
        ),
        "minimum_pairwise_probe_sweep_reduction": min(
            row["probe_sweep_reduction"] for row in rows
        ),
        "maximum_pairwise_probe_sweep_reduction": max(
            row["probe_sweep_reduction"] for row in rows
        ),
        "aggregate_staged_green_seconds": sum(
            row["timings_seconds"]["staged_green_queries"] for row in rows
        ),
        "aggregate_streaming_centerline_seconds": sum(
            row["timings_seconds"]["streaming_centerline"] for row in rows
        ),
        "aggregate_case_seconds": sum(row["timings_seconds"]["total"] for row in rows),
        "wall_seconds": elapsed,
        "panel_result_sha256": EXPECTED_PANEL_SHA256,
        "protocol_sha256": sha256(PROTOCOL),
        "source_sha256": sha256(Path(__file__)),
        "dependency_sha256": dependency_hashes(),
        "combined_family_failure_upper": 2.0e-6,
        "outcome_files_read": sum(row["outcome_files_read"] for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    assert_protocol_frozen()
    panel = panel_index()
    expected = {(s, g, a) for s, g, a, _, _ in CASE_ROWS}
    if set(panel) != expected:
        raise RuntimeError("direct-image candidate family differs from panel")
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "DIRECT-IMAGE PANEL INPUT VALIDATION PASSED",
                    "cases": len(panel),
                    "panel_sha256": sha256(PANEL),
                    "dependency_sha256": dependency_hashes(),
                    "protocol_sha256": sha256(PROTOCOL),
                    "outcome_files_read": 0,
                },
                indent=2,
            )
        )
        return

    cases = list(CASE_ROWS)
    if args.seed is not None:
        cases = [row for row in cases if row[0] == args.seed]
        if not cases:
            raise ValueError("seed is outside the frozen panel")
    started = time.perf_counter()
    rows = []
    pending = []
    for case in cases:
        candidate = Candidate(case[0], case[1], case[2])
        destination = cache_path(candidate)
        if destination.exists():
            cached = safe_json(destination)
            if valid_cache(case, cached):
                rows.append(cached)
                print(f"reused {candidate}", flush=True)
                continue
        pending.append(case)
    with ProcessPoolExecutor(
        max_workers=min(max(1, args.workers), max(1, len(pending)))
    ) as pool:
        futures = {pool.submit(audit_case, case): case for case in pending}
        for future in as_completed(futures):
            case = futures[future]
            row = future.result()
            candidate = Candidate(case[0], case[1], case[2])
            write_cache(candidate, row)
            rows.append(row)
            print(
                f"audited {candidate} route={row['route']} prefix={row['prefix']}",
                flush=True,
            )
    if args.seed is not None:
        print(json.dumps({"seed": args.seed, "rows": rows}, indent=2))
        return
    payload = aggregate(rows, time.perf_counter() - started)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
