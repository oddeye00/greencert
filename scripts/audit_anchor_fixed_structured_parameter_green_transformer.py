#!/usr/bin/env python3
"""Outcome-blind audit of the anchor-fixed structured Green operator.

This post-release audit restricts nonlinear forcing to updates 1..H-1.  The
update-0 remainder is exactly zero because the realized anchor is fixed.  The
method, cohort, probe schedule, family budget, nonce, and promotion rule are
sealed in ANCHOR_FIXED_STRUCTURED_PARAMETER_GREEN_AUDIT_PROTOCOL.md.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

import audit_structured_parameter_green_transformer as base
from direct_image_green_bound import direct_image_rows
from prefix_gram_enclosure import equal_family_stage_delta, prefix_gram_rows
from structured_parameter_green_v2 import (
    make_batched_anchor_fixed_profiled_structured_parameter_green_products,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "anchor_fixed_structured_parameter_green_transformer_audit.json"
CACHE = ROOT / "results" / "anchor_fixed_structured_parameter_green_transformer_cache"
PROTOCOL = ROOT / "ANCHOR_FIXED_STRUCTURED_PARAMETER_GREEN_AUDIT_PROTOCOL.md"
THEOREM = ROOT / "STRUCTURED_PARAMETER_GREEN_THEOREM_V2.md"
INDEXING_NOTE = ROOT / "STRUCTURED_PARAMETER_GREEN_THEOREM_V1_INDEXING_NOTE.md"
STRUCTURED_BASELINE = ROOT / "results" / "structured_parameter_green_transformer_audit.json"
VERSION = 3
MASTER_NONCE = "d784fcf9c34ecb9372c4c20a492838406017c3a7cad9f0c42eff766b10ced7be"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def dependency_paths() -> tuple[Path, ...]:
    paths = (
        Path(__file__),
        ROOT / "scripts" / "structured_parameter_green_v2.py",
        ROOT / "scripts" / "test_structured_parameter_green_v2.py",
        THEOREM,
        INDEXING_NOTE,
        STRUCTURED_BASELINE,
        *base.dependency_paths(),
    )
    # Keep first occurrence while preserving the reviewable order.
    return tuple(dict.fromkeys(paths))


def dependency_hashes() -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in dependency_paths()
    }


def assert_protocol_frozen() -> None:
    if not PROTOCOL.is_file():
        raise RuntimeError("anchor-fixed protocol is absent; refusing probes")
    text = PROTOCOL.read_text(encoding="utf-8").upper()
    required = {
        "MASTER_NONCE": MASTER_NONCE.upper(),
        "CASE_SET_SHA256": base.case_set_sha256(),
        **{f"DEPENDENCY:{name}": value for name, value in dependency_hashes().items()},
    }
    missing = [name for name, value in required.items() if value not in text]
    if missing:
        raise RuntimeError("anchor-fixed protocol mismatch: " + ", ".join(missing))


def probe_seed(candidate: base.Candidate, horizon: int) -> int:
    payload = (
        f"{MASTER_NONCE}|{candidate.seed}|{candidate.gate_index}|"
        f"{candidate.anchor}|{horizon}|anchor-fixed-structured-v{VERSION}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def cache_path(candidate: base.Candidate) -> Path:
    return CACHE / (
        f"seed_{candidate.seed}_gate_{candidate.gate_index}_"
        f"anchor_{candidate.anchor}_v{VERSION}.json"
    )


def baseline_index() -> dict[tuple[int, float, int], dict]:
    payload = base.safe_json(STRUCTURED_BASELINE)
    return {
        (
            int(row["candidate"]["seed"]),
            float(row["candidate"]["threshold"]),
            int(row["candidate"]["anchor"]),
        ): row
        for row in payload["rows"]
    }


def audit_case(case: tuple[int, float, int, int, str]) -> dict:
    assert_protocol_frozen()
    started = time.perf_counter()
    seed, threshold, anchor, horizon, certificate_sha = case
    candidate = base.Candidate(seed, threshold, anchor)
    source_row = base.panel_index()[(seed, threshold, anchor)]
    structured_row = baseline_index()[(seed, threshold, anchor)]
    certificate_path = base.output_path(candidate)
    if sha256(certificate_path) != certificate_sha:
        raise RuntimeError(f"certificate hash mismatch for {candidate}")
    certificate = base.safe_json(certificate_path)
    config, template, spec, data, parameter, velocity = base.load_candidate(candidate)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = data
    dimension = int(parameter.numel())

    path = base.build_streaming_transformer_centerline(
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
        base.make_scaled_optimizer_jvp_vjp(
            center[step, :dimension],
            train_pairs,
            train_labels,
            template,
            spec,
            config,
        )
        for step in range(horizon)
    ]
    old_apply, _ = base.make_causal_green_products(
        [row[0] for row in old_products],
        [row[1] for row in old_products],
        2 * dimension,
    )
    correction_rows = old_apply(residual.reshape(-1)).reshape(horizon, -1)
    correction = torch.cat(
        (torch.zeros_like(correction_rows[:1]), correction_rows), dim=0
    )
    corrected_scaled = scaled_center + correction
    if base.tensor_sha256(corrected_scaled) != source_row["corrected_path_sha256"]:
        raise RuntimeError(f"corrected path mismatch for {candidate}")
    corrected = base.from_scaled(
        corrected_scaled, dimension, config.learning_rate
    )

    products = [
        base.make_batched_scaled_optimizer_products(
            corrected[step, :dimension],
            train_pairs,
            train_labels,
            template,
            spec,
            config,
        )
        for step in range(horizon)
    ]
    apply, transpose = (
        make_batched_anchor_fixed_profiled_structured_parameter_green_products(
            [row[0] for row in products],
            [row[1] for row in products],
            dimension,
            config.learning_rate,
            [1.0] * horizon,
        )
    )

    generator = torch.Generator(device=corrected.device).manual_seed(
        probe_seed(candidate, horizon)
    )
    stage_delta = equal_family_stage_delta(
        family_failure=base.FAMILY_FAILURE,
        operators=len(base.CASE_ROWS),
        prefixes=base.PREFIXES,
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
    reduced_dimension = max(0, horizon - 1) * dimension
    for prefix in base.PREFIXES:
        count = prefix - len(initial_norms)
        vectors = torch.stack(
            [
                torch.randn(
                    reduced_dimension,
                    generator=generator,
                    dtype=corrected.dtype,
                    device=corrected.device,
                )
                for _ in range(count)
            ]
        )
        probe_hashes.extend(
            hashlib.sha256(
                row.detach().cpu().numpy().tobytes(order="C")
            ).hexdigest().upper()
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
        direct_attempt = base.evaluate_attempt(
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
        stage = {
            "prefix": prefix,
            "direct": {**direct, **direct_attempt},
            "gram": None,
        }
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
        gram_attempt = base.evaluate_attempt(
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

    structured_sweeps = int(structured_row["logical_total_green_sweeps"])
    total = forward + transposed
    return {
        "version": VERSION,
        "evidence_boundary": "post-v1.0.1 outcome-blind theorem audit",
        "candidate": candidate.__dict__,
        "horizon": horizon,
        "operator_input_blocks": max(0, horizon - 1),
        "anchor_block_omitted": True,
        "route": route,
        "issued": final_attempt is not None and final_attempt["issued"],
        "bracket": None if final_attempt is None else final_attempt["bracket"],
        "inherited_bracket": source_row["bracket"],
        "bracket_preserved": (
            final_attempt is not None
            and final_attempt["bracket"] == source_row["bracket"]
        ),
        "prefix": len(initial_norms),
        "logical_forward_green_sweeps": forward,
        "logical_transpose_green_sweeps": transposed,
        "logical_total_green_sweeps": total,
        "full_state_staged_green_sweeps": int(
            structured_row["full_state_staged_green_sweeps"]
        ),
        "unrestricted_structured_green_sweeps": structured_sweeps,
        "reduction_vs_unrestricted_structured": structured_sweeps / total,
        "reduction_vs_full_state": (
            int(structured_row["full_state_staged_green_sweeps"]) / total
        ),
        "probe_seed": probe_seed(candidate, horizon),
        "probe_hashes": probe_hashes,
        "initial_probe_norms": initial_norms,
        "direct_image_norms": image_norms,
        "gram_norms": gram_norms,
        "stages": stages,
        "corrected_path_sha256": base.tensor_sha256(corrected_scaled),
        "certificate_sha256": certificate_sha,
        "outcome_files_read": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }


def save_case(row: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    candidate = base.Candidate(**row["candidate"])
    cache_path(candidate).write_text(
        json.dumps(row, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def aggregate(rows: list[dict]) -> dict:
    rows.sort(
        key=lambda row: (
            row["candidate"]["seed"],
            row["candidate"]["threshold"],
            row["candidate"]["anchor"],
        )
    )
    full_total = sum(row["full_state_staged_green_sweeps"] for row in rows)
    unrestricted_total = sum(
        row["unrestricted_structured_green_sweeps"] for row in rows
    )
    restricted_total = sum(row["logical_total_green_sweeps"] for row in rows)
    payload = {
        "status": "anchor-fixed structured parameter Green audit complete",
        "version": VERSION,
        "evidence_boundary": "post-v1.0.1 outcome-blind theorem audit",
        "cases": len(rows),
        "issued": sum(row["issued"] for row in rows),
        "brackets_preserved": sum(row["bracket_preserved"] for row in rows),
        "route_distribution": dict(Counter(row["route"] for row in rows)),
        "full_state_staged_green_sweeps": full_total,
        "unrestricted_structured_green_sweeps": unrestricted_total,
        "anchor_fixed_structured_green_sweeps": restricted_total,
        "reduction_vs_unrestricted_structured": unrestricted_total / restricted_total,
        "reduction_vs_full_state": full_total / restricted_total,
        "sweeps_saved_vs_unrestricted_structured": unrestricted_total - restricted_total,
        "median_pairwise_reduction_vs_unrestricted": statistics.median(
            row["reduction_vs_unrestricted_structured"] for row in rows
        ),
        "minimum_pairwise_reduction_vs_unrestricted": min(
            row["reduction_vs_unrestricted_structured"] for row in rows
        ),
        "maximum_pairwise_reduction_vs_unrestricted": max(
            row["reduction_vs_unrestricted_structured"] for row in rows
        ),
        "combined_family_failure_upper": base.FAMILY_FAILURE + 1.0e-6,
        "outcome_files_read": sum(row["outcome_files_read"] for row in rows),
        "promotion_gate_passed": (
            len(rows) == len(base.CASE_ROWS)
            and all(row["bracket_preserved"] for row in rows)
            and restricted_total < unrestricted_total
        ),
        "case_set_sha256": base.case_set_sha256(),
        "protocol_sha256": sha256(PROTOCOL),
        "dependency_sha256": dependency_hashes(),
        "rows": rows,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    assert_protocol_frozen()
    cases = list(base.CASE_ROWS)
    if args.case_index is not None:
        cases = [cases[args.case_index]]

    rows = []
    if args.workers == 1 or len(cases) == 1:
        for case in cases:
            row = audit_case(case)
            save_case(row)
            rows.append(row)
            print(
                json.dumps(
                    {
                        key: row[key]
                        for key in (
                            "candidate",
                            "route",
                            "bracket",
                            "logical_total_green_sweeps",
                            "reduction_vs_unrestricted_structured",
                        )
                    },
                    indent=2,
                ),
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(audit_case, case): case for case in cases}
            for future in as_completed(futures):
                row = future.result()
                save_case(row)
                rows.append(row)
                print(
                    json.dumps(
                        {
                            key: row[key]
                            for key in (
                                "candidate",
                                "route",
                                "bracket",
                                "logical_total_green_sweeps",
                                "reduction_vs_unrestricted_structured",
                            )
                        },
                        indent=2,
                    ),
                    flush=True,
                )
    if len(rows) == len(base.CASE_ROWS):
        payload = aggregate(rows)
        OUTPUT.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        print(
            json.dumps(
                {
                    key: payload[key]
                    for key in payload
                    if key not in ("rows", "dependency_sha256")
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
