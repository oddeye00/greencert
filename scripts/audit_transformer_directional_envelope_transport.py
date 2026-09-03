#!/usr/bin/env python3
"""Frozen four-case audit of directionally transported neural envelopes."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch

from analytic_jet_release import logit_margin_radius, scaled_momentum_jacobian_drift
from audit_transformer_adaptive_sweep_cohort import raw_slacks
from audit_transformer_direct_image_green_panel import tensor_sha256
from audit_transformer_relinearized_prefix_panel import from_scaled
from corrected_path_closure import exact_corrected_path_closure
from streaming_variational_centerline_v15 import build_streaming_transformer_centerline
from transformer_block_envelope_v15 import (
    ball_valid_envelope,
    exact_stage_values,
    parameter_geometry,
)
from transformer_certificate_protocol import Candidate
from transformer_hvp_grokking import logits
from transformer_mixed_directional_jet_v15 import (
    directionally_transported_envelope_inputs,
    mixed_directional_objective_fourth_bound,
)
from transformer_optimizer_probe_v15 import scaled_optimizer_map_and_jvp
from transformer_v3_certificate import (
    _logic_slack,
    _persistent_bracket,
    load_candidate,
    output_path,
    safe_json,
    verify_method_seal,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CLOSURE_PARENT = RESULTS / "transformer_directional_block_remainder_diagnostic.json"
FULL_PARENT = RESULTS / "transformer_fully_recentered_three_sweep_audit.json"
EVENT_PARENT = RESULTS / "transformer_directional_three_sweep_event_audit.json"
PROTOCOL = ROOT / "DIRECTIONAL_ENVELOPE_TRANSPORT_AUDIT_PROTOCOL.md"
THEOREM = ROOT / "DIRECTIONAL_ENVELOPE_TRANSPORT_THEOREM.md"
AMENDMENT = ROOT / "DIRECTIONAL_ENVELOPE_TRANSPORT_SOURCE_ISOLATION_AMENDMENT.md"
OUTPUT = RESULTS / "transformer_directional_envelope_transport_audit.json"
SWEEPS = 3
DEVELOPMENT = (366, 0.8, 1120)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def identity(candidate: dict) -> tuple[int, float, int]:
    return (
        int(candidate["seed"]),
        float(candidate["threshold"]),
        int(candidate["anchor"]),
    )


def close(left: float, right: float, tolerance: float = 3.0e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=1.0e-300)


def dominates(upper: float, lower: float) -> bool:
    return float(upper) >= float(lower) * (1.0 - 3.0e-13) - 1.0e-14


def run_case(task: dict) -> dict:
    closure_parent = task["closure"]
    full_parent = task["full"]
    event_parent = task["event"]
    candidate = Candidate(**closure_parent["candidate"])
    config, template, spec, data, parameter, velocity = load_candidate(candidate)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = data
    dimension = int(parameter.numel())
    eta = float(config.learning_rate)
    horizon = int(closure_parent["horizon"])
    timings: dict[str, float] = {}
    started = time.perf_counter()

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
        sweeps=SWEEPS,
        fused_derivatives=True,
    )
    timings["fused_three_sweep_centerline"] = time.perf_counter() - phase
    if path["centerline_sha256"] != full_parent["centerline_sha256"]:
        raise RuntimeError(f"centerline hash changed for {candidate}")
    center = path["center"]
    scaled_center = path["scaled_center"]

    phase = time.perf_counter()
    prior = torch.zeros_like(scaled_center[0])
    correction_rows = []
    for step in range(horizon):
        mapped, linear = scaled_optimizer_map_and_jvp(
            center[step],
            prior,
            train_pairs,
            train_labels,
            template,
            spec,
            config,
        )
        residual = mapped - scaled_center[step + 1]
        current = linear + residual
        correction_rows.append(current)
        prior = current
    correction = torch.cat(
        (torch.zeros_like(torch.stack(correction_rows)[:1]), torch.stack(correction_rows)),
        dim=0,
    )
    corrected_scaled = scaled_center + correction
    corrected = from_scaled(corrected_scaled, dimension, eta)
    timings["fused_signed_correction"] = time.perf_counter() - phase
    corrected_hash = tensor_sha256(corrected_scaled)
    if corrected_hash != closure_parent["corrected_path_sha256"]:
        raise RuntimeError(f"corrected path hash changed for {candidate}")
    if corrected_hash != full_parent["corrected_path_sha256"]:
        raise RuntimeError(f"full-parent path hash changed for {candidate}")

    phase = time.perf_counter()
    domain = float(full_parent["domain_radius_about_corrected_path"])
    blocks = []
    terms = []
    checks = 0
    max_stage_ratio = 1.0
    max_geometry_ratio = 1.0
    max_derivative_ratio = 1.0
    for step in range(1, horizon + 1):
        point = center[step, :dimension]
        direction = correction[step, :dimension]
        centre_values = exact_stage_values(point, spec, config)
        geometry = parameter_geometry(point, spec, config)
        mixed = mixed_directional_objective_fourth_bound(
            point,
            direction,
            spec,
            config,
            centre_values=centre_values,
            parameter_norms=geometry,
        )
        transported_values, transported_geometry = (
            directionally_transported_envelope_inputs(centre_values, geometry, mixed)
        )
        transported = ball_valid_envelope(
            point,
            spec,
            config,
            epsilon=domain,
            centre_values=transported_values,
            parameter_norms=transported_geometry,
        )
        if not bool(transported["fixed_point_consistent"]):
            raise RuntimeError(f"transported fixed point failed for {candidate}, step {step}")

        exact_point = corrected[step, :dimension]
        exact_values = exact_stage_values(exact_point, spec, config)
        exact_geometry = parameter_geometry(exact_point, spec, config)
        exact_envelope = ball_valid_envelope(
            exact_point,
            spec,
            config,
            epsilon=domain,
            centre_values=exact_values,
            parameter_norms=exact_geometry,
        )
        for name, exact in exact_values.items():
            upper = float(transported_values[name])
            if not dominates(upper, exact):
                raise RuntimeError(
                    f"stage transport failed for {candidate}, step {step}, {name}"
                )
            max_stage_ratio = max(max_stage_ratio, upper / max(float(exact), 1.0e-300))
            checks += 1
        for name, exact in exact_geometry.items():
            upper = float(transported_geometry[name])
            if not dominates(upper, exact):
                raise RuntimeError(
                    f"geometry transport failed for {candidate}, step {step}, {name}"
                )
            max_geometry_ratio = max(
                max_geometry_ratio, upper / max(float(exact), 1.0e-300)
            )
            checks += 1
        for name in ("first", "second", "third"):
            upper = float(transported[name])
            exact = float(exact_envelope[name])
            if not dominates(upper, exact):
                raise RuntimeError(
                    f"derivative transport failed for {candidate}, step {step}, {name}"
                )
            max_derivative_ratio = max(max_derivative_ratio, upper / max(exact, 1.0e-300))
            checks += 1
        if step < horizon:
            terms.append(
                math.sqrt(2.0)
                * eta
                * float(mixed["gradient_taylor_remainder_upper"])
            )
        blocks.append(transported)
    timings["transported_and_exact_audit_envelopes"] = time.perf_counter() - phase
    sequence = math.sqrt(sum(value * value for value in terms))
    if not close(sequence, closure_parent["directional_block_taylor_sequence_upper"]):
        raise RuntimeError(f"mixed directional sequence changed for {candidate}")

    phase = time.perf_counter()
    injection = (
        float(full_parent["response_recurrence_residual_norm"])
        + float(full_parent["quadratic_surrogate_injection_norm"])
        + sequence
    )
    if not close(injection, closure_parent["directional_cancellation_safe_injection_upper"]):
        raise RuntimeError(f"directional injection changed for {candidate}")
    kappa = float(closure_parent["unchanged_green_operator_norm_upper_bound"])
    drift = max(
        scaled_momentum_jacobian_drift(
            first=float(block["first"]),
            second=float(block["second"]),
            third=float(block["third"]),
            learning_rate=eta,
        )
        for block in blocks[:-1]
    )
    closure = exact_corrected_path_closure(
        kappa=kappa,
        derivative_drift=drift,
        defect_response_bound=kappa * injection,
        domain_radius=domain,
    )
    if not closure.closure_passed or closure.remainder_radius is None:
        raise RuntimeError(f"transported closure abstained for {candidate}")
    certificate = safe_json(output_path(candidate))
    required = int(certificate["required_correct"])
    raw = [
        raw_slacks(
            logits(corrected[step, :dimension], cert_pairs, template, spec),
            cert_labels,
            required,
        )
        for step in range(horizon + 1)
    ]
    margins = [0.0] + [
        logit_margin_radius(
            first=float(block["first"]), state_radius=float(closure.remainder_radius)
        )
        for block in blocks
    ]
    guarantee = [float(pair[0]) - margin for pair, margin in zip(raw, margins)]
    exclusion = [float(pair[1]) - margin for pair, margin in zip(raw, margins)]
    bracket = _persistent_bracket(guarantee, exclusion)
    logic_slack = _logic_slack(bracket, guarantee, exclusion)
    if bracket != event_parent["sealed_four_sweep_bracket"]:
        raise RuntimeError(f"transported bracket changed for {candidate}: {bracket}")
    timings["closure_and_event"] = time.perf_counter() - phase
    timings["end_to_end_including_exact_dominance_audit"] = time.perf_counter() - started

    return {
        "candidate": candidate.__dict__,
        "development_row": identity(candidate.__dict__) == DEVELOPMENT,
        "horizon": horizon,
        "same_centerline": True,
        "same_corrected_path": True,
        "corrected_path_sha256": corrected_hash,
        "mixed_sequence_upper": sequence,
        "transport_checks": checks,
        "maximum_stage_majorant_ratio": max_stage_ratio,
        "maximum_geometry_majorant_ratio": max_geometry_ratio,
        "maximum_derivative_envelope_ratio": max_derivative_ratio,
        "transported_derivative_drift": drift,
        "fresh_corrected_center_derivative_drift": float(
            closure_parent["closure"]["derivative_drift"]
        ),
        "closure": closure.as_dict(),
        "issued": True,
        "bracket": bracket,
        "logic_slack": logic_slack,
        "maximum_output_margin_radius": max(margins),
        "timings_seconds": timings,
        "outcome_files_read": 0,
    }


def summarize(rows: list[dict]) -> dict:
    return {
        "cases": len(rows),
        "issued": sum(bool(row["issued"]) for row in rows),
        "same_centerline": all(row["same_centerline"] for row in rows),
        "same_corrected_path": all(row["same_corrected_path"] for row in rows),
        "maximum_stage_majorant_ratio": max(
            row["maximum_stage_majorant_ratio"] for row in rows
        ),
        "maximum_geometry_majorant_ratio": max(
            row["maximum_geometry_majorant_ratio"] for row in rows
        ),
        "maximum_derivative_envelope_ratio": max(
            row["maximum_derivative_envelope_ratio"] for row in rows
        ),
        "median_end_to_end_seconds": statistics.median(
            row["timings_seconds"]["end_to_end_including_exact_dominance_audit"]
            for row in rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    verify_method_seal()
    closure_payload = safe_json(CLOSURE_PARENT)
    full_by_id = {
        identity(row["candidate"]): row for row in safe_json(FULL_PARENT)["rows"]
    }
    event_by_id = {
        identity(row["candidate"]): row for row in safe_json(EVENT_PARENT)["rows"]
    }
    closure_rows = [row for row in closure_payload["rows"] if row["closure_passed"]]
    if len(closure_rows) != 4:
        raise RuntimeError("frozen directional closure cohort changed")
    tasks = [
        {
            "closure": row,
            "full": full_by_id[identity(row["candidate"])],
            "event": event_by_id[identity(row["candidate"])],
        }
        for row in closure_rows
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_case, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                json.dumps(
                    {
                        "candidate": row["candidate"],
                        "bracket": row["bracket"],
                        "derivative_ratio": row["maximum_derivative_envelope_ratio"],
                        "seconds": row["timings_seconds"][
                            "end_to_end_including_exact_dominance_audit"
                        ],
                    }
                ),
                flush=True,
            )
    rows.sort(key=lambda row: identity(row["candidate"]))
    holdouts = [row for row in rows if not row["development_row"]]
    all_summary = summarize(rows)
    holdout_summary = summarize(holdouts)
    passed = (
        all_summary["issued"] == 4
        and all_summary["same_centerline"]
        and all_summary["same_corrected_path"]
        and len(holdouts) == 3
        and holdout_summary["issued"] == 3
        and all(row["bracket"] == event_by_id[identity(row["candidate"])]["sealed_four_sweep_bracket"] for row in rows)
        and all(row["outcome_files_read"] == 0 for row in rows)
    )
    source_hashes = {
        "closure_parent": sha256(CLOSURE_PARENT),
        "full_parent": sha256(FULL_PARENT),
        "event_parent": sha256(EVENT_PARENT),
        "protocol": sha256(PROTOCOL),
        "theorem": sha256(THEOREM),
        "source_isolation_amendment": sha256(AMENDMENT),
        "block_envelope_v15": sha256(
            ROOT / "scripts" / "transformer_block_envelope_v15.py"
        ),
        "hvp_v15": sha256(ROOT / "scripts" / "transformer_hvp_grokking_v15.py"),
        "modal_v15": sha256(ROOT / "scripts" / "transformer_modal_forecast_v15.py"),
        "mixed_jet_v15": sha256(
            ROOT / "scripts" / "transformer_mixed_directional_jet_v15.py"
        ),
        "streaming_centerline_v15": sha256(
            ROOT / "scripts" / "streaming_variational_centerline_v15.py"
        ),
        "historical_mixed_jet": sha256(
            ROOT / "scripts" / "transformer_mixed_directional_jet.py"
        ),
        "historical_streaming_centerline": sha256(
            ROOT / "scripts" / "streaming_variational_centerline.py"
        ),
        "optimizer_probe_v15": sha256(
            ROOT / "scripts" / "transformer_optimizer_probe_v15.py"
        ),
        "v3_method_seal": sha256(ROOT / "TRANSFORMER_V3_METHOD_SEAL.json"),
        "script": sha256(Path(__file__).resolve()),
    }
    result = {
        "status": "DIRECTIONAL ENVELOPE TRANSPORT AUDIT PASSED" if passed else "FAILED",
        "evidence_boundary": (
            "Post-release outcome-blind equivalence audit; reuses frozen forcing and "
            "Green constants, verifies the historical v3 source seal, reads no future "
            "outcome files, and is not new model evidence."
        ),
        "source_isolated_replay": True,
        "source_hashes": source_hashes,
        "all_cases": all_summary,
        "nondevelopment_cases": holdout_summary,
        "protocol_gates_passed": passed,
        "rows": rows,
        "outcome_files_read": 0,
    }
    OUTPUT.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    if not passed:
        raise RuntimeError("directional envelope transport protocol failed")


if __name__ == "__main__":
    main()
