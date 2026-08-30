#!/usr/bin/env python3
"""Outcome-blind post-seal audit of a residualized second Green response.

The frozen v3 certificate bounds ``||K N(z)||`` by ``kappa*M*p*Z/2``.
Theorem 1 already permits a sharper response ``y=K q`` with
``q=N(z)``.  Directly evaluating ``N(z)`` is catastrophically ill-conditioned
at the observed response scale, so this audit uses the cancellation-safe
center quadratic term ``D^2G(c)[z,z]/2``.  It measures whether a practical
two-response implementation has enough potential to justify adding a verified
fourth-derivative remainder in future work.

This is method-development evidence only: it reads no future outcomes, changes
no frozen certificate, and does not claim that the float64 quadratic surrogate
is an outward enclosure of the exact nonlinear defect.
"""

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

from one_shot_recenter_closure import exact_one_shot_closure
from transformer_four_sweep_development_audit import to_scaled
from transformer_green_development_audit import build_frozen_centerline
from transformer_green_operator import make_transformer_green_products
from transformer_two_response import optimizer_center_quadratic_defect
from transformer_v3_certificate import (
    _bracket_at_radius,
    _gate_raw_slacks,
    load_candidate,
    safe_json,
)
from transformer_hvp_grokking import logits
from transformer_v3_protocol import PROBES


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "transformer_v3_two_response_postseal_audit.json"
CACHE = RESULTS / "transformer_v3_two_response_cache"
METHOD_VERSION = 1


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def relative_error(left: float, right: float) -> float:
    return abs(left - right) / max(abs(right), torch.finfo(torch.float64).tiny)


def maximum_forcing_for_closure(
    *, kappa: float, derivative_drift: float, response_max: float, domain_radius: float
) -> float:
    """Largest total recentered forcing compatible with scalar closure."""

    b = float(kappa) * float(derivative_drift)
    linear = b * float(response_max)
    if linear >= 1.0:
        return 0.0
    available = float(domain_radius) - float(response_max)
    if available < 0.0:
        return 0.0
    if b == 0.0:
        return math.inf
    maximizing_radius = (1.0 - linear) / b
    radius = min(available, maximizing_radius)
    return max(0.0, (1.0 - linear) * radius - 0.5 * b * radius * radius)


def audit_certificate(path: Path) -> dict:
    certificate = safe_json(path)
    candidate = certificate["candidate"]
    row = {
        "candidate": candidate,
        "certificate_path": path.relative_to(ROOT).as_posix(),
        "certificate_sha256": sha256(path),
        "old_certificate_issued": bool(certificate["certificate_issued"]),
        "old_earliest_power": certificate.get("earliest_issuing_power"),
        "old_bracket": certificate.get("certified_bracket"),
        "outcome_files_read": 0,
        "randomized_queries_added": 0,
    }
    if certificate.get("green_trace") is None:
        row.update({"evaluable": False, "reason": "no sealed Green trace"})
        return row

    from transformer_certificate_protocol import Candidate

    coordinate = Candidate(
        int(candidate["seed"]),
        float(candidate["threshold"]),
        int(candidate["anchor"]),
    )
    config, template, spec, data, parameter, velocity = load_candidate(coordinate)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = data
    horizon = int(certificate["protocol"]["horizon"])
    dimension = int(parameter.numel())
    timings: dict[str, float] = {}

    started = time.perf_counter()
    phase = time.perf_counter()
    path_data = build_frozen_centerline(
        config,
        template,
        spec,
        train_pairs,
        train_labels,
        parameter,
        velocity,
    )
    timings["centerline"] = time.perf_counter() - phase
    if path_data["centerline_sha256"] != certificate["centerline_sha256"]:
        raise RuntimeError(f"centerline hash mismatch for {candidate}")
    center = path_data["center"][: horizon + 1]
    scaled_center = path_data["scaled_center"][: horizon + 1]

    phase = time.perf_counter()
    residual = torch.stack(
        [
            to_scaled(
                path_data["map_step"](center[step]),
                dimension,
                config.learning_rate,
            )
            - scaled_center[step + 1]
            for step in range(horizon)
        ]
    )
    apply_green, _ = make_transformer_green_products(
        center[:horizon, :dimension],
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )
    signed = apply_green(residual.reshape(-1)).reshape(horizon, -1)
    timings["first_signed_response"] = time.perf_counter() - phase
    response_norm = float(torch.linalg.vector_norm(signed))
    response_max = float(torch.linalg.vector_norm(signed, dim=1).max())
    if relative_error(
        response_norm, float(certificate["signed_response_sequence_norm"])
    ) > 2.0e-12:
        raise RuntimeError(f"signed-response sequence mismatch for {candidate}")
    if relative_error(
        response_max, float(certificate["signed_response_max_state_norm"])
    ) > 2.0e-12:
        raise RuntimeError(f"signed-response maximum mismatch for {candidate}")

    # q_j=N_j(z_j), with the exact anchor forcing q_0=0.  Each nonzero row is
    # evaluated as D^2G(c_j)[z_j,z_j]/2, avoiding subtraction of optimizer maps.
    phase = time.perf_counter()
    zero = torch.zeros_like(signed[0])
    q_rows = [zero]
    for step in range(1, horizon):
        q_rows.append(
            optimizer_center_quadratic_defect(
                center[step, :dimension],
                signed[step - 1],
                train_pairs,
                train_labels,
                template,
                spec,
                config,
            )
        )
    q_surrogate = torch.stack(q_rows)
    timings["quadratic_defect"] = time.perf_counter() - phase

    phase = time.perf_counter()
    second_response = apply_green(q_surrogate.reshape(-1)).reshape(horizon, -1)
    timings["second_signed_response"] = time.perf_counter() - phase
    q_norm = float(torch.linalg.vector_norm(q_surrogate))
    q_max = float(torch.linalg.vector_norm(q_surrogate, dim=1).max())
    second_norm = float(torch.linalg.vector_norm(second_response))
    second_max = float(torch.linalg.vector_norm(second_response, dim=1).max())

    output_rows = certificate["output_rows"]
    required = int(certificate["required_correct"])
    raw_zero = _gate_raw_slacks(
        logits(center[0, :dimension], cert_pairs, template, spec),
        cert_labels,
        required,
    )
    power_audits = []
    for power_row in certificate["power_rows"]:
        power = int(power_row["power"])
        kappa = float(power_row["kappa_upper"])
        drift = float(power_row["maximum_optimizer_derivative_drift_upper"])
        old_closure = power_row["one_shot_closure"]
        old_beta = float(old_closure["corrected_defect_response_bound"])
        domain_radius = float(old_closure["domain_radius"])
        closure = exact_one_shot_closure(
            kappa=kappa,
            derivative_drift=drift,
            response_sequence_norm=response_norm,
            response_max_state_norm=response_max,
            corrected_defect_response_bound=second_norm,
            domain_radius=domain_radius,
        )
        output_uppers = [
            float(output["trace"]["rows"][power - 1]["operator_norm_upper_bound"])
            for output in output_rows
        ]
        trial_bracket = None
        trial_slack = None
        if closure.closure_passed:
            trial_bracket, trial_slack, _ = _bracket_at_radius(
                radius=float(closure.total_pointwise_radius),
                output_uppers=output_uppers,
                output_rows=output_rows,
                raw_zero=raw_zero,
            )
        forcing_cap = maximum_forcing_for_closure(
            kappa=kappa,
            derivative_drift=drift,
            response_max=response_max,
            domain_radius=domain_radius,
        )
        response_error_headroom = max(0.0, forcing_cap - second_norm)
        injection_error_headroom = response_error_headroom / kappa
        injection_bound = 0.5 * drift * response_max * response_norm
        power_audits.append(
            {
                "power": power,
                "kappa_upper": kappa,
                "derivative_drift_upper": drift,
                "old_corrected_defect_response_bound": old_beta,
                "quadratic_surrogate_response_norm": second_norm,
                "surrogate_to_old_response_bound_ratio": (
                    second_norm / old_beta if old_beta > 0.0 else 0.0
                ),
                "quadratic_surrogate_injection_norm": q_norm,
                "taylor_injection_bound": injection_bound,
                "surrogate_injection_to_taylor_bound_ratio": (
                    q_norm / injection_bound if injection_bound > 0.0 else 0.0
                ),
                "observed_response_to_kappa_injection_ratio": (
                    second_norm / (kappa * q_norm)
                    if kappa > 0.0 and q_norm > 0.0
                    else 0.0
                ),
                "maximum_total_forcing_for_closure": forcing_cap,
                "admissible_additive_response_error": response_error_headroom,
                "admissible_sigma_q_plus_tau_q": injection_error_headroom,
                "surrogate_closure": closure.as_dict(),
                "surrogate_bracket": trial_bracket,
                "surrogate_logic_slack": trial_slack,
                "surrogate_issued": trial_bracket is not None,
                "old_issued": bool(power_row["certificate_issued"]),
                "old_bracket": power_row["certified_bracket"],
            }
        )

    trial_rows = [audit for audit in power_audits if audit["surrogate_issued"]]
    primary = trial_rows[0] if trial_rows else None
    old_power = certificate.get("earliest_issuing_power")
    new_power = None if primary is None else int(primary["power"])
    saved_power_levels = (
        None if old_power is None or new_power is None else int(old_power) - new_power
    )
    saved_hvp_calls = (
        None
        if saved_power_levels is None
        else saved_power_levels * 2 * PROBES * horizon
    )
    net_hvp_calls_after_second_response = (
        None if saved_hvp_calls is None else saved_hvp_calls - horizon
    )
    row.update(
        {
            "two_response_method_version": METHOD_VERSION,
            "evaluable": True,
            "horizon": horizon,
            "quadratic_surrogate_scope": (
                "ordinary-float center quadratic term; exact q requires a verified "
                "fourth-derivative remainder"
            ),
            "response_sequence_norm": response_norm,
            "response_max_state_norm": response_max,
            "quadratic_surrogate_injection_norm": q_norm,
            "quadratic_surrogate_injection_max_state_norm": q_max,
            "quadratic_surrogate_second_response_norm": second_norm,
            "quadratic_surrogate_second_response_max_state_norm": second_max,
            "surrogate_issued": primary is not None,
            "surrogate_earliest_power": new_power,
            "surrogate_bracket": None if primary is None else primary["surrogate_bracket"],
            "saved_progressive_power_levels_vs_old": saved_power_levels,
            "saved_objective_hvp_calls_vs_old": saved_hvp_calls,
            "net_saved_objective_hvp_calls_after_second_response": (
                net_hvp_calls_after_second_response
            ),
            "added_third_derivative_contractions": max(0, horizon - 1),
            "added_second_response_hvp_calls": horizon,
            "timings_seconds": timings,
            "elapsed_seconds": time.perf_counter() - started,
            "power_audits": power_audits,
        }
    )
    return row


def cache_path(certificate_path: Path) -> Path:
    return CACHE / f"{certificate_path.stem}.two_response_v{METHOD_VERSION}.json"


def valid_cached_row(certificate_path: Path, row: dict) -> bool:
    return (
        bool(row.get("evaluable"))
        and int(row.get("two_response_method_version", METHOD_VERSION)) == METHOD_VERSION
        and row.get("certificate_sha256") == sha256(certificate_path)
        and row.get("quadratic_surrogate_scope") is not None
        and row.get("power_audits")
    )


def write_cache(certificate_path: Path, row: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    destination = cache_path(certificate_path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def prior_diagnostic_rows() -> dict[str, dict]:
    """Recover completed rows from earlier same-method diagnostic runs."""

    recovered: dict[str, dict] = {}
    for path in RESULTS.glob("transformer_v3_two_response_*_diagnostic.json"):
        payload = safe_json(path)
        for row in payload.get("rows", []):
            certificate_path = ROOT / str(row.get("certificate_path", ""))
            if certificate_path.is_file() and valid_cached_row(certificate_path, row):
                row.setdefault("two_response_method_version", METHOD_VERSION)
                recovered[row["certificate_sha256"]] = row
    return recovered


def aggregate(rows: list[dict]) -> dict:
    evaluable = [row for row in rows if row.get("evaluable")]
    converted = [
        row for row in evaluable if row["surrogate_issued"] and not row["old_certificate_issued"]
    ]
    earlier = [
        row
        for row in evaluable
        if row["surrogate_earliest_power"] is not None
        and row["old_earliest_power"] is not None
        and row["surrogate_earliest_power"] < row["old_earliest_power"]
    ]
    retained = [
        row
        for row in evaluable
        if row["old_certificate_issued"] and row["surrogate_issued"]
    ]
    ratios = [
        audit["surrogate_to_old_response_bound_ratio"]
        for row in evaluable
        for audit in row["power_audits"]
        if audit["power"] == (row["surrogate_earliest_power"] or 8)
    ]
    return {
        "status": "OUTCOME-BLIND POST-SEAL TWO-RESPONSE POTENTIAL AUDIT COMPLETED",
        "evidence_boundary": (
            "The center quadratic defect is cancellation-safe but is not an outward "
            "enclosure of the exact defect. Results below are method-development "
            "potential only; prospective certificate counts are unchanged."
        ),
        "certificate_records": len(rows),
        "evaluable_records": len(evaluable),
        "old_issued_evaluable": sum(row["old_certificate_issued"] for row in evaluable),
        "surrogate_issued": sum(row["surrogate_issued"] for row in evaluable),
        "old_issued_retained": len(retained),
        "converted_old_abstentions": len(converted),
        "converted_candidates": [row["candidate"] for row in converted],
        "earlier_power_cases": len(earlier),
        "earlier_power_candidates": [row["candidate"] for row in earlier],
        "median_saved_power_levels_among_earlier": (
            statistics.median(row["saved_progressive_power_levels_vs_old"] for row in earlier)
            if earlier
            else None
        ),
        "median_surrogate_to_old_response_bound_ratio": (
            statistics.median(ratios) if ratios else None
        ),
        "maximum_surrogate_to_old_response_bound_ratio": max(ratios) if ratios else None,
        "aggregate_elapsed_seconds": sum(row.get("elapsed_seconds", 0.0) for row in evaluable),
        "outcome_files_read": 0,
        "randomized_queries_added": 0,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    paths = sorted(RESULTS.glob("transformer_v3_certificate_seed_*.json"))
    if args.seed is not None:
        paths = [
            path
            for path in paths
            if int(safe_json(path)["candidate"]["seed"]) == int(args.seed)
        ]
    if not paths:
        raise RuntimeError("no matching v3 certificate records")

    rows: list[dict] = []
    pending: list[Path] = []
    recovered = {} if args.refresh else prior_diagnostic_rows()
    for path in paths:
        cached = cache_path(path)
        row = None
        if not args.refresh and cached.exists():
            candidate_row = safe_json(cached)
            if valid_cached_row(path, candidate_row):
                row = candidate_row
        if row is None and not args.refresh:
            row = recovered.get(sha256(path))
        if row is None:
            pending.append(path)
        else:
            rows.append(row)
            write_cache(path, row)
            candidate = row["candidate"]
            print(
                "reused "
                f"seed={candidate['seed']} gate={candidate['threshold']:.1f} "
                f"anchor={candidate['anchor']}"
            )

    with ProcessPoolExecutor(max_workers=min(args.workers, max(1, len(pending)))) as pool:
        futures = {pool.submit(audit_certificate, path): path for path in pending}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            if row.get("evaluable"):
                write_cache(futures[future], row)
            candidate = row["candidate"]
            print(
                "audited "
                f"seed={candidate['seed']} gate={candidate['threshold']:.1f} "
                f"anchor={candidate['anchor']} evaluable={row.get('evaluable')}"
            )
    rows.sort(
        key=lambda row: (
            int(row["candidate"]["seed"]),
            float(row["candidate"]["threshold"]),
            int(row["candidate"]["anchor"]),
        )
    )
    payload = aggregate(rows)
    destination = args.output
    if not destination.is_absolute():
        destination = ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        "PASS: two-response potential audit completed for "
        f"{payload['evaluable_records']} evaluable records; "
        f"earlier-power cases={payload['earlier_power_cases']}; "
        f"converted abstentions={payload['converted_old_abstentions']}."
    )


if __name__ == "__main__":
    main()
