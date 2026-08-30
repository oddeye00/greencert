#!/usr/bin/env python3
"""Post-seal audit of time-resolved recentering on one v3 candidate.

This script rebuilds only the deterministic centerline and signed response.  It
reuses every sealed output/Green trace and does not alter the prospective v3
result.  The purpose is to measure whether retaining checkpointwise drift
bounds improves the first issuing power or radius.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

import torch

from heterogeneous_recenter_closure import heterogeneous_one_shot_closure
from transformer_block_envelope import objective_hessian_lipschitz
from transformer_certificate_protocol import Candidate
from transformer_four_sweep_development_audit import to_scaled
from transformer_green_development_audit import build_frozen_centerline, load_candidate
from transformer_green_operator import make_transformer_green_products
from transformer_v3_certificate import (
    _bracket_at_radius,
    _gate_raw_slacks,
    _q_geometry,
    output_path,
    safe_json,
)
from transformer_hvp_grokking import logits


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CANDIDATE_SEAL = ROOT / "TRANSFORMER_V3_CANDIDATE_SEAL.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def relative_error(left: float, right: float) -> float:
    return abs(left - right) / max(abs(right), torch.finfo(torch.float64).tiny)


def gaussian_calibration(delta: float, probes: int) -> float:
    return NormalDist().inv_cdf((1.0 + delta ** (1.0 / probes)) / 2.0)


def audit(candidate: Candidate) -> dict:
    certificate_path = output_path(candidate)
    certificate = safe_json(certificate_path)
    if certificate.get("green_trace") is None:
        raise ValueError("candidate has no sealed Green trace")

    config, template, spec, data, parameter, velocity = load_candidate(candidate)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = data
    horizon = int(certificate["protocol"]["horizon"])
    dimension = parameter.numel()
    path = build_frozen_centerline(
        config,
        template,
        spec,
        train_pairs,
        train_labels,
        parameter,
        velocity,
    )
    if path["centerline_sha256"] != certificate["centerline_sha256"]:
        raise RuntimeError("rebuilt centerline differs from sealed record")
    center = path["center"]
    scaled_center = path["scaled_center"]
    residual = torch.stack(
        [
            to_scaled(path["map_step"](center[step]), dimension, config.learning_rate)
            - scaled_center[step + 1]
            for step in range(horizon)
        ]
    )
    green, _ = make_transformer_green_products(
        center[:horizon, :dimension],
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )
    response = green(residual.reshape(-1)).reshape(horizon, -1)
    response_norms = torch.linalg.vector_norm(response, dim=1)
    response_norm = float(torch.linalg.vector_norm(response))
    response_max = float(response_norms.max())
    domain_radius = float(certificate["outer_domain_radius"])
    for key, observed in (
        ("signed_response_sequence_norm", response_norm),
        ("signed_response_max_state_norm", response_max),
    ):
        if relative_error(observed, float(certificate[key])) > 2.0e-12:
            raise RuntimeError(f"rebuilt response differs at {key}")

    required = int(certificate["required_correct"])
    raw_zero = _gate_raw_slacks(
        logits(center[0, :dimension], cert_pairs, template, spec),
        cert_labels,
        required,
    )
    output_rows = certificate["output_rows"]
    candidate_count = len(safe_json(CANDIDATE_SEAL)["candidates"])
    candidate_budget = float(certificate["protocol"]["family_failure_probability"]) / candidate_count
    # A role-stratified future policy: half of each sealed candidate's budget
    # goes to its one Green operator and half is shared by its H output
    # operators.  This allocation is evaluated post-seal here; it did not
    # govern the prospective v3 certificates.
    role_green_delta = candidate_budget / 2.0
    role_output_delta = candidate_budget / (2.0 * horizon)
    probes = int(certificate["protocol"]["probe_config"]["probes"])
    role_green_c = gaussian_calibration(role_green_delta, probes)
    role_output_c = gaussian_calibration(role_output_delta, probes)

    rows = []
    role_rows = []
    for power_row in certificate["power_rows"]:
        power = int(power_row["power"])
        scalar_drift, output_uppers = _q_geometry(
            power=power,
            output_rows=output_rows,
            config=config,
            domain_radius=domain_radius,
        )
        drift = []
        for row, output_upper in zip(output_rows[:-1], output_uppers[:-1]):
            first_ball = output_upper + float(row["block_second"]) * domain_radius
            objective_drift = objective_hessian_lipschitz(
                first_ball,
                float(row["block_second"]),
                float(row["block_third"]),
            )
            drift.append(math.sqrt(2.0) * config.learning_rate * objective_drift)
        if drift and relative_error(max(drift), scalar_drift) > 2.0e-12:
            raise RuntimeError("time-resolved drift does not reproduce scalar maximum")

        kappa = float(power_row["kappa_upper"])
        closure = heterogeneous_one_shot_closure(
            kappa=kappa,
            drift_bounds=drift,
            response_input_state_norms=response_norms[:-1].tolist(),
            response_sequence_norm=response_norm,
            response_max_state_norm=response_max,
            domain_radius=domain_radius,
        )
        bracket = None
        logic_slack = None
        if closure.closure_passed:
            bracket, logic_slack, _ = _bracket_at_radius(
                radius=float(closure.total_pointwise_radius),
                output_uppers=output_uppers,
                output_rows=output_rows,
                raw_zero=raw_zero,
            )
        old = power_row["one_shot_closure"]
        rows.append(
            {
                "power": power,
                "heterogeneous_closure": closure.as_dict(),
                "heterogeneous_bracket": bracket,
                "heterogeneous_issued": bracket is not None,
                "heterogeneous_logic_slack": logic_slack,
                "scalar_closure_passed": bool(old["closure_passed"]),
                "scalar_total_pointwise_radius": old["total_pointwise_radius"],
                "scalar_issued": bool(power_row["certificate_issued"]),
                "scalar_bracket": power_row["certified_bracket"],
                "forcing_ratio_to_scalar": (
                    closure.corrected_defect_response_bound
                    / float(old["corrected_defect_response_bound"])
                    if float(old["corrected_defect_response_bound"]) > 0.0
                    else 0.0
                ),
                "linear_ratio_to_scalar": (
                    closure.linearized_remainder_coefficient
                    / float(old["linearized_remainder_coefficient"])
                    if float(old["linearized_remainder_coefficient"]) > 0.0
                    else 0.0
                ),
            }
        )

        role_output_uppers = [
            (float(row["trace"]["rows"][power - 1]["Y"]) / role_output_c)
            ** (1.0 / (2.0 * power))
            for row in output_rows
        ]
        role_green_y = float(
            certificate["green_trace"]["rows"][power - 1]["Y"]
        )
        role_kappa = (role_green_y / role_green_c) ** (1.0 / (2.0 * power))
        role_drift = []
        for row, output_upper in zip(output_rows[:-1], role_output_uppers[:-1]):
            first_ball = output_upper + float(row["block_second"]) * domain_radius
            objective_drift = objective_hessian_lipschitz(
                first_ball,
                float(row["block_second"]),
                float(row["block_third"]),
            )
            role_drift.append(
                math.sqrt(2.0) * config.learning_rate * objective_drift
            )
        role_closure = heterogeneous_one_shot_closure(
            kappa=role_kappa,
            drift_bounds=role_drift,
            response_input_state_norms=response_norms[:-1].tolist(),
            response_sequence_norm=response_norm,
            response_max_state_norm=response_max,
            domain_radius=domain_radius,
        )
        role_bracket = None
        role_logic_slack = None
        if role_closure.closure_passed:
            role_bracket, role_logic_slack, _ = _bracket_at_radius(
                radius=float(role_closure.total_pointwise_radius),
                output_uppers=role_output_uppers,
                output_rows=output_rows,
                raw_zero=raw_zero,
            )
        role_rows.append(
            {
                "power": power,
                "kappa_upper": role_kappa,
                "heterogeneous_closure": role_closure.as_dict(),
                "bracket": role_bracket,
                "issued": role_bracket is not None,
                "logic_slack": role_logic_slack,
            }
        )

    first = next((row for row in rows if row["heterogeneous_issued"]), None)
    role_first = next((row for row in role_rows if row["issued"]), None)
    result = {
        "status": "POST-SEAL TIME-RESOLVED RECENTER AUDIT",
        "prospective_v3_result_changed": False,
        "candidate": candidate.__dict__,
        "certificate_path": str(certificate_path.relative_to(ROOT)),
        "certificate_sha256": sha256(certificate_path),
        "sealed_scalar_earliest_power": certificate["earliest_issuing_power"],
        "sealed_scalar_bracket": certificate["certified_bracket"],
        "heterogeneous_earliest_power": None if first is None else first["power"],
        "heterogeneous_bracket": None if first is None else first["heterogeneous_bracket"],
        "role_stratified_budget_audit": {
            "status": "post-seal diagnostic; not prospective v3 evidence",
            "sealed_candidate_count": candidate_count,
            "candidate_budget": candidate_budget,
            "green_delta": role_green_delta,
            "per_output_delta": role_output_delta,
            "total_budget_upper_bound": candidate_count
            * (role_green_delta + horizon * role_output_delta),
            "earliest_power": None if role_first is None else role_first["power"],
            "bracket": None if role_first is None else role_first["bracket"],
            "rows": role_rows,
        },
        "rows": rows,
        "interpretation": (
            "Post-seal theorem audit using sealed probes; it is not a new "
            "prospective certificate or coverage observation."
        ),
    }
    destination = RESULTS / (
        f"transformer_v3_heterogeneous_recenter_seed_{candidate.seed}_"
        f"gate_{candidate.gate_index}_anchor_{candidate.anchor}_role_budget.json"
    )
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite audit: {destination}")
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["output"] = str(destination.relative_to(ROOT))
    result["sha256"] = sha256(destination)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=372)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--anchor", type=int, default=3440)
    args = parser.parse_args()
    result = audit(Candidate(args.seed, args.threshold, args.anchor))
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
