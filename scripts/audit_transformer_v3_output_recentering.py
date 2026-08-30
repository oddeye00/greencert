#!/usr/bin/env python3
"""Outcome-blind post-seal audit of response-centered event transport.

The prospective v3 records center output Taylor bounds at the reference path
``c`` and charge the complete radius ``p + E``.  The state theorem proves the
stronger statement ``||x - (c + z)|| <= E``.  This audit evaluates logits at
the known corrected center ``c + z`` and charges only ``E``.  It reads no
future trajectory or outcome file and never changes a prospective record.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from response_centered_event_transport import (
    classification_margin_origin_radius,
    classification_margin_remainder_radius,
)
from transformer_four_sweep_development_audit import to_scaled
from transformer_green_development_audit import build_frozen_centerline
from transformer_green_operator import make_transformer_green_products
from transformer_hvp_grokking import logits
from transformer_v3_certificate import load_candidate


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "transformer_v3_output_recentering_postseal_audit.json"
PERSISTENCE = 25


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load(path: Path) -> dict:
    lowered = path.name.lower()
    if lowered.endswith(".outcomes.json") or lowered.endswith(".sealed.log"):
        raise RuntimeError(f"outcome-blind audit attempted forbidden read: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def gate_raw_slacks(
    values: torch.Tensor, labels: torch.Tensor, required: int
) -> tuple[float, float]:
    true = values.gather(1, labels[:, None])
    margins = true - values
    rows = torch.arange(len(labels))
    margins[rows, labels] = torch.inf
    per_example = torch.min(margins, dim=1).values
    guarantee = torch.sort(per_example, descending=True).values[required - 1]
    incorrect_needed = len(labels) - required + 1
    exclusion = -torch.sort(per_example).values[incorrect_needed - 1]
    return float(guarantee), float(exclusion)


def first_persistent(values: list[bool]) -> int | None:
    for start in range(max(0, len(values) - PERSISTENCE + 1)):
        if all(values[start : start + PERSISTENCE]):
            return start
    return None


def bracket(
    guarantee_slacks: list[float], exclusion_slacks: list[float]
) -> list[int] | None:
    lower = first_persistent([value <= 0.0 for value in exclusion_slacks])
    upper = first_persistent([value > 0.0 for value in guarantee_slacks])
    if lower is None or upper is None or lower > upper:
        return None
    return [int(lower), int(upper)]


def logic_slack(
    event_bracket: list[int] | None,
    guarantee_slacks: list[float],
    exclusion_slacks: list[float],
) -> float | None:
    if event_bracket is None:
        return None
    lower, upper = event_bracket
    upper_slack = min(guarantee_slacks[upper : upper + PERSISTENCE])
    prior = [
        max(exclusion_slacks[start : start + PERSISTENCE])
        for start in range(lower)
    ]
    lower_slack = math.inf if not prior else min(prior)
    return float(min(lower_slack, upper_slack))


def audit_certificate(path: Path) -> dict:
    certificate = load(path)
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

    started = time.perf_counter()
    from transformer_certificate_protocol import Candidate

    coordinate = Candidate(
        int(candidate["seed"]),
        float(candidate["threshold"]),
        int(candidate["anchor"]),
    )
    config, template, spec, data, parameter, velocity = load_candidate(coordinate)
    torch.set_num_threads(config.threads)
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = data
    path_data = build_frozen_centerline(
        config,
        template,
        spec,
        train_pairs,
        train_labels,
        parameter,
        velocity,
    )
    if path_data["centerline_sha256"] != certificate["centerline_sha256"]:
        raise RuntimeError(f"centerline hash mismatch for {candidate}")
    horizon = int(certificate["protocol"]["horizon"])
    center = path_data["center"][: horizon + 1]
    scaled_center = path_data["scaled_center"][: horizon + 1]
    dimension = parameter.numel()
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
    observed_response_norm = float(torch.linalg.vector_norm(signed))
    if not math.isclose(
        observed_response_norm,
        float(certificate["signed_response_sequence_norm"]),
        rel_tol=2.0e-12,
        abs_tol=1.0e-28,
    ):
        raise RuntimeError(f"signed-response mismatch for {candidate}")

    required = int(certificate["required_correct"])
    anchor_slacks = gate_raw_slacks(
        logits(center[0, :dimension], cert_pairs, template, spec),
        cert_labels,
        required,
    )
    corrected_raw = []
    response_parameter_norms = []
    for step in range(1, horizon + 1):
        correction = signed[step - 1, :dimension]
        corrected_theta = center[step, :dimension] + correction
        corrected_raw.append(
            gate_raw_slacks(
                logits(corrected_theta, cert_pairs, template, spec),
                cert_labels,
                required,
            )
        )
        response_parameter_norms.append(float(torch.linalg.vector_norm(correction)))

    output_rows = certificate["output_rows"]
    if len(output_rows) != horizon:
        raise RuntimeError(f"output-row horizon mismatch for {candidate}")
    power_audits = []
    for power_row in certificate.get("power_rows", []):
        power = int(power_row["power"])
        closure = power_row["one_shot_closure"]
        remainder = closure.get("remainder_radius")
        total = closure.get("total_pointwise_radius")
        if not closure["closure_passed"] or remainder is None or total is None:
            power_audits.append(
                {
                    "power": power,
                    "closure_passed": False,
                    "recentered_bracket": None,
                    "hybrid_bracket": None,
                }
            )
            continue

        old_guarantee = [anchor_slacks[0]]
        old_exclusion = [anchor_slacks[1]]
        new_guarantee = [anchor_slacks[0]]
        new_exclusion = [anchor_slacks[1]]
        old_margins = []
        new_margins = []
        for index, output_row in enumerate(output_rows):
            upper = float(
                output_row["trace"]["rows"][power - 1][
                    "operator_norm_upper_bound"
                ]
            )
            second = float(output_row["block_second"])
            old_margin = classification_margin_origin_radius(
                output_jacobian_upper=upper,
                output_hessian_upper=second,
                total_radius=float(total),
            )
            new_margin = classification_margin_remainder_radius(
                output_jacobian_upper=upper,
                output_hessian_upper=second,
                response_norm=response_parameter_norms[index],
                remainder_radius=float(remainder),
            )
            if new_margin > old_margin + 2.0e-14 * max(1.0, old_margin):
                raise AssertionError("response-centered radius lost dominance")
            old_margins.append(old_margin)
            new_margins.append(new_margin)
            old_guarantee.append(float(output_row["raw_guarantee_slack"]) - old_margin)
            old_exclusion.append(float(output_row["raw_exclusion_slack"]) - old_margin)
            raw_g, raw_e = corrected_raw[index]
            new_guarantee.append(raw_g - new_margin)
            new_exclusion.append(raw_e - new_margin)

        hybrid_guarantee = [max(old, new) for old, new in zip(old_guarantee, new_guarantee)]
        hybrid_exclusion = [max(old, new) for old, new in zip(old_exclusion, new_exclusion)]
        recentered_bracket = bracket(new_guarantee, new_exclusion)
        hybrid_bracket = bracket(hybrid_guarantee, hybrid_exclusion)
        power_audits.append(
            {
                "power": power,
                "closure_passed": True,
                "old_recomputed_bracket": bracket(old_guarantee, old_exclusion),
                "recentered_bracket": recentered_bracket,
                "hybrid_bracket": hybrid_bracket,
                "recentered_logic_slack": logic_slack(
                    recentered_bracket, new_guarantee, new_exclusion
                ),
                "hybrid_logic_slack": logic_slack(
                    hybrid_bracket, hybrid_guarantee, hybrid_exclusion
                ),
                "maximum_origin_margin_radius": max(old_margins, default=0.0),
                "maximum_recentered_margin_radius": max(new_margins, default=0.0),
                "maximum_margin_radius_ratio": (
                    max(new_margins) / max(old_margins)
                    if old_margins and max(old_margins) > 0.0
                    else 0.0
                ),
            }
        )

    recentered_rows = [r for r in power_audits if r.get("recentered_bracket")]
    hybrid_rows = [r for r in power_audits if r.get("hybrid_bracket")]
    primary_recentered = recentered_rows[0] if recentered_rows else None
    primary_hybrid = hybrid_rows[0] if hybrid_rows else None
    row.update(
        {
            "evaluable": True,
            "horizon": horizon,
            "deterministic_corrected_forward_evaluations": horizon,
            "observed_signed_response_sequence_norm": observed_response_norm,
            "recentered_issued": primary_recentered is not None,
            "recentered_earliest_power": (
                None if primary_recentered is None else primary_recentered["power"]
            ),
            "recentered_bracket": (
                None
                if primary_recentered is None
                else primary_recentered["recentered_bracket"]
            ),
            "hybrid_issued": primary_hybrid is not None,
            "hybrid_earliest_power": (
                None if primary_hybrid is None else primary_hybrid["power"]
            ),
            "hybrid_bracket": (
                None if primary_hybrid is None else primary_hybrid["hybrid_bracket"]
            ),
            "old_issued_retained_by_hybrid": (
                not row["old_certificate_issued"] or primary_hybrid is not None
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "power_audits": power_audits,
        }
    )
    return row


def main() -> None:
    paths = sorted(RESULTS.glob("transformer_v3_certificate_seed_*.json"))
    rows = []
    with ProcessPoolExecutor(max_workers=min(4, len(paths))) as pool:
        futures = {pool.submit(audit_certificate, path): path for path in paths}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            candidate = row["candidate"]
            print(
                "audited "
                f"seed={candidate['seed']} gate={candidate['threshold']:.1f} "
                f"anchor={candidate['anchor']}"
            )
    rows.sort(
        key=lambda row: (
            int(row["candidate"]["seed"]),
            float(row["candidate"]["threshold"]),
            int(row["candidate"]["anchor"]),
        )
    )
    evaluable = [row for row in rows if row["evaluable"]]
    hybrid = [row for row in evaluable if row["hybrid_issued"]]
    recentered = [row for row in evaluable if row["recentered_issued"]]
    old = [row for row in rows if row["old_certificate_issued"]]
    if not all(row["old_issued_retained_by_hybrid"] for row in evaluable):
        raise AssertionError("hybrid output transport lost a prospective certificate")
    ratios = [
        audit["maximum_margin_radius_ratio"]
        for row in hybrid
        for audit in row["power_audits"]
        if audit.get("power") == row["hybrid_earliest_power"]
    ]
    earlier = [
        row
        for row in hybrid
        if row["old_earliest_power"] is not None
        and row["hybrid_earliest_power"] < row["old_earliest_power"]
    ]
    converted = [row for row in hybrid if not row["old_certificate_issued"]]
    payload = {
        "status": "OUTCOME-BLIND POST-SEAL RESPONSE-CENTERED OUTPUT AUDIT PASSED",
        "evidence_boundary": (
            "The output-recentering corollary was introduced after the v3 seal. "
            "This audit reads no future outcomes and changes no prospective record; "
            "any issuance gain is method-development evidence until frozen confirmation."
        ),
        "construction": (
            "evaluate logits at c+z; charge only the theorem remainder E; intersect "
            "with the original valid output enclosure"
        ),
        "certificate_records": len(rows),
        "evaluable_records": len(evaluable),
        "old_issued": len(old),
        "recentered_only_issued": len(recentered),
        "hybrid_issued": len(hybrid),
        "hybrid_converted_old_abstentions": len(converted),
        "hybrid_converted_candidates": [row["candidate"] for row in converted],
        "old_issued_retained": sum(
            row["old_certificate_issued"] and row["hybrid_issued"] for row in evaluable
        ),
        "earlier_power_cases": len(earlier),
        "earlier_power_candidates": [row["candidate"] for row in earlier],
        "median_maximum_margin_radius_ratio_at_hybrid_issue": (
            statistics.median(ratios) if ratios else None
        ),
        "maximum_margin_radius_ratio_at_hybrid_issue": max(ratios) if ratios else None,
        "randomized_queries_added": 0,
        "outcome_files_read": 0,
        "deterministic_forward_evaluations": sum(
            row.get("deterministic_corrected_forward_evaluations", 0) for row in evaluable
        ),
        "aggregate_elapsed_seconds": sum(row.get("elapsed_seconds", 0.0) for row in evaluable),
        "audit_script_sha256": sha256(Path(__file__)),
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        "PASS: response-centered output transport retained "
        f"{payload['old_issued_retained']}/{payload['old_issued']} old certificates, "
        f"issues {payload['hybrid_issued']} hybrid records, and adds no randomized query."
    )


if __name__ == "__main__":
    main()
