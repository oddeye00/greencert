#!/usr/bin/env python3
"""Post-seal asynchronous-power and numerical-padding audit for v3.

The audit never changes an issued certificate.  It reuses each sealed
same-probe q=1..8 trace to answer two implementation questions:

1. Must output-Jacobian and Green operators use the same power?
2. How much deterministic radius padding can the frozen event margins absorb?

All q pairs are valid on the already-budgeted per-operator Gaussian events;
the grid search is retrospective and is reported only as computational
headroom, not as a prospective issuance result.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import median

import numpy as np
import torch

from one_shot_recenter_closure import conservative_one_shot_closure
from transformer_hvp_grokking import (
    TransformerConfig,
    artifact_paths,
    flat_spec,
    logits,
    make_disjoint_split,
    make_template,
)
from transformer_v3_certificate import (
    _bracket_at_radius,
    _gate_raw_slacks,
    _q_geometry,
    frozen_candidates,
    output_path,
    safe_json,
)
from transformer_v3_protocol import MAXIMUM_POWER, PROBES


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CERTIFICATE_SEAL = ROOT / "TRANSFORMER_V3_CERTIFICATE_SEAL.json"
JOIN_SEAL = ROOT / "TRANSFORMER_V3_EXECUTION_AMENDMENT_JOIN_SEAL.json"
AGGREGATE = RESULTS / "transformer_v3_confirmation_audit.json"
OUTPUT = RESULTS / "transformer_v3_power_grid_postseal_audit.json"
PADDING_GRID = (
    0.0,
    1.0e-15,
    1.0e-14,
    1.0e-13,
    1.0e-12,
    1.0e-11,
    1.0e-10,
    1.0e-9,
    1.0e-8,
    1.0e-7,
    1.0e-6,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json_exclusive(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite post-seal audit: {path}")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def raw_anchor_slacks(candidate, required: int) -> tuple[TransformerConfig, tuple[float, float]]:
    blind_path, checkpoint_path = artifact_paths(candidate.seed, development=False)
    payload = safe_json(blind_path)
    config = TransformerConfig(**payload["config"])
    template = make_template(config)
    spec = flat_spec(template)
    cert_pairs, cert_labels = make_disjoint_split(config)[4:]
    checkpoints = np.load(checkpoint_path)
    parameter = torch.from_numpy(checkpoints[f"step_{candidate.anchor}"]).clone()
    center_logits = logits(parameter, cert_pairs, template, spec)
    return config, _gate_raw_slacks(center_logits, cert_labels, required)


def pair_result(
    *,
    q_output: int,
    q_green: int,
    certificate: dict,
    config: TransformerConfig,
    raw_zero: tuple[float, float],
) -> dict:
    output_rows = certificate["output_rows"]
    map_drift, output_uppers = _q_geometry(
        power=q_output,
        output_rows=output_rows,
        config=config,
        domain_radius=float(certificate["outer_domain_radius"]),
    )
    kappa = float(
        certificate["green_trace"]["rows"][q_green - 1][
            "operator_norm_upper_bound"
        ]
    )
    closure = conservative_one_shot_closure(
        kappa=kappa,
        derivative_drift=map_drift,
        response_sequence_norm=float(certificate["signed_response_sequence_norm"]),
        response_max_state_norm=float(certificate["signed_response_max_state_norm"]),
        domain_radius=float(certificate["outer_domain_radius"]),
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
    output_seconds = sum(
        float(row["trace"]["rows"][q_output - 1]["cumulative_operator_seconds"])
        for row in output_rows
    )
    green_seconds = float(
        certificate["green_trace"]["rows"][q_green - 1][
            "cumulative_operator_seconds"
        ]
    )
    horizon = int(certificate["protocol"]["horizon"])
    return {
        "q_output": q_output,
        "q_green": q_green,
        "certificate_issued": bracket is not None,
        "certified_bracket": bracket,
        "certificate_logic_slack": logic_slack,
        "total_pointwise_radius": closure.total_pointwise_radius,
        "logical_gram_applications": PROBES * (horizon * q_output + q_green),
        "measured_cumulative_operator_seconds": output_seconds + green_seconds,
        "output_operator_seconds": output_seconds,
        "green_operator_seconds": green_seconds,
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"post-seal audit already exists: {OUTPUT}")
    certificate_seal = safe_json(CERTIFICATE_SEAL)
    aggregate = safe_json(AGGREGATE)
    join_seal = safe_json(JOIN_SEAL)
    if sha256(CERTIFICATE_SEAL) != aggregate["certificate_seal_sha256"]:
        raise RuntimeError("aggregate points to another certificate seal")
    if sha256(AGGREGATE) != join_seal["aggregate_sha256"]:
        raise RuntimeError("join seal points to another aggregate")

    audit_by_candidate = {
        (
            int(row["candidate"]["seed"]),
            float(row["candidate"]["threshold"]),
            int(row["candidate"]["anchor"]),
        ): row
        for row in aggregate["rows"]
    }
    candidates, _, _ = frozen_candidates()
    rows = []
    issued_rows = []
    for candidate in candidates:
        certificate = safe_json(output_path(candidate))
        audit = audit_by_candidate[
            (candidate.seed, candidate.threshold, candidate.anchor)
        ]
        if certificate.get("green_trace") is None:
            rows.append(
                {
                    "candidate": candidate.__dict__,
                    "original_issued": bool(certificate["certificate_issued"]),
                    "grid_available": False,
                    "reason": (
                        "deterministic early abstention or reference construction failure"
                    ),
                }
            )
            continue

        config, raw_zero = raw_anchor_slacks(
            candidate, int(certificate["required_correct"])
        )
        grid = [
            pair_result(
                q_output=q_output,
                q_green=q_green,
                certificate=certificate,
                config=config,
                raw_zero=raw_zero,
            )
            for q_output in range(1, MAXIMUM_POWER + 1)
            for q_green in range(1, MAXIMUM_POWER + 1)
        ]
        lockstep = [row for row in grid if row["q_output"] == row["q_green"]]
        lockstep_issued = [row for row in lockstep if row["certificate_issued"]]
        observed_lockstep = None if not lockstep_issued else lockstep_issued[0]
        expected_power = certificate.get("earliest_issuing_power")
        if expected_power is None:
            if observed_lockstep is not None:
                raise AssertionError("grid issued a lockstep certificate absent from record")
        else:
            if observed_lockstep is None or observed_lockstep["q_output"] != expected_power:
                raise AssertionError("grid failed to reproduce frozen earliest power")
            if observed_lockstep["certified_bracket"] != certificate["certified_bracket"]:
                raise AssertionError("grid failed to reproduce frozen bracket")

        issuing = [row for row in grid if row["certificate_issued"]]
        minimum_logical = (
            None
            if not issuing
            else min(
                issuing,
                key=lambda row: (
                    row["logical_gram_applications"],
                    row["measured_cumulative_operator_seconds"],
                ),
            )
        )
        minimum_measured = (
            None
            if not issuing
            else min(
                issuing,
                key=lambda row: (
                    row["measured_cumulative_operator_seconds"],
                    row["logical_gram_applications"],
                ),
            )
        )
        q_output_one = [
            row for row in grid if row["q_output"] == 1 and row["certificate_issued"]
        ]
        q_output_one_first = None if not q_output_one else q_output_one[0]
        full = next(
            row
            for row in grid
            if row["q_output"] == MAXIMUM_POWER
            and row["q_green"] == MAXIMUM_POWER
        )

        padding = {}
        if certificate["certificate_issued"]:
            primary_power = int(certificate["earliest_issuing_power"])
            map_drift, output_uppers = _q_geometry(
                power=primary_power,
                output_rows=certificate["output_rows"],
                config=config,
                domain_radius=float(certificate["outer_domain_radius"]),
            )
            del map_drift
            base_radius = float(certificate["certified_total_pointwise_radius"])
            actual_event = audit["actual_persistent_event"]
            for value in PADDING_GRID:
                bracket, logic_slack, _ = _bracket_at_radius(
                    radius=base_radius + value,
                    output_uppers=output_uppers,
                    output_rows=certificate["output_rows"],
                    raw_zero=raw_zero,
                )
                padding[f"{value:.1e}"] = {
                    "issued": bracket is not None,
                    "bracket": bracket,
                    "contains_actual": bool(
                        bracket is not None
                        and actual_event is not None
                        and bracket[0] <= actual_event <= bracket[1]
                    ),
                    "logic_slack": logic_slack,
                }

        row = {
            "candidate": candidate.__dict__,
            "horizon": int(certificate["protocol"]["horizon"]),
            "original_issued": bool(certificate["certificate_issued"]),
            "original_covered": audit["bracket_contains_actual"],
            "grid_available": True,
            "frozen_lockstep_first": observed_lockstep,
            "minimum_logical_pair": minimum_logical,
            "minimum_measured_pair": minimum_measured,
            "q_output_one_first": q_output_one_first,
            "full_q8_pair": full,
            "padding_sensitivity": padding,
        }
        rows.append(row)
        if certificate["certificate_issued"]:
            issued_rows.append(row)

    padding_summary = {}
    for value in PADDING_GRID:
        key = f"{value:.1e}"
        retained = [
            row["padding_sensitivity"][key]
            for row in issued_rows
            if key in row["padding_sensitivity"]
        ]
        padding_summary[key] = {
            "issued": sum(item["issued"] for item in retained),
            "covered": sum(item["contains_actual"] for item in retained),
            "maximum_bracket_width": (
                None
                if not retained or not any(item["bracket"] for item in retained)
                else max(
                    item["bracket"][1] - item["bracket"][0]
                    for item in retained
                    if item["bracket"] is not None
                )
            ),
        }

    logical_speedups = [
        row["full_q8_pair"]["logical_gram_applications"]
        / row["minimum_logical_pair"]["logical_gram_applications"]
        for row in issued_rows
    ]
    measured_speedups = [
        row["full_q8_pair"]["measured_cumulative_operator_seconds"]
        / row["minimum_measured_pair"]["measured_cumulative_operator_seconds"]
        for row in issued_rows
    ]
    q_output_one = [row for row in issued_rows if row["q_output_one_first"] is not None]
    response_deficits = [
        float(row["observed_response_centered_sequence_error"])
        - float(row["certified_remainder_sequence_radius"])
        for row in aggregate["rows"]
        if row["certificate_issued"]
    ]
    payload = {
        "status": "POST-SEAL DIAGNOSTIC; DOES NOT CHANGE V3 ISSUANCE",
        "certificate_seal_sha256": sha256(CERTIFICATE_SEAL),
        "aggregate_sha256": sha256(AGGREGATE),
        "join_seal_sha256": sha256(JOIN_SEAL),
        "candidate_count": len(candidates),
        "grid_evaluable_candidates": sum(row["grid_available"] for row in rows),
        "frozen_issued": len(issued_rows),
        "q_output_one_retains_frozen_issued": len(q_output_one),
        "median_minimum_logical_speedup_over_full_q8": median(logical_speedups),
        "minimum_minimum_logical_speedup_over_full_q8": min(logical_speedups),
        "maximum_minimum_logical_speedup_over_full_q8": max(logical_speedups),
        "median_trace_operator_time_speedup_over_full_q8": median(measured_speedups),
        "minimum_trace_operator_time_speedup_over_full_q8": min(measured_speedups),
        "maximum_trace_operator_time_speedup_over_full_q8": max(measured_speedups),
        "maximum_float64_response_remainder_deficit": max(response_deficits),
        "padding_sensitivity_on_frozen_issued": padding_summary,
        "interpretation": (
            "q-pair optimization and padding are retrospective audits. Same-event "
            "validity permits asynchronous powers, but a prospective online policy "
            "must be frozen and implemented before claiming realized wall-time gains."
        ),
        "rows": rows,
    }
    write_json_exclusive(OUTPUT, payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
