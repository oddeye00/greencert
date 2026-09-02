#!/usr/bin/env python3
"""Combine disjoint Gaussian blocks into one staged causal-row certificate."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Sequence

import torch

from analytic_jet_release import logit_margin_radius
from audit_transformer_adaptive_sweep_cohort import first_persistent
from causal_row_green import causal_row_quadratic_envelope


PERSISTENCE = 25


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def persistent_bracket(
    guarantee: Sequence[float], exclusion: Sequence[float]
) -> tuple[list[int] | None, float | None]:
    lower = first_persistent([value <= 0.0 for value in exclusion])
    upper = first_persistent([value > 0.0 for value in guarantee])
    if lower is None or upper is None or lower > upper:
        return None, None
    prior = [max(exclusion[start : start + PERSISTENCE]) for start in range(lower)]
    lower_slack = math.inf if not prior else min(prior)
    upper_slack = min(guarantee[upper : upper + PERSISTENCE])
    return [lower, upper], min(lower_slack, upper_slack)


def _same(left, right, name: str) -> None:
    if left != right:
        raise ValueError(f"probe blocks disagree at {name}")


def combine_probe_blocks(records: Sequence[dict], *, stage_delta: float) -> dict:
    if len(records) < 2:
        raise ValueError("at least two probe blocks are required")
    ordered = sorted(records, key=lambda row: int(row["probe_offset"]))
    reference = ordered[0]
    stream_size = int(reference["probe_stream_size"])
    offsets = []
    total = 0
    for row in ordered:
        for key in (
            "candidate",
            "sweeps",
            "defect_route",
            "closure_channel",
            "horizon",
            "probe_seed",
            "probe_stream_size",
            "centerline_sha256",
            "corrected_path_sha256",
            "domain_radius",
            "sealed_four_sweep_bracket",
            "raw_event_slacks",
            "output_first_derivative_bounds",
            "active_curvature_bounds",
            "active_forcing_error_bounds",
        ):
            _same(reference[key], row[key], key)
        offset = int(row["probe_offset"])
        count = int(row["probes"])
        offsets.extend(range(offset, offset + count))
        total += count
    if sorted(offsets) != list(range(total)) or total != stream_size:
        raise ValueError("probe blocks do not partition the declared stream")

    horizon = int(reference["horizon"])
    delta = float(stage_delta)
    if not 0.0 < delta < 1.0:
        raise ValueError("stage_delta must lie in (0,1)")
    row_budget = delta / horizon
    calibration = NormalDist().inv_cdf(
        0.5 * (1.0 + row_budget ** (1.0 / total))
    )
    raw_maxima = [
        max(float(row["row_image_maxima"][step]) for row in ordered)
        for step in range(horizon)
    ]
    gains = torch.tensor(
        [value / calibration for value in raw_maxima], dtype=torch.float64
    )
    signed = torch.tensor(
        [
            max(float(row["signed_response_row_norms"][step]) for row in ordered)
            for step in range(horizon)
        ],
        dtype=torch.float64,
    )
    errors = torch.tensor(
        reference["active_forcing_error_bounds"], dtype=torch.float64
    )
    affine = signed + gains * torch.sqrt(torch.cumsum(errors.square(), dim=0))
    radii = causal_row_quadratic_envelope(
        affine, gains, reference["active_curvature_bounds"]
    )
    domain = float(reference["domain_radius"])
    domain_passed = bool(torch.isfinite(radii).all() and (radii <= domain).all())

    raw = reference["raw_event_slacks"]
    first = reference["output_first_derivative_bounds"]
    margins = [0.0]
    bracket = None
    logic_slack = None
    if domain_passed:
        margins.extend(
            logit_margin_radius(
                first=float(first[step]), state_radius=float(radii[step - 1])
            )
            for step in range(1, horizon + 1)
        )
        guarantee = [float(pair[0]) - margin for pair, margin in zip(raw, margins)]
        exclusion = [float(pair[1]) - margin for pair, margin in zip(raw, margins)]
        bracket, logic_slack = persistent_bracket(guarantee, exclusion)
    issued = bracket is not None and logic_slack is not None and logic_slack > 0.0
    return {
        "status": "combined staged causal row-Green certificate",
        "candidate": reference["candidate"],
        "sweeps": reference["sweeps"],
        "closure_channel": reference["closure_channel"],
        "horizon": horizon,
        "probe_seed": reference["probe_seed"],
        "probe_stream_size": stream_size,
        "probe_blocks": [
            {"offset": int(row["probe_offset"]), "count": int(row["probes"])}
            for row in ordered
        ],
        "stage_delta": delta,
        "row_budget": row_budget,
        "row_calibration": calibration,
        "row_image_maxima": raw_maxima,
        "row_gain_bounds": gains.tolist(),
        "row_affine_bounds": affine.tolist(),
        "row_radii": radii.tolist(),
        "maximum_row_radius": float(radii.max()),
        "domain_radius": domain,
        "row_domain_passed": domain_passed,
        "issued": issued,
        "bracket": bracket,
        "logic_slack": logic_slack,
        "sealed_four_sweep_bracket": reference["sealed_four_sweep_bracket"],
        "retains_sealed_bracket": issued
        and bracket == reference["sealed_four_sweep_bracket"],
        "maximum_margin_radius": max(margins),
        "centerline_sha256": reference["centerline_sha256"],
        "corrected_path_sha256": reference["corrected_path_sha256"],
        "released_corrected_path_match": reference["released_corrected_path_match"],
        "logical_forward_probe_applications": stream_size,
        "logical_signed_response_applications": 1,
        "logical_transpose_applications": 0,
        "logical_total_linearized_sweeps": stream_size + 1,
        "outcome_files_read": sum(int(row["outcome_files_read"]) for row in ordered),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--stage-delta", type=float, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = [json.loads(path.read_text(encoding="utf-8")) for path in args.records]
    result = combine_probe_blocks(records, stage_delta=args.stage_delta)
    if args.output is not None:
        args.output.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
