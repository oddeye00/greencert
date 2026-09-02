#!/usr/bin/env python3
"""Deterministic interface tests for staged causal-row probe combination."""
from __future__ import annotations

import copy

from combine_causal_row_probe_blocks import combine_probe_blocks


def block(offset: int, maxima: list[float]) -> dict:
    horizon = 3
    return {
        "candidate": {"seed": 1, "threshold": 0.7, "anchor": 2},
        "sweeps": 4,
        "defect_route": "quadratic",
        "closure_channel": "structured_parameter",
        "horizon": horizon,
        "probes": 2,
        "probe_seed": 91,
        "probe_stream_size": 4,
        "probe_offset": offset,
        "centerline_sha256": "A",
        "corrected_path_sha256": "B",
        "released_corrected_path_match": True,
        "domain_radius": 1.0,
        "sealed_four_sweep_bracket": [0, 0],
        "raw_event_slacks": [[1.0, -1.0]] * (horizon + 1),
        "output_first_derivative_bounds": [0.0] * (horizon + 1),
        "active_curvature_bounds": [0.0] * horizon,
        "active_forcing_error_bounds": [0.0] * horizon,
        "signed_response_row_norms": [0.0] * horizon,
        "row_image_maxima": maxima,
        "outcome_files_read": 0,
    }


def main() -> None:
    left = block(0, [1.0, 4.0, 2.0])
    right = block(2, [3.0, 2.0, 5.0])
    result = combine_probe_blocks((left, right), stage_delta=1.0e-4)
    assert result["row_image_maxima"] == [3.0, 4.0, 5.0]
    assert result["row_domain_passed"]
    assert result["logical_total_linearized_sweeps"] == 5
    assert result["outcome_files_read"] == 0

    broken = copy.deepcopy(right)
    broken["corrected_path_sha256"] = "C"
    try:
        combine_probe_blocks((left, broken), stage_delta=1.0e-4)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched paths were accepted")
    print("PASS: staged causal-row probe blocks combine conservatively")


if __name__ == "__main__":
    main()
