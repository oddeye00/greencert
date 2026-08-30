#!/usr/bin/env python3
"""Arithmetic checks for the matrix-free Transformer scaling benchmark."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "results" / "transformer_scaling_benchmark.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def close(left: float, right: float) -> None:
    assert math.isclose(float(left), float(right), rel_tol=2e-14, abs_tol=1e-300)


def main() -> None:
    payload = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    profiles = payload["profiles"]
    assert [row["profile"] for row in profiles] == ["paper", "100k", "1m"]
    assert profiles[0]["parameter_count"] == 13_792
    assert 100_000 <= profiles[1]["parameter_count"] <= 130_000
    assert profiles[2]["parameter_count"] >= 1_000_000

    for profile in profiles:
        source = ROOT / "results" / f"transformer_scaling_benchmark_{profile['profile']}.json"
        assert sha256(source) == payload["profile_source_hashes"][profile["profile"]]
        timing = profile["timings_seconds"]
        assert timing["objective_hvp_median"] > 0.0
        assert timing["output_jacobian_gram_median"] > 0.0
        assert profile["observed_process_peak_rss_bytes"] > 0
        projection = profile["projection_h300"]
        assert projection["centerline_objective_hvp_calls"] == 1_500
        assert projection["green_probe_objective_hvp_calls"] == 76_800
        assert projection["output_probe_gram_calls"] == 38_528
        expected_core = (
            (1_500 + 76_800) * timing["objective_hvp_median"]
            + 38_528 * timing["output_jacobian_gram_median"]
        )
        close(projection["projected_core_certificate_seconds"], expected_core)
        close(
            projection["projected_300_step_training_seconds"],
            300.0 * timing["gradient_median"],
        )

    scaling = payload["scaling"]
    close(
        scaling["paper_to_largest_parameter_ratio"],
        profiles[-1]["parameter_count"] / profiles[0]["parameter_count"],
    )
    assert scaling["largest_profile_completes_matrix_free_hvp"] is True
    assert scaling["largest_profile_completes_output_gram_product"] is True
    costs = payload["existing_paper_scale_cost"]
    assert costs["candidate_certificates_with_runtime"] == 22
    assert costs["training_runs"] == 24
    assert costs["aggregate_candidate_hours"] > 19.0
    assert costs["certificate_to_300_step_continuation_ratio"] > 600.0
    print(
        "PASS: matrix-free HVP/output Gram reach 1,008,864 parameters; "
        f"measured paper-batch certificate/continuation cost ratio is "
        f"{costs['certificate_to_300_step_continuation_ratio']:.1f}x."
    )


if __name__ == "__main__":
    main()
