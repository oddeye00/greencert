#!/usr/bin/env python3
"""Independent arithmetic audit of the low-rank causal-resolvent record."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "transformer_low_rank_resolvent_diagnostic.json"
OUTPUT = (
    ROOT
    / "results"
    / "transformer_low_rank_resolvent_independent_audit.json"
)
EXPECTED_CONFIGURATIONS = ((0, 26), (4, 26), (8, 26), (4, 7))
EXPECTED_UNION_DIMENSIONS = (0, 4, 8, 16)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def close(left: float, right: float, tolerance: float = 3.0e-11) -> bool:
    return math.isclose(
        float(left), float(right), rel_tol=tolerance, abs_tol=1.0e-300
    )


def c_delta(delta: float, probes: int) -> float:
    return NormalDist().inv_cdf(
        0.5 * (1.0 + float(delta) ** (1.0 / int(probes)))
    )


def finite_sum(alpha: float, horizon: int) -> float:
    term = 1.0
    total = 0.0
    for _ in range(int(horizon)):
        total += term
        term *= float(alpha)
    return total


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    require(
        payload["status"]
        == "low-rank union-subspace causal-resolvent diagnostic complete",
        "diagnostic is incomplete",
    )
    require(payload["outcome_files_read"] == 0, "outcome boundary changed")
    require(
        tuple(map(tuple, payload["configurations_predeclared"]))
        == EXPECTED_CONFIGURATIONS,
        "predeclared configuration family changed",
    )
    require(
        tuple(map(tuple, payload["configurations_executed"]))
        == EXPECTED_CONFIGURATIONS,
        "not every predeclared configuration executed",
    )
    require(len(payload["rows"]) == 4, "row count changed")
    bridge = payload["corrected_path_bridge"]
    require(
        bridge["status"]
        == "recomputed-to-sealed corrected-parameter bridge passed",
        "corrected-path bridge failed",
    )
    require(bridge["outcome_files_read"] == 0, "bridge read an outcome")
    lower_bounds = []
    gain_ratios = []
    adjoint_residuals = []
    for index, row in enumerate(payload["rows"]):
        require(
            (int(row["rank"]), int(row["block_size"]))
            == EXPECTED_CONFIGURATIONS[index],
            "configuration order changed",
        )
        require(row["outcome_files_read"] == 0, "row read an outcome")
        require(not row["issued"], "negative audit unexpectedly issued")
        require(not row["domain_passed"], "negative audit passed its domain")
        require(
            int(row["union_reduction"]["union_dimension"])
            == EXPECTED_UNION_DIMENSIONS[index],
            "union dimension changed",
        )
        probes = int(row["probes_per_residual"])
        delta = float(row["mismatch_operator_stage_delta"])
        calibration = c_delta(delta, probes)
        direct = max(map(float, row["mismatch_image_norms"])) / calibration
        require(
            close(direct, row["mismatch_direct_gain_upper"]),
            "direct mismatch bound changed",
        )
        gram_upper = []
        gram_lower = []
        for gram in row["mismatch_gram_rows"]:
            power = int(gram["power"])
            require(close(gram["c_delta"], calibration), "calibration changed")
            upper = (float(gram["Y"]) / calibration) ** (
                1.0 / (2.0 * power)
            )
            require(
                close(upper, gram["operator_norm_upper_bound"]),
                "Gram upper bound changed",
            )
            gram_upper.append(upper)
            gram_lower.append(float(gram["operator_norm_lower_estimate"]))
        mismatch = min([direct] + gram_upper)
        lower = max(
            [
                max(
                    image / initial
                    for image, initial in zip(
                        map(float, row["mismatch_image_norms"]),
                        map(float, row["initial_probe_norms"]),
                    )
                )
            ]
            + gram_lower
        )
        require(close(mismatch, row["mismatch_gain_upper"]), "route minimum changed")
        require(
            close(lower, row["mismatch_gain_lower_estimate"]),
            "mismatch lower estimate changed",
        )
        multiplier = finite_sum(mismatch, int(payload["horizon"]))
        require(
            close(multiplier, row["finite_resolvent_multiplier_upper"]),
            "finite multiplier changed",
        )
        structured = float(row["approximate_structured_gain_upper"]) * multiplier
        require(
            close(structured, row["structured_gain_upper"]),
            "structured gain changed",
        )
        require(lower > 1.0, "sampled mismatch lower bound no longer exceeds one")
        require(
            float(row["mismatch_adjoint_relative_residual"]) <= 1.0e-9,
            "mismatch adjoint residual is too large",
        )
        lower_bounds.append(lower)
        gain_ratios.append(float(row["gain_ratio_to_released"]))
        adjoint_residuals.append(
            float(row["mismatch_adjoint_relative_residual"])
        )
    audit = {
        "status": "independent low-rank causal-resolvent arithmetic audit passed",
        "source": SOURCE.relative_to(ROOT).as_posix(),
        "source_sha256": sha256(SOURCE),
        "configurations_verified": len(payload["rows"]),
        "all_outcome_files_read": 0,
        "all_configurations_abstained": True,
        "minimum_sampled_mismatch_norm_lower_bound": min(lower_bounds),
        "maximum_gain_ratio_to_released": max(gain_ratios),
        "maximum_adjoint_relative_residual": max(adjoint_residuals),
        "interpretation": (
            "The exact union-subspace reduction removes optimizer-skeleton "
            "inflation, but ranks 0, 4, and 8 leave a mismatch operator whose "
            "norm already exceeds one by a deterministic sampled lower bound."
        ),
        "outcome_files_read": 0,
    }
    OUTPUT.write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
