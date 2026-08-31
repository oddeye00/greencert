#!/usr/bin/env python3
"""Independent hash/probability/root audit of the segmented diagnostic."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "transformer_segmented_resolvent_diagnostic.json"
OUTPUT = (
    ROOT / "results" / "transformer_segmented_resolvent_independent_audit.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def close(left: float, right: float, tolerance: float = 3.0e-12) -> bool:
    return abs(float(left) - float(right)) <= tolerance * max(
        1.0, abs(float(left)), abs(float(right))
    )


def c_delta(delta: float, probes: int) -> float:
    return NormalDist().inv_cdf(0.5 * (1.0 + delta ** (1.0 / probes)))


def finite_sum(alpha: float, horizon: int) -> float:
    term = 1.0
    total = 0.0
    for _ in range(horizon):
        total += term
        term *= alpha
    return total


def root(response: float, gain: float, lipschitz: float) -> float | None:
    coefficient = gain * lipschitz
    discriminant = 1.0 - 2.0 * coefficient * response
    if discriminant < 0.0:
        return None
    return 2.0 * response / (1.0 + math.sqrt(discriminant))


def main() -> None:
    payload = load(SOURCE)
    if payload["status"] != "segmented causal-resolvent diagnostic complete":
        raise AssertionError("diagnostic is incomplete")
    if payload["outcome_files_read"] != 0:
        raise AssertionError("diagnostic reports an outcome read")
    source_path = ROOT / payload["source"]
    if sha256(source_path) != payload["source_sha256"]:
        raise AssertionError("structured source hash mismatch")
    rows = payload["rows"]
    if [int(row["block_size"]) for row in rows] != [26, 7, 2, 1]:
        raise AssertionError("unexpected segment grid")
    horizon = int(payload["horizon"])
    stage_delta = float(payload["stage_delta"])
    if not close(stage_delta, 1.0e-6 / 6.0):
        raise AssertionError("joint probability allocation changed")
    calibration = c_delta(stage_delta, int(payload["probes"]))

    arithmetic_rows = 0
    for row in rows:
        if int(row["block_size"]) == 1:
            if (
                float(row["mismatch_gain_upper"]) != 0.0
                or not row["issued"]
                or not row["inherited_exact_operator_bound"]
            ):
                raise AssertionError("exact per-step calibration row changed")
            continue
        alpha = (
            max(float(value) for value in row["gram_image_norms"])
            / calibration
        ) ** 0.5
        kappa0 = max(
            float(value) for value in row["approximate_green_image_norms"]
        ) / calibration
        multiplier = finite_sum(alpha, horizon)
        gain = kappa0 * multiplier
        if not close(alpha, row["mismatch_gain_upper"]):
            raise AssertionError("mismatch gain arithmetic changed")
        if not close(kappa0, row["approximate_structured_gain_upper"]):
            raise AssertionError("approximate gain arithmetic changed")
        if not close(multiplier, row["finite_resolvent_multiplier_upper"]):
            raise AssertionError("finite resolvent multiplier changed")
        if not close(gain, row["preconditioned_structured_gain_upper"]):
            raise AssertionError("preconditioned gain arithmetic changed")
        response = gain * float(row["parameter_forcing_upper"])
        radius = root(
            response,
            gain,
            float(row["objective_hessian_lipschitz_upper"]),
        )
        stored_radius = row["parameter_remainder_radius"]
        if radius is None:
            if stored_radius is not None:
                raise AssertionError("failed root was stored as finite")
        elif stored_radius is None or not close(radius, stored_radius):
            raise AssertionError("quadratic root arithmetic changed")
        arithmetic_rows += 1

    by_block = {int(row["block_size"]): row for row in rows}
    if by_block[26]["issued"]:
        raise AssertionError("one-segment diagnostic unexpectedly issued")
    if not by_block[7]["issued"] or by_block[7]["bracket"] != [2, 2]:
        raise AssertionError("four-segment bracket changed")
    if not by_block[2]["issued"] or by_block[2]["bracket"] != [2, 2]:
        raise AssertionError("thirteen-segment bracket changed")
    if not close(
        by_block[7]["preconditioned_to_released_gain_ratio"],
        11.261278901380102,
    ):
        raise AssertionError("four-segment gain ratio changed")
    if not close(
        by_block[2]["preconditioned_to_released_gain_ratio"],
        0.7349481168415238,
    ):
        raise AssertionError("thirteen-segment gain ratio changed")

    result = {
        "status": "independent segmented causal-resolvent arithmetic audit passed",
        "source": SOURCE.relative_to(ROOT).as_posix(),
        "source_sha256": sha256(SOURCE),
        "structured_source_sha256": sha256(source_path),
        "arithmetic_rows_verified": arithmetic_rows,
        "four_segment_issued": bool(by_block[7]["issued"]),
        "four_segment_bracket": by_block[7]["bracket"],
        "four_segment_mismatch_gain_upper": by_block[7][
            "mismatch_gain_upper"
        ],
        "four_segment_resolvent_multiplier_upper": by_block[7][
            "finite_resolvent_multiplier_upper"
        ],
        "thirteen_segment_issued": bool(by_block[2]["issued"]),
        "thirteen_segment_bracket": by_block[2]["bracket"],
        "thirteen_segment_gain_ratio_to_released": by_block[2][
            "preconditioned_to_released_gain_ratio"
        ],
        "outcome_files_read": 0,
    }
    OUTPUT.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
