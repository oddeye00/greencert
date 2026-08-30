#!/usr/bin/env python3
"""Independent arithmetic audit of the mixed-precision residual replay."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

from flint import ctx

from one_shot_recenter_closure import conservative_one_shot_closure
from outward_inexact_anytime_gram import (
    folded_normal_calibration_lower,
    outward_inexact_gram_operator_upper_bound,
    outward_operator_supersolution_value,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "transformer_v3_mixed_precision_residual_postseal_audit.json"
DESTINATION = ROOT / "results" / "transformer_v3_mixed_precision_residual_independent_audit.json"
RUNNER = ROOT / "scripts" / "audit_transformer_v3_mixed_precision_residual.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def close(left: float, right: float, tolerance: float = 2.0e-12) -> bool:
    return abs(float(left) - float(right)) <= tolerance * max(
        abs(float(left)), abs(float(right)), 1.0e-300
    )


def main() -> None:
    if DESTINATION.exists():
        raise FileExistsError(f"refusing to overwrite {DESTINATION}")
    result = load(RESULT)
    source = ROOT / result["source"]
    tolerance_source = ROOT / result["tolerance_source"]
    if sha256(source) != result["source_sha256"]:
        raise AssertionError("combined benchmark hash mismatch")
    if sha256(tolerance_source) != result["tolerance_source_sha256"]:
        raise AssertionError("tolerance audit hash mismatch")
    source_data = load(source)
    tolerance = load(tolerance_source)
    output_calibration = folded_normal_calibration_lower(
        delta=float(source_data["family_budget"]["role_output_per_operator_delta"]),
        probes=16,
    )
    green_calibration = folded_normal_calibration_lower(
        delta=float(source_data["family_budget"]["green_per_operator_delta"]),
        probes=16,
    )

    rows = list(result["output_rows"]) + [result["green"]]
    ratios = []
    maximum_supersolution_lower = math.inf
    for row in rows:
        if int(row["timing_repetitions"]) != 5:
            raise AssertionError("mixed-precision timing repetition count changed")
        if not row["exact_repeat_outputs_identical"]:
            raise AssertionError("binary64 repeat output changed")
        if not row["float32_repeat_outputs_identical"]:
            raise AssertionError("float32 repeat output changed")
        terminal = float(row["terminal_norm_inflated_upper"])
        residual = float(row["inflated_residual_upper"])
        if row is result["green"]:
            delta_key = "green"
            calibration = green_calibration
        else:
            delta_key = "output"
            calibration = output_calibration
        recorded_operator = float(row["outward_residual_corrected_operator_upper"])
        recomputed_operator = outward_inexact_gram_operator_upper_bound(
            terminal_norm=terminal,
            calibration_lower=calibration,
            residual_norms=[residual],
        )
        if recorded_operator < recomputed_operator:
            raise AssertionError(f"{delta_key} recorded root is below recomputation")
        old_precision = ctx.prec
        ctx.prec = 256
        try:
            value = outward_operator_supersolution_value(
                recorded_operator,
                terminal_norm=terminal,
                calibration_lower=calibration,
                residual_norms=[residual],
            )
            lower = float(value.lower())
        finally:
            ctx.prec = old_precision
        if lower < 0.0:
            raise AssertionError(f"{delta_key} outward root is not a supersolution")
        maximum_supersolution_lower = min(maximum_supersolution_lower, lower)
        ratios.append(float(row["measured_residual_to_terminal_ratio"]))
        expected_ratio = float(row["maximum_measured_residual_norm"]) / float(
            row["terminal_norm_float32_promoted"]
        )
        if not close(expected_ratio, ratios[-1]):
            raise AssertionError("residual ratio mismatch")

    maximum_ratio = max(ratios)
    admissible = float(
        tolerance["tolerances"]["common"][
            "lower_passing_relative_gram_residual"
        ]
    )
    if not close(maximum_ratio, result["maximum_measured_relative_residual"]):
        raise AssertionError("maximum residual ratio mismatch")
    if not maximum_ratio < admissible:
        raise AssertionError("measured residual exceeds admissible sensitivity")
    if not close(admissible / maximum_ratio, result["admissible_to_measured_ratio"]):
        raise AssertionError("headroom ratio mismatch")

    closure_row = result["closure"]
    closure = conservative_one_shot_closure(
        kappa=float(result["green"]["outward_residual_corrected_operator_upper"]),
        derivative_drift=float(closure_row["derivative_drift"]),
        response_sequence_norm=float(closure_row["response_sequence_norm"]),
        response_max_state_norm=float(closure_row["response_max_state_norm"]),
        domain_radius=float(closure_row["domain_radius"]),
    )
    for key, value in closure.as_dict().items():
        recorded = closure_row[key]
        if isinstance(value, bool):
            if bool(value) != bool(recorded):
                raise AssertionError(f"closure boolean changed: {key}")
        elif value is None:
            if recorded is not None:
                raise AssertionError(f"closure null changed: {key}")
        elif not close(value, recorded):
            raise AssertionError(f"closure scalar changed: {key}")

    by_step = {int(row["step"]): row for row in result["output_rows"]}
    event = int(result["bracket"][0])
    radius = float(closure.total_pointwise_radius)
    minimum_slack = math.inf
    for step in source_data["decision"]["query_order"]:
        row = by_step[int(step)]
        margin = math.sqrt(2.0) * (
            float(row["outward_residual_corrected_operator_upper"]) * radius
            + 0.5 * float(row["block_second"]) * radius * radius
        )
        if event <= int(step) < event + 25:
            slack = float(row["raw_guarantee_slack"]) - margin
        else:
            slack = float(row["raw_exclusion_slack"]) - margin
        if slack <= 0.0:
            raise AssertionError(f"queried logic fails at step {step}")
        minimum_slack = min(minimum_slack, slack)
    if not close(minimum_slack, result["minimum_queried_logic_slack"]):
        raise AssertionError("minimum logic slack mismatch")
    if result["bracket"] != source_data["sealed_bracket"]:
        raise AssertionError("bracket changed")

    timings = result["timings_seconds"]
    repetitions = int(timings["timing_repetitions"])
    if repetitions != 5:
        raise AssertionError("aggregate timing repetition count changed")
    output_exact = [float(value) for value in timings["output_gram_binary64_trials"]]
    output_approximate = [float(value) for value in timings["output_gram_float32_trials"]]
    green_exact = [float(value) for value in timings["green_gram_binary64_trials"]]
    green_approximate = [float(value) for value in timings["green_gram_float32_trials"]]
    vectors = (output_exact, output_approximate, green_exact, green_approximate)
    if any(len(vector) != repetitions for vector in vectors):
        raise AssertionError("aggregate timing vector length changed")
    output_speedups = [
        exact / approximate
        for exact, approximate in zip(output_exact, output_approximate)
    ]
    green_speedups = [
        exact / approximate
        for exact, approximate in zip(green_exact, green_approximate)
    ]
    combined_speedups = [
        (output_exact[index] + green_exact[index])
        / (output_approximate[index] + green_approximate[index])
        for index in range(repetitions)
    ]
    speedups = result["measured_operator_speedups"]
    checks = (
        ("output_median_paired", statistics.median(output_speedups)),
        ("green_median_paired", statistics.median(green_speedups)),
        ("combined_q1_kernel", statistics.median(combined_speedups)),
        ("combined_q1_kernel_minimum", min(combined_speedups)),
        ("combined_q1_kernel_maximum", max(combined_speedups)),
    )
    for key, expected in checks:
        if not close(expected, speedups[key]):
            raise AssertionError(f"speedup mismatch: {key}")
    for recorded, expected in zip(
        speedups["combined_q1_kernel_trials"], combined_speedups
    ):
        if not close(recorded, expected):
            raise AssertionError("paired combined timing trace changed")
    expected_combined_speedup = statistics.median(combined_speedups)

    payload = {
        "status": "independent mixed-precision residual audit passed",
        "result": str(RESULT.relative_to(ROOT)),
        "result_sha256": sha256(RESULT),
        "runner_sha256": sha256(RUNNER),
        "source_sha256": sha256(source),
        "tolerance_source_sha256": sha256(tolerance_source),
        "same_bracket": True,
        "maximum_measured_relative_residual": maximum_ratio,
        "admissible_relative_residual": admissible,
        "admissible_to_measured_ratio": admissible / maximum_ratio,
        "combined_q1_kernel_speedup": expected_combined_speedup,
        "combined_q1_kernel_speedup_minimum": min(combined_speedups),
        "combined_q1_kernel_speedup_maximum": max(combined_speedups),
        "timing_repetitions": repetitions,
        "minimum_queried_logic_slack": minimum_slack,
        "all_outward_scalar_supersolutions_rechecked": True,
        "claim_boundary": result["claim_boundary"],
    }
    DESTINATION.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**payload, "output": str(DESTINATION), "sha256": sha256(DESTINATION)}, indent=2))


if __name__ == "__main__":
    main()
