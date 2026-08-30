#!/usr/bin/env python3
"""Post-seal mixed-precision audit for the inexact Gram theorem.

The exact target operators are the binary64 Green/output operators along the
immutable response-centered Transformer path.  For each committed q=1 probe
block, this audit evaluates the same Gram product in float32, measures the
binary64 discrepancy probe by probe, and feeds an inflated discrepancy norm to
the residual-corrected Gram root.

This is implementation-development evidence, not a computer-assisted proof:
the binary64 reference products and the measured discrepancies are not outward
enclosures of the exact-real neural kernels.  Its purpose is to test whether a
real approximate kernel lies comfortably inside the theorem's independently
computed admissible-residual budget.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from batched_green_operator import (
    make_batched_output_gram_operator,
    make_batched_transformer_green_products,
)
from benchmark_transformer_v3_role_sparse import role_identity
from one_shot_recenter_closure import conservative_one_shot_closure
from outward_inexact_anytime_gram import (
    folded_normal_calibration_lower,
    outward_inexact_gram_operator_upper_bound,
)
from probe_jacobian_bound import ProbeConfig, namespaced_probe_seed
from transformer_block_envelope import objective_hessian_lipschitz
from transformer_certificate_protocol import Candidate
from transformer_four_sweep_development_audit import to_scaled
from transformer_green_development_audit import build_frozen_centerline
from transformer_green_operator import make_transformer_green_products
from transformer_hvp_grokking import flat_spec, make_template
from transformer_v3_certificate import METHOD_SEAL, load_candidate, safe_json
from transformer_v3_protocol import (
    FAMILY_FAILURE_PROBABILITY,
    HORIZON,
    MAXIMUM_POWER,
    PROBES,
    green_identity,
    maximum_operator_count,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOURCE = RESULTS / (
    "transformer_v3_combined_online_role_seed_366_gate_1_anchor_1120_"
    "matched-combined-v5.json"
)
TOLERANCE_SOURCE = RESULTS / "transformer_v3_inexact_operator_tolerance_postseal_audit.json"
DESTINATION = RESULTS / "transformer_v3_mixed_precision_residual_postseal_audit.json"
CANDIDATE = Candidate(366, 0.80, 1120)
MEASUREMENT_INFLATION = 1.0 + 1.0e-9
TIMING_REPETITIONS = 5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def relative_error(left: float, right: float) -> float:
    return abs(float(left) - float(right)) / max(
        abs(float(right)), torch.finfo(torch.float64).tiny
    )


def committed_rows(*, dimension: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.stack(
        [
            torch.randn(dimension, generator=generator, dtype=torch.float64)
            for _ in range(PROBES)
        ]
    )


def inflated_upper(value: float) -> float:
    return float(np.nextafter(float(value) * MEASUREMENT_INFLATION, math.inf))


def compare_q1(
    *,
    exact_apply,
    approximate_apply,
    vectors: torch.Tensor,
    calibration_lower: float,
) -> dict:
    exact_started = time.perf_counter()
    exact_terminal = exact_apply(vectors)
    exact_seconds = time.perf_counter() - exact_started

    approximate_started = time.perf_counter()
    approximate_terminal32 = approximate_apply(vectors.to(torch.float32))
    approximate_seconds = time.perf_counter() - approximate_started
    approximate_terminal = approximate_terminal32.to(torch.float64)

    exact_repeat_seconds: list[float] = []
    approximate_repeat_seconds: list[float] = []
    exact_repeat_identical = True
    approximate_repeat_identical = True

    def run_exact() -> None:
        nonlocal exact_repeat_identical
        started = time.perf_counter()
        value = exact_apply(vectors)
        exact_repeat_seconds.append(time.perf_counter() - started)
        exact_repeat_identical = exact_repeat_identical and torch.equal(
            value, exact_terminal
        )

    def run_approximate() -> None:
        nonlocal approximate_repeat_identical
        started = time.perf_counter()
        value = approximate_apply(vectors.to(torch.float32))
        approximate_repeat_seconds.append(time.perf_counter() - started)
        approximate_repeat_identical = approximate_repeat_identical and torch.equal(
            value, approximate_terminal32
        )

    # The first pair above warms both paths.  Alternate order thereafter so a
    # cache/order advantage cannot systematically favor one precision.
    for repetition in range(TIMING_REPETITIONS):
        if repetition % 2 == 0:
            run_exact()
            run_approximate()
        else:
            run_approximate()
            run_exact()
    if not exact_repeat_identical or not approximate_repeat_identical:
        raise RuntimeError("repeated deterministic kernel output changed")

    exact_norms = torch.linalg.vector_norm(exact_terminal, dim=1)
    approximate_norms = torch.linalg.vector_norm(approximate_terminal, dim=1)
    residual_norms = torch.linalg.vector_norm(
        approximate_terminal - exact_terminal, dim=1
    )
    terminal_upper = inflated_upper(float(approximate_norms.max()))
    residual_upper = inflated_upper(float(residual_norms.max()))
    operator_upper = outward_inexact_gram_operator_upper_bound(
        terminal_norm=terminal_upper,
        calibration_lower=calibration_lower,
        residual_norms=[residual_upper],
    )
    exact_float_root = math.sqrt(float(exact_norms.max()) / calibration_lower)
    return {
        "terminal_norm_exact_binary64": float(exact_norms.max()),
        "terminal_norm_float32_promoted": float(approximate_norms.max()),
        "terminal_norm_inflated_upper": terminal_upper,
        "maximum_measured_residual_norm": float(residual_norms.max()),
        "inflated_residual_upper": residual_upper,
        "measured_residual_to_terminal_ratio": (
            float(residual_norms.max()) / float(approximate_norms.max())
        ),
        "outward_residual_corrected_operator_upper": operator_upper,
        "binary64_reference_operator_root": exact_float_root,
        "operator_bound_inflation": operator_upper / exact_float_root,
        "warmup_exact_operator_seconds": exact_seconds,
        "warmup_float32_operator_seconds": approximate_seconds,
        "timing_repetitions": TIMING_REPETITIONS,
        "timing_order": "alternating after one warm-up pair",
        "exact_repeat_seconds": exact_repeat_seconds,
        "float32_repeat_seconds": approximate_repeat_seconds,
        "exact_repeat_outputs_identical": exact_repeat_identical,
        "float32_repeat_outputs_identical": approximate_repeat_identical,
        "exact_operator_seconds": statistics.median(exact_repeat_seconds),
        "float32_operator_seconds": statistics.median(approximate_repeat_seconds),
        "measured_kernel_speedup": statistics.median(
            exact / approximate
            for exact, approximate in zip(
                exact_repeat_seconds, approximate_repeat_seconds
            )
        ),
    }


def main() -> None:
    if DESTINATION.exists():
        raise FileExistsError(f"refusing to overwrite {DESTINATION}")
    source = safe_json(SOURCE)
    tolerance = safe_json(TOLERANCE_SOURCE)
    if source["candidate"] != CANDIDATE.__dict__:
        raise RuntimeError("combined source candidate changed")
    if not source["same_bracket"] or source["fallback_rows"]:
        raise RuntimeError("expected an issued singleton with no fallback rows")

    config64, template64, spec64, data, parameter, velocity = load_candidate(CANDIDATE)
    torch.set_num_threads(config64.threads)
    torch.use_deterministic_algorithms(True)
    train_pairs, train_labels, _, _, cert_pairs, _ = data
    dimension = int(parameter.numel())
    horizon = int(source["horizon"])

    center_started = time.perf_counter()
    path = build_frozen_centerline(
        config64,
        template64,
        spec64,
        train_pairs,
        train_labels,
        parameter,
        velocity,
    )
    center64 = path["center"]
    scaled_center = path["scaled_center"]
    residual = torch.stack(
        [
            to_scaled(path["map_step"](center64[step]), dimension, config64.learning_rate)
            - scaled_center[step + 1]
            for step in range(horizon)
        ]
    )
    signed_green, _ = make_transformer_green_products(
        center64[:horizon, :dimension],
        train_pairs,
        train_labels,
        template64,
        spec64,
        config64,
    )
    signed_response = signed_green(residual.reshape(-1)).reshape(horizon, -1)
    response_norm = float(torch.linalg.vector_norm(signed_response))
    response_max = float(torch.linalg.vector_norm(signed_response, dim=1).max())
    domain_radius = 2.0 * response_norm
    center_seconds = time.perf_counter() - center_started
    if path["centerline_sha256"] != source["centerline_sha256"]:
        raise RuntimeError("centerline hash changed")

    config32 = replace(config64, dtype="float32")
    template32 = make_template(config32)
    spec32 = flat_spec(template32)
    if spec32.shapes != spec64.shapes or spec32.sizes != spec64.sizes:
        raise RuntimeError("float32 template changed parameter layout")
    center32 = center64.to(torch.float32)

    maximum_candidates = int(maximum_operator_count()["maximum_candidates"])
    role_outputs_per_candidate = HORIZON + (HORIZON + 1)
    output_delta = (
        0.5
        * FAMILY_FAILURE_PROBABILITY
        / (maximum_candidates * role_outputs_per_candidate)
    )
    green_delta = 0.5 * FAMILY_FAILURE_PROBABILITY / maximum_candidates
    output_calibration = folded_normal_calibration_lower(
        delta=output_delta, probes=PROBES
    )
    green_calibration = folded_normal_calibration_lower(
        delta=green_delta, probes=PROBES
    )
    master_nonce = str(safe_json(METHOD_SEAL)["master_nonce"])

    train_cert_pairs = torch.cat((train_pairs, cert_pairs), dim=0)
    output_audits: list[dict] = []
    maximum_map_drift = 0.0
    output_rows = {int(row["step"]): row for row in source["output_rows"]}
    for step in range(1, horizon + 1):
        row = output_rows[step]
        supports_training = bool(row["supports_training"])
        supports_certificate = bool(row["supports_certificate"])
        if supports_training and supports_certificate:
            pairs = train_cert_pairs
        elif supports_training:
            pairs = train_pairs
        elif supports_certificate:
            pairs = cert_pairs
        else:
            raise RuntimeError(f"output row {step} supports no role")
        if len(pairs) != int(row["pairs"]):
            raise RuntimeError(f"pair count changed at step {step}")

        exact_apply = make_batched_output_gram_operator(
            center64[step, :dimension], pairs, template64, spec64
        )
        approximate_apply = make_batched_output_gram_operator(
            center32[step, :dimension], pairs, template32, spec32
        )
        vectors = committed_rows(
            dimension=dimension,
            seed=namespaced_probe_seed(
                master_nonce,
                role_identity(CANDIDATE, step, int(row["role"])),
            ),
        )
        # Guard against a future identity refactor by checking the committed Y.
        audit = compare_q1(
            exact_apply=exact_apply,
            approximate_apply=approximate_apply,
            vectors=vectors,
            calibration_lower=output_calibration,
        )
        if relative_error(
            audit["terminal_norm_exact_binary64"], float(row["Y"])
        ) > 2.0e-12:
            raise RuntimeError(f"committed output trace changed at step {step}")
        audit.update(
            {
                "step": step,
                "pairs": int(len(pairs)),
                "role": int(row["role"]),
                "supports_training": supports_training,
                "supports_certificate": supports_certificate,
                "block_second": float(row["block_second"]),
                "block_third": float(row["block_third"]),
                "raw_guarantee_slack": float(row["raw_guarantee_slack"]),
                "raw_exclusion_slack": float(row["raw_exclusion_slack"]),
            }
        )
        output_audits.append(audit)
        if supports_training and step < horizon:
            first_ball = (
                float(audit["outward_residual_corrected_operator_upper"])
                + float(row["block_second"]) * domain_radius
            )
            objective_drift = objective_hessian_lipschitz(
                first_ball,
                float(row["block_second"]),
                float(row["block_third"]),
            )
            maximum_map_drift = max(
                maximum_map_drift,
                math.sqrt(2.0) * config64.learning_rate * objective_drift,
            )

    green64, green64_t = make_batched_transformer_green_products(
        center64[:horizon, :dimension],
        train_pairs,
        train_labels,
        template64,
        spec64,
        config64,
    )
    green32, green32_t = make_batched_transformer_green_products(
        center32[:horizon, :dimension],
        train_pairs,
        train_labels,
        template32,
        spec32,
        config32,
    )
    green_vectors = committed_rows(
        dimension=horizon * 2 * dimension,
        seed=namespaced_probe_seed(
            master_nonce, green_identity(CANDIDATE, horizon)
        ),
    )
    green_audit = compare_q1(
        exact_apply=lambda rows: green64_t(green64(rows)),
        approximate_apply=lambda rows: green32_t(green32(rows)),
        vectors=green_vectors,
        calibration_lower=green_calibration,
    )
    frozen_green = source["green_rows"][0]
    if relative_error(
        green_audit["terminal_norm_exact_binary64"], float(frozen_green["Y"])
    ) > 2.0e-12:
        raise RuntimeError("committed Green trace changed")

    closure = conservative_one_shot_closure(
        kappa=float(green_audit["outward_residual_corrected_operator_upper"]),
        derivative_drift=maximum_map_drift,
        response_sequence_norm=response_norm,
        response_max_state_norm=response_max,
        domain_radius=domain_radius,
    )
    if not closure.closure_passed:
        raise RuntimeError("mixed-precision residual-corrected closure failed")

    radius = float(closure.total_pointwise_radius)
    query_order = [int(step) for step in source["decision"]["query_order"]]
    event = int(source["frozen_predicted_persistent_event"])
    persistence = 25
    guarantees: dict[int, bool] = {}
    exclusions: dict[int, bool] = {0: True}
    by_step = {int(row["step"]): row for row in output_audits}
    logic_slacks: list[float] = []
    for step in query_order:
        row = by_step[step]
        if not row["supports_certificate"]:
            raise RuntimeError(f"queried step {step} lacks certificate role")
        margin = math.sqrt(2.0) * (
            float(row["outward_residual_corrected_operator_upper"]) * radius
            + 0.5 * float(row["block_second"]) * radius * radius
        )
        guarantee_slack = float(row["raw_guarantee_slack"]) - margin
        exclusion_slack = float(row["raw_exclusion_slack"]) - margin
        guarantees[step] = guarantee_slack > 0.0
        exclusions[step] = exclusion_slack > 0.0
        logic_slacks.append(
            guarantee_slack if event <= step < event + persistence else exclusion_slack
        )
    success = all(guarantees.get(step, False) for step in range(event, event + persistence))
    failure_witnesses = [
        int(step) for step in source["decision"]["failure_witnesses"]
    ]
    earlier_excluded = all(exclusions.get(step, False) for step in failure_witnesses)
    if not success or not earlier_excluded:
        raise RuntimeError("mixed-precision output margins do not reproduce bracket")

    residual_ratios = [
        float(row["measured_residual_to_terminal_ratio"])
        for row in output_audits
    ] + [float(green_audit["measured_residual_to_terminal_ratio"])]
    admissible = float(
        tolerance["tolerances"]["common"][
            "lower_passing_relative_gram_residual"
        ]
    )
    output_exact_trials = [
        sum(float(row["exact_repeat_seconds"][trial]) for row in output_audits)
        for trial in range(TIMING_REPETITIONS)
    ]
    output_float32_trials = [
        sum(float(row["float32_repeat_seconds"][trial]) for row in output_audits)
        for trial in range(TIMING_REPETITIONS)
    ]
    green_exact_trials = [float(value) for value in green_audit["exact_repeat_seconds"]]
    green_float32_trials = [float(value) for value in green_audit["float32_repeat_seconds"]]
    combined_exact_trials = [
        output_exact_trials[trial] + green_exact_trials[trial]
        for trial in range(TIMING_REPETITIONS)
    ]
    combined_float32_trials = [
        output_float32_trials[trial] + green_float32_trials[trial]
        for trial in range(TIMING_REPETITIONS)
    ]
    output_paired_speedups = [
        exact / approximate
        for exact, approximate in zip(output_exact_trials, output_float32_trials)
    ]
    green_paired_speedups = [
        exact / approximate
        for exact, approximate in zip(green_exact_trials, green_float32_trials)
    ]
    combined_paired_speedups = [
        exact / approximate
        for exact, approximate in zip(combined_exact_trials, combined_float32_trials)
    ]
    result = {
        "status": "post-seal mixed-precision residual audit passed",
        "scope": (
            "Measured float32-vs-binary64 q=1 Gram residuals on one immutable "
            "candidate; residual measurements are not outward exact-real kernel bounds."
        ),
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "tolerance_source": str(TOLERANCE_SOURCE.relative_to(ROOT)),
        "tolerance_source_sha256": sha256(TOLERANCE_SOURCE),
        "candidate": CANDIDATE.__dict__,
        "centerline_sha256": path["centerline_sha256"],
        "precision_path": "float32 HVP/JVP/VJP and output Gram, promoted to float64 for residual measurement",
        "measurement_inflation": MEASUREMENT_INFLATION,
        "admissible_common_relative_residual_threshold": admissible,
        "maximum_measured_relative_residual": max(residual_ratios),
        "admissible_to_measured_ratio": admissible / max(residual_ratios),
        "all_measured_residuals_below_admissible_threshold": max(residual_ratios) < admissible,
        "green": green_audit,
        "closure": closure.as_dict(),
        "bracket": [event, event],
        "same_bracket": [event, event] == source["sealed_bracket"],
        "minimum_queried_logic_slack": min(logic_slacks),
        "timings_seconds": {
            "centerline_and_signed_response_binary64": center_seconds,
            "timing_repetitions": TIMING_REPETITIONS,
            "timing_order": "alternating after one warm-up pair",
            "output_gram_binary64_trials": output_exact_trials,
            "output_gram_float32_trials": output_float32_trials,
            "green_gram_binary64_trials": green_exact_trials,
            "green_gram_float32_trials": green_float32_trials,
            "combined_q1_binary64_trials": combined_exact_trials,
            "combined_q1_float32_trials": combined_float32_trials,
            "output_gram_binary64_median": statistics.median(output_exact_trials),
            "output_gram_float32_median": statistics.median(output_float32_trials),
            "green_gram_binary64_median": statistics.median(green_exact_trials),
            "green_gram_float32_median": statistics.median(green_float32_trials),
        },
        "measured_operator_speedups": {
            "output_median_paired": statistics.median(output_paired_speedups),
            "green_median_paired": statistics.median(green_paired_speedups),
            "combined_q1_kernel": statistics.median(combined_paired_speedups),
            "combined_q1_kernel_minimum": min(combined_paired_speedups),
            "combined_q1_kernel_maximum": max(combined_paired_speedups),
            "combined_q1_kernel_trials": combined_paired_speedups,
        },
        "output_rows": output_audits,
        "claim_boundary": (
            "The theorem would make this path rigorous if the displayed residual "
            "uppers were independently verified. This audit establishes empirical "
            "headroom and a same-bracket mixed-precision path, not those verified bounds."
        ),
    }
    DESTINATION.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key != "output_rows"
            },
            indent=2,
        )
    )
    print(json.dumps({"output": str(DESTINATION), "sha256": sha256(DESTINATION)}, indent=2))


if __name__ == "__main__":
    main()
