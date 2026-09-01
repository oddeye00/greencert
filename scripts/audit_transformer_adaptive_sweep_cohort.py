#!/usr/bin/env python3
"""Outcome-blind cohort audit of one-, two-, and three-sweep references."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from statistics import NormalDist

import torch

from analytic_jet_release import analytic_jet_release, logit_margin_radius
from audit_transformer_direct_image_green_panel import tensor_sha256
from batched_green_operator import make_batched_transformer_green_products
from transformer_block_envelope import ball_valid_envelope
from transformer_certificate_protocol import Candidate
from transformer_fourth_jet_bound import objective_fourth_derivative_bound
from transformer_hvp_grokking import logits
from transformer_modal_forecast import optimizer_jvp, optimizer_map
from transformer_optimizer_probe import make_scaled_optimizer_jvp_vjp
from transformer_two_response import optimizer_center_quadratic_defect
from transformer_v3_certificate import load_candidate, output_path, safe_json


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PARENT = RESULTS / "transformer_v3_relinearized_prefix_panel_audit.json"
EXPECTED_PARENT_SHA256 = (
    "08E501B51FEAC3D96FFE02BE0B5D84E0E682C2E73CB906C083ED0FEF7E75E12B"
)
PROTOCOL = ROOT / "ADAPTIVE_SWEEP_COHORT_PROTOCOL.md"
OUTPUT = RESULTS / "transformer_adaptive_sweep_cohort_audit.json"
CACHE = RESULTS / "transformer_adaptive_sweep_cohort_cache"
SWEEP_GRID = (1, 2, 3)
PROBES = 4
PERSISTENCE = 25
FAMILY_FAILURE_UPPER = 1.0e-6
MASTER_NONCE = (
    "3631e2479441793c5bc31596f3697f5eab60bec95e53984c5aadbffaf0aa4460"
)
DEVELOPMENT = (366, 0.8, 1120)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def scaled(path: torch.Tensor, dimension: int, eta: float) -> torch.Tensor:
    return torch.cat((path[..., :dimension], eta * path[..., dimension:]), dim=-1)


def unscaled(path: torch.Tensor, dimension: int, eta: float) -> torch.Tensor:
    return torch.cat((path[..., :dimension], path[..., dimension:] / eta), dim=-1)


def identity(candidate: Candidate, horizon: int, sweeps: int) -> tuple[int, ...]:
    return (
        117,
        int(candidate.seed),
        int(candidate.gate_index),
        int(candidate.anchor),
        int(horizon),
        int(sweeps),
        int(PROBES),
    )


def probe_seed(operator_identity: tuple[int, ...]) -> int:
    payload = (
        "greencert/adaptive-sweep-cohort-v1\0"
        + MASTER_NONCE
        + "\0"
        + "|".join(str(value) for value in operator_identity)
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (
        2**63 - 1
    )


def first_persistent(values: list[bool]) -> int | None:
    for start in range(max(0, len(values) - PERSISTENCE + 1)):
        if all(values[start : start + PERSISTENCE]):
            return start
    return None


def raw_slacks(
    values: torch.Tensor, labels: torch.Tensor, required: int
) -> tuple[float, float]:
    true_values = values.gather(1, labels[:, None])
    margins = true_values - values
    rows = torch.arange(len(labels))
    margins[rows, labels] = torch.inf
    per_example = torch.min(margins, dim=1).values
    guarantee = torch.sort(per_example, descending=True).values[required - 1]
    incorrect_needed = len(labels) - required + 1
    exclusion = -torch.sort(per_example).values[incorrect_needed - 1]
    return float(guarantee), float(exclusion)


@torch.no_grad()
def all_reduced_paths(
    config,
    template,
    spec,
    train_pairs: torch.Tensor,
    train_labels: torch.Tensor,
    parameter: torch.Tensor,
    velocity: torch.Tensor,
    horizon: int,
) -> tuple[list[torch.Tensor], list[dict]]:
    """Retain all reduced paths from one causal three-sweep pipeline."""

    anchor = torch.cat((parameter, velocity))

    def map_step(state: torch.Tensor) -> torch.Tensor:
        return optimizer_map(
            state, train_pairs, train_labels, template, spec, config
        )

    def jvp(center: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        return optimizer_jvp(
            center,
            direction,
            train_pairs,
            train_labels,
            template,
            spec,
            config,
        )

    affine_defect = map_step(anchor) - anchor
    displacement = torch.zeros_like(anchor)
    raw_current = anchor.clone()
    corrections = [torch.zeros_like(anchor) for _ in SWEEP_GRID]
    paths = [[anchor.clone()] for _ in SWEEP_GRID]
    defect_maxima = [0.0 for _ in SWEEP_GRID]
    correction_maxima = [0.0 for _ in SWEEP_GRID]

    for step in range(horizon):
        displacement = jvp(anchor, displacement) + affine_defect
        raw_next = anchor + displacement
        reference_current = raw_current
        reference_next = raw_next
        next_corrections = []
        for index, _ in enumerate(SWEEP_GRID):
            correction_current = corrections[index]
            defect = map_step(reference_current) - reference_next
            next_correction = jvp(reference_current, correction_current) + defect
            if not bool(torch.isfinite(next_correction).all()):
                raise RuntimeError(
                    f"nonfinite sweep {index + 1} at step {step + 1}"
                )
            defect_maxima[index] = max(
                defect_maxima[index], float(torch.linalg.vector_norm(defect))
            )
            correction_maxima[index] = max(
                correction_maxima[index],
                float(torch.linalg.vector_norm(next_correction)),
            )
            center_current = reference_current + correction_current
            center_next = reference_next + next_correction
            paths[index].append(center_next.clone())
            next_corrections.append(next_correction)
            reference_current = center_current
            reference_next = center_next
        corrections = next_corrections
        raw_current = raw_next

    diagnostics = [
        {
            "sweeps": index + 1,
            "maximum_uncorrected_defect_norm": defect_maxima[index],
            "maximum_correction_norm": correction_maxima[index],
            "centerline_hvp_calls": (index + 1) * horizon,
        }
        for index in range(len(SWEEP_GRID))
    ]
    return [torch.stack(rows) for rows in paths], diagnostics


def evaluate_path(
    *,
    candidate: Candidate,
    sweeps: int,
    center: torch.Tensor,
    prefix: dict,
    certificate: dict,
    config,
    template,
    spec,
    train_pairs: torch.Tensor,
    train_labels: torch.Tensor,
    cert_pairs: torch.Tensor,
    cert_labels: torch.Tensor,
    stage_delta: float,
) -> dict:
    started = time.perf_counter()
    timings: dict[str, float] = {}
    horizon = int(prefix["horizon"])
    dimension = center.shape[1] // 2
    eta = float(config.learning_rate)
    scaled_center = scaled(center, dimension, eta)

    phase = time.perf_counter()
    mapped = [
        optimizer_map(
            center[step], train_pairs, train_labels, template, spec, config
        )
        for step in range(horizon)
    ]
    residual = torch.stack(
        [scaled(mapped[step], dimension, eta) - scaled_center[step + 1]
         for step in range(horizon)]
    )
    products = [
        make_scaled_optimizer_jvp_vjp(
            center[step, :dimension],
            train_pairs,
            train_labels,
            template,
            spec,
            config,
        )
        for step in range(horizon)
    ]
    correction_rows = []
    prior = torch.zeros_like(residual[0])
    recurrence_rows = []
    for step in range(horizon):
        current = products[step][0](prior) + residual[step]
        correction_rows.append(current)
        recurrence_rows.append(current - (products[step][0](prior) + residual[step]))
        prior = current
    correction_rows_tensor = torch.stack(correction_rows)
    correction = torch.cat(
        (torch.zeros_like(correction_rows_tensor[:1]), correction_rows_tensor), dim=0
    )
    corrected_scaled = scaled_center + correction
    corrected = unscaled(corrected_scaled, dimension, eta)
    timings["signed_correction"] = time.perf_counter() - phase

    phase = time.perf_counter()
    mapped_corrected = [
        optimizer_map(
            corrected[step], train_pairs, train_labels, template, spec, config
        )
        for step in range(horizon)
    ]
    mapped_corrected_scaled = torch.stack(
        [scaled(row, dimension, eta) for row in mapped_corrected]
    )
    literal_defect = mapped_corrected_scaled - corrected_scaled[1:]
    literal_defect_norm = float(torch.linalg.vector_norm(literal_defect))
    subtraction_scale = float(
        torch.linalg.vector_norm(
            mapped_corrected_scaled.abs() + corrected_scaled[1:].abs()
        )
    )
    cancellation_ratio = literal_defect_norm / max(
        torch.finfo(center.dtype).eps * subtraction_scale,
        torch.finfo(center.dtype).tiny,
    )
    timings["literal_defect_diagnostic"] = time.perf_counter() - phase

    correction_sequence_norm = float(
        torch.linalg.vector_norm(correction_rows_tensor)
    )
    correction_max = float(
        torch.linalg.vector_norm(correction_rows_tensor, dim=1).max()
    )
    domain = float(prefix["domain_radius"])
    row = {
        "candidate": candidate.__dict__,
        "sweeps": sweeps,
        "horizon": horizon,
        "centerline_sha256": tensor_sha256(scaled_center),
        "corrected_path_sha256": tensor_sha256(corrected_scaled),
        "correction_sequence_norm": correction_sequence_norm,
        "correction_max_state_norm": correction_max,
        "domain_radius": domain,
        "domain_passed": correction_max <= domain,
        "literal_corrected_defect_norm_float64": literal_defect_norm,
        "literal_defect_to_roundoff_scale_ratio": cancellation_ratio,
        "probe_identity": list(identity(candidate, horizon, sweeps)),
        "probe_seed": probe_seed(identity(candidate, horizon, sweeps)),
        "stage_delta": stage_delta,
        "green_queried": False,
        "closure_passed": False,
        "issued": False,
        "bracket": None,
        "sealed_four_sweep_bracket": prefix["bracket"],
        "retains_sealed_bracket": False,
        "logic_slack": None,
        "sequential_vector_hvp_sweeps_if_deployed": sweeps + 1,
        "batched_operator_passes_if_deployed": sweeps + 1,
        "timings_seconds": timings,
        "outcome_files_read": 0,
    }
    if correction_max > domain:
        row["abstention_reason"] = "signed correction leaves derivative domain"
        timings["evaluation_total"] = time.perf_counter() - started
        return row

    phase = time.perf_counter()
    recurrence_norm = float(
        torch.linalg.vector_norm(torch.stack(recurrence_rows))
    )
    quadratic_rows = [torch.zeros_like(correction_rows_tensor[0])]
    taylor_terms = []
    fourth_bounds = []
    direction_norms = []
    for step in range(1, horizon):
        direction_norm = float(
            torch.linalg.vector_norm(correction[step, :dimension])
        )
        direction_norms.append(direction_norm)
        quadratic_rows.append(
            optimizer_center_quadratic_defect(
                center[step, :dimension],
                correction[step],
                train_pairs,
                train_labels,
                template,
                spec,
                config,
            )
        )
        fourth = objective_fourth_derivative_bound(
            center[step, :dimension],
            template,
            spec,
            config,
            radius=direction_norm,
        )
        fourth_bounds.append(fourth)
        taylor_terms.append(
            math.sqrt(2.0) * eta * fourth * direction_norm**3 / 6.0
        )
    quadratic = torch.stack(quadratic_rows)
    quadratic_norm = float(torch.linalg.vector_norm(quadratic))
    taylor_upper = math.sqrt(sum(value * value for value in taylor_terms))
    injection_upper = recurrence_norm + quadratic_norm + taylor_upper
    timings["cancellation_safe_forcing"] = time.perf_counter() - phase

    phase = time.perf_counter()
    blocks = [
        ball_valid_envelope(
            center[step, :dimension],
            spec,
            config,
            epsilon=domain,
            exact_values=True,
            sphere=True,
        )
        for step in range(1, horizon + 1)
    ]
    neural_domain = all(bool(block["fixed_point_consistent"]) for block in blocks)
    timings["analytic_neural_jets"] = time.perf_counter() - phase
    row.update(
        {
            "response_recurrence_residual_norm": recurrence_norm,
            "quadratic_surrogate_injection_norm": quadratic_norm,
            "quadratic_surrogate_sha256": tensor_sha256(quadratic),
            "directional_quadratic_taylor_error_upper": taylor_upper,
            "cancellation_safe_injection_upper": injection_upper,
            "maximum_local_objective_fourth_derivative_upper": max(
                fourth_bounds, default=0.0
            ),
            "maximum_parameter_direction_norm": max(direction_norms, default=0.0),
            "neural_jet_domain_passed": neural_domain,
        }
    )
    if not neural_domain:
        row["abstention_reason"] = "analytic neural-jet domain failed"
        timings["evaluation_total"] = time.perf_counter() - started
        return row

    phase = time.perf_counter()
    batch_apply, _ = make_batched_transformer_green_products(
        corrected[:horizon, :dimension],
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )
    generator = torch.Generator(device=center.device).manual_seed(row["probe_seed"])
    probes = torch.stack(
        [
            torch.randn(
                horizon * 2 * dimension,
                generator=generator,
                dtype=center.dtype,
                device=center.device,
            )
            for _ in range(PROBES)
        ]
    )
    images = batch_apply(probes)
    initial_norms = [float(value) for value in torch.linalg.vector_norm(probes, dim=1)]
    image_norms = [float(value) for value in torch.linalg.vector_norm(images, dim=1)]
    calibration = NormalDist().inv_cdf(
        0.5 * (1.0 + stage_delta ** (1.0 / PROBES))
    )
    kappa = max(image_norms) / calibration
    timings["direct_image_green"] = time.perf_counter() - phase

    phase = time.perf_counter()
    forcing_response = kappa * injection_upper
    release = analytic_jet_release(
        kappa=kappa,
        corrected_defect_response_bound=forcing_response,
        correction_max_state_norm=correction_max,
        domain_radius=domain,
        learning_rate=eta,
        transition_jets=[
            (float(block["first"]), float(block["second"]), float(block["third"]))
            for block in blocks[:-1]
        ],
        output_first_bounds=[float(block["first"]) for block in blocks],
    )
    bracket = None
    logic_slack = None
    maximum_margin = None
    if release.closure.closure_passed:
        state_radius = float(release.state_radius_about_original_reference)
        required = int(certificate["required_correct"])
        raw = [
            raw_slacks(
                logits(center[step, :dimension], cert_pairs, template, spec),
                cert_labels,
                required,
            )
            for step in range(horizon + 1)
        ]
        margins = [0.0] + [
            logit_margin_radius(first=float(block["first"]), state_radius=state_radius)
            for block in blocks
        ]
        guarantee = [float(pair[0]) - margin for pair, margin in zip(raw, margins)]
        exclusion = [float(pair[1]) - margin for pair, margin in zip(raw, margins)]
        lower = first_persistent([value <= 0.0 for value in exclusion])
        upper = first_persistent([value > 0.0 for value in guarantee])
        if lower is not None and upper is not None and lower <= upper:
            bracket = [lower, upper]
            prior = [
                max(exclusion[start : start + PERSISTENCE])
                for start in range(lower)
            ]
            lower_slack = math.inf if not prior else min(prior)
            upper_slack = min(guarantee[upper : upper + PERSISTENCE])
            logic_slack = min(lower_slack, upper_slack)
        maximum_margin = max(margins)
    issued = bracket is not None and logic_slack is not None and logic_slack > 0.0
    retained = issued and bracket == prefix["bracket"]
    timings["closure_and_event"] = time.perf_counter() - phase
    timings["evaluation_total"] = time.perf_counter() - started
    row.update(
        {
            "green_queried": True,
            "initial_probe_norms": initial_norms,
            "green_image_norms": image_norms,
            "green_operator_norm_upper_bound": kappa,
            "cancellation_safe_response_upper": forcing_response,
            "analytic_release": release.as_dict(),
            "closure_passed": release.closure.closure_passed,
            "maximum_margin_radius": maximum_margin,
            "issued": issued,
            "bracket": bracket,
            "retains_sealed_bracket": retained,
            "logic_slack": logic_slack,
            "sequential_vector_hvp_sweeps_if_deployed": sweeps + 1 + PROBES,
            "batched_operator_passes_if_deployed": sweeps + 2,
        }
    )
    if not release.closure.closure_passed:
        row["abstention_reason"] = "nonlinear closure failed"
    elif not issued:
        row["abstention_reason"] = "persistent event margins did not issue"
    elif not retained:
        row["abstention_reason"] = "issued bracket differs from sealed comparator"
    return row


def cache_path(candidate: Candidate) -> Path:
    return CACHE / (
        f"seed_{candidate.seed}_gate_{candidate.gate_index}_"
        f"anchor_{candidate.anchor}.json"
    )


def run_case(task: dict) -> dict:
    candidate = Candidate(**task["candidate"])
    destination = cache_path(candidate)
    if destination.is_file():
        cached = safe_json(destination)
        if (
            cached.get("protocol_sha256") == task["protocol_sha256"]
            and cached.get("script_sha256") == task["script_sha256"]
            and cached.get("prefix_certificate_sha256")
            == task["prefix"]["certificate_sha256"]
        ):
            return cached

    prefix = task["prefix"]
    certificate_path = output_path(candidate)
    if sha256(certificate_path) != prefix["certificate_sha256"]:
        raise RuntimeError(f"certificate hash mismatch for {candidate}")
    certificate = safe_json(certificate_path)
    config, template, spec, data, parameter, velocity = load_candidate(candidate)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = data
    started = time.perf_counter()
    paths, pipeline_diagnostics = all_reduced_paths(
        config,
        template,
        spec,
        train_pairs,
        train_labels,
        parameter,
        velocity,
        int(prefix["horizon"]),
    )
    pipeline_seconds = time.perf_counter() - started
    rows = []
    for sweeps, center in zip(SWEEP_GRID, paths):
        rows.append(
            evaluate_path(
                candidate=candidate,
                sweeps=sweeps,
                center=center,
                prefix=prefix,
                certificate=certificate,
                config=config,
                template=template,
                spec=spec,
                train_pairs=train_pairs,
                train_labels=train_labels,
                cert_pairs=cert_pairs,
                cert_labels=cert_labels,
                stage_delta=float(task["stage_delta"]),
            )
        )
    result = {
        "candidate": candidate.__dict__,
        "development_row": (
            candidate.seed, candidate.threshold, candidate.anchor
        ) == DEVELOPMENT,
        "horizon": int(prefix["horizon"]),
        "prefix_certificate_sha256": prefix["certificate_sha256"],
        "sealed_four_sweep_bracket": prefix["bracket"],
        "pipeline_diagnostics": pipeline_diagnostics,
        "three_sweep_pipeline_seconds": pipeline_seconds,
        "rows": rows,
        "protocol_sha256": task["protocol_sha256"],
        "script_sha256": task["script_sha256"],
        "outcome_files_read": 0,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return result


def summarize(cases: list[dict], *, exclude_development: bool) -> dict:
    selected = [case for case in cases if not exclude_development or not case["development_row"]]
    by_sweep = {}
    for sweeps in SWEEP_GRID:
        rows = [
            next(row for row in case["rows"] if row["sweeps"] == sweeps)
            for case in selected
        ]
        by_sweep[str(sweeps)] = {
            "cases": len(rows),
            "domain_passed": sum(bool(row["domain_passed"]) for row in rows),
            "green_queried": sum(bool(row["green_queried"]) for row in rows),
            "closure_passed": sum(bool(row["closure_passed"]) for row in rows),
            "issued": sum(bool(row["issued"]) for row in rows),
            "retained_sealed_bracket": sum(
                bool(row["retains_sealed_bracket"]) for row in rows
            ),
            "median_logic_slack_when_issued": (
                statistics.median(
                    row["logic_slack"] for row in rows if row["issued"]
                )
                if any(row["issued"] for row in rows)
                else None
            ),
        }
    practical_sweeps = []
    for case in selected:
        row3 = next(row for row in case["rows"] if row["sweeps"] == 3)
        practical_sweeps.append(3 if row3["retains_sealed_bracket"] else 4)
    return {
        "cases": len(selected),
        "by_sweep": by_sweep,
        "three_then_four_effective_sweeps": practical_sweeps,
        "three_sweep_retentions": practical_sweeps.count(3),
        "four_sweep_fallbacks": practical_sweeps.count(4),
        "mean_centerline_sweeps": statistics.mean(practical_sweeps),
        "total_centerline_sweeps_saved_vs_four": sum(4 - value for value in practical_sweeps),
        "fractional_centerline_sweep_reduction": (
            sum(4 - value for value in practical_sweeps) / (4 * len(practical_sweeps))
            if practical_sweeps
            else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if sha256(PARENT) != EXPECTED_PARENT_SHA256:
        raise RuntimeError("fixed prefix-panel parent hash changed")
    parent = safe_json(PARENT)
    prefixes = parent["rows"]
    if len(prefixes) != 15:
        raise RuntimeError("protocol requires exactly 15 parent rows")
    protocol_sha = sha256(PROTOCOL)
    script_sha = sha256(Path(__file__))
    stage_delta = FAMILY_FAILURE_UPPER / (len(prefixes) * len(SWEEP_GRID))
    identities = [
        identity(Candidate(**prefix["candidate"]), int(prefix["horizon"]), sweeps)
        for prefix in prefixes
        for sweeps in SWEEP_GRID
    ]
    seeds = [probe_seed(value) for value in identities]
    if len(set(identities)) != len(identities) or len(set(seeds)) != len(seeds):
        raise RuntimeError("operator identities or probe streams collide")
    tasks = [
        {
            "candidate": prefix["candidate"],
            "prefix": prefix,
            "stage_delta": stage_delta,
            "protocol_sha256": protocol_sha,
            "script_sha256": script_sha,
        }
        for prefix in prefixes
    ]
    cases = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_case, task): task for task in tasks}
        for future in as_completed(futures):
            case = future.result()
            cases.append(case)
            retained = [
                row["sweeps"] for row in case["rows"] if row["retains_sealed_bracket"]
            ]
            print(
                json.dumps(
                    {
                        "completed": case["candidate"],
                        "horizon": case["horizon"],
                        "retained_sweeps": retained,
                    }
                ),
                flush=True,
            )
    cases.sort(
        key=lambda case: (
            case["candidate"]["seed"],
            case["candidate"]["threshold"],
            case["candidate"]["anchor"],
        )
    )
    all_summary = summarize(cases, exclude_development=False)
    holdout_summary = summarize(cases, exclude_development=True)
    result = {
        "status": "outcome-blind adaptive-sweep cohort audit complete",
        "evidence_boundary": (
            "Post-release method-development audit. The development row is "
            "reported separately; no revealed trajectory or event time was read."
        ),
        "parent_sha256": sha256(PARENT),
        "protocol_sha256": protocol_sha,
        "script_sha256": script_sha,
        "master_nonce": MASTER_NONCE,
        "declared_operator_count": len(identities),
        "unique_probe_streams": len(set(seeds)),
        "probes_per_operator": PROBES,
        "family_failure_upper": FAMILY_FAILURE_UPPER,
        "stage_delta": stage_delta,
        "all_cases": all_summary,
        "nondevelopment_cases": holdout_summary,
        "universal_three_sweep_replacement": (
            all_summary["three_sweep_retentions"] == all_summary["cases"]
        ),
        "cases": cases,
        "outcome_files_read": 0,
    }
    OUTPUT.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
