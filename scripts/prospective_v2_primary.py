#!/usr/bin/env python3
"""Prospective primary confirmation for the frozen recentered-v2 method.

Trigger selection and first-passage observation share one chronological replay.
The trigger state machine never reads an eventual-crossing table.  Certificate
construction is blind to the replayed future and writes paired v1/v2 caches
that contain no observed event time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta as beta_distribution

from generate_smooth_mlp_seed import artifact_paths, frozen_config
from modular_accuracy_certificate import event_bracket
from replay_smooth_mlp_thresholds import THRESHOLDS, required_counts
from smooth_mlp_certificate import (
    exact_objective_hessian,
    modal_path,
    objective_hessian_lipschitz,
)
from smooth_mlp_modular_grokking import analytic_gradient, initialize, logits, make_split
from variational_mlp_certificate import audit_actual_path, certified_accuracy_counts
from variational_shadowing import StepLinearization, residual_centered_tube


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "PROSPECTIVE_V2_PRIMARY_PROTOCOL.md"
MANIFEST = ROOT / "PROSPECTIVE_V2_CODE_MANIFEST.json"
SCAN_OUT = ROOT / "results" / "prospective_v2_primary_triggers_blind.json"
OUTCOMES_OUT = ROOT / "results" / "prospective_v2_primary_outcomes.json"
OUT = ROOT / "results" / "prospective_v2_primary.json"
CACHE_DIR = ROOT / "results" / "prospective_v2_primary_cache"
SEEDS = tuple(range(17, 25))
HORIZON = 250
NUMERIC_CAP = 1e4
METHOD_VERSION = "recentered-variational-v2-frozen-2026-08-21"
PIPELINE_VERSION = "prospective-primary-v1-2026-08-21"


def protocol_sha256() -> str:
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def manifest_sha256() -> str:
    return hashlib.sha256(MANIFEST.read_bytes()).hexdigest()


def verify_manifest() -> dict:
    """Fail closed if any frozen protocol or implementation file changed."""
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if payload.get("protocol_sha256") != protocol_sha256():
        raise RuntimeError("prospective protocol does not match the frozen manifest")
    for record in payload.get("files", []):
        relative = Path(record["path"])
        target = (ROOT / relative).resolve()
        try:
            target.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise RuntimeError(f"manifest path escapes research root: {relative}") from exc
        observed = hashlib.sha256(target.read_bytes()).hexdigest()
        if observed != record["sha256"]:
            raise RuntimeError(f"frozen file changed: {relative}")
    return payload


def threshold_keys() -> list[str]:
    return [f"{threshold:.2f}" for threshold in THRESHOLDS]


def prospective_state_machine(
    steps_and_counts: list[tuple[int, int]],
    required: dict[float, int],
    raw_count_provider,
    *,
    checkpoint_every: int = 250,
    horizon: int = HORIZON,
    last_anchor_step: int | None = None,
) -> tuple[dict[str, int | None], dict[str, int | None], dict[str, int | None], list[int]]:
    """Chronologically select triggers and record first passages.

    This pure state machine is intentionally independent of training artifacts.
    A gate remains eligible until its first observed crossing, including when it
    never crosses by the stopping time.
    """
    crossings = {key: None for key in threshold_keys()}
    triggers = {key: None for key in threshold_keys()}
    raw_offsets = {key: None for key in threshold_keys()}
    raw_anchors: list[int] = []
    for step, count in steps_and_counts:
        for threshold in THRESHOLDS:
            key = f"{threshold:.2f}"
            if crossings[key] is None and count >= required[threshold]:
                crossings[key] = int(step)

        if step % checkpoint_every != 0 or (
            last_anchor_step is not None and step > last_anchor_step
        ):
            continue
        eligible = [
            (threshold, f"{threshold:.2f}")
            for threshold in THRESHOLDS
            if crossings[f"{threshold:.2f}"] is None
            and triggers[f"{threshold:.2f}"] is None
            and count == required[threshold] - 1
        ]
        if not eligible:
            continue
        raw_counts = np.asarray(raw_count_provider(int(step)), dtype=np.int64)
        if raw_counts.shape != (horizon + 1,):
            raise ValueError("raw count provider returned the wrong horizon")
        raw_anchors.append(int(step))
        for threshold, key in eligible:
            candidates = np.flatnonzero(raw_counts[1:] >= required[threshold]) + 1
            if len(candidates) == 0:
                continue
            triggers[key] = int(step)
            raw_offsets[key] = int(candidates[0])
    return triggers, raw_offsets, crossings, raw_anchors


@torch.no_grad()
def modal_reference_counts_from_parameter(
    parameter: torch.Tensor,
    train_pairs: torch.Tensor,
    train_labels: torch.Tensor,
    test_pairs: torch.Tensor,
    test_labels: torch.Tensor,
    config,
    horizon: int = HORIZON,
) -> np.ndarray:
    hessian, _ = exact_objective_hessian(parameter, train_pairs, train_labels, config)
    gradient = analytic_gradient(parameter, train_pairs, train_labels, config)
    eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
    modal, _ = modal_path(
        eigenvalues,
        eigenvectors,
        gradient,
        config.learning_rate,
        horizon,
    )
    reference = parameter[None, :] + modal @ eigenvectors.T
    counts = np.zeros(len(reference), dtype=np.int64)
    for offset, center in enumerate(reference):
        counts[offset] = int(
            torch.sum(torch.argmax(logits(center, test_pairs, config), dim=1) == test_labels)
        )
    return counts


@torch.no_grad()
def prospective_replay(seed: int) -> dict:
    """Replay one seed without consulting a crossing artifact."""
    torch.set_num_threads(4)
    config = frozen_config(seed)
    result_path, checkpoint_path = artifact_paths(seed)
    if not result_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(f"missing frozen training artifacts for seed {seed}")
    training = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoints = np.load(checkpoint_path)
    train_pairs, train_labels, test_pairs, test_labels = make_split(config)
    required = required_counts(len(test_pairs))
    parameter = initialize(config)
    crossings = {key: None for key in threshold_keys()}
    triggers = {key: None for key in threshold_keys()}
    raw_offsets = {key: None for key in threshold_keys()}
    raw_anchors: list[int] = []
    maximum_checkpoint_error = 0.0
    maximum_correct = 0

    for step in range(config.steps + 1):
        if step % config.checkpoint_every == 0:
            saved = torch.from_numpy(checkpoints[f"step_{step}"])
            maximum_checkpoint_error = max(
                maximum_checkpoint_error,
                float(torch.linalg.vector_norm(parameter - saved)),
            )

        count = int(
            torch.sum(torch.argmax(logits(parameter, test_pairs, config), dim=1) == test_labels)
        )
        maximum_correct = max(maximum_correct, count)
        for threshold in THRESHOLDS:
            key = f"{threshold:.2f}"
            if crossings[key] is None and count >= required[threshold]:
                crossings[key] = int(step)

        if (
            step % config.checkpoint_every == 0
            and step <= config.steps - HORIZON
        ):
            eligible = [
                (threshold, f"{threshold:.2f}")
                for threshold in THRESHOLDS
                if crossings[f"{threshold:.2f}"] is None
                and triggers[f"{threshold:.2f}"] is None
                and count == required[threshold] - 1
            ]
            if eligible:
                print(f"  seed {seed}: raw modal anchor {step}", flush=True)
                raw_counts = modal_reference_counts_from_parameter(
                    parameter,
                    train_pairs,
                    train_labels,
                    test_pairs,
                    test_labels,
                    config,
                )
                raw_anchors.append(int(step))
                for threshold, key in eligible:
                    candidates = np.flatnonzero(raw_counts[1:] >= required[threshold]) + 1
                    if len(candidates):
                        triggers[key] = int(step)
                        raw_offsets[key] = int(candidates[0])

        if step == config.steps:
            break
        parameter.add_(
            analytic_gradient(parameter, train_pairs, train_labels, config),
            alpha=-config.learning_rate,
        )

    return {
        "seed": seed,
        "training_summary": training["summary"],
        "required_correct": {f"{key:.2f}": value for key, value in required.items()},
        "triggers": triggers,
        "raw_offsets": raw_offsets,
        "crossings": crossings,
        "raw_anchors_scanned": raw_anchors,
        "maximum_correct": maximum_correct,
        "maximum_checkpoint_parameter_error": maximum_checkpoint_error,
        "all_checkpoint_replays_exact": maximum_checkpoint_error == 0.0,
    }


def scan() -> dict:
    verify_manifest()
    seed_rows = []
    for seed in SEEDS:
        print(f"prospective replay seed {seed}", flush=True)
        seed_rows.append(prospective_replay(seed))
    blind_payload = {
        "protocol": PROTOCOL.name,
        "protocol_sha256": protocol_sha256(),
        "manifest_sha256": manifest_sha256(),
        "pipeline_version": PIPELINE_VERSION,
        "seeds": list(SEEDS),
        "horizon": HORIZON,
        "seed_rows": [
            {
                "seed": row["seed"],
                "required_correct": row["required_correct"],
                "triggers": row["triggers"],
                "raw_offsets": row["raw_offsets"],
                "raw_anchors_scanned": row["raw_anchors_scanned"],
                "maximum_checkpoint_parameter_error": row[
                    "maximum_checkpoint_parameter_error"
                ],
                "all_checkpoint_replays_exact": row["all_checkpoint_replays_exact"],
            }
            for row in seed_rows
        ],
    }
    outcomes_payload = {
        "protocol": PROTOCOL.name,
        "protocol_sha256": protocol_sha256(),
        "manifest_sha256": manifest_sha256(),
        "pipeline_version": PIPELINE_VERSION,
        "seeds": list(SEEDS),
        "seed_rows": [
            {
                "seed": row["seed"],
                "training_summary": row["training_summary"],
                "crossings": row["crossings"],
                "maximum_correct": row["maximum_correct"],
            }
            for row in seed_rows
        ],
    }
    SCAN_OUT.parent.mkdir(parents=True, exist_ok=True)
    SCAN_OUT.write_text(json.dumps(blind_payload, indent=2) + "\n", encoding="utf-8")
    OUTCOMES_OUT.write_text(
        json.dumps(outcomes_payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "blind_triggers": str(SCAN_OUT),
                "sealed_outcomes": str(OUTCOMES_OUT),
                "seeds": list(SEEDS),
            },
            indent=2,
        )
    )
    return blind_payload


def _linearizer(parameter, train_pairs, train_labels, config):
    def linearize(center: torch.Tensor) -> StepLinearization:
        gradient = analytic_gradient(center, train_pairs, train_labels, config)
        hessian, _ = exact_objective_hessian(center, train_pairs, train_labels, config)
        hessian_eigenvalues = torch.linalg.eigvalsh(hessian)
        jacobian = torch.eye(parameter.numel(), dtype=parameter.dtype) - config.learning_rate * hessian
        beta = float(torch.max(torch.abs(1.0 - config.learning_rate * hessian_eigenvalues)))
        return StepLinearization(
            mapped_center=center - config.learning_rate * gradient,
            jacobian=jacobian,
            jacobian_operator_norm=beta,
            jacobian_lipschitz=lambda radius, center=center: (
                config.learning_rate * objective_hessian_lipschitz(center, config, radius)
            ),
        )

    return linearize


@torch.no_grad()
def build_paired_tubes(seed: int, anchor: int, horizon: int = HORIZON):
    """Build paired v1/v2 tubes while sharing modal-center Hessians."""
    config = frozen_config(seed)
    _, checkpoint_path = artifact_paths(seed)
    checkpoints = np.load(checkpoint_path)
    parameter = torch.from_numpy(checkpoints[f"step_{anchor}"]).clone()
    train_pairs, train_labels, test_pairs, test_labels = make_split(config)
    anchor_hessian, _ = exact_objective_hessian(parameter, train_pairs, train_labels, config)
    anchor_gradient = analytic_gradient(parameter, train_pairs, train_labels, config)
    eigenvalues, eigenvectors = torch.linalg.eigh(anchor_hessian)
    modal, _ = modal_path(
        eigenvalues,
        eigenvectors,
        anchor_gradient,
        config.learning_rate,
        horizon,
    )
    modal_reference = parameter[None, :] + modal @ eigenvectors.T
    linearize = _linearizer(parameter, train_pairs, train_labels, config)

    correction = torch.zeros_like(modal_reference)
    correction_reached = 0
    v1_radius = np.zeros(horizon + 1, dtype=np.float64)
    v1_reached = 0
    v1_active = True
    for step in range(horizon):
        geometry = linearize(modal_reference[step])
        defect = geometry.mapped_center - modal_reference[step + 1]
        if v1_active:
            omega = float(v1_radius[step])
            radius = float(torch.linalg.vector_norm(correction[step])) + omega
            nonlinear = 0.5 * float(geometry.jacobian_lipschitz(radius)) * radius**2
            next_omega = geometry.jacobian_operator_norm * omega + nonlinear
            next_total = float(
                torch.linalg.vector_norm(geometry.jacobian @ correction[step] + defect)
            ) + next_omega
            if (
                np.isfinite(next_omega)
                and np.isfinite(next_total)
                and next_omega <= NUMERIC_CAP
                and next_total <= NUMERIC_CAP
            ):
                v1_radius[step + 1] = next_omega
                v1_reached = step + 1
            else:
                v1_active = False
        next_correction = geometry.jacobian @ correction[step] + defect
        next_norm = float(torch.linalg.vector_norm(next_correction))
        if not np.isfinite(next_norm) or next_norm > NUMERIC_CAP:
            break
        correction[step + 1] = next_correction
        correction_reached = step + 1

    corrected_reference = modal_reference[: correction_reached + 1] + correction[: correction_reached + 1]
    v2_tube = residual_centered_tube(
        corrected_reference,
        linearize,
        numeric_cap=NUMERIC_CAP,
    )
    v1_keep = min(v1_reached, correction_reached) + 1
    return {
        "parameter": parameter,
        "corrected_reference": corrected_reference,
        "v1_reference": corrected_reference[:v1_keep],
        "v1_radius": v1_radius[:v1_keep],
        "v1_reached": v1_keep - 1,
        "v2_reference": v2_tube.reference,
        "v2_radius": v2_tube.error_radius,
        "v2_reached": v2_tube.reached_horizon,
        "correction_reached": correction_reached,
        "data": (train_pairs, train_labels, test_pairs, test_labels),
        "config": config,
    }


def cache_paths(seed: int, anchor: int) -> tuple[Path, Path]:
    stem = CACHE_DIR / f"seed_{seed}_anchor_{anchor}_h{HORIZON}"
    return stem.with_suffix(".json"), stem.with_suffix(".npz")


@torch.no_grad()
def blind_certificate(seed: int, anchor: int, *, use_cache: bool = True) -> dict:
    verify_manifest()
    json_path, array_path = cache_paths(seed, anchor)
    if use_cache and json_path.exists() and array_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if (
            payload.get("method_version") == METHOD_VERSION
            and payload.get("protocol_sha256") == protocol_sha256()
            and payload.get("manifest_sha256") == manifest_sha256()
        ):
            return payload

    torch.set_num_threads(4)
    built = build_paired_tubes(seed, anchor)
    _, _, test_pairs, test_labels = built["data"]
    config = built["config"]
    required = required_counts(len(test_pairs))
    v1_g, v1_p, v1_center = certified_accuracy_counts(
        built["v1_reference"], built["v1_radius"], test_pairs, test_labels, config
    )
    v2_g, v2_p, v2_center = certified_accuracy_counts(
        built["v2_reference"], built["v2_radius"], test_pairs, test_labels, config
    )
    events = {}
    for threshold in THRESHOLDS:
        key = f"{threshold:.2f}"
        v1_indices = np.flatnonzero(v1_center >= required[threshold])
        v2_indices = np.flatnonzero(v2_center >= required[threshold])
        events[key] = {
            "v1_bracket": event_bracket(v1_g, v1_p, required[threshold]),
            "v2_bracket": event_bracket(v2_g, v2_p, required[threshold]),
            "v1_reference_crossing": None if not len(v1_indices) else int(v1_indices[0]),
            "v2_reference_crossing": None if not len(v2_indices) else int(v2_indices[0]),
        }
    payload = {
        "status": "blind certificate; no future crossing data read",
        "protocol_sha256": protocol_sha256(),
        "manifest_sha256": manifest_sha256(),
        "pipeline_version": PIPELINE_VERSION,
        "method_version": METHOD_VERSION,
        "seed": seed,
        "anchor": anchor,
        "requested_horizon": HORIZON,
        "correction_reached_horizon": int(built["correction_reached"]),
        "v1_reached_horizon": int(built["v1_reached"]),
        "v2_reached_horizon": int(built["v2_reached"]),
        "events": events,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        array_path,
        anchor_parameter=built["parameter"].numpy(),
        corrected_reference=built["corrected_reference"].numpy(),
        v1_radius=built["v1_radius"],
        v2_radius=built["v2_radius"],
    )
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "seed": seed,
                "anchor": anchor,
                "v1_horizon": payload["v1_reached_horizon"],
                "v2_horizon": payload["v2_reached_horizon"],
            },
            indent=2,
        ),
        flush=True,
    )
    return payload


def certify_all() -> None:
    verify_manifest()
    scan_payload = json.loads(SCAN_OUT.read_text(encoding="utf-8"))
    if scan_payload["protocol_sha256"] != protocol_sha256():
        raise RuntimeError("protocol changed after prospective scan")
    if scan_payload["manifest_sha256"] != manifest_sha256():
        raise RuntimeError("implementation changed after prospective scan")
    anchors = sorted(
        {
            (int(row["seed"]), int(anchor))
            for row in scan_payload["seed_rows"]
            for anchor in row["triggers"].values()
            if anchor is not None
        }
    )
    for index, (seed, anchor) in enumerate(anchors, start=1):
        print(f"blind certificate {index}/{len(anchors)}: seed {seed}, anchor {anchor}", flush=True)
        blind_certificate(seed, anchor)


def clopper_pearson(successes: int, trials: int, confidence: float = 0.95):
    if trials == 0:
        return None
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(
        beta_distribution.ppf(alpha / 2.0, successes, trials - successes + 1)
    )
    upper = 1.0 if successes == trials else float(
        beta_distribution.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes)
    )
    return [lower, upper]


def classify_failure(row: dict) -> str:
    if row["anchor"] is None:
        return "no_trigger"
    if row["v2_certificate_issued"]:
        return "issued"
    if row["actual_crossing"] is None:
        return "event_unobserved"
    if row["actual_lead"] > HORIZON:
        return "event_beyond_window"
    if row["v2_reached_horizon"] < row["actual_lead"]:
        return "state_horizon"
    return "output_margin"


@torch.no_grad()
def aggregate() -> dict:
    verify_manifest()
    scan_payload = json.loads(SCAN_OUT.read_text(encoding="utf-8"))
    outcome_payload = json.loads(OUTCOMES_OUT.read_text(encoding="utf-8"))
    if scan_payload["protocol_sha256"] != protocol_sha256():
        raise RuntimeError("protocol changed after prospective scan")
    if outcome_payload["protocol_sha256"] != protocol_sha256():
        raise RuntimeError("protocol changed after outcome replay")
    if scan_payload["manifest_sha256"] != manifest_sha256():
        raise RuntimeError("implementation changed after prospective scan")
    if outcome_payload["manifest_sha256"] != manifest_sha256():
        raise RuntimeError("implementation changed after outcome replay")
    outcomes_by_seed = {
        int(row["seed"]): row for row in outcome_payload["seed_rows"]
    }
    seed_training_rows = []
    for seed in SEEDS:
        outcome = outcomes_by_seed[seed]
        training_summary = outcome["training_summary"]
        fit_step = training_summary["fit_step"]
        crossing_95 = outcome["crossings"]["0.95"]
        delay_ratio = (
            None
            if fit_step is None or crossing_95 is None
            else float(crossing_95 / max(int(fit_step), 1))
        )
        seed_training_rows.append(
            {
                "seed": seed,
                "fit_step_first_logged": fit_step,
                "exact_95_crossing": crossing_95,
                "exact_95_to_logged_fit_ratio": delay_ratio,
                "natural_grokking_ratio_gt_10": (
                    delay_ratio is not None and delay_ratio > 10.0
                ),
            }
        )
    rows = []
    anchor_audits = {}
    for seed_row in scan_payload["seed_rows"]:
        seed = int(seed_row["seed"])
        outcome_row = outcomes_by_seed[seed]
        for threshold in THRESHOLDS:
            key = f"{threshold:.2f}"
            anchor = seed_row["triggers"][key]
            crossing = outcome_row["crossings"][key]
            row = {
                "seed": seed,
                "threshold": threshold,
                "anchor": anchor,
                "raw_trigger_prediction": seed_row["raw_offsets"][key],
                "actual_crossing": crossing,
            }
            if anchor is None:
                row.update(
                    {
                        "actual_lead": None,
                        "v1_reached_horizon": None,
                        "v2_reached_horizon": None,
                        "v1_certificate_issued": False,
                        "v2_certificate_issued": False,
                        "v1_bracket": None,
                        "v2_bracket": None,
                        "v1_covered": None,
                        "v2_covered": None,
                        "v2_reference_prediction": None,
                    }
                )
                rows.append(row)
                continue
            anchor = int(anchor)
            certificate = blind_certificate(seed, anchor)
            event = certificate["events"][key]
            actual_lead = None if crossing is None else int(crossing) - anchor
            v1_bracket = event["v1_bracket"]
            v2_bracket = event["v2_bracket"]
            row.update(
                {
                    "actual_lead": actual_lead,
                    "v1_reached_horizon": int(certificate["v1_reached_horizon"]),
                    "v2_reached_horizon": int(certificate["v2_reached_horizon"]),
                    "v1_certificate_issued": v1_bracket is not None,
                    "v2_certificate_issued": v2_bracket is not None,
                    "v1_bracket": v1_bracket,
                    "v2_bracket": v2_bracket,
                    "v1_covered": (
                        None
                        if v1_bracket is None or actual_lead is None
                        else bool(v1_bracket[0] <= actual_lead <= v1_bracket[1])
                    ),
                    "v2_covered": (
                        None
                        if v2_bracket is None or actual_lead is None
                        else bool(v2_bracket[0] <= actual_lead <= v2_bracket[1])
                    ),
                    "v2_reference_prediction": event["v2_reference_crossing"],
                }
            )
            rows.append(row)

            audit_key = (seed, anchor)
            if audit_key not in anchor_audits:
                _, array_path = cache_paths(seed, anchor)
                arrays = np.load(array_path)
                config = frozen_config(seed)
                train_pairs, train_labels, _, _ = make_split(config)
                reference = torch.from_numpy(arrays["corrected_reference"][: len(arrays["v2_radius"])])
                anchor_parameter = torch.from_numpy(arrays["anchor_parameter"])
                anchor_audits[audit_key] = audit_actual_path(
                    anchor_parameter,
                    reference,
                    arrays["v2_radius"],
                    train_pairs,
                    train_labels,
                    config,
                )

    triggered = [row for row in rows if row["anchor"] is not None]
    v1_issued = [row for row in triggered if row["v1_certificate_issued"]]
    issued = [row for row in triggered if row["v2_certificate_issued"]]
    v1_covered = sum(row["v1_covered"] is True for row in v1_issued)
    covered = sum(row["v2_covered"] is True for row in issued)
    false_issued = [row for row in issued if row["v2_covered"] is not True]
    gains = [
        row["v2_reached_horizon"] / row["v1_reached_horizon"]
        for row in triggered
        if row["v1_reached_horizon"] > 0
    ]
    issuing_seeds = sorted({row["seed"] for row in issued})
    all_covered_by_seed = [
        seed
        for seed in issuing_seeds
        if all(row["v2_covered"] is True for row in issued if row["seed"] == seed)
    ]
    issued_leads = [
        int(row["actual_lead"])
        for row in issued
        if row["actual_lead"] is not None
    ]
    bracket_spans = [
        int(row["v2_bracket"][1] - row["v2_bracket"][0]) for row in issued
    ]
    raw_errors = [
        abs(int(row["raw_trigger_prediction"]) - int(row["actual_lead"]))
        for row in triggered
        if row["raw_trigger_prediction"] is not None
        and row["actual_lead"] is not None
    ]
    summary = {
        "models_trained": len(SEEDS),
        "models_fitted": sum(row["fit_step_first_logged"] is not None for row in seed_training_rows),
        "models_with_95_event": sum(row["exact_95_crossing"] is not None for row in seed_training_rows),
        "natural_grokking_models_ratio_gt_10": sum(
            row["natural_grokking_ratio_gt_10"] for row in seed_training_rows
        ),
        "seed_threshold_pairs": len(rows),
        "events_observed": sum(row["actual_crossing"] is not None for row in rows),
        "triggers_available": len(triggered),
        "unique_trigger_anchors": len(anchor_audits),
        "v1_certificates_issued": len(v1_issued),
        "v1_coverage_count": [v1_covered, len(v1_issued)],
        "v2_certificates_issued": len(issued),
        "v2_issuance_per_trigger": None if not triggered else len(issued) / len(triggered),
        "v2_abstention_rate_when_triggered": None if not triggered else 1.0 - len(issued) / len(triggered),
        "v2_coverage_count": [covered, len(issued)],
        "v2_coverage_exact_95_interval_iid_event_working_model": clopper_pearson(covered, len(issued)),
        "issuing_seeds": issuing_seeds,
        "all_covered_issuing_seeds": all_covered_by_seed,
        "seed_level_all_covered_count": [len(all_covered_by_seed), len(issuing_seeds)],
        "seed_level_exact_95_interval_iid_seed_working_model": clopper_pearson(
            len(all_covered_by_seed), len(issuing_seeds)
        ),
        "false_issued_brackets": len(false_issued),
        "v1_median_reached_horizon": None if not triggered else float(
            np.median([row["v1_reached_horizon"] for row in triggered])
        ),
        "v2_median_reached_horizon": None if not triggered else float(
            np.median([row["v2_reached_horizon"] for row in triggered])
        ),
        "v2_full_250_step_tubes": sum(
            row["v2_reached_horizon"] == HORIZON for row in triggered
        ),
        "median_pairwise_horizon_gain": None if not gains else float(np.median(gains)),
        "v2_median_bracket_span": None if not bracket_spans else float(np.median(bracket_spans)),
        "v2_maximum_bracket_span": None if not bracket_spans else int(max(bracket_spans)),
        "v2_median_lead": None if not issued_leads else float(np.median(issued_leads)),
        "v2_minimum_lead": None if not issued_leads else int(min(issued_leads)),
        "v2_maximum_lead": None if not issued_leads else int(max(issued_leads)),
        "raw_modal_median_absolute_timing_error": None if not raw_errors else float(np.median(raw_errors)),
        "state_tube_violations": sum(audit["violations"] for audit in anchor_audits.values()),
        "maximum_observed_error_to_bound_ratio": max(
            (audit["maximum_error_to_bound_ratio"] for audit in anchor_audits.values()),
            default=0.0,
        ),
        "failure_modes": dict(Counter(classify_failure(row) for row in rows)),
    }
    payload = {
        "protocol": PROTOCOL.name,
        "protocol_sha256": protocol_sha256(),
        "manifest_sha256": manifest_sha256(),
        "pipeline_version": PIPELINE_VERSION,
        "method_version": METHOD_VERSION,
        "seeds": list(SEEDS),
        "summary": summary,
        "issued_brackets": [row for row in issued],
        "rows": rows,
        "seed_training_rows": seed_training_rows,
        "anchor_audits": [
            {"seed": seed, "anchor": anchor, **audit}
            for (seed, anchor), audit in sorted(anchor_audits.items())
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--scan", action="store_true")
    action.add_argument("--certify-all", action="store_true")
    action.add_argument("--aggregate", action="store_true")
    action.add_argument("--certificate", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--anchor", type=int)
    args = parser.parse_args()
    if args.scan:
        scan()
    elif args.certify_all:
        certify_all()
    elif args.aggregate:
        aggregate()
    else:
        if args.seed not in SEEDS or args.anchor is None:
            parser.error("--certificate requires a frozen seed and --anchor")
        blind_certificate(args.seed, args.anchor, use_cache=False)


if __name__ == "__main__":
    main()
