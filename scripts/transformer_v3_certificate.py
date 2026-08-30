#!/usr/bin/env python3
"""Outcome-blind Transformer v3 response-centered certificate construction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from batched_green_operator import (
    make_batched_output_gram_operator,
    make_batched_transformer_green_products,
    progressive_batched_gram_norm_bounds,
)
from one_shot_recenter_closure import conservative_one_shot_closure
from probe_jacobian_bound import namespaced_probe_seed
from transformer_block_envelope import ball_valid_envelope, objective_hessian_lipschitz
from transformer_certificate_protocol import Candidate
from transformer_four_sweep_development_audit import first_persistent, to_scaled
from transformer_green_development_audit import build_frozen_centerline
from transformer_green_operator import make_transformer_green_products
from transformer_hvp_grokking import (
    TransformerConfig,
    artifact_paths,
    flat_spec,
    logits,
    make_disjoint_split,
    make_template,
)
from transformer_v3_protocol import (
    FAMILY_FAILURE_PROBABILITY,
    MAXIMUM_POWER,
    PERSISTENCE,
    SWEEPS,
    candidate_universe,
    green_identity,
    make_registry,
    maximum_operator_count,
    output_identity,
    probe_config,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METHOD_SEAL = ROOT / "TRANSFORMER_V3_METHOD_SEAL.json"
CANDIDATE_MANIFEST = RESULTS / "transformer_v3_candidates_blind.json"
CANDIDATE_SEAL = ROOT / "TRANSFORMER_V3_CANDIDATE_SEAL.json"
NUMERIC_RADIUS_CAP = 1.0e3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def safe_json(path: Path) -> dict:
    lowered = path.name.lower()
    if lowered.endswith(".outcomes.json") or lowered.endswith(".sealed.log"):
        raise RuntimeError(f"outcome-blind process attempted forbidden read: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_method_seal() -> dict:
    seal = safe_json(METHOD_SEAL)
    for name, expected in seal["code_manifest"].items():
        observed = sha256(ROOT / name)
        if observed != expected:
            raise RuntimeError(
                f"v3 claim-relevant file changed after freeze: {name}: "
                f"{observed} != {expected}"
            )
    return seal


def frozen_candidates() -> tuple[tuple[Candidate, ...], dict[Candidate, int], dict]:
    seal = safe_json(CANDIDATE_SEAL)
    if seal["method_seal_sha256"] != sha256(METHOD_SEAL):
        raise RuntimeError("v3 candidate seal points to a different method")
    if sha256(CANDIDATE_MANIFEST) != seal["candidate_manifest_sha256"]:
        raise RuntimeError("v3 candidate manifest hash mismatch")
    manifest = safe_json(CANDIDATE_MANIFEST)
    selected = [
        row for row in manifest["records"] if row["disposition"] == "candidate frozen"
    ]
    candidates = tuple(
        Candidate(int(row["seed"]), float(row["threshold"]), int(row["anchor"]))
        for row in selected
    )
    horizons = {
        candidate: int(row["predicted_offset"]) + PERSISTENCE - 1
        for candidate, row in zip(candidates, selected)
    }
    observed = [
        {
            "seed": candidate.seed,
            "threshold": candidate.threshold,
            "anchor": candidate.anchor,
            "horizon": horizons[candidate],
        }
        for candidate in candidates
    ]
    if observed != seal["candidates"]:
        raise RuntimeError("v3 candidate coordinates differ from the seal")
    return candidates, horizons, seal


def output_path(candidate: Candidate) -> Path:
    return RESULTS / (
        f"transformer_v3_certificate_seed_{candidate.seed}_"
        f"gate_{candidate.gate_index}_anchor_{candidate.anchor}.json"
    )


def cache_path(candidate: Candidate) -> Path:
    return RESULTS / "transformer_v3_cache" / (
        f"seed_{candidate.seed}_gate_{candidate.gate_index}_"
        f"anchor_{candidate.anchor}.json"
    )


def _write_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_cache(
    candidate: Candidate,
    horizon: int,
    centerline_sha256: str,
    master_nonce: str,
) -> dict:
    path = cache_path(candidate)
    if not path.exists():
        return {"output_rows": [], "green_trace": None}
    payload = safe_json(path)
    expected = (horizon, centerline_sha256, master_nonce, probe_config().__dict__)
    observed = (
        int(payload.get("horizon", -1)),
        payload.get("centerline_sha256"),
        payload.get("master_nonce"),
        payload.get("probe_config"),
    )
    if observed != expected:
        raise RuntimeError(f"stale v3 cache: {observed} != {expected}")
    return payload


def _save_cache(
    candidate: Candidate,
    horizon: int,
    centerline_sha256: str,
    master_nonce: str,
    output_rows: list[dict],
    green_trace: dict | None,
) -> None:
    _write_cache(
        cache_path(candidate),
        {
            "status": "outcome-blind Transformer v3 resumable cache",
            "candidate": candidate.__dict__,
            "horizon": horizon,
            "centerline_sha256": centerline_sha256,
            "master_nonce": master_nonce,
            "probe_config": probe_config().__dict__,
            "output_rows": output_rows,
            "green_trace": green_trace,
        },
    )


def load_candidate(candidate: Candidate):
    blind_path, checkpoint_path = artifact_paths(candidate.seed, development=False)
    payload = safe_json(blind_path)
    if any("certificate" in name.lower() for name in payload["trajectory_columns"]):
        raise RuntimeError("v3 blind training artifact contains certification outcomes")
    config = TransformerConfig(**payload["config"])
    template = make_template(config)
    spec = flat_spec(template)
    data = make_disjoint_split(config)
    checkpoints = np.load(checkpoint_path)
    parameter = torch.from_numpy(checkpoints[f"step_{candidate.anchor}"]).clone()
    velocity = torch.from_numpy(checkpoints[f"velocity_{candidate.anchor}"]).clone()
    return config, template, spec, data, parameter, velocity


def _gate_raw_slacks(
    center_logits: torch.Tensor,
    labels: torch.Tensor,
    required: int,
) -> tuple[float, float]:
    true = center_logits.gather(1, labels[:, None])
    margins = true - center_logits
    rows = torch.arange(len(labels))
    margins[rows, labels] = torch.inf
    per_example = torch.min(margins, dim=1).values
    guarantee = torch.sort(per_example, descending=True).values[required - 1]
    definitely_incorrect_needed = len(labels) - required + 1
    exclusion = -torch.sort(per_example).values[definitely_incorrect_needed - 1]
    return float(guarantee), float(exclusion)


def _first_persistent_true(values: list[bool]) -> int | None:
    for start in range(max(0, len(values) - PERSISTENCE + 1)):
        if all(values[start : start + PERSISTENCE]):
            return int(start)
    return None


def _persistent_bracket(
    guarantee_slacks: list[float],
    exclusion_slacks: list[float],
) -> list[int] | None:
    guaranteed = [value > 0.0 for value in guarantee_slacks]
    possible = [value <= 0.0 for value in exclusion_slacks]
    lower = _first_persistent_true(possible)
    upper = _first_persistent_true(guaranteed)
    if lower is None or upper is None or lower > upper:
        return None
    return [lower, upper]


def _logic_slack(
    bracket: list[int] | None,
    guarantee_slacks: list[float],
    exclusion_slacks: list[float],
) -> float | None:
    if bracket is None:
        return None
    lower, upper = bracket
    upper_slack = min(guarantee_slacks[upper : upper + PERSISTENCE])
    prior = [
        max(exclusion_slacks[start : start + PERSISTENCE])
        for start in range(lower)
    ]
    lower_slack = math.inf if not prior else min(prior)
    return min(lower_slack, upper_slack)


def _q_geometry(
    *,
    power: int,
    output_rows: list[dict],
    config: TransformerConfig,
    domain_radius: float,
) -> tuple[float, list[float]]:
    maximum_map_drift = 0.0
    output_uppers = []
    for index, row in enumerate(output_rows):
        output_upper = float(
            row["trace"]["rows"][power - 1]["operator_norm_upper_bound"]
        )
        output_uppers.append(output_upper)
        first_ball = output_upper + float(row["block_second"]) * domain_radius
        objective_drift = objective_hessian_lipschitz(
            first_ball,
            float(row["block_second"]),
            float(row["block_third"]),
        )
        map_drift = math.sqrt(2.0) * config.learning_rate * objective_drift
        if index + 1 < len(output_rows):
            maximum_map_drift = max(maximum_map_drift, map_drift)
    return maximum_map_drift, output_uppers


def _bracket_at_radius(
    *,
    radius: float,
    output_uppers: list[float],
    output_rows: list[dict],
    raw_zero: tuple[float, float],
) -> tuple[list[int] | None, float | None, float]:
    guarantee_slacks = [raw_zero[0]]
    exclusion_slacks = [raw_zero[1]]
    maximum_margin = 0.0
    for row, output_upper in zip(output_rows, output_uppers):
        margin = math.sqrt(2.0) * (
            output_upper * radius
            + 0.5 * float(row["block_second"]) * radius * radius
        )
        maximum_margin = max(maximum_margin, margin)
        guarantee_slacks.append(float(row["raw_guarantee_slack"]) - margin)
        exclusion_slacks.append(float(row["raw_exclusion_slack"]) - margin)
    bracket = _persistent_bracket(guarantee_slacks, exclusion_slacks)
    return (
        bracket,
        _logic_slack(bracket, guarantee_slacks, exclusion_slacks),
        maximum_margin,
    )


def certify(candidate: Candidate) -> dict:
    method = verify_method_seal()
    candidates, horizons, candidate_seal = frozen_candidates()
    if candidate not in horizons:
        raise ValueError(f"candidate is outside the frozen v3 set: {candidate}")
    horizon = horizons[candidate]
    master_nonce = str(method["master_nonce"])
    config, template, spec, data, parameter, velocity = load_candidate(candidate)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = data
    started = time.perf_counter()

    center_started = time.perf_counter()
    path = build_frozen_centerline(
        config,
        template,
        spec,
        train_pairs,
        train_labels,
        parameter,
        velocity,
    )
    center_seconds = time.perf_counter() - center_started
    center = path["center"]
    scaled_center = path["scaled_center"]
    dimension = parameter.numel()
    center_counts = np.asarray(
        [
            int(
                (
                    logits(state[:dimension], cert_pairs, template, spec).argmax(1)
                    == cert_labels
                ).sum()
            )
            for state in center
        ],
        dtype=np.int64,
    )
    required = int(math.ceil(candidate.threshold * len(cert_pairs)))
    predicted_event = first_persistent(center_counts, required)
    if predicted_event is None or predicted_event + PERSISTENCE - 1 != horizon:
        raise RuntimeError("v3 centerline no longer matches its sealed modal event")

    residual = torch.stack(
        [
            to_scaled(path["map_step"](center[step]), dimension, config.learning_rate)
            - scaled_center[step + 1]
            for step in range(horizon)
        ]
    )
    defect_norm = float(torch.linalg.vector_norm(residual))
    scalar_green, _ = make_transformer_green_products(
        center[:horizon, :dimension],
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )
    signed_response = scalar_green(residual.reshape(-1)).reshape(horizon, -1)
    response_norm = float(torch.linalg.vector_norm(signed_response))
    response_max_state = float(
        torch.linalg.vector_norm(signed_response, dim=1).max()
    )
    domain_radius = 2.0 * response_norm
    if not math.isfinite(domain_radius) or domain_radius > NUMERIC_RADIUS_CAP:
        raise RuntimeError(f"v3 outer domain radius is unusable: {domain_radius}")

    registry = make_registry(candidates, horizons, master_nonce)
    probe = probe_config()
    cache = _load_cache(
        candidate, horizon, path["centerline_sha256"], master_nonce
    )
    rows_by_step = {
        int(row["step"]): row for row in cache.get("output_rows", [])
    }
    green_trace = cache.get("green_trace")
    for row in rows_by_step.values():
        registry.claim(tuple(row["identity"]))
    if green_trace is not None:
        registry.claim(tuple(green_trace["identity"]))

    logits_zero = logits(center[0, :dimension], cert_pairs, template, spec)
    raw_zero = _gate_raw_slacks(logits_zero, cert_labels, required)
    all_pairs = torch.cartesian_prod(
        torch.arange(config.modulus), torch.arange(config.modulus)
    ).long()
    output_started = time.perf_counter()
    for step in range(1, horizon + 1):
        if step in rows_by_step:
            continue
        theta = center[step, :dimension]
        block = ball_valid_envelope(
            theta,
            spec,
            config,
            epsilon=domain_radius,
            exact_values=True,
            sphere=True,
        )
        output_apply = make_batched_output_gram_operator(
            theta, all_pairs, template, spec
        )
        identity = output_identity(candidate, step)
        seed = registry.claim(identity)
        trace = progressive_batched_gram_norm_bounds(
            output_apply,
            dimension=dimension,
            dtype=theta.dtype,
            device=theta.device,
            config=probe,
            seed=seed,
        )
        center_logits = logits(theta, cert_pairs, template, spec)
        guarantee, exclusion = _gate_raw_slacks(
            center_logits, cert_labels, required
        )
        rows_by_step[step] = {
            "step": step,
            "identity": list(identity),
            "trace": trace,
            "block_first": float(block["first"]),
            "block_second": float(block["second"]),
            "block_third": float(block["third"]),
            "block_fixed_point_consistent": bool(
                block["fixed_point_consistent"]
            ),
            "block_fixed_point_iterations": int(
                block["fixed_point_iterations_used"]
            ),
            "raw_guarantee_slack": guarantee,
            "raw_exclusion_slack": exclusion,
        }
        if step % 10 == 0 or step == horizon:
            _save_cache(
                candidate,
                horizon,
                path["centerline_sha256"],
                master_nonce,
                [rows_by_step[key] for key in sorted(rows_by_step)],
                green_trace,
            )
    output_seconds = time.perf_counter() - output_started
    output_rows = [rows_by_step[key] for key in sorted(rows_by_step)]
    if [row["step"] for row in output_rows] != list(range(1, horizon + 1)):
        raise RuntimeError("v3 output cache is incomplete")
    fixed_points_consistent = all(
        row["block_fixed_point_consistent"] for row in output_rows
    )

    q_geometry = {}
    for power in range(1, MAXIMUM_POWER + 1):
        q_geometry[power] = _q_geometry(
            power=power,
            output_rows=output_rows,
            config=config,
            domain_radius=domain_radius,
        )

    directional_kappa_lower = max(
        1.0, response_norm / max(defect_norm, 1.0e-300)
    )
    tightest_map_drift = q_geometry[MAXIMUM_POWER][0]
    optimistic = conservative_one_shot_closure(
        kappa=directional_kappa_lower,
        derivative_drift=tightest_map_drift,
        response_sequence_norm=response_norm,
        response_max_state_norm=response_max_state,
        domain_radius=domain_radius,
    )
    early_abstention = not fixed_points_consistent or not optimistic.closure_passed

    green_seconds = 0.0
    if not early_abstention and green_trace is None:
        batch_green, batch_green_t = make_batched_transformer_green_products(
            center[:horizon, :dimension],
            train_pairs,
            train_labels,
            template,
            spec,
            config,
        )

        def green_gram(rows: torch.Tensor) -> torch.Tensor:
            return batch_green_t(batch_green(rows))

        identity = green_identity(candidate, horizon)
        seed = registry.claim(identity)
        green_started = time.perf_counter()
        green_trace = progressive_batched_gram_norm_bounds(
            green_gram,
            dimension=horizon * 2 * dimension,
            dtype=parameter.dtype,
            device=parameter.device,
            config=probe,
            seed=seed,
        )
        green_seconds = time.perf_counter() - green_started
        green_trace["identity"] = list(identity)
        _save_cache(
            candidate,
            horizon,
            path["centerline_sha256"],
            master_nonce,
            output_rows,
            green_trace,
        )

    power_rows = []
    if green_trace is not None:
        for power in range(1, MAXIMUM_POWER + 1):
            map_drift, output_uppers = q_geometry[power]
            kappa = float(
                green_trace["rows"][power - 1]["operator_norm_upper_bound"]
            )
            closure = conservative_one_shot_closure(
                kappa=kappa,
                derivative_drift=map_drift,
                response_sequence_norm=response_norm,
                response_max_state_norm=response_max_state,
                domain_radius=domain_radius,
            )
            bracket = None
            logic_slack = None
            maximum_margin = None
            if closure.closure_passed and fixed_points_consistent:
                bracket, logic_slack, maximum_margin = _bracket_at_radius(
                    radius=float(closure.total_pointwise_radius),
                    output_uppers=output_uppers,
                    output_rows=output_rows,
                    raw_zero=raw_zero,
                )
            power_rows.append(
                {
                    "power": power,
                    "kappa_upper": kappa,
                    "maximum_optimizer_derivative_drift_upper": map_drift,
                    "maximum_output_jacobian_upper": max(output_uppers),
                    "one_shot_closure": closure.as_dict(),
                    "certified_bracket": bracket,
                    "certificate_issued": bracket is not None,
                    "certificate_logic_slack": logic_slack,
                    "maximum_margin_radius": maximum_margin,
                    "logical_output_gram_applications": (
                        len(output_rows) * probe.probes * power
                    ),
                    "logical_green_gram_applications": probe.probes * power,
                }
            )

    issued_rows = [row for row in power_rows if row["certificate_issued"]]
    primary = issued_rows[0] if issued_rows else None

    baseline = None
    if green_trace is not None:
        map_drift, output_uppers = q_geometry[MAXIMUM_POWER]
        kappa = float(
            green_trace["rows"][MAXIMUM_POWER - 1]["operator_norm_upper_bound"]
        )
        baseline_closure = 2.0 * kappa * map_drift * response_norm
        baseline_bracket = None
        baseline_logic_slack = None
        baseline_max_margin = None
        if fixed_points_consistent and baseline_closure <= 1.0:
            (
                baseline_bracket,
                baseline_logic_slack,
                baseline_max_margin,
            ) = _bracket_at_radius(
                radius=domain_radius,
                output_uppers=output_uppers,
                output_rows=output_rows,
                raw_zero=raw_zero,
            )
        baseline = {
            "method": "preceding fixed R=2Z q=8 rule",
            "closure_statistic": baseline_closure,
            "closure_passed": fixed_points_consistent and baseline_closure <= 1.0,
            "certified_bracket": baseline_bracket,
            "certificate_issued": baseline_bracket is not None,
            "certificate_logic_slack": baseline_logic_slack,
            "maximum_margin_radius": baseline_max_margin,
        }

    registry_summary = registry.summary()
    result = {
        "status": "FROZEN V3 OUTCOME-BLIND HIGH-CONFIDENCE FLOAT64",
        "candidate": candidate.__dict__,
        "method_seal_sha256": sha256(METHOD_SEAL),
        "candidate_seal_sha256": sha256(CANDIDATE_SEAL),
        "candidate_manifest_sha256": candidate_seal["candidate_manifest_sha256"],
        "protocol": {
            "sweeps": SWEEPS,
            "horizon": horizon,
            "persistence": PERSISTENCE,
            "outer_domain_radius_rule": "rho = 2 ||K_H s||_X",
            "primary_radius_rule": "p + smaller recentered remainder root",
            "progressive_powers": list(range(1, MAXIMUM_POWER + 1)),
            "same_probe_block_across_powers": True,
            "probe_config": probe.__dict__,
            "family_failure_probability": FAMILY_FAILURE_PROBABILITY,
            "maximum_operator_accounting": maximum_operator_count(),
            "instantiated_operator_universe": len(
                candidate_universe(candidates, horizons)
            ),
            "master_nonce": master_nonce,
        },
        "centerline_sha256": path["centerline_sha256"],
        "sweep_diagnostics": path["diagnostics"],
        "required_correct": required,
        "predicted_persistent_event": predicted_event,
        "defect_sequence_norm": defect_norm,
        "signed_response_sequence_norm": response_norm,
        "signed_response_max_state_norm": response_max_state,
        "outer_domain_radius": domain_radius,
        "directional_green_norm_lower_bound": directional_kappa_lower,
        "optimistic_one_shot_closure": optimistic.as_dict(),
        "early_abstention_before_green_probe": early_abstention,
        "block_fixed_points_all_consistent": fixed_points_consistent,
        "green_trace": green_trace,
        "power_rows": power_rows,
        "earliest_issuing_power": None if primary is None else primary["power"],
        "certified_total_pointwise_radius": (
            None
            if primary is None
            else primary["one_shot_closure"]["total_pointwise_radius"]
        ),
        "certified_remainder_sequence_radius": (
            None
            if primary is None
            else primary["one_shot_closure"]["remainder_radius"]
        ),
        "certified_bracket": None if primary is None else primary["certified_bracket"],
        "certificate_issued": primary is not None,
        "certificate_logic_slack": (
            None if primary is None else primary["certificate_logic_slack"]
        ),
        "matched_fixed_radius_baseline": baseline,
        "output_rows": output_rows,
        "probability_budget": {
            "queried_operators": registry_summary["queried_operator_count"],
            "queried_union_bound": (
                registry_summary["queried_operator_count"] * probe.delta
            ),
            "maximum_family_union_bound": FAMILY_FAILURE_PROBABILITY,
            "no_union_over_power_levels": True,
            **registry_summary,
        },
        "timings_seconds": {
            "centerline": center_seconds,
            "output_phase_this_process": output_seconds,
            "green_phase_this_process": green_seconds,
            "total_this_process": time.perf_counter() - started,
        },
        "outcome_joined": False,
    }
    destination = output_path(candidate)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite v3 certificate: {destination}")
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["output"] = str(destination)
    result["sha256"] = sha256(destination)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--anchor", type=int, required=True)
    args = parser.parse_args()
    candidate = Candidate(args.seed, args.threshold, args.anchor)
    result = certify(candidate)
    print(
        json.dumps(
            {
                "candidate": result["candidate"],
                "certificate_issued": result["certificate_issued"],
                "earliest_issuing_power": result["earliest_issuing_power"],
                "certified_bracket": result["certified_bracket"],
                "baseline_issued": (
                    None
                    if result["matched_fixed_radius_baseline"] is None
                    else result["matched_fixed_radius_baseline"][
                        "certificate_issued"
                    ]
                ),
                "output": result["output"],
                "sha256": result["sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
