#!/usr/bin/env python3
"""Outcome-blind signed-Green certificates for the frozen fresh population."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from probe_jacobian_bound import jacobian_norm_bound
from transformer_block_envelope import ball_valid_envelope, objective_hessian_lipschitz
from transformer_certificate_protocol import Candidate
from transformer_four_sweep_development_audit import (
    count_envelope,
    first_persistent,
    persistent_bracket,
    to_scaled,
)
from transformer_green_development_audit import (
    build_frozen_centerline,
    gate_slacks,
    persistent_certificate_slack,
)
from transformer_green_operator import green_norm_bound, make_transformer_green_products
from transformer_green_confirmation_protocol import (
    FAMILY_FAILURE_PROBABILITY,
    HORIZON,
    MASTER_NONCE,
    PERSISTENCE,
    SWEEPS,
    candidate_universe,
    green_identity,
    make_registry,
    maximum_operator_count,
    output_identity,
    probe_config,
)
from transformer_hvp_grokking import (
    TransformerConfig,
    artifact_paths,
    flat_spec,
    logits,
    make_disjoint_split,
    make_template,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CANDIDATE_MANIFEST = RESULTS / "transformer_green_confirmation_candidates_blind.json"
CANDIDATE_SEAL = ROOT / "TRANSFORMER_GREEN_CONFIRMATION_CANDIDATE_SEAL.json"
METHOD_SEAL = ROOT / "TRANSFORMER_GREEN_CONFIRMATION_METHOD_SEAL.json"
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
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"claim-relevant file changed after freeze: {name}")
    return seal


def frozen_candidates() -> tuple[tuple[Candidate, ...], dict[Candidate, int], dict]:
    seal = safe_json(CANDIDATE_SEAL)
    if seal["method_seal_sha256"] != sha256(METHOD_SEAL):
        raise RuntimeError("candidate seal points to a different frozen method")
    if sha256(CANDIDATE_MANIFEST) != seal["candidate_manifest_sha256"]:
        raise RuntimeError("fresh candidate-manifest hash mismatch")
    manifest = safe_json(CANDIDATE_MANIFEST)
    rows = [row for row in manifest["records"] if row["disposition"] == "candidate frozen"]
    candidates = tuple(
        Candidate(int(row["seed"]), float(row["threshold"]), int(row["anchor"]))
        for row in rows
    )
    horizons = {
        candidate: int(row["predicted_offset"]) + PERSISTENCE - 1
        for candidate, row in zip(candidates, rows)
    }
    for candidate, horizon in horizons.items():
        if not 1 <= horizon <= HORIZON:
            raise RuntimeError(f"sealed candidate horizon is invalid: {candidate} -> {horizon}")
    declared = seal["candidates"]
    observed = [
        {
            "seed": candidate.seed,
            "threshold": candidate.threshold,
            "anchor": candidate.anchor,
            "horizon": horizons[candidate],
        }
        for candidate in candidates
    ]
    if observed != declared:
        raise RuntimeError("candidate coordinates/horizons differ from the seal")
    return candidates, horizons, seal


def cache_path(candidate: Candidate) -> Path:
    return RESULTS / "transformer_green_confirmation_cache" / (
        f"seed_{candidate.seed}_gate_{candidate.gate_index}_anchor_{candidate.anchor}.json"
    )


def output_path(candidate: Candidate) -> Path:
    return RESULTS / (
        f"transformer_green_confirmation_certificate_seed_{candidate.seed}_"
        f"gate_{candidate.gate_index}_anchor_{candidate.anchor}.json"
    )


def load_candidate(candidate: Candidate):
    blind_path, checkpoint_path = artifact_paths(candidate.seed, development=False)
    payload = safe_json(blind_path)
    if any("certificate" in name.lower() for name in payload["trajectory_columns"]):
        raise RuntimeError("blind training artifact contains certification outcomes")
    config = TransformerConfig(**payload["config"])
    template = make_template(config)
    spec = flat_spec(template)
    data = make_disjoint_split(config)
    checkpoints = np.load(checkpoint_path)
    parameter = torch.from_numpy(checkpoints[f"step_{candidate.anchor}"]).clone()
    velocity = torch.from_numpy(checkpoints[f"velocity_{candidate.anchor}"]).clone()
    return config, template, spec, data, parameter, velocity


def load_cache(candidate: Candidate, horizon: int, centerline_sha256: str) -> dict:
    path = cache_path(candidate)
    if not path.exists():
        return {"output_rows": []}
    payload = safe_json(path)
    expected = (MASTER_NONCE, horizon, centerline_sha256)
    observed = (
        payload.get("master_nonce"),
        int(payload.get("horizon", -1)),
        payload.get("centerline_sha256"),
    )
    if observed != expected:
        raise RuntimeError(f"stale confirmation cache: {observed} != {expected}")
    return payload


def save_cache(
    candidate: Candidate,
    horizon: int,
    centerline_sha256: str,
    green_probe: dict | None,
    output_rows: list[dict],
) -> None:
    path = cache_path(candidate)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "outcome-blind signed-Green confirmation cache",
        "candidate": candidate.__dict__,
        "horizon": horizon,
        "master_nonce": MASTER_NONCE,
        "centerline_sha256": centerline_sha256,
        "probe_config": probe_config().__dict__,
        "green_probe": green_probe,
        "output_rows": output_rows,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def certify(candidate: Candidate) -> dict:
    method_seal = verify_method_seal()
    started = time.perf_counter()
    candidates, horizons, seal = frozen_candidates()
    if candidate not in horizons:
        raise ValueError(f"candidate is outside the sealed confirmation set: {candidate}")
    horizon = horizons[candidate]
    config, template, spec, data, parameter, velocity = load_candidate(candidate)
    torch.set_num_threads(config.threads)
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = data
    path = build_frozen_centerline(
        config,
        template,
        spec,
        train_pairs,
        train_labels,
        parameter,
        velocity,
    )
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
        raise RuntimeError("fresh centerline no longer matches the sealed modal event")

    residual = torch.stack(
        [
            to_scaled(path["map_step"](center[step]), dimension, config.learning_rate)
            - scaled_center[step + 1]
            for step in range(horizon)
        ]
    )
    defect_norm = float(torch.linalg.vector_norm(residual))
    green_apply, _ = make_transformer_green_products(
        center[:horizon, :dimension],
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )
    signed_response = green_apply(residual.reshape(-1)).reshape(horizon, -1)
    response_norm = float(torch.linalg.vector_norm(signed_response))
    response_max_state = float(torch.linalg.vector_norm(signed_response, dim=1).max())
    radius = 2.0 * response_norm
    if not math.isfinite(radius) or radius > NUMERIC_RADIUS_CAP:
        raise RuntimeError(f"fixed signed-Green radius is numerically unusable: {radius}")

    registry = make_registry(candidates, horizons)
    probe = probe_config()
    cache = load_cache(candidate, horizon, path["centerline_sha256"])
    green_probe = cache.get("green_probe")
    rows_by_step = {int(row["step"]): row for row in cache.get("output_rows", [])}
    if green_probe is not None:
        registry.claim(tuple(green_probe["identity"]))
    for row in rows_by_step.values():
        registry.claim(tuple(row["output_probe"]["identity"]))

    all_pairs = torch.cartesian_prod(
        torch.arange(config.modulus), torch.arange(config.modulus)
    ).long()
    logits_zero = logits(center[0, :dimension], cert_pairs, template, spec)
    zero_guarantee, zero_exclusion = gate_slacks(
        logits_zero, cert_labels, 0.0, required
    )
    guaranteed = [int(center_counts[0])]
    possible = [int(center_counts[0])]
    guarantee_slacks = [zero_guarantee]
    exclusion_slacks = [zero_exclusion]
    geometry = []
    maximum_map_drift = 0.0
    fixed_points_consistent = True

    # Output operators come first.  This makes the ||K||>=1 early-abstention
    # gate available before the expensive Green probe without changing the
    # mathematical issuance rule.
    for step in range(1, horizon + 1):
        theta = center[step, :dimension]
        cached = rows_by_step.get(step)
        if cached is None:
            cached = {
                "step": step,
                "output_probe": jacobian_norm_bound(
                    theta,
                    all_pairs,
                    template,
                    spec,
                    probe,
                    output_identity(candidate, step),
                    registry=registry,
                ),
            }
            rows_by_step[step] = cached
            save_cache(
                candidate,
                horizon,
                path["centerline_sha256"],
                green_probe,
                [rows_by_step[key] for key in sorted(rows_by_step)],
            )

        block = ball_valid_envelope(
            theta,
            spec,
            config,
            epsilon=radius,
            exact_values=True,
            sphere=True,
        )
        fixed_points_consistent &= bool(block["fixed_point_consistent"])
        output_upper = float(cached["output_probe"]["jacobian_norm_upper_bound"])
        first_ball = output_upper + block["second"] * radius
        objective_lipschitz = objective_hessian_lipschitz(
            first_ball, block["second"], block["third"]
        )
        map_drift = math.sqrt(2.0) * config.learning_rate * objective_lipschitz
        if step < horizon:
            maximum_map_drift = max(maximum_map_drift, map_drift)
        margin_radius = math.sqrt(2.0) * (
            output_upper * radius + 0.5 * block["second"] * radius * radius
        )
        center_logits = logits(theta, cert_pairs, template, spec)
        lower_count, upper_count = count_envelope(
            center_logits, cert_labels, margin_radius
        )
        guarantee_slack, exclusion_slack = gate_slacks(
            center_logits, cert_labels, margin_radius, required
        )
        guaranteed.append(lower_count)
        possible.append(upper_count)
        guarantee_slacks.append(guarantee_slack)
        exclusion_slacks.append(exclusion_slack)
        geometry.append(
            {
                "step": step,
                "output_probe": cached["output_probe"],
                "block_first": block["first"],
                "block_second": block["second"],
                "block_third": block["third"],
                "block_fixed_point_consistent": block["fixed_point_consistent"],
                "block_fixed_point_iterations": block["fixed_point_iterations_used"],
                "first_ball": first_ball,
                "objective_hessian_lipschitz_upper": objective_lipschitz,
                "optimizer_derivative_drift_upper": map_drift,
                "margin_radius": margin_radius,
                "guaranteed_correct": lower_count,
                "possibly_correct": upper_count,
                "guaranteed_gate_slack": guarantee_slack,
                "possible_below_gate_slack": exclusion_slack,
            }
        )

    minimum_closure_lhs = 2.0 * maximum_map_drift * response_norm
    early_abstention = minimum_closure_lhs > 1.0 or not fixed_points_consistent
    if not early_abstention and green_probe is None:
        green_probe = green_norm_bound(
            center[:horizon, :dimension],
            train_pairs,
            train_labels,
            template,
            spec,
            config,
            probe,
            green_identity(candidate, horizon),
            registry,
        )
        save_cache(
            candidate,
            horizon,
            path["centerline_sha256"],
            green_probe,
            [rows_by_step[key] for key in sorted(rows_by_step)],
        )

    kappa = None if green_probe is None else float(
        green_probe["green_operator_norm_upper_bound"]
    )
    closure_lhs = None if kappa is None else 2.0 * kappa * maximum_map_drift * response_norm
    closure_passed = (
        fixed_points_consistent
        and closure_lhs is not None
        and closure_lhs <= 1.0
    )
    raw_bracket = persistent_bracket(
        np.asarray(guaranteed, dtype=np.int64),
        np.asarray(possible, dtype=np.int64),
        required,
    )
    bracket = raw_bracket if closure_passed else None
    certificate_slack = persistent_certificate_slack(
        bracket, guarantee_slacks, exclusion_slacks
    )
    registry_summary = registry.summary()
    queried = registry_summary["queried_operator_count"]
    result = {
        "status": "FROZEN FRESH signed-Green certificate; outcomes unopened",
        "candidate": candidate.__dict__,
        "method_seal_sha256": sha256(METHOD_SEAL),
        "candidate_seal_sha256": sha256(CANDIDATE_SEAL),
        "candidate_manifest_sha256": seal["candidate_manifest_sha256"],
        "protocol": {
            "sweeps": SWEEPS,
            "horizon": horizon,
            "persistence": PERSISTENCE,
            "radius_rule": "R = 2 ||K_H s||_sequence",
            "safe_early_abstention": "2 M Z > 1 because ||K_H|| >= 1",
            "probe_config": probe.__dict__,
            "family_failure_probability": FAMILY_FAILURE_PROBABILITY,
            "maximum_operator_accounting": maximum_operator_count(),
            "instantiated_operator_universe": len(candidate_universe(candidates, horizons)),
            "master_nonce": MASTER_NONCE,
        },
        "centerline_sha256": path["centerline_sha256"],
        "sweep_diagnostics": path["diagnostics"],
        "required_correct": required,
        "predicted_persistent_event": predicted_event,
        "defect_sequence_norm": defect_norm,
        "signed_response_sequence_norm": response_norm,
        "signed_response_max_state_norm": response_max_state,
        "signed_radius": radius,
        "green_probe": green_probe,
        "maximum_optimizer_derivative_drift_upper": maximum_map_drift,
        "minimum_closure_lhs_using_kappa_ge_1": minimum_closure_lhs,
        "early_abstention_before_green_probe": early_abstention,
        "closure_lhs_2_kappa_M_Z": closure_lhs,
        "closure_slack": None if closure_lhs is None else 1.0 - closure_lhs,
        "closure_passed": closure_passed,
        "block_fixed_points_all_consistent": fixed_points_consistent,
        "raw_margin_bracket": raw_bracket,
        "certified_bracket": bracket,
        "certificate_issued": bracket is not None,
        "certificate_output_logic_slack": certificate_slack,
        "guaranteed_correct": guaranteed,
        "possibly_correct": possible,
        "geometry": geometry,
        "probability_budget": {
            "queried_operators": queried,
            "queried_union_bound": queried * probe.delta,
            "maximum_family_union_bound": FAMILY_FAILURE_PROBABILITY,
            **registry_summary,
        },
        "outcome_joined": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    destination = output_path(candidate)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite fresh certificate: {destination}")
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
    result = certify(Candidate(args.seed, args.threshold, args.anchor))
    print(json.dumps({
        "candidate": result["candidate"],
        "predicted_persistent_event": result["predicted_persistent_event"],
        "closure_lhs": result["closure_lhs_2_kappa_M_Z"],
        "early_abstention": result["early_abstention_before_green_probe"],
        "certified_bracket": result["certified_bracket"],
        "output": result["output"],
        "sha256": result["sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
