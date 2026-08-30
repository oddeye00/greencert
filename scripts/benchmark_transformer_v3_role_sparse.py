#!/usr/bin/env python3
"""Matched post-seal wall-time benchmark for role-sparse output transport.

The benchmark never opens a future outcome.  It compares the frozen all-pairs
output operator with separate training and certification operators on one
sealed v3 candidate.  Certification-output times are selected by the
predictable adaptive witness policy.  The sealed Green trace is reused so the
experiment isolates the output-transport multiplier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from adaptive_witness_policy import WitnessQuery, acquire_witnesses
from batched_green_operator import make_batched_output_gram_operator
from one_shot_recenter_closure import conservative_one_shot_closure
from online_progressive_gram import OnlineGramState
from probe_jacobian_bound import ProbeConfig, namespaced_probe_seed
from transformer_block_envelope import objective_hessian_lipschitz
from transformer_certificate_protocol import Candidate
from transformer_green_development_audit import build_frozen_centerline
from transformer_v3_certificate import (
    frozen_candidates,
    load_candidate,
    output_path,
    safe_json,
    verify_method_seal,
)
from transformer_v3_protocol import (
    FAMILY_FAILURE_PROBABILITY,
    HORIZON,
    MAXIMUM_POWER,
    PROBES,
    SWEEPS,
    maximum_operator_count,
    output_identity,
    probe_config,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_CANDIDATE = Candidate(369, 0.80, 4480)
ROLE_PROTOCOL = 903
TRAIN_ROLE = 11
CERT_ROLE = 12
FUSED_ROLE = 13


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_cache(path: Path, expected: dict) -> dict:
    if not path.exists():
        return {
            **expected,
            "baseline": {},
            "train": {},
            "cert": {},
            "fused_train": {},
            "fused": {},
            "fused_cert_extra": {},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"benchmark cache mismatch for {key}")
    for key in (
        "baseline",
        "train",
        "cert",
        "fused_train",
        "fused",
        "fused_cert_extra",
    ):
        payload.setdefault(key, {})
    return payload


def save_cache(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def role_identity(candidate: Candidate, step: int, role: int) -> tuple[int, ...]:
    return (
        ROLE_PROTOCOL,
        candidate.seed,
        candidate.gate_index,
        candidate.anchor,
        int(step),
        SWEEPS,
        int(role),
    )


def run_operator(
    *,
    theta: torch.Tensor,
    pairs: torch.Tensor,
    template,
    spec,
    config: ProbeConfig,
    seed: int,
    power: int,
) -> dict:
    started = time.perf_counter()
    apply = make_batched_output_gram_operator(theta, pairs, template, spec)
    state = OnlineGramState.initialize(
        dimension=theta.numel(),
        dtype=theta.dtype,
        device=theta.device,
        config=config,
        seed=seed,
    )
    rows = [state.step(apply) for _ in range(power)]
    return {
        "power": power,
        "operator_norm_upper_bound": rows[-1]["operator_norm_upper_bound"],
        "trace": rows,
        "operator_seconds": state.cumulative_operator_seconds,
        "wall_seconds": time.perf_counter() - started,
        "pairs": int(len(pairs)),
        "seed": int(seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_CANDIDATE.seed)
    parser.add_argument("--threshold", type=float, default=DEFAULT_CANDIDATE.threshold)
    parser.add_argument("--anchor", type=int, default=DEFAULT_CANDIDATE.anchor)
    parser.add_argument(
        "--mode",
        choices=("baseline", "role", "fused", "both", "all"),
        default="all",
    )
    args = parser.parse_args()
    candidate = Candidate(args.seed, args.threshold, args.anchor)

    method = verify_method_seal()
    candidates, horizons, _ = frozen_candidates()
    if candidate not in horizons:
        raise ValueError(f"candidate is outside the sealed v3 set: {candidate}")
    horizon = int(horizons[candidate])
    certificate_path = output_path(candidate)
    certificate = safe_json(certificate_path)
    if not bool(certificate.get("certificate_issued")):
        raise ValueError("benchmark requires a sealed issued certificate")
    power = int(certificate["earliest_issuing_power"])
    event = int(certificate["certified_bracket"][0])
    if certificate["certified_bracket"] != [event, event]:
        raise ValueError("benchmark expects a singleton bracket")

    config, template, spec, data, parameter, velocity = load_candidate(candidate)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    train_pairs, train_labels, _, _, cert_pairs, _ = data
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
    if path["centerline_sha256"] != certificate["centerline_sha256"]:
        raise RuntimeError("rebuilt centerline differs from the sealed certificate")
    center = path["center"]
    dimension = int(parameter.numel())
    all_pairs = torch.cartesian_prod(
        torch.arange(config.modulus), torch.arange(config.modulus)
    ).long()
    master_nonce = str(method["master_nonce"])

    maximum_candidates = int(maximum_operator_count()["maximum_candidates"])
    green_delta = 0.5 * FAMILY_FAILURE_PROBABILITY / maximum_candidates
    role_outputs_per_candidate = HORIZON + (HORIZON + 1)
    output_delta = (
        0.5
        * FAMILY_FAILURE_PROBABILITY
        / (maximum_candidates * role_outputs_per_candidate)
    )
    role_config = ProbeConfig(PROBES, MAXIMUM_POWER, output_delta)
    green_config = ProbeConfig(PROBES, MAXIMUM_POWER, green_delta)

    stem = (
        f"transformer_v3_role_sparse_seed_{candidate.seed}_"
        f"gate_{candidate.gate_index}_anchor_{candidate.anchor}"
    )
    cache_path = RESULTS / f"{stem}_cache.json"
    destination = RESULTS / f"{stem}_audit.json"
    expected = {
        "status": "post-seal matched output-transport benchmark cache",
        "candidate": candidate.__dict__,
        "horizon": horizon,
        "power": power,
        "centerline_sha256": path["centerline_sha256"],
        "certificate_sha256": sha256(certificate_path),
        "output_delta": output_delta,
        "green_delta": green_delta,
    }
    cache = load_cache(cache_path, expected)

    frozen_rows = {int(row["step"]): row for row in certificate["output_rows"]}
    raw_exclusions = {
        step: float(row["raw_exclusion_slack"])
        for step, row in frozen_rows.items()
    }

    def optimistic_query(step: int) -> WitnessQuery:
        return WitnessQuery(
            step,
            event <= step < event + int(certificate["protocol"]["persistence"]),
            raw_exclusions.get(step, -math.inf) > 0.0,
        )

    preplan = acquire_witnesses(
        event=event,
        persistence=int(certificate["protocol"]["persistence"]),
        horizon=horizon,
        raw_exclusion_slacks=raw_exclusions,
        query=optimistic_query,
        exact_failures={0},
    )
    if not preplan.issued:
        raise RuntimeError(f"deterministic raw-margin preplan failed: {preplan.reason}")
    planned_event_times = set(preplan.query_order)

    if args.mode in ("baseline", "both", "all"):
        baseline_config = probe_config()
        for step in range(1, horizon + 1):
            key = str(step)
            if key in cache["baseline"]:
                continue
            seed = namespaced_probe_seed(master_nonce, output_identity(candidate, step))
            row = run_operator(
                theta=center[step, :dimension],
                pairs=all_pairs,
                template=template,
                spec=spec,
                config=baseline_config,
                seed=seed,
                power=power,
            )
            frozen_upper = float(
                frozen_rows[step]["trace"]["rows"][power - 1][
                    "operator_norm_upper_bound"
                ]
            )
            relative = abs(row["operator_norm_upper_bound"] - frozen_upper) / max(
                abs(frozen_upper), 1.0e-300
            )
            if relative > 2.0e-12:
                raise RuntimeError(f"baseline trace mismatch at step {step}: {relative}")
            row["frozen_relative_error"] = relative
            cache["baseline"][key] = row
            if step % 10 == 0 or step == horizon:
                save_cache(cache_path, cache)
                print(f"baseline {step}/{horizon}", flush=True)

    if args.mode in ("role", "both", "all"):
        for step in range(1, horizon):
            key = str(step)
            if key in cache["train"]:
                continue
            identity = role_identity(candidate, step, TRAIN_ROLE)
            cache["train"][key] = run_operator(
                theta=center[step, :dimension],
                pairs=train_pairs,
                template=template,
                spec=spec,
                config=role_config,
                seed=namespaced_probe_seed(master_nonce, identity),
                power=power,
            )
            if step % 10 == 0 or step == horizon - 1:
                save_cache(cache_path, cache)
                print(f"training role {step}/{horizon - 1}", flush=True)

        maximum_map_drift = 0.0
        domain_radius = float(certificate["outer_domain_radius"])
        for step in range(1, horizon):
            role_upper = float(cache["train"][str(step)]["operator_norm_upper_bound"])
            sealed = frozen_rows[step]
            first_ball = role_upper + float(sealed["block_second"]) * domain_radius
            objective_drift = objective_hessian_lipschitz(
                first_ball,
                float(sealed["block_second"]),
                float(sealed["block_third"]),
            )
            maximum_map_drift = max(
                maximum_map_drift,
                math.sqrt(2.0) * config.learning_rate * objective_drift,
            )

        green_row = certificate["green_trace"]["rows"][power - 1]
        green_y = float(green_row["Y"])
        kappa = (
            0.0
            if green_y <= 0.0
            else (green_y / green_config.c_delta()) ** (1.0 / (2.0 * power))
        )
        closure = conservative_one_shot_closure(
            kappa=kappa,
            derivative_drift=maximum_map_drift,
            response_sequence_norm=float(certificate["signed_response_sequence_norm"]),
            response_max_state_norm=float(certificate["signed_response_max_state_norm"]),
            domain_radius=domain_radius,
        )
        if not closure.closure_passed:
            raise RuntimeError("role-separated optimizer closure failed")
        radius = float(closure.total_pointwise_radius)

        def query(step: int) -> WitnessQuery:
            key = str(step)
            if key not in cache["cert"]:
                identity = role_identity(candidate, step, CERT_ROLE)
                cache["cert"][key] = run_operator(
                    theta=center[step, :dimension],
                    pairs=cert_pairs,
                    template=template,
                    spec=spec,
                    config=role_config,
                    seed=namespaced_probe_seed(master_nonce, identity),
                    power=power,
                )
                save_cache(cache_path, cache)
                print(f"certification role queried step {step}", flush=True)
            upper = float(cache["cert"][key]["operator_norm_upper_bound"])
            sealed = frozen_rows[step]
            margin = math.sqrt(2.0) * (
                upper * radius
                + 0.5 * float(sealed["block_second"]) * radius * radius
            )
            return WitnessQuery(
                step,
                float(sealed["raw_guarantee_slack"]) - margin > 0.0,
                float(sealed["raw_exclusion_slack"]) - margin > 0.0,
            )

        policy = acquire_witnesses(
            event=event,
            persistence=int(certificate["protocol"]["persistence"]),
            horizon=horizon,
            raw_exclusion_slacks=raw_exclusions,
            query=query,
            exact_failures={0},
        )
        if not policy.issued:
            raise RuntimeError(f"adaptive role-sparse policy abstained: {policy.reason}")
        cache["policy"] = {
            "issued": policy.issued,
            "reason": policy.reason,
            "success_times": list(policy.success_times),
            "failure_witnesses": list(policy.failure_witnesses),
            "query_order": list(policy.query_order),
            "certified_bracket": [event, event],
            "kappa_upper": kappa,
            "maximum_optimizer_derivative_drift_upper": maximum_map_drift,
            "closure": closure.as_dict(),
        }
        save_cache(cache_path, cache)

    if args.mode in ("fused", "all"):
        train_cert_pairs = torch.cat((train_pairs, cert_pairs), dim=0)
        for step in range(1, horizon):
            key = str(step)
            planned = step in planned_event_times
            bucket = "fused" if planned else "fused_train"
            if key in cache[bucket]:
                continue
            pairs = train_cert_pairs if planned else train_pairs
            role = FUSED_ROLE if planned else TRAIN_ROLE
            identity = role_identity(candidate, step, role)
            cache[bucket][key] = run_operator(
                theta=center[step, :dimension],
                pairs=pairs,
                template=template,
                spec=spec,
                config=role_config,
                seed=namespaced_probe_seed(master_nonce, identity),
                power=power,
            )
            if step % 10 == 0 or step == horizon - 1:
                save_cache(cache_path, cache)
                print(f"fused training path {step}/{horizon - 1}", flush=True)

        if horizon in planned_event_times and str(horizon) not in cache["fused"]:
            identity = role_identity(candidate, horizon, CERT_ROLE)
            cache["fused"][str(horizon)] = run_operator(
                theta=center[horizon, :dimension],
                pairs=cert_pairs,
                template=template,
                spec=spec,
                config=role_config,
                seed=namespaced_probe_seed(master_nonce, identity),
                power=power,
            )
            save_cache(cache_path, cache)
            print(f"fused final certification step {horizon}", flush=True)

        maximum_map_drift = 0.0
        domain_radius = float(certificate["outer_domain_radius"])
        for step in range(1, horizon):
            key = str(step)
            source = cache["fused"] if step in planned_event_times else cache["fused_train"]
            upper = float(source[key]["operator_norm_upper_bound"])
            sealed = frozen_rows[step]
            first_ball = upper + float(sealed["block_second"]) * domain_radius
            objective_drift = objective_hessian_lipschitz(
                first_ball,
                float(sealed["block_second"]),
                float(sealed["block_third"]),
            )
            maximum_map_drift = max(
                maximum_map_drift,
                math.sqrt(2.0) * config.learning_rate * objective_drift,
            )

        green_row = certificate["green_trace"]["rows"][power - 1]
        green_y = float(green_row["Y"])
        kappa = (
            0.0
            if green_y <= 0.0
            else (green_y / green_config.c_delta()) ** (1.0 / (2.0 * power))
        )
        closure = conservative_one_shot_closure(
            kappa=kappa,
            derivative_drift=maximum_map_drift,
            response_sequence_norm=float(certificate["signed_response_sequence_norm"]),
            response_max_state_norm=float(certificate["signed_response_max_state_norm"]),
            domain_radius=domain_radius,
        )
        if not closure.closure_passed:
            raise RuntimeError("fused role-separated optimizer closure failed")
        radius = float(closure.total_pointwise_radius)
        success_times = set(
            range(event, event + int(certificate["protocol"]["persistence"]))
        )

        def fused_query(step: int) -> WitnessQuery:
            key = str(step)
            if step in planned_event_times:
                primary = cache["fused"][key]
            else:
                primary = None

            def verdict(row: dict) -> tuple[bool, bool]:
                upper = float(row["operator_norm_upper_bound"])
                sealed = frozen_rows[step]
                margin = math.sqrt(2.0) * (
                    upper * radius
                    + 0.5 * float(sealed["block_second"]) * radius * radius
                )
                return (
                    float(sealed["raw_guarantee_slack"]) - margin > 0.0,
                    float(sealed["raw_exclusion_slack"]) - margin > 0.0,
                )

            guarantee, exclusion = (False, False) if primary is None else verdict(primary)
            needed = (step in success_times and not guarantee) or (
                step not in success_times and not exclusion
            )
            if needed:
                if key not in cache["fused_cert_extra"]:
                    identity = role_identity(candidate, step, CERT_ROLE)
                    cache["fused_cert_extra"][key] = run_operator(
                        theta=center[step, :dimension],
                        pairs=cert_pairs,
                        template=template,
                        spec=spec,
                        config=role_config,
                        seed=namespaced_probe_seed(master_nonce, identity),
                        power=power,
                    )
                    save_cache(cache_path, cache)
                    print(f"fused fallback certification step {step}", flush=True)
                extra_guarantee, extra_exclusion = verdict(
                    cache["fused_cert_extra"][key]
                )
                guarantee = guarantee or extra_guarantee
                exclusion = exclusion or extra_exclusion
            return WitnessQuery(step, guarantee, exclusion)

        fused_policy = acquire_witnesses(
            event=event,
            persistence=int(certificate["protocol"]["persistence"]),
            horizon=horizon,
            raw_exclusion_slacks=raw_exclusions,
            query=fused_query,
            exact_failures={0},
        )
        if not fused_policy.issued:
            raise RuntimeError(f"fused adaptive policy abstained: {fused_policy.reason}")
        cache["fused_policy"] = {
            "issued": fused_policy.issued,
            "reason": fused_policy.reason,
            "preplanned_query_order": list(preplan.query_order),
            "query_order": list(fused_policy.query_order),
            "success_times": list(fused_policy.success_times),
            "failure_witnesses": list(fused_policy.failure_witnesses),
            "certified_bracket": [event, event],
            "kappa_upper": kappa,
            "maximum_optimizer_derivative_drift_upper": maximum_map_drift,
            "closure": closure.as_dict(),
        }
        save_cache(cache_path, cache)

    if not cache["baseline"] or "fused_policy" not in cache:
        print(json.dumps({"cache": str(cache_path), "status": "partial"}, indent=2))
        return

    baseline_rows = list(cache["baseline"].values())
    baseline_operator = sum(float(row["operator_seconds"]) for row in baseline_rows)
    baseline_wall = sum(float(row["wall_seconds"]) for row in baseline_rows)
    fused_rows = list(cache["fused_train"].values()) + list(cache["fused"].values())
    fused_extra_steps = {
        str(step) for step in cache["fused_policy"]["query_order"]
    }
    fused_extra_rows = [
        row
        for step, row in cache["fused_cert_extra"].items()
        if step in fused_extra_steps
    ]
    fused_rows += fused_extra_rows
    fused_operator = sum(float(row["operator_seconds"]) for row in fused_rows)
    fused_wall = sum(float(row["wall_seconds"]) for row in fused_rows)
    separate = None
    if cache["train"] and "policy" in cache:
        queried_cert = [str(step) for step in cache["policy"]["query_order"]]
        train_rows = list(cache["train"].values())
        cert_rows = [cache["cert"][step] for step in queried_cert]
        separate_operator = sum(
            float(row["operator_seconds"]) for row in train_rows + cert_rows
        )
        separate_wall = sum(float(row["wall_seconds"]) for row in train_rows + cert_rows)
        separate = {
            "training_role_times": len(train_rows),
            "adaptive_certification_role_times": len(cert_rows),
            "operator_seconds": separate_operator,
            "output_wall_seconds": separate_wall,
            "operator_speedup": baseline_operator / separate_operator,
            "output_wall_speedup": baseline_wall / separate_wall,
        }
    payload = {
        "status": "post-seal matched wall-time benchmark; prospective counts unchanged",
        "scope": (
            "output-Jacobian transport only; centerline and sealed Green response are "
            "shared and excluded from the speedup denominator"
        ),
        "candidate": candidate.__dict__,
        "certificate_sha256": sha256(certificate_path),
        "centerline_sha256": path["centerline_sha256"],
        "horizon": horizon,
        "power": power,
        "sealed_bracket": certificate["certified_bracket"],
        "role_sparse_bracket": cache["fused_policy"]["certified_bracket"],
        "same_bracket": cache["fused_policy"]["certified_bracket"]
        == certificate["certified_bracket"],
        "family_budget": {
            "family_failure_probability": FAMILY_FAILURE_PROBABILITY,
            "green_fraction": 0.5,
            "output_fraction": 0.5,
            "green_per_operator_delta": green_delta,
            "role_output_per_operator_delta": output_delta,
            "maximum_candidates": maximum_candidates,
            "maximum_role_outputs_per_candidate": role_outputs_per_candidate,
        },
        "query_counts": {
            "baseline_all_pair_times": len(baseline_rows),
            "preplanned_event_times": len(planned_event_times),
            "fused_or_training_path_times": horizon - 1,
            "final_certification_times": int(horizon in planned_event_times),
            "fallback_certification_times": len(fused_extra_rows),
            "adaptive_query_order": cache["fused_policy"]["query_order"],
        },
        "pair_work": {
            "baseline": sum(int(row["pairs"]) for row in baseline_rows),
            "fused_role_sparse": sum(int(row["pairs"]) for row in fused_rows),
        },
        "timings_seconds": {
            "centerline_shared": center_seconds,
            "baseline_operator": baseline_operator,
            "fused_role_sparse_operator": fused_operator,
            "operator_speedup": baseline_operator / fused_operator,
            "baseline_output_wall": baseline_wall,
            "fused_role_sparse_output_wall": fused_wall,
            "output_wall_speedup": baseline_wall / fused_wall,
        },
        "naive_separate_role_result": separate,
        "role_sparse_geometry": cache["fused_policy"],
        "cache": str(cache_path.relative_to(ROOT)),
        "interpretation": (
            "This converts the witness/role accounting result into a measured "
            "output-transport speedup on an immutable sealed candidate.  It is a "
            "post-seal implementation audit, not a new prospective certificate."
        ),
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(destination),
                "sha256": sha256(destination),
                "same_bracket": payload["same_bracket"],
                **payload["timings_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
