#!/usr/bin/env python3
"""Execute the asynchronous online power cascade on one sealed v3 candidate."""

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
)
from one_shot_recenter_closure import conservative_one_shot_closure
from online_progressive_gram import OnlineGramState
from transformer_block_envelope import ball_valid_envelope
from transformer_certificate_protocol import Candidate
from transformer_four_sweep_development_audit import to_scaled
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
from transformer_v3_certificate import (
    METHOD_SEAL,
    _bracket_at_radius,
    _gate_raw_slacks,
    _q_geometry,
    frozen_candidates,
    output_path,
    safe_json,
)
from transformer_v3_protocol import (
    MAXIMUM_POWER,
    green_identity,
    make_registry,
    output_identity,
    probe_config,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CERTIFICATE_SEAL = ROOT / "TRANSFORMER_V3_CERTIFICATE_SEAL.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def relative_error(left: float, right: float) -> float:
    return abs(left - right) / max(abs(right), torch.finfo(torch.float64).tiny)


def benchmark(
    candidate: Candidate,
    *,
    force_full_q8: bool = False,
    run_label: str = "matched-online",
) -> dict:
    method = safe_json(METHOD_SEAL)
    candidates, horizons, _ = frozen_candidates()
    if candidate not in horizons:
        raise ValueError(f"candidate is outside the frozen v3 set: {candidate}")
    horizon = horizons[candidate]
    certificate_path = output_path(candidate)
    certificate = safe_json(certificate_path)
    if not certificate["certificate_issued"] or certificate.get("green_trace") is None:
        raise ValueError("online benchmark requires an originally issued Green certificate")
    certificate_seal = safe_json(CERTIFICATE_SEAL)
    sealed = next(
        row
        for row in certificate_seal["certificate_files"]
        if row["candidate"] == candidate.__dict__
    )
    if sealed["sha256"] != sha256(certificate_path):
        raise RuntimeError("candidate certificate changed after seal")

    blind_path, checkpoint_path = artifact_paths(candidate.seed, development=False)
    payload = safe_json(blind_path)
    config = TransformerConfig(**payload["config"])
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    template = make_template(config)
    spec = flat_spec(template)
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = make_disjoint_split(config)
    checkpoints = np.load(checkpoint_path)
    parameter = torch.from_numpy(checkpoints[f"step_{candidate.anchor}"]).clone()
    velocity = torch.from_numpy(checkpoints[f"velocity_{candidate.anchor}"]).clone()
    dimension = parameter.numel()
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
    if path["centerline_sha256"] != certificate["centerline_sha256"]:
        raise RuntimeError("online benchmark centerline hash mismatch")
    center = path["center"]
    scaled_center = path["scaled_center"]
    residual = torch.stack(
        [
            to_scaled(path["map_step"](center[step]), dimension, config.learning_rate)
            - scaled_center[step + 1]
            for step in range(horizon)
        ]
    )
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
    response_max = float(torch.linalg.vector_norm(signed_response, dim=1).max())
    domain_radius = 2.0 * response_norm
    for key, observed in (
        ("signed_response_sequence_norm", response_norm),
        ("signed_response_max_state_norm", response_max),
        ("outer_domain_radius", domain_radius),
    ):
        if relative_error(observed, float(certificate[key])) > 2.0e-12:
            raise RuntimeError(f"online benchmark geometry differs at {key}")

    registry = make_registry(candidates, horizons, str(method["master_nonce"]))
    probe = probe_config()
    all_pairs = torch.cartesian_prod(
        torch.arange(config.modulus), torch.arange(config.modulus)
    ).long()
    required = int(certificate["required_correct"])
    raw_zero = _gate_raw_slacks(
        logits(center[0, :dimension], cert_pairs, template, spec),
        cert_labels,
        required,
    )

    output_started = time.perf_counter()
    output_entries = []
    maximum_relative_trace_deviation = 0.0
    for step in range(1, horizon + 1):
        theta = center[step, :dimension]
        block = ball_valid_envelope(
            theta,
            spec,
            config,
            epsilon=domain_radius,
            exact_values=True,
            sphere=True,
        )
        apply = make_batched_output_gram_operator(theta, all_pairs, template, spec)
        identity = output_identity(candidate, step)
        state = OnlineGramState.initialize(
            dimension=dimension,
            dtype=theta.dtype,
            device=theta.device,
            config=probe,
            seed=registry.claim(identity),
        )
        trace_row = state.step(apply)
        frozen_row = certificate["output_rows"][step - 1]["trace"]["rows"][0]
        maximum_relative_trace_deviation = max(
            maximum_relative_trace_deviation,
            relative_error(
                trace_row["operator_norm_upper_bound"],
                float(frozen_row["operator_norm_upper_bound"]),
            ),
        )
        guarantee, exclusion = _gate_raw_slacks(
            logits(theta, cert_pairs, template, spec), cert_labels, required
        )
        output_entries.append(
            {
                "state": state,
                "apply": apply,
                "row": {
                    "step": step,
                    "trace": {"rows": [trace_row]},
                    "block_first": float(block["first"]),
                    "block_second": float(block["second"]),
                    "block_third": float(block["third"]),
                    "block_fixed_point_consistent": bool(block["fixed_point_consistent"]),
                    "raw_guarantee_slack": guarantee,
                    "raw_exclusion_slack": exclusion,
                },
            }
        )
    output_seconds = time.perf_counter() - output_started

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

    green_state = OnlineGramState.initialize(
        dimension=horizon * 2 * dimension,
        dtype=parameter.dtype,
        device=parameter.device,
        config=probe,
        seed=registry.claim(green_identity(candidate, horizon)),
    )
    green_rows = []

    def evaluate(q_output: int, q_green: int) -> dict:
        output_rows = [entry["row"] for entry in output_entries]
        map_drift, output_uppers = _q_geometry(
            power=q_output,
            output_rows=output_rows,
            config=config,
            domain_radius=domain_radius,
        )
        kappa = float(green_rows[q_green - 1]["operator_norm_upper_bound"])
        closure = conservative_one_shot_closure(
            kappa=kappa,
            derivative_drift=map_drift,
            response_sequence_norm=response_norm,
            response_max_state_norm=response_max,
            domain_radius=domain_radius,
        )
        bracket = None
        logic_slack = None
        if closure.closure_passed and all(
            entry["row"]["block_fixed_point_consistent"] for entry in output_entries
        ):
            bracket, logic_slack, _ = _bracket_at_radius(
                radius=float(closure.total_pointwise_radius),
                output_uppers=output_uppers,
                output_rows=output_rows,
                raw_zero=raw_zero,
            )
        return {
            "q_output": q_output,
            "q_green": q_green,
            "certificate_issued": bracket is not None,
            "certified_bracket": bracket,
            "certificate_logic_slack": logic_slack,
            "total_pointwise_radius": closure.total_pointwise_radius,
        }

    green_started = time.perf_counter()
    decision = None
    policy_trace = []
    for q_green in range(1, MAXIMUM_POWER + 1):
        row = green_state.step(green_gram)
        green_rows.append(row)
        frozen_row = certificate["green_trace"]["rows"][q_green - 1]
        maximum_relative_trace_deviation = max(
            maximum_relative_trace_deviation,
            relative_error(
                row["operator_norm_upper_bound"],
                float(frozen_row["operator_norm_upper_bound"]),
            ),
        )
        trial = evaluate(1, q_green)
        policy_trace.append(trial)
        if trial["certificate_issued"] and decision is None:
            decision = trial
            if not force_full_q8:
                break

    if decision is None or force_full_q8:
        for q_output in range(2, MAXIMUM_POWER + 1):
            for step, entry in enumerate(output_entries, start=1):
                trace_row = entry["state"].step(entry["apply"])
                entry["row"]["trace"]["rows"].append(trace_row)
                frozen_row = certificate["output_rows"][step - 1]["trace"]["rows"][
                    q_output - 1
                ]
                maximum_relative_trace_deviation = max(
                    maximum_relative_trace_deviation,
                    relative_error(
                        trace_row["operator_norm_upper_bound"],
                        float(frozen_row["operator_norm_upper_bound"]),
                    ),
                )
            for q_green in range(1, len(green_rows) + 1):
                trial = evaluate(q_output, q_green)
                policy_trace.append(trial)
                if trial["certificate_issued"] and decision is None:
                    decision = trial
                    if not force_full_q8:
                        break
            if decision is not None and not force_full_q8:
                break
    green_seconds = time.perf_counter() - green_started

    if decision is None:
        raise RuntimeError("online cascade failed to reproduce an issued certificate")
    if decision["certified_bracket"] != certificate["certified_bracket"]:
        raise RuntimeError("online cascade produced a different bracket")
    if maximum_relative_trace_deviation > 2.0e-12:
        raise RuntimeError(
            f"online trace differs from sealed trace by {maximum_relative_trace_deviation}"
        )

    online_operator_seconds = sum(
        entry["state"].cumulative_operator_seconds for entry in output_entries
    ) + green_state.cumulative_operator_seconds
    frozen_full_operator_seconds = sum(
        float(row["trace"]["rows"][MAXIMUM_POWER - 1]["cumulative_operator_seconds"])
        for row in certificate["output_rows"]
    ) + float(
        certificate["green_trace"]["rows"][MAXIMUM_POWER - 1][
            "cumulative_operator_seconds"
        ]
    )
    allocated_probe_bytes = sum(
        entry["state"].allocated_bytes for entry in output_entries
    ) + green_state.allocated_bytes
    result = {
        "status": "POST-SEAL EXECUTABLE ONLINE-STOPPING BENCHMARK",
        "execution_mode": "full-q8 control" if force_full_q8 else "online stopping",
        "run_label": run_label,
        "benchmark_script_sha256": sha256(Path(__file__).resolve()),
        "candidate": candidate.__dict__,
        "certificate_sha256": sha256(certificate_path),
        "certificate_seal_sha256": sha256(CERTIFICATE_SEAL),
        "policy": (
            "q_output=1; advance Green until issue/q=8; then advance outputs and "
            "recheck stored Green powers"
        ),
        "decision": decision,
        "frozen_earliest_lockstep_power": certificate["earliest_issuing_power"],
        "frozen_bracket": certificate["certified_bracket"],
        "maximum_relative_trace_deviation": maximum_relative_trace_deviation,
        "queried_output_power": output_entries[0]["state"].power,
        "queried_green_power": green_state.power,
        "logical_gram_applications": probe.probes
        * (horizon * output_entries[0]["state"].power + green_state.power),
        "full_q8_logical_gram_applications": probe.probes
        * (horizon + 1)
        * MAXIMUM_POWER,
        "logical_application_speedup": (
            (horizon + 1) * MAXIMUM_POWER
            / (horizon * output_entries[0]["state"].power + green_state.power)
        ),
        "online_measured_operator_seconds": online_operator_seconds,
        "frozen_full_trace_operator_seconds": frozen_full_operator_seconds,
        "measured_operator_time_speedup": (
            frozen_full_operator_seconds / online_operator_seconds
        ),
        "allocated_live_probe_bytes": allocated_probe_bytes,
        "allocated_live_probe_gib": allocated_probe_bytes / (1024**3),
        "timings_seconds": {
            "centerline_and_signed_response": center_seconds,
            "output_q1_and_envelopes": output_seconds,
            "green_and_output_refinement": green_seconds,
            "end_to_end": time.perf_counter() - started,
        },
        "policy_trace": policy_trace,
    }
    destination = RESULTS / (
        f"transformer_v3_online_policy_seed_{candidate.seed}_"
        f"gate_{candidate.gate_index}_anchor_{candidate.anchor}_{run_label}.json"
    )
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite online benchmark: {destination}")
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["output"] = str(destination)
    result["sha256"] = sha256(destination)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=366)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--anchor", type=int, default=1120)
    parser.add_argument("--force-full-q8", action="store_true")
    parser.add_argument("--run-label", default="matched-online")
    args = parser.parse_args()
    result = benchmark(
        Candidate(args.seed, args.threshold, args.anchor),
        force_full_q8=args.force_full_q8,
        run_label=args.run_label,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "policy_trace"}, indent=2))


if __name__ == "__main__":
    main()
