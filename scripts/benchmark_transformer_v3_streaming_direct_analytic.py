#!/usr/bin/env python3
"""Complete streamed, direct-image, analytic-jet Transformer certificate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from analytic_jet_release import analytic_jet_release, logit_margin_radius
from audit_transformer_direct_image_green_panel import (
    OUTPUT as DIRECT_PANEL,
    assert_protocol_frozen,
    panel_index,
)
from audit_transformer_relinearized_prefix_panel import from_scaled
from batched_green_operator import make_batched_transformer_green_products
from direct_image_green_bound import direct_image_rows
from streaming_variational_centerline import build_streaming_transformer_centerline
from transformer_block_envelope import ball_valid_envelope
from transformer_certificate_protocol import Candidate
from transformer_green_operator import make_causal_green_products
from transformer_hvp_grokking import logits
from transformer_optimizer_probe import make_scaled_optimizer_jvp_vjp
from transformer_v3_certificate import (
    _gate_raw_slacks,
    _logic_slack,
    _persistent_bracket,
    load_candidate,
    output_path,
    safe_json,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
IDENTITY = RESULTS / "transformer_seed_366_streaming_prefix_identity.json"
CONTINUATION = RESULTS / "transformer_seed_366_matched_continuation.json"
CANDIDATE = Candidate(366, 0.8, 1120)
PROBES = 4


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes(order="C")
    ).hexdigest().upper()


def close(left: float, right: float, tolerance: float = 3.0e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=1.0e-300)


def direct_row() -> dict:
    payload = safe_json(DIRECT_PANEL)
    for row in payload["rows"]:
        if row["candidate"] == CANDIDATE.__dict__:
            return row
    raise RuntimeError("candidate is absent from direct-image panel")


def benchmark(run_label: str) -> dict:
    assert_protocol_frozen()
    identity = safe_json(IDENTITY)
    continuation = safe_json(CONTINUATION)
    prefix = panel_index()[
        (CANDIDATE.seed, CANDIDATE.threshold, CANDIDATE.anchor)
    ]
    prior_direct = direct_row()
    certificate_path = output_path(CANDIDATE)
    certificate = safe_json(certificate_path)
    horizon = int(prefix["horizon"])
    if horizon != int(identity["horizon"]):
        raise RuntimeError("identity and corrected-prefix horizons differ")

    config, template, spec, data, parameter, velocity = load_candidate(CANDIDATE)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = data
    dimension = int(parameter.numel())
    started = time.perf_counter()
    timings = {}

    phase = time.perf_counter()
    path = build_streaming_transformer_centerline(
        config,
        template,
        spec,
        train_pairs,
        train_labels,
        parameter,
        velocity,
        maximum_horizon=horizon,
    )
    timings["streaming_centerline"] = time.perf_counter() - phase
    if tensor_sha256(path["center"]) != identity["state_prefix_sha256"]:
        raise RuntimeError("streaming state prefix differs from identity bridge")
    if tensor_sha256(path["scaled_center"]) != identity["scaled_prefix_sha256"]:
        raise RuntimeError("streaming scaled prefix differs from identity bridge")
    center = path["center"]
    scaled_center = path["scaled_center"]

    phase = time.perf_counter()
    mapped = [path["map_step"](center[step]) for step in range(horizon)]
    residual = torch.stack(
        [
            torch.cat(
                (
                    mapped[step][:dimension],
                    config.learning_rate * mapped[step][dimension:],
                )
            )
            - scaled_center[step + 1]
            for step in range(horizon)
        ]
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
    old_apply, _ = make_causal_green_products(
        [row[0] for row in products],
        [row[1] for row in products],
        2 * dimension,
    )
    correction_rows = old_apply(residual.reshape(-1)).reshape(horizon, -1)
    correction = torch.cat(
        (torch.zeros_like(correction_rows[:1]), correction_rows), dim=0
    )
    corrected_scaled = scaled_center + correction
    corrected = from_scaled(corrected_scaled, dimension, config.learning_rate)
    timings["signed_correction_and_corrected_path"] = time.perf_counter() - phase
    if tensor_sha256(corrected_scaled) != prefix["corrected_path_sha256"]:
        raise RuntimeError("corrected path differs from sealed prefix panel")
    if not close(
        float(torch.linalg.vector_norm(correction_rows)),
        float(prefix["correction_sequence_norm"]),
    ):
        raise RuntimeError("signed correction norm changed")

    phase = time.perf_counter()
    domain_radius = float(prefix["domain_radius"])
    blocks = []
    for step in range(1, horizon + 1):
        block = ball_valid_envelope(
            center[step, :dimension],
            spec,
            config,
            epsilon=domain_radius,
            exact_values=True,
            sphere=True,
        )
        sealed = certificate["output_rows"][step - 1]
        for key in ("first", "second", "third"):
            if not close(float(block[key]), float(sealed[f"block_{key}"])):
                raise RuntimeError(f"analytic jet changed at step {step}: {key}")
        if not bool(block["fixed_point_consistent"]):
            raise RuntimeError(f"analytic jet domain failed at step {step}")
        blocks.append(block)
    timings["analytic_neural_jets"] = time.perf_counter() - phase

    phase = time.perf_counter()
    batch_apply, _ = make_batched_transformer_green_products(
        corrected[:horizon, :dimension],
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )
    generator = torch.Generator(device=corrected.device).manual_seed(
        int(prefix["probe_seed"])
    )
    probes = []
    for index in range(PROBES):
        probe = torch.randn(
            horizon * 2 * dimension,
            generator=generator,
            dtype=corrected.dtype,
            device=corrected.device,
        )
        if tensor_sha256(probe) != prefix["probe_hashes"][index]:
            raise RuntimeError(f"Green probe identity changed at index {index}")
        probes.append(probe)
    probe_block = torch.stack(probes)
    images = batch_apply(probe_block)
    initial_norms = [
        float(value) for value in torch.linalg.vector_norm(probe_block, dim=1)
    ]
    image_norms = [float(value) for value in torch.linalg.vector_norm(images, dim=1)]
    direct = direct_image_rows(
        image_norms=image_norms,
        initial_norms=initial_norms,
        prefixes=(PROBES,),
        stage_delta=float(prefix["stage_delta"]),
    )[0]
    timings["direct_image_green"] = time.perf_counter() - phase
    prior_stage = prior_direct["stage_rows"][0]["direct"]
    if not close(
        float(direct["operator_norm_upper_bound"]),
        float(prior_stage["operator_norm_upper_bound"]),
    ):
        raise RuntimeError("direct-image Green bound changed")

    phase = time.perf_counter()
    kappa = float(direct["operator_norm_upper_bound"])
    forcing_response = kappa * float(prefix["total_corrected_injection_upper"])
    release = analytic_jet_release(
        kappa=kappa,
        corrected_defect_response_bound=forcing_response,
        correction_max_state_norm=float(prefix["correction_max_state_norm"]),
        domain_radius=domain_radius,
        learning_rate=float(config.learning_rate),
        transition_jets=[
            (float(row["first"]), float(row["second"]), float(row["third"]))
            for row in blocks[:-1]
        ],
        output_first_bounds=[float(row["first"]) for row in blocks],
    )
    if not release.closure.closure_passed:
        raise RuntimeError("direct-image analytic-jet closure abstained")
    state_radius = float(release.state_radius_about_original_reference)
    required = int(certificate["required_correct"])
    raw = [
        _gate_raw_slacks(
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
    guarantee_slacks = [float(pair[0]) - margin for pair, margin in zip(raw, margins)]
    exclusion_slacks = [float(pair[1]) - margin for pair, margin in zip(raw, margins)]
    bracket = _persistent_bracket(guarantee_slacks, exclusion_slacks)
    logic_slack = _logic_slack(bracket, guarantee_slacks, exclusion_slacks)
    timings["analytic_closure_and_event"] = time.perf_counter() - phase
    if bracket != certificate["certified_bracket"] or bracket != prior_direct["bracket"]:
        raise RuntimeError("optimized path changed the sealed bracket")
    total_seconds = time.perf_counter() - started

    result = {
        "status": "streamed direct-image analytic-jet certificate benchmark complete",
        "evidence_boundary": (
            "Post-release outcome-blind composition of previously audited exact "
            "streaming, direct-image Green, and deterministic analytic-jet routes."
        ),
        "candidate": CANDIDATE.__dict__,
        "run_label": run_label,
        "horizon": horizon,
        "certified_bracket": bracket,
        "same_bracket": True,
        "logic_slack": logic_slack,
        "state_radius_about_original_reference": state_radius,
        "green_operator_norm_upper_bound": kappa,
        "green_forward_probes": PROBES,
        "green_transpose_probes": 0,
        "randomized_output_operators": 0,
        "analytic_release": release.as_dict(),
        "timings_seconds": {**timings, "end_to_end": total_seconds},
        "matched_continuation": {
            "source": str(CONTINUATION.relative_to(ROOT)).replace("\\", "/"),
            "source_sha256": sha256(CONTINUATION),
            "median_26_step_seconds": float(continuation["median_short_seconds"]),
            "median_300_step_seconds": float(continuation["median_full_seconds"]),
            "certificate_to_26_step_ratio": total_seconds
            / float(continuation["median_short_seconds"]),
            "certificate_to_300_step_ratio": total_seconds
            / float(continuation["median_full_seconds"]),
        },
        "identity_record_sha256": sha256(IDENTITY),
        "certificate_sha256": sha256(certificate_path),
        "direct_panel_sha256": sha256(DIRECT_PANEL),
        "corrected_prefix_panel_sha256": sha256(
            RESULTS / "transformer_v3_relinearized_prefix_panel_audit.json"
        ),
        "script_sha256": sha256(Path(__file__).resolve()),
        "combined_family_failure_upper": 1.0e-6,
        "outcome_files_read": 0,
    }
    destination = RESULTS / (
        "transformer_v3_streaming_direct_analytic_seed_366_gate_1_"
        f"anchor_1120_{run_label}.json"
    )
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite benchmark: {destination}")
    destination.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return {**result, "output": str(destination), "sha256": sha256(destination)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-label", default="replicate-1")
    args = parser.parse_args()
    result = benchmark(args.run_label)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("analytic_release",)
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
