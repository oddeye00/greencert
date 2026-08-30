#!/usr/bin/env python3
"""Full-sequence outward Arb audit for four amplified-secant projections."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from flint import arb, ctx

from arb_transformer_multijet import (
    arb_transformer_objective_jet,
    make_parameter_jet,
)
from audit_transformer_v3_amplified_secant_full import maximum_forcing_for_closure
from audit_transformer_v3_response_free_probe import probe_seed
from one_shot_recenter_closure import exact_one_shot_closure
from transformer_certificate_protocol import Candidate
from transformer_four_sweep_development_audit import to_scaled
from transformer_green_development_audit import build_frozen_centerline
from transformer_green_operator import make_transformer_green_products
from transformer_hvp_grokking import FlatSpec, TransformerConfig, logits
from transformer_v3_certificate import (
    _bracket_at_radius,
    _gate_raw_slacks,
    load_candidate,
    output_path,
    safe_json,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PROTOCOL = ROOT / "AMPLIFIED_SECANT_OUTWARD_EXECUTION_PROTOCOL_V2.md"
FOUR_PROTOCOL = ROOT / "AMPLIFIED_SECANT_FOUR_PROBE_PROTOCOL.md"
OUTPUT = RESULTS / "transformer_v3_arb_secant_full_v2_audit.json"
NONCE = "5a37e5ccaf6834c438fde251d52ec1de313329314377d70cc1cb25e62fc52f2a"
DOMAIN = "greencert-response-free-secant-four-probe-v1|"
LAMBDA = 4096.0
HORIZON = 52
PROBES = 4
PRECISION = 192
WORKERS = 4
EXPECTED_SOURCE_HASHES = {
    "scripts/arb_transformer_objective.py": "59DC6B9889669725CD4257EE6E6F6263C46B0EA3C7D7DAF5A514AF714DF5C736",
    "scripts/arb_transformer_multijet.py": "186C6C3A7CCA06AC322778C242FA882D662EE9349F536045D50470623F92754C",
    "scripts/test_arb_transformer_objective.py": "0A04A910F4DAAFEC4D3D69FB21DA8933C756C15E96F2290C9DDD4903F3092A63",
    "scripts/test_arb_transformer_multijet.py": "4008532246552EF440B6854EAF1CCCF3871B93A8CBB5513E0A7AB62CDBD87F20",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def outward_float_upper(value: arb) -> float:
    return math.nextafter(float(value.upper()), math.inf)


def _worker(task: dict) -> dict:
    ctx.prec = int(task["precision"])
    config = TransformerConfig(**task["config"])
    spec = FlatSpec(
        names=tuple(task["spec"]["names"]),
        shapes=tuple(torch.Size(row) for row in task["spec"]["shapes"]),
        sizes=tuple(int(value) for value in task["spec"]["sizes"]),
    )
    point = task["point"]
    direction = task["direction"]
    theta_probes = task["theta_probes"]
    velocity_probes = task["velocity_probes"]
    probe_terms = [
        [(1.0, velocity), (-1.0, theta)]
        for theta, velocity in zip(theta_probes, velocity_probes)
    ]
    pairs = np.asarray(task["train_pairs"], dtype=np.int64)
    labels = np.asarray(task["train_labels"], dtype=np.int64)
    started = time.perf_counter()
    base = arb_transformer_objective_jet(
        make_parameter_jet(
            point,
            [],
            spec,
            x_direction=direction,
            y_direction_terms=probe_terms,
        ),
        pairs,
        labels,
        config,
    )
    base_seconds = time.perf_counter() - started
    started = time.perf_counter()
    shifted = arb_transformer_objective_jet(
        make_parameter_jet(
            point,
            [],
            spec,
            base_terms=[(LAMBDA, direction)],
            y_direction_terms=probe_terms,
        ),
        pairs,
        labels,
        config,
    )
    shifted_seconds = time.perf_counter() - started
    if base.xy is None:
        raise RuntimeError("mixed jet missing")
    intervals = []
    for index in range(PROBES):
        raw = shifted.y[index] - base.y[index] - arb(LAMBDA) * base.xy[index]
        projected = arb(config.learning_rate) * raw / arb(LAMBDA) ** 2
        intervals.append(projected.str(90, radius=True, more=True))
    return {
        "step": int(task["step"]),
        "intervals": intervals,
        "base_seconds": base_seconds,
        "shifted_seconds": shifted_seconds,
        "total_seconds": base_seconds + shifted_seconds,
    }


def main() -> None:
    for relative, expected in EXPECTED_SOURCE_HASHES.items():
        observed = sha256(ROOT / relative)
        if observed != expected:
            raise RuntimeError(f"frozen source hash changed: {relative}: {observed}")
    ctx.prec = PRECISION
    candidate = Candidate(366, 0.7, 1040)
    certificate_path = output_path(candidate)
    certificate = safe_json(certificate_path)
    config, template, spec, data, parameter, velocity = load_candidate(candidate)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = data
    dimension = int(parameter.numel())
    wall_started = time.perf_counter()
    path = build_frozen_centerline(
        config,
        template,
        spec,
        train_pairs,
        train_labels,
        parameter,
        velocity,
    )
    if path["centerline_sha256"] != certificate["centerline_sha256"]:
        raise RuntimeError("centerline hash mismatch")
    center = path["center"][: HORIZON + 1]
    scaled_center = path["scaled_center"][: HORIZON + 1]
    mapped = [path["map_step"](center[step]) for step in range(HORIZON)]
    residual = torch.stack(
        [
            to_scaled(mapped[step], dimension, config.learning_rate)
            - scaled_center[step + 1]
            for step in range(HORIZON)
        ]
    )
    apply_green, _ = make_transformer_green_products(
        center[:HORIZON, :dimension],
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )
    signed = apply_green(residual.reshape(-1)).reshape(HORIZON, -1)
    response_norm = float(torch.linalg.vector_norm(signed))
    response_max = float(torch.linalg.vector_norm(signed, dim=1).max())
    setup_seconds = time.perf_counter() - wall_started

    generator = torch.Generator(device="cpu").manual_seed(
        probe_seed(nonce=NONCE, domain=DOMAIN)
    )
    gaussian = torch.randn(
        PROBES,
        HORIZON * 2 * dimension,
        generator=generator,
        dtype=center.dtype,
    ).reshape(PROBES, HORIZON, 2 * dimension)
    tasks = []
    config_row = asdict(config)
    spec_row = {
        "names": list(spec.names),
        "shapes": [list(shape) for shape in spec.shapes],
        "sizes": list(spec.sizes),
    }
    pair_rows = train_pairs.cpu().numpy().tolist()
    label_rows = train_labels.cpu().numpy().tolist()
    for step in range(1, HORIZON):
        theta_probe, velocity_probe = gaussian[:, step, :].split(dimension, dim=1)
        tasks.append(
            {
                "step": step,
                "precision": PRECISION,
                "config": config_row,
                "spec": spec_row,
                "point": center[step, :dimension].tolist(),
                "direction": signed[step - 1, :dimension].tolist(),
                "theta_probes": theta_probe.tolist(),
                "velocity_probes": velocity_probe.tolist(),
                "train_pairs": pair_rows,
                "train_labels": label_rows,
            }
        )

    jet_started = time.perf_counter()
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(_worker, task) for task in tasks]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(
                json.dumps(
                    {
                        "progress": f"{completed}/{HORIZON - 1}",
                        "step": row["step"],
                        "jet_seconds": row["total_seconds"],
                    }
                ),
                flush=True,
            )
    jet_wall_seconds = time.perf_counter() - jet_started
    rows.sort(key=lambda row: int(row["step"]))

    ctx.prec = PRECISION
    sums = [arb(0) for _ in range(PROBES)]
    checkpoint_rows = []
    for row in rows:
        parsed = [arb(value) for value in row["intervals"]]
        for index, value in enumerate(parsed):
            sums[index] += value
        checkpoint_rows.append(
            {
                "step": int(row["step"]),
                "intervals": row["intervals"],
                "base_seconds": float(row["base_seconds"]),
                "shifted_seconds": float(row["shifted_seconds"]),
                "total_seconds": float(row["total_seconds"]),
            }
        )
    magnitudes = [abs(value) for value in sums]
    largest = max(range(PROBES), key=lambda index: float(magnitudes[index].upper()))
    projection_upper = magnitudes[largest].upper()
    delta = arb(1) / 10**6
    calibration = arb(2).sqrt() * (delta.root(PROBES)).erfinv()
    calibration_lower = calibration.lower()
    norm_bound = projection_upper / calibration_lower
    norm_upper = outward_float_upper(norm_bound)

    amplified = safe_json(RESULTS / "transformer_v3_amplified_secant_full_audit.json")
    amplified_row = next(
        row for row in amplified["rows"] if float(row["amplification"]) == LAMBDA
    )
    sigma = math.nextafter(
        float(amplified_row["analytic_secant_discrepancy_upper"]), math.inf
    )
    power_row = certificate["power_rows"][0]
    kappa = math.nextafter(float(power_row["kappa_upper"]), math.inf)
    drift = math.nextafter(
        float(power_row["maximum_optimizer_derivative_drift_upper"]), math.inf
    )
    domain_radius = float(power_row["one_shot_closure"]["domain_radius"])
    beta = math.nextafter(kappa * (sigma + norm_upper), math.inf)
    closure = exact_one_shot_closure(
        kappa=kappa,
        derivative_drift=drift,
        response_sequence_norm=response_norm,
        response_max_state_norm=response_max,
        corrected_defect_response_bound=beta,
        domain_radius=domain_radius,
    )
    raw_zero = _gate_raw_slacks(
        logits(center[0, :dimension], cert_pairs, template, spec),
        cert_labels,
        int(certificate["required_correct"]),
    )
    output_rows = certificate["output_rows"]
    output_uppers = [
        float(row["trace"]["rows"][0]["operator_norm_upper_bound"])
        for row in output_rows
    ]
    bracket = None
    logic_slack = None
    if closure.closure_passed:
        bracket, logic_slack, _ = _bracket_at_radius(
            radius=float(closure.total_pointwise_radius),
            output_uppers=output_uppers,
            output_rows=output_rows,
            raw_zero=raw_zero,
        )
    forcing_cap = maximum_forcing_for_closure(
        kappa=kappa,
        derivative_drift=drift,
        response_max=response_max,
        domain_radius=domain_radius,
    ) / kappa
    payload = {
        "status": "FULL-SEQUENCE OUTWARD ARB SECANT PROBE AUDIT V2 PASSED",
        "evidence_boundary": (
            "Outcome-blind post-seal hardening v2. The four exact scalar secant "
            "projections are outward-enclosed conditional on stored dyadic "
            "center/response/probe values; state-probe differences are formed "
            "inside Arb. Under the ideal-PRNG model, upstream "
            "Green construction, derivative envelopes, and output margins retain "
            "their existing numerical boundary, so this is not a complete "
            "computer-assisted Transformer event proof."
        ),
        "candidate": candidate.__dict__,
        "horizon": HORIZON,
        "checkpoints": HORIZON - 1,
        "probes": PROBES,
        "scalar_intervals": (HORIZON - 1) * PROBES,
        "delta_exact": "1/1000000",
        "precision_bits": PRECISION,
        "workers": WORKERS,
        "amplification": LAMBDA,
        "setup_seconds": setup_seconds,
        "jet_wall_seconds": jet_wall_seconds,
        "jet_cpu_seconds_sum": sum(row["total_seconds"] for row in rows),
        "total_wall_seconds": time.perf_counter() - wall_started,
        "summed_projection_intervals": [
            value.str(90, radius=True, more=True) for value in sums
        ],
        "maximum_projection_absolute_upper": outward_float_upper(projection_upper),
        "calibration_interval": calibration.str(90, radius=True, more=True),
        "calibration_lower": float(calibration_lower),
        "secant_forcing_norm_upper": norm_upper,
        "analytic_secant_discrepancy_upper": sigma,
        "response_free_beta_upper": beta,
        "forcing_cap": forcing_cap,
        "forcing_headroom_ratio": forcing_cap / (sigma + norm_upper),
        "closure": closure.as_dict(),
        "bracket": bracket,
        "logic_slack": logic_slack,
        "checkpoint_rows": checkpoint_rows,
        "source_sha256": EXPECTED_SOURCE_HASHES,
        "protocol_sha256": sha256(PROTOCOL),
        "four_probe_protocol_sha256": sha256(FOUR_PROTOCOL),
        "certificate_sha256": sha256(certificate_path),
        "centerline_sha256": path["centerline_sha256"],
        "outcome_files_read": 0,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "checkpoint_rows"}, indent=2))


if __name__ == "__main__":
    main()
