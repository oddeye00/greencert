#!/usr/bin/env python3
"""Independent recomputation of the frozen four-probe corrected-path result."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from probe_jacobian_bound import c_delta, namespaced_probe_seed
from transformer_certificate_protocol import Candidate, PERSISTENCE
from transformer_four_sweep_development_audit import to_scaled
from transformer_green_development_audit import build_frozen_centerline
from transformer_green_operator import make_transformer_green_products
from transformer_hvp_grokking import logits
from transformer_v3_certificate import load_candidate, output_path, safe_json


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "transformer_v3_relinearized_secant_four_probe_audit.json"
OUTPUT = ROOT / "results" / "transformer_v3_relinearized_secant_four_probe_independent_audit.json"
PROTOCOL = ROOT / "RELINEARIZED_SECANT_FOUR_PROBE_PROTOCOL.md"
WRAPPER = ROOT / "scripts" / "audit_transformer_relinearized_secant_four_probe.py"
BASE = ROOT / "scripts" / "audit_transformer_relinearized_secant.py"
AMPLIFIED = ROOT / "results" / "transformer_v3_amplified_secant_full_audit.json"
OUTWARD = ROOT / "results" / "transformer_v3_arb_secant_full_v2_independent_audit.json"

CANDIDATE = Candidate(366, 0.70, 1040)
HORIZON = 52
PROBES = 4
DELTA = 4.59896983075791e-11
NONCE = "611fda4bd0aa71d5a3ea2c4158a103cb32330ed279660ffe9dc35232aea14360"
IDENTITY = (93, 366, 0, 1040, 52, 4, 1)
EXPECTED_HASHES = {
    WRAPPER: "893CDBFDDA9D9AB1E53FE8D9F72D242E59E2F3B0FCB9631C7653CE150333CF1A",
    BASE: "D6459D702F62AF11427FA26E197AA0C8F5D6D2EFEFC1C05C36FEA1EB0E90C8F6",
    PROTOCOL: "A4B94DB67A78758BF0DA3824816D25B8B00D4CA0FE8F384B8AA7B12BB257E47A",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def close(left: float, right: float, tolerance: float = 2.0e-12) -> None:
    if not math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=1e-30):
        raise AssertionError(f"numeric mismatch: {left!r} != {right!r}")


def from_scaled(path: torch.Tensor, dimension: int, eta: float) -> torch.Tensor:
    return torch.cat((path[..., :dimension], path[..., dimension:] / eta), dim=-1)


def gate_slacks(values: torch.Tensor, labels: torch.Tensor, required: int) -> tuple[float, float]:
    true = values.gather(1, labels[:, None])
    margins = true - values
    rows = torch.arange(len(labels))
    margins[rows, labels] = torch.inf
    per_example = torch.min(margins, dim=1).values
    guarantee = torch.sort(per_example, descending=True).values[required - 1]
    excluded_needed = len(labels) - required + 1
    exclusion = -torch.sort(per_example).values[excluded_needed - 1]
    return float(guarantee), float(exclusion)


def first_persistent(values: list[bool]) -> int | None:
    for start in range(max(0, len(values) - PERSISTENCE + 1)):
        if all(values[start : start + PERSISTENCE]):
            return start
    return None


def main() -> None:
    for path, expected in EXPECTED_HASHES.items():
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"frozen hash changed: {path}: {observed}")
    stored = safe_json(RESULT)
    certificate_path = output_path(CANDIDATE)
    certificate = safe_json(certificate_path)
    amplified = safe_json(AMPLIFIED)
    outward = safe_json(OUTWARD)
    config, template, spec, data, parameter, velocity = load_candidate(CANDIDATE)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = data
    dimension = int(parameter.numel())
    path = build_frozen_centerline(
        config, template, spec, train_pairs, train_labels, parameter, velocity
    )
    center = path["center"][: HORIZON + 1]
    scaled = path["scaled_center"][: HORIZON + 1]
    mapped = [path["map_step"](center[j]) for j in range(HORIZON)]
    defect = torch.stack(
        [
            to_scaled(mapped[j], dimension, config.learning_rate) - scaled[j + 1]
            for j in range(HORIZON)
        ]
    )
    apply_old, _ = make_transformer_green_products(
        center[:HORIZON, :dimension],
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )
    response_rows = apply_old(defect.reshape(-1)).reshape(HORIZON, -1)
    response = torch.cat((torch.zeros_like(response_rows[:1]), response_rows), dim=0)
    corrected_scaled = scaled + response
    corrected = from_scaled(corrected_scaled, dimension, config.learning_rate)
    apply_new, transpose_new = make_transformer_green_products(
        corrected[:HORIZON, :dimension],
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )

    seed = namespaced_probe_seed(NONCE, IDENTITY)
    generator = torch.Generator(device=corrected.device).manual_seed(seed)
    best = 0.0
    lower = 0.0
    for _ in range(PROBES):
        vector = torch.randn(
            HORIZON * 2 * dimension,
            generator=generator,
            dtype=corrected.dtype,
            device=corrected.device,
        )
        initial = float(torch.linalg.vector_norm(vector))
        vector = transpose_new(apply_new(vector))
        final = float(torch.linalg.vector_norm(vector))
        best = max(best, final)
        lower = max(lower, math.sqrt(final / initial))
    calibration = c_delta(DELTA, PROBES)
    kappa = math.sqrt(best / calibration)
    close(best, stored["probe"]["Y"])
    close(calibration, stored["probe"]["c_delta"], 2e-14)
    close(kappa, stored["relinearized_kappa"])
    close(lower, stored["probe"]["operator_norm_lower_estimate"])
    if seed != int(stored["probe"]["rng_seed"]):
        raise AssertionError("probe seed mismatch")

    secant_row = next(
        row for row in amplified["rows"] if float(row["amplification"]) == 4096.0
    )
    injection = float(secant_row["analytic_secant_discrepancy_upper"]) + float(
        outward["secant_forcing_norm_upper"]
    )
    forcing = kappa * injection
    power_row = certificate["power_rows"][0]
    drift = float(power_row["maximum_optimizer_derivative_drift_upper"])
    domain = float(power_row["one_shot_closure"]["domain_radius"])
    correction_max = float(torch.linalg.vector_norm(response, dim=1).max())
    discriminant = 1.0 - 2.0 * kappa * drift * forcing
    if discriminant < 0.0:
        raise AssertionError("independent corrected-path closure failed")
    radius = 2.0 * forcing / (1.0 + math.sqrt(discriminant))
    if correction_max + radius > domain:
        raise AssertionError("independent corrected-path domain failed")
    close(injection, stored["total_injection_forcing_upper"])
    close(forcing, stored["corrected_defect_response_bound"])
    close(discriminant, stored["closure"]["discriminant"])
    close(radius, stored["closure"]["remainder_radius"])

    required = int(certificate["required_correct"])
    guarantee: list[float] = []
    exclusion: list[float] = []
    for step in range(HORIZON + 1):
        raw = gate_slacks(
            logits(corrected[step, :dimension], cert_pairs, template, spec),
            cert_labels,
            required,
        )
        margin = 0.0
        if step:
            output = certificate["output_rows"][step - 1]
            first = float(output["trace"]["rows"][0]["operator_norm_upper_bound"])
            second = float(output["block_second"])
            shift = float(torch.linalg.vector_norm(response[step, :dimension]))
            margin = math.sqrt(2.0) * (
                (first + second * shift) * radius + 0.5 * second * radius * radius
            )
        guarantee.append(raw[0] - margin)
        exclusion.append(raw[1] - margin)
    lower_index = first_persistent([value <= 0.0 for value in exclusion])
    upper_index = first_persistent([value > 0.0 for value in guarantee])
    bracket = None if lower_index is None or upper_index is None or lower_index > upper_index else [lower_index, upper_index]
    if bracket != stored["bracket"] or bracket != [28, 28]:
        raise AssertionError(f"bracket mismatch: {bracket} != {stored['bracket']}")
    if int(stored["outcome_files_read"]) != 0:
        raise AssertionError("result reports outcome access")

    payload = {
        "status": "INDEPENDENT FOUR-PROBE RELINEARIZED SECANT AUDIT PASSED",
        "result_sha256": sha256(RESULT),
        "protocol_sha256": sha256(PROTOCOL),
        "probe_seed": seed,
        "probes_recomputed": PROBES,
        "gram_power": 1,
        "Y": best,
        "c_delta": calibration,
        "kappa_upper": kappa,
        "forcing_upper": injection,
        "closure_radius": radius,
        "bracket": bracket,
        "outcome_files_read": 0,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

