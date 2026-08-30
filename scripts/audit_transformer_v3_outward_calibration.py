#!/usr/bin/env python3
"""Outward Arb audit of Transformer folded-normal calibration and scalar roots.

This post-seal audit treats each stored binary64 probe norm Y as an exact dyadic
input, outward-encloses c_delta and (Y/c_delta)^(1/(2q)), and replays every
issued v3 decision.  It does not enclose the HVP/VJP that produced Y.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import flint
import numpy as np
from flint import arb, ctx

from audit_transformer_v3_power_grid import pair_result, raw_anchor_slacks
from transformer_v3_certificate import frozen_candidates, output_path, safe_json


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "transformer_v3_outward_calibration_postseal_audit.json"
PRECISION_BITS = 256


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def exact_arb(value: float) -> arb:
    numerator, denominator = float(value).as_integer_ratio()
    return arb(numerator) / arb(denominator)


def outward_float_lower(value: arb) -> float:
    return float(np.nextafter(float(value.lower()), -math.inf))


def outward_float_upper(value: arb) -> float:
    return float(np.nextafter(float(value.upper()), math.inf))


def calibration_interval(delta: float, probes: int) -> tuple[arb, float, float]:
    # Phi^{-1}((1+delta^(1/m))/2) = sqrt(2) erfinv(delta^(1/m)).
    value = arb(2).sqrt() * exact_arb(delta).root(probes).erfinv()
    lower = outward_float_lower(value)
    upper = outward_float_upper(value)
    if not (0.0 < lower <= upper):
        raise AssertionError("invalid outward calibration interval")
    return value, lower, upper


def outward_probe_bound(y_value: float, calibration: arb, power: int) -> float:
    if y_value <= 0.0:
        return 0.0
    enclosed = (exact_arb(y_value) / calibration).root(2 * power)
    return outward_float_upper(enclosed)


def harden_trace(trace: dict, calibration: arb) -> tuple[float, float, float, int]:
    maximum_inflation = 1.0
    maximum_stored_deficit = 0.0
    maximum_stored_surplus = 0.0
    repaired_rows = 0
    for row in trace["rows"]:
        outward = outward_probe_bound(float(row["Y"]), calibration, int(row["power"]))
        stored = float(row["operator_norm_upper_bound"])
        hardened = max(stored, outward)
        row["operator_norm_upper_bound"] = hardened
        if stored > 0.0:
            maximum_inflation = max(maximum_inflation, hardened / stored)
            maximum_stored_deficit = max(maximum_stored_deficit, outward - stored)
            maximum_stored_surplus = max(maximum_stored_surplus, stored - outward)
            repaired_rows += int(outward > stored)
    return (
        maximum_inflation,
        maximum_stored_deficit,
        maximum_stored_surplus,
        repaired_rows,
    )


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite post-seal audit: {OUTPUT}")
    old_precision = ctx.prec
    ctx.prec = PRECISION_BITS
    try:
        candidates, _, _ = frozen_candidates()
        rows = []
        issued = 0
        retained = 0
        maximum_inflation = 1.0
        maximum_deficit = 0.0
        maximum_surplus = 0.0
        repaired_rows = 0
        calibration_records: dict[tuple[float, int], dict] = {}

        for candidate in candidates:
            certificate_path = output_path(candidate)
            certificate = safe_json(certificate_path)
            if not bool(certificate.get("certificate_issued")):
                continue
            issued += 1
            protocol = certificate["protocol"]["probe_config"]
            delta = float(protocol["delta"])
            probes = int(protocol["probes"])
            key = (delta, probes)
            if key not in calibration_records:
                interval, lower, upper = calibration_interval(delta, probes)
                stored = float(certificate["green_trace"]["rows"][0]["c_delta"])
                conservative = min(stored, lower)
                calibration_records[key] = {
                    "quantile_interval": interval,
                    "conservative_calibration": exact_arb(conservative),
                    "delta_exact_binary64": delta.hex(),
                    "probes": probes,
                    "stored_c_delta": stored,
                    "outward_lower": lower,
                    "outward_upper": upper,
                    "stored_is_no_larger_than_exact_quantile": stored <= lower,
                    "stored_offset_from_outward_lower_ulps": (
                        (stored - lower) / float(np.spacing(stored))
                    ),
                    "calibration_used_for_replay": conservative,
                    "relative_interval_width": (upper - lower) / stored,
                }
            calibration = calibration_records[key]["conservative_calibration"]

            hardened = copy.deepcopy(certificate)
            local_inflation, local_deficit, local_surplus, local_repairs = harden_trace(
                hardened["green_trace"], calibration
            )
            maximum_inflation = max(maximum_inflation, local_inflation)
            maximum_deficit = max(maximum_deficit, local_deficit)
            maximum_surplus = max(maximum_surplus, local_surplus)
            repaired_rows += local_repairs
            for output_row in hardened["output_rows"]:
                (
                    local_inflation,
                    local_deficit,
                    local_surplus,
                    local_repairs,
                ) = harden_trace(
                    output_row["trace"], calibration
                )
                maximum_inflation = max(maximum_inflation, local_inflation)
                maximum_deficit = max(maximum_deficit, local_deficit)
                maximum_surplus = max(maximum_surplus, local_surplus)
                repaired_rows += local_repairs

            config, raw_zero = raw_anchor_slacks(
                candidate, int(certificate["required_correct"])
            )
            power = int(certificate["earliest_issuing_power"])
            replay = pair_result(
                q_output=power,
                q_green=power,
                certificate=hardened,
                config=config,
                raw_zero=raw_zero,
            )
            same = (
                replay["certificate_issued"]
                and replay["certified_bracket"] == certificate["certified_bracket"]
            )
            retained += int(same)
            rows.append(
                {
                    "candidate": candidate.__dict__,
                    "certificate_sha256": sha256(certificate_path),
                    "power": power,
                    "sealed_bracket": certificate["certified_bracket"],
                    "outward_calibration_bracket": replay["certified_bracket"],
                    "same_bracket": same,
                    "certificate_logic_slack": replay["certificate_logic_slack"],
                    "total_pointwise_radius": replay["total_pointwise_radius"],
                }
            )

        if retained != issued:
            raise AssertionError(f"outward calibration retained {retained}/{issued}")
        serializable_calibrations = []
        for record in calibration_records.values():
            serializable_calibrations.append(
                {
                    key: value
                    for key, value in record.items()
                    if key not in {"quantile_interval", "conservative_calibration"}
                }
            )
        payload = {
            "status": "POST-SEAL OUTWARD CALIBRATION AUDIT; PROSPECTIVE COUNTS UNCHANGED",
            "scope": (
                "Arb encloses the folded-normal quantile and every scalar probe-root "
                "evaluation, conditional on stored binary64 Y. HVP/VJP, norm "
                "accumulation, neural jets, and margins remain non-outward float64."
            ),
            "arb_version": flint.__version__,
            "precision_bits": PRECISION_BITS,
            "calibrations": serializable_calibrations,
            "issued_certificates": issued,
            "retained_same_bracket": retained,
            "maximum_probe_bound_inflation_factor": maximum_inflation,
            "maximum_absolute_stored_bound_deficit": maximum_deficit,
            "maximum_absolute_stored_bound_surplus": maximum_surplus,
            "probe_rows_repaired_upward": repaired_rows,
            "minimum_replayed_logic_slack": min(
                float(row["certificate_logic_slack"]) for row in rows
            ),
            "rows": rows,
        }
        OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(OUTPUT), "sha256": sha256(OUTPUT), **payload}, indent=2))
    finally:
        ctx.prec = old_precision


if __name__ == "__main__":
    main()
