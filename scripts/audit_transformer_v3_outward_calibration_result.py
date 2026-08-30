#!/usr/bin/env python3
"""Independent integrity audit of the outward calibration replay record."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from flint import arb, ctx


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOURCE = RESULTS / "transformer_v3_outward_calibration_postseal_audit.json"
OUTPUT = RESULTS / "transformer_v3_outward_calibration_independent_audit.json"
GATE_INDEX = {0.7: 0, 0.8: 1, 0.9: 2}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def exact_arb(value: float) -> arb:
    numerator, denominator = value.as_integer_ratio()
    return arb(numerator) / arb(denominator)


def main() -> None:
    source = load(SOURCE)
    if source["issued_certificates"] != 11 or source["retained_same_bracket"] != 11:
        raise AssertionError("unexpected issuance/retention counts")
    if len(source["rows"]) != 11:
        raise AssertionError("candidate rows are incomplete")
    if not (1.0 <= source["maximum_probe_bound_inflation_factor"] < 1.0 + 2e-15):
        raise AssertionError("unexpected scalar inflation scale")
    if source["minimum_replayed_logic_slack"] <= 0.0:
        raise AssertionError("non-strict replayed certificate")

    old_precision = ctx.prec
    ctx.prec = int(source["precision_bits"])
    try:
        for calibration in source["calibrations"]:
            delta = float.fromhex(calibration["delta_exact_binary64"])
            probes = int(calibration["probes"])
            exact = arb(2).sqrt() * exact_arb(delta).root(probes).erfinv()
            lower = float(np.nextafter(float(exact.lower()), -math.inf))
            upper = float(np.nextafter(float(exact.upper()), math.inf))
            if lower != float(calibration["outward_lower"]):
                raise AssertionError("calibration lower endpoint mismatch")
            if upper != float(calibration["outward_upper"]):
                raise AssertionError("calibration upper endpoint mismatch")
            if float(calibration["calibration_used_for_replay"]) > lower:
                raise AssertionError("replay calibration is not conservative")

        distinct = set()
        for row in source["rows"]:
            candidate = row["candidate"]
            gate = GATE_INDEX[float(candidate["threshold"])]
            certificate_path = RESULTS / (
                f"transformer_v3_certificate_seed_{candidate['seed']}_"
                f"gate_{gate}_anchor_{candidate['anchor']}.json"
            )
            certificate = load(certificate_path)
            if sha256(certificate_path) != row["certificate_sha256"]:
                raise AssertionError("certificate hash mismatch")
            if row["sealed_bracket"] != certificate["certified_bracket"]:
                raise AssertionError("sealed bracket mismatch")
            if row["outward_calibration_bracket"] != certificate["certified_bracket"]:
                raise AssertionError("outward replay bracket mismatch")
            if not row["same_bracket"] or float(row["certificate_logic_slack"]) <= 0.0:
                raise AssertionError("outward replay did not issue strictly")
            distinct.add(int(candidate["seed"]))
    finally:
        ctx.prec = old_precision

    payload = {
        "status": "independent outward-calibration integrity audit passed",
        "source_sha256": sha256(SOURCE),
        "issued_and_retained": 11,
        "distinct_seeds": len(distinct),
        "stored_calibration_is_conservative": all(
            bool(row["stored_is_no_larger_than_exact_quantile"])
            for row in source["calibrations"]
        ),
        "minimum_logic_slack": source["minimum_replayed_logic_slack"],
        "scope_preserved": (
            "conditional on stored binary64 Y; no HVP/VJP or neural-jet "
            "outward-enclosure claim"
        ),
    }
    if OUTPUT.exists():
        if load(OUTPUT) != payload:
            raise AssertionError("existing independent audit differs from replay")
    else:
        OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "sha256": sha256(OUTPUT), **payload}, indent=2))


if __name__ == "__main__":
    main()
