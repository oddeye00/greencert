#!/usr/bin/env python3
"""Seal the same-code online-versus-full-q8 benchmark comparison."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ONLINE = RESULTS / (
    "transformer_v3_online_policy_seed_366_gate_1_anchor_1120_matched-online.json"
)
FULL = RESULTS / (
    "transformer_v3_online_policy_seed_366_gate_1_anchor_1120_matched-full-q8.json"
)
OUTPUT = RESULTS / "transformer_v3_online_policy_matched_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"matched audit already exists: {OUTPUT}")
    online = json.loads(ONLINE.read_text(encoding="utf-8"))
    full = json.loads(FULL.read_text(encoding="utf-8"))
    invariants = (
        "benchmark_script_sha256",
        "candidate",
        "certificate_sha256",
        "certificate_seal_sha256",
        "frozen_bracket",
        "decision",
        "allocated_live_probe_bytes",
    )
    for key in invariants:
        if online[key] != full[key]:
            raise AssertionError(f"matched benchmark differs at {key}")
    if online["execution_mode"] != "online stopping":
        raise AssertionError("online record has the wrong execution mode")
    if full["execution_mode"] != "full-q8 control":
        raise AssertionError("full record has the wrong execution mode")
    if online["maximum_relative_trace_deviation"] != 0.0:
        raise AssertionError("online trace differs from the frozen trace")
    if full["maximum_relative_trace_deviation"] != 0.0:
        raise AssertionError("full trace differs from the frozen trace")
    if (online["queried_output_power"], online["queried_green_power"]) != (1, 1):
        raise AssertionError("online policy did not stop at q=(1,1)")
    if (full["queried_output_power"], full["queried_green_power"]) != (8, 8):
        raise AssertionError("full control did not exhaust q=(8,8)")

    operator_speedup = (
        float(full["online_measured_operator_seconds"])
        / float(online["online_measured_operator_seconds"])
    )
    end_to_end_speedup = (
        float(full["timings_seconds"]["end_to_end"])
        / float(online["timings_seconds"]["end_to_end"])
    )
    if not math.isclose(operator_speedup, 8.0, rel_tol=0.02):
        raise AssertionError("matched operator speedup is inconsistent with 8x work")
    payload = {
        "status": "POST-SEAL MATCHED ONLINE-STOPPING BENCHMARK",
        "scope": "one sealed horizon-26 Transformer candidate; implementation evidence",
        "prospective_issuance_changed": False,
        "candidate": online["candidate"],
        "certificate_sha256": online["certificate_sha256"],
        "certificate_seal_sha256": online["certificate_seal_sha256"],
        "benchmark_script_sha256": online["benchmark_script_sha256"],
        "online_record": str(ONLINE.relative_to(ROOT)),
        "online_record_sha256": sha256(ONLINE),
        "full_q8_record": str(FULL.relative_to(ROOT)),
        "full_q8_record_sha256": sha256(FULL),
        "identical_decision": online["decision"],
        "maximum_relative_trace_deviation": 0.0,
        "online_powers": [
            online["queried_output_power"],
            online["queried_green_power"],
        ],
        "full_q8_powers": [full["queried_output_power"], full["queried_green_power"]],
        "logical_application_speedup": (
            float(full["logical_gram_applications"])
            / float(online["logical_gram_applications"])
        ),
        "measured_operator_time_speedup": operator_speedup,
        "measured_end_to_end_speedup": end_to_end_speedup,
        "online_operator_seconds": online["online_measured_operator_seconds"],
        "full_q8_operator_seconds": full["online_measured_operator_seconds"],
        "online_end_to_end_seconds": online["timings_seconds"]["end_to_end"],
        "full_q8_end_to_end_seconds": full["timings_seconds"]["end_to_end"],
        "online_live_probe_gib": online["allocated_live_probe_gib"],
        "interpretation": (
            "The same executable and probe streams reproduce the sealed bracket. "
            "The comparison establishes realized stopping speedup on this candidate, "
            "not a population-wide wall-time estimate."
        ),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
