#!/usr/bin/env python3
"""Post-seal replay of predictable adaptive witness acquisition on v3."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import median

from adaptive_witness_policy import WitnessQuery, acquire_witnesses


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "transformer_v3_adaptive_witness_postseal_audit.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def slacks(row: dict, *, power: int, radius: float) -> tuple[float, float]:
    upper = float(row["trace"]["rows"][power - 1]["operator_norm_upper_bound"])
    margin = math.sqrt(2.0) * (
        upper * radius + 0.5 * float(row["block_second"]) * radius * radius
    )
    return (
        float(row["raw_guarantee_slack"]) - margin,
        float(row["raw_exclusion_slack"]) - margin,
    )


def audit(path: Path) -> dict | None:
    payload = load(path)
    if not bool(payload.get("certificate_issued")):
        return None
    power = int(payload["earliest_issuing_power"])
    radius = float(payload["certified_total_pointwise_radius"])
    bracket = tuple(map(int, payload["certified_bracket"]))
    if bracket[0] != bracket[1]:
        raise AssertionError("v3 adaptive audit expects singleton brackets")
    event = bracket[0]
    horizon = int(payload["protocol"]["horizon"])
    persistence = int(payload["protocol"]["persistence"])
    rows = {int(row["step"]): row for row in payload["output_rows"]}
    evaluated = {
        step: slacks(row, power=power, radius=radius)
        for step, row in rows.items()
    }
    raw_exclusions = {
        step: float(row["raw_exclusion_slack"])
        for step, row in rows.items()
    }

    def query(step: int) -> WitnessQuery:
        guarantee, exclusion = evaluated[step]
        return WitnessQuery(step, guarantee > 0.0, exclusion > 0.0)

    result = acquire_witnesses(
        event=event,
        persistence=persistence,
        horizon=horizon,
        raw_exclusion_slacks=raw_exclusions,
        query=query,
        exact_failures={0},
    )
    if not result.issued:
        raise AssertionError(f"adaptive witness policy failed on {path.name}: {result.reason}")
    if set(result.success_times) & set(result.failure_witnesses):
        raise AssertionError("success and failure witnesses overlap")

    candidate = payload["candidate"]
    queried_nonanchor_failures = set(result.failure_witnesses) - {0}
    return {
        "record": str(path.relative_to(ROOT)),
        "seed": int(candidate["seed"]),
        "threshold": float(candidate["threshold"]),
        "anchor": int(candidate["anchor"]),
        "horizon": horizon,
        "event": event,
        "issuing_power": power,
        "query_order": list(result.query_order),
        "success_queries": len(result.success_times),
        "failure_queries": len(result.query_order) - len(result.success_times),
        "certified_nonanchor_failure_witnesses": len(queried_nonanchor_failures),
        "total_output_queries": len(result.query_order),
        "full_output_queries": horizon,
        "query_reduction": horizon / len(result.query_order),
        "bracket_reconstructed": True,
    }


def main() -> None:
    records = []
    pattern = "transformer_v3_certificate_seed_*_gate_*_anchor_*.json"
    for path in sorted(RESULTS.glob(pattern)):
        row = audit(path)
        if row is not None:
            records.append(row)
    if len(records) != 11:
        raise AssertionError(f"expected 11 issued records, found {len(records)}")
    reductions = [row["query_reduction"] for row in records]
    payload = {
        "status": "post-seal predictable-policy replay; no prospective count changed",
        "policy": (
            "query the fixed predicted success window; then choose the unqueried "
            "positive-raw-slack time covering the most uncovered earlier windows, "
            "breaking ties by raw exclusion slack and latest time"
        ),
        "summary": {
            "issued_records": len(records),
            "all_brackets_reconstructed": True,
            "median_query_reduction": median(reductions),
            "minimum_query_reduction": min(reductions),
            "maximum_query_reduction": max(reductions),
            "total_full_queries": sum(row["full_output_queries"] for row in records),
            "total_adaptive_queries": sum(row["total_output_queries"] for row in records),
        },
        "interpretation": (
            "Unlike the minimum-witness accounting audit, this policy discovers "
            "its witnesses from centerline slacks and prior query results.  A future "
            "prospective implementation must use fresh domain-separated blocks and "
            "predictable failure budgets for adaptively requested operators."
        ),
        "records": records,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest().upper()
    print(json.dumps({"output": str(OUTPUT), "sha256": digest, **payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()

