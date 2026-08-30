#!/usr/bin/env python3
"""Post-seal accounting audit for witness-sparse Transformer event transport.

This script does not change any prospective certificate.  It reuses the sealed
v3 radii and output-norm traces to identify a minimum set of already certified
failure times that witnesses absence of an earlier persistent event.  It then
reports the output-operator workload of a future role-separated implementation:
training-output operators at transition inputs, certification-output operators
only at event-witness times, and no trigger-output operators after selection.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "transformer_v3_witness_sparse_postseal_audit.json"

TRAIN_EXAMPLES = 173
TRIGGER_EXAMPLES = 58
CERTIFICATION_EXAMPLES = 58
ALL_EXAMPLES = TRAIN_EXAMPLES + TRIGGER_EXAMPLES + CERTIFICATION_EXAMPLES


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _minimum_interval_witnesses(
    available: set[int], *, starts: range, persistence: int
) -> list[int]:
    """Greedy minimum hitting set for equal-length intervals and fixed points."""

    ordered = sorted(available)
    chosen: list[int] = []
    last: int | None = None
    for start in starts:
        end = start + persistence - 1
        if last is not None and start <= last <= end:
            continue
        candidates = [point for point in ordered if start <= point <= end]
        if not candidates:
            raise AssertionError(f"no certified failure witness for [{start}, {end}]")
        last = candidates[-1]
        chosen.append(last)
    return chosen


def _row_slacks(row: dict, *, power: int, radius: float) -> tuple[float, float]:
    upper = float(row["trace"]["rows"][power - 1]["operator_norm_upper_bound"])
    margin = math.sqrt(2.0) * (
        upper * radius + 0.5 * float(row["block_second"]) * radius * radius
    )
    return (
        float(row["raw_guarantee_slack"]) - margin,
        float(row["raw_exclusion_slack"]) - margin,
    )


def _audit_record(path: Path) -> dict | None:
    payload = _load(path)
    if not bool(payload.get("certificate_issued")):
        return None

    power = int(payload["earliest_issuing_power"])
    radius = float(payload["certified_total_pointwise_radius"])
    lower, upper = map(int, payload["certified_bracket"])
    if lower != upper:
        raise AssertionError("the frozen v3 issued brackets are expected to be singleton")
    event = lower
    horizon = int(payload["protocol"]["horizon"])
    persistence = int(payload["protocol"]["persistence"])
    if event + persistence - 1 != horizon:
        raise AssertionError("sealed horizon must end at the persistent success window")

    guarantees = {0: False}
    failures = {0: True}  # exact anchor count is below gate by the sealed selector
    for row in payload["output_rows"]:
        step = int(row["step"])
        guarantee, exclusion = _row_slacks(row, power=power, radius=radius)
        guarantees[step] = guarantee > 0.0
        failures[step] = exclusion > 0.0

    success_times = list(range(event, event + persistence))
    if not all(guarantees[step] for step in success_times):
        raise AssertionError("stored issued bracket lacks a certified success window")
    available_failures = {step for step, value in failures.items() if value}
    failure_witnesses = _minimum_interval_witnesses(
        available_failures,
        starts=range(event),
        persistence=persistence,
    )

    event_query_times = sorted((set(success_times) | set(failure_witnesses)) - {0})
    for start in range(event):
        if not any(start <= point < start + persistence for point in failure_witnesses):
            raise AssertionError("greedy witnesses do not block every earlier window")

    current_pair_work = horizon * ALL_EXAMPLES
    role_separated_pair_work = (
        max(0, horizon - 1) * TRAIN_EXAMPLES
        + len(event_query_times) * CERTIFICATION_EXAMPLES
    )
    if role_separated_pair_work > current_pair_work:
        raise AssertionError("role separation unexpectedly increases pair workload")

    candidate = payload["candidate"]
    return {
        "record": str(path.relative_to(ROOT)),
        "seed": int(candidate["seed"]),
        "threshold": float(candidate["threshold"]),
        "anchor": int(candidate["anchor"]),
        "horizon": horizon,
        "persistence": persistence,
        "event": event,
        "issuing_power": power,
        "failure_witness_times": failure_witnesses,
        "success_times": success_times,
        "event_query_times": event_query_times,
        "current_output_operator_times": horizon,
        "witness_event_operator_times": len(event_query_times),
        "event_operator_time_reduction": horizon / len(event_query_times),
        "current_example_operator_work": current_pair_work,
        "role_separated_example_operator_work": role_separated_pair_work,
        "projected_example_operator_speedup": (
            current_pair_work / role_separated_pair_work
        ),
        "projected_example_operator_reduction_fraction": (
            1.0 - role_separated_pair_work / current_pair_work
        ),
    }


def main() -> None:
    records = []
    for path in sorted(RESULTS.glob("transformer_v3_certificate_seed_*_gate_*_anchor_*.json")):
        row = _audit_record(path)
        if row is not None:
            records.append(row)
    if len(records) != 11:
        raise AssertionError(f"expected 11 issued v3 certificates, found {len(records)}")

    event_speedups = [row["event_operator_time_reduction"] for row in records]
    pair_speedups = [row["projected_example_operator_speedup"] for row in records]
    reductions = [row["projected_example_operator_reduction_fraction"] for row in records]
    payload = {
        "status": "post-seal theorem/accounting audit; no prospective count changed",
        "scope": {
            "issued_records": len(records),
            "train_examples": TRAIN_EXAMPLES,
            "trigger_examples": TRIGGER_EXAMPLES,
            "certification_examples": CERTIFICATION_EXAMPLES,
            "current_operator_uses_all_examples": ALL_EXAMPLES,
            "anchor_is_exact_certified_failure": True,
        },
        "summary": {
            "all_brackets_reconstructed_by_sparse_witnesses": True,
            "median_event_operator_time_reduction": median(event_speedups),
            "minimum_event_operator_time_reduction": min(event_speedups),
            "maximum_event_operator_time_reduction": max(event_speedups),
            "median_projected_example_operator_speedup": median(pair_speedups),
            "minimum_projected_example_operator_speedup": min(pair_speedups),
            "maximum_projected_example_operator_speedup": max(pair_speedups),
            "median_projected_example_operator_reduction_fraction": median(reductions),
        },
        "interpretation": (
            "The event logic needs only the persistent success window and a minimum "
            "interval-stabbing set of certified failures.  End-to-end realization also "
            "requires separate train-output and certification-output norm operators; "
            "the reported pair-work ratios are accounting projections, not wall time."
        ),
        "records": records,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest().upper()
    print(json.dumps({"output": str(OUTPUT), "sha256": digest, **payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
