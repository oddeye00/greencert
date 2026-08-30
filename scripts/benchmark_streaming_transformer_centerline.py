#!/usr/bin/env python3
"""Matched full-300 versus prefix-local Transformer centerline benchmark."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from streaming_variational_centerline import build_streaming_transformer_centerline
from transformer_certificate_protocol import Candidate, PERSISTENCE
from transformer_green_development_audit import build_frozen_centerline
from transformer_hvp_grokking import logits
from transformer_v3_certificate import load_candidate


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "transformer_streaming_centerline_benchmark.json"
CASES = (
    (Candidate(366, 0.8, 1120), 26, "full_first"),
    (Candidate(360, 0.7, 3480), 131, "stream_first"),
    (Candidate(370, 0.7, 2280), 299, "full_first"),
)


def first_persistent(values: list[bool]) -> int | None:
    for start in range(max(0, len(values) - PERSISTENCE + 1)):
        if all(values[start : start + PERSISTENCE]):
            return start
    return None


def main() -> None:
    rows = []
    for candidate, horizon, order in CASES:
        config, template, spec, data, parameter, velocity = load_candidate(candidate)
        torch.set_num_threads(config.threads)
        torch.use_deterministic_algorithms(True)
        train_pairs, train_labels, _, _, cert_pairs, cert_labels = data

        def run_full():
            started = time.perf_counter()
            value = build_frozen_centerline(
                config,
                template,
                spec,
                train_pairs,
                train_labels,
                parameter,
                velocity,
            )
            return value, time.perf_counter() - started

        def run_stream():
            started = time.perf_counter()
            value = build_streaming_transformer_centerline(
                config,
                template,
                spec,
                train_pairs,
                train_labels,
                parameter,
                velocity,
                maximum_horizon=horizon,
            )
            return value, time.perf_counter() - started

        if order == "full_first":
            full, full_seconds = run_full()
            streamed, stream_seconds = run_stream()
        else:
            streamed, stream_seconds = run_stream()
            full, full_seconds = run_full()

        full_prefix = full["center"][: horizon + 1]
        full_scaled_prefix = full["scaled_center"][: horizon + 1]
        bitwise_center = bool(torch.equal(streamed["center"], full_prefix))
        bitwise_scaled = bool(
            torch.equal(streamed["scaled_center"], full_scaled_prefix)
        )
        if not bitwise_center or not bitwise_scaled:
            raise RuntimeError(f"streaming prefix mismatch for {candidate}")
        required = int(math.ceil(candidate.threshold * len(cert_pairs)))
        counts = [
            int(
                (
                    logits(state[: parameter.numel()], cert_pairs, template, spec).argmax(1)
                    == cert_labels
                ).sum()
            )
            for state in streamed["center"]
        ]
        event = first_persistent([value >= required for value in counts])
        if event is None or event + PERSISTENCE - 1 != horizon:
            raise RuntimeError(f"streamed predicted event changed for {candidate}")
        state_bytes = int(streamed["center"][0].numel() * streamed["center"][0].element_size())
        rows.append(
            {
                "candidate": candidate.__dict__,
                "horizon": horizon,
                "order": order,
                "bitwise_center_prefix_equal": bitwise_center,
                "bitwise_scaled_prefix_equal": bitwise_scaled,
                "predicted_persistent_event": event,
                "full_300_seconds": full_seconds,
                "streamed_prefix_seconds": stream_seconds,
                "speedup": full_seconds / stream_seconds,
                "full_intermediate_centerline_bytes_estimate": 5 * 301 * state_bytes,
                "streaming_live_plus_final_bytes_estimate": (
                    (horizon + 1 + 2 * 4 + 3) * state_bytes
                ),
                "estimated_centerline_memory_reduction": (
                    (5 * 301)
                    / (horizon + 1 + 2 * 4 + 3)
                ),
                "outcome_files_read": 0,
            }
        )
        print(json.dumps(rows[-1]), flush=True)

    payload = {
        "status": "STREAMING PREFIX-LOCAL CENTERLINE BENCHMARK COMPLETED",
        "evidence_boundary": (
            "Post-seal deterministic implementation benchmark. It reads no "
            "revealed future trajectory and changes no certificate count."
        ),
        "cases": len(rows),
        "all_bitwise_equal": all(
            row["bitwise_center_prefix_equal"]
            and row["bitwise_scaled_prefix_equal"]
            for row in rows
        ),
        "rows": rows,
        "outcome_files_read": 0,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
