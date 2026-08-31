#!/usr/bin/env python3
"""Matched direct-continuation control for the optimized Transformer certificate."""
from __future__ import annotations

import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from transformer_certificate_protocol import Candidate
from transformer_modal_forecast import optimizer_map
from transformer_v3_certificate import load_candidate


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "transformer_seed_366_matched_continuation.json"
CANDIDATE = Candidate(366, 0.8, 1120)
SHORT_HORIZON = 26
FULL_HORIZON = 300
ORDERS = ((SHORT_HORIZON, FULL_HORIZON), (FULL_HORIZON, SHORT_HORIZON), (SHORT_HORIZON, FULL_HORIZON))


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes(order="C")
    ).hexdigest().upper()


def main() -> None:
    config, template, spec, data, parameter, velocity = load_candidate(CANDIDATE)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    train_pairs, train_labels, _, _, _, _ = data
    anchor = torch.cat((parameter, velocity))

    def continue_for(horizon: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        state = anchor.clone()
        short_endpoint = None
        started = time.perf_counter()
        for step in range(1, horizon + 1):
            state = optimizer_map(
                state,
                train_pairs,
                train_labels,
                template,
                spec,
                config,
            )
            if step == SHORT_HORIZON:
                short_endpoint = state.clone()
        elapsed = time.perf_counter() - started
        if short_endpoint is None:
            raise RuntimeError("short endpoint was not reached")
        return state, short_endpoint, elapsed

    rows = []
    short_hashes = []
    for replicate, order in enumerate(ORDERS, start=1):
        for horizon in order:
            endpoint, short_endpoint, elapsed = continue_for(horizon)
            short_hash = tensor_sha256(short_endpoint)
            short_hashes.append(short_hash)
            rows.append(
                {
                    "replicate": replicate,
                    "order": list(order),
                    "horizon": horizon,
                    "elapsed_seconds": elapsed,
                    "endpoint_sha256": tensor_sha256(endpoint),
                    "short_endpoint_sha256": short_hash,
                }
            )
    if len(set(short_hashes)) != 1:
        raise RuntimeError("matched direct continuations disagree at step 26")
    short_times = [row["elapsed_seconds"] for row in rows if row["horizon"] == SHORT_HORIZON]
    full_times = [row["elapsed_seconds"] for row in rows if row["horizon"] == FULL_HORIZON]
    payload = {
        "status": "matched direct-continuation benchmark complete",
        "evidence_boundary": (
            "Post-release deterministic timing control; no event outcome or "
            "certificate decision is used."
        ),
        "candidate": CANDIDATE.__dict__,
        "replicates": len(ORDERS),
        "short_horizon": SHORT_HORIZON,
        "full_horizon": FULL_HORIZON,
        "short_seconds": short_times,
        "full_seconds": full_times,
        "median_short_seconds": statistics.median(short_times),
        "median_full_seconds": statistics.median(full_times),
        "common_short_endpoint_sha256": short_hashes[0],
        "rows": rows,
        "outcome_files_read": 0,
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
