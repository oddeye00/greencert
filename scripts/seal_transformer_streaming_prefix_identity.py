#!/usr/bin/env python3
"""Seal a bitwise identity bridge from the frozen path to its streamed prefix."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

from streaming_variational_centerline import build_streaming_transformer_centerline
from transformer_certificate_protocol import Candidate
from transformer_green_development_audit import build_frozen_centerline
from transformer_v3_certificate import load_candidate, output_path, safe_json


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "transformer_seed_366_streaming_prefix_identity.json"
CANDIDATE = Candidate(366, 0.8, 1120)
HORIZON = 26


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes(order="C")
    ).hexdigest().upper()


def main() -> None:
    certificate_path = output_path(CANDIDATE)
    certificate = safe_json(certificate_path)
    if int(certificate["protocol"]["horizon"]) != HORIZON:
        raise RuntimeError("sealed certificate horizon changed")
    config, template, spec, data, parameter, velocity = load_candidate(CANDIDATE)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    train_pairs, train_labels, _, _, _, _ = data

    full_started = time.perf_counter()
    full = build_frozen_centerline(
        config,
        template,
        spec,
        train_pairs,
        train_labels,
        parameter,
        velocity,
    )
    full_seconds = time.perf_counter() - full_started
    stream_started = time.perf_counter()
    streamed = build_streaming_transformer_centerline(
        config,
        template,
        spec,
        train_pairs,
        train_labels,
        parameter,
        velocity,
        maximum_horizon=HORIZON,
    )
    stream_seconds = time.perf_counter() - stream_started

    full_center_prefix = full["center"][: HORIZON + 1]
    full_scaled_prefix = full["scaled_center"][: HORIZON + 1]
    if not torch.equal(streamed["center"], full_center_prefix):
        raise RuntimeError("streamed state prefix is not bitwise identical")
    if not torch.equal(streamed["scaled_center"], full_scaled_prefix):
        raise RuntimeError("streamed scaled prefix is not bitwise identical")
    if full["centerline_sha256"] != certificate["centerline_sha256"]:
        raise RuntimeError("rebuilt full centerline differs from sealed certificate")

    payload = {
        "status": "streaming-prefix identity bridge sealed",
        "evidence_boundary": (
            "Post-release deterministic implementation identity; no future "
            "trajectory or event outcome is read."
        ),
        "candidate": CANDIDATE.__dict__,
        "horizon": HORIZON,
        "certificate": str(certificate_path.relative_to(ROOT)).replace("\\", "/"),
        "certificate_sha256": sha256(certificate_path),
        "full_centerline_sha256": full["centerline_sha256"],
        "state_prefix_sha256": tensor_sha256(full_center_prefix),
        "scaled_prefix_sha256": tensor_sha256(full_scaled_prefix),
        "streamed_state_prefix_sha256": tensor_sha256(streamed["center"]),
        "streamed_scaled_prefix_sha256": tensor_sha256(streamed["scaled_center"]),
        "bitwise_state_prefix_equal": True,
        "bitwise_scaled_prefix_equal": True,
        "full_300_seconds": full_seconds,
        "streamed_prefix_seconds": stream_seconds,
        "measured_speedup": full_seconds / stream_seconds,
        "frozen_builder_sha256": sha256(
            ROOT / "scripts" / "transformer_green_development_audit.py"
        ),
        "streaming_builder_sha256": sha256(
            ROOT / "scripts" / "streaming_variational_centerline.py"
        ),
        "outcome_files_read": 0,
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
