#!/usr/bin/env python3
"""Fast anchor-state regression for strict Transformer ball inflation."""
from __future__ import annotations

import json
from pathlib import Path

from strict_transformer_block_envelope import strict_ball_valid_envelope
from transformer_block_envelope import ball_valid_envelope
from transformer_certificate_protocol import Candidate
from transformer_v3_certificate import load_candidate


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "results" / "transformer_v3_certificate_seed_366_gate_1_anchor_1120.json"


def main() -> None:
    payload = json.loads(RECORD.read_text(encoding="utf-8"))
    raw = payload["candidate"]
    candidate = Candidate(int(raw["seed"]), float(raw["threshold"]), int(raw["anchor"]))
    config, _, spec, _, parameter, _ = load_candidate(candidate)
    epsilon = float(payload["outer_domain_radius"])
    baseline = ball_valid_envelope(
        parameter, spec, config, epsilon=epsilon, exact_values=True, sphere=True
    )
    strict = strict_ball_valid_envelope(
        parameter, spec, config, epsilon=epsilon, exact_values=True, sphere=True
    )
    assert strict["strict_binary64_postfixed"]
    assert all(
        float(strict["stage_first"][name]) * epsilon
        <= float(strict["inflation"][name])
        for name in strict["stage_first"]
    )
    for key in ("first", "second", "third"):
        assert float(strict[key]) >= float(baseline[key])
    print("strict Transformer block-envelope regression passed")


if __name__ == "__main__":
    main()
