#!/usr/bin/env python3
"""Regression gate for the independently aggregated hardening evidence."""
from __future__ import annotations

from audit_postseal_hardening import audit_batched, audit_digits, audit_modern


def main() -> None:
    digits = audit_digits()
    batched = audit_batched()
    modern = audit_modern()
    assert digits["issued"] == digits["covered"] == digits["brackets_identical"] == 7
    assert digits["signed_only_issued_and_covered"] == 1
    assert batched["complete_replays"] == batched["exact_replay_matches"] == 2
    assert batched["median_end_to_end_speedup"] > 2.5
    assert batched["million_parameter_batched_hours"] < 12.0
    assert modern["parameters"] >= 100_000
    assert modern["depth"] == 2 and modern["normalization"] == "layernorm"
    assert modern["optimizer"] == "AdamW"
    assert modern["optimizer_adjoint_relative_error"] < 1e-12
    assert modern["green_adjoint_relative_error"] < 1e-12
    print(
        "PASS: 7/7 digits outward brackets, two exact accelerated replays, "
        "and 100k-parameter LayerNorm/AdamW products independently verify."
    )


if __name__ == "__main__":
    main()
