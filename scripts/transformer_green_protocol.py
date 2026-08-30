#!/usr/bin/env python3
"""Operator accounting for the finite-window Green certificate."""
from __future__ import annotations

from probe_jacobian_bound import ProbeConfig, ProbeRegistry
from transformer_certificate_protocol import (
    FAMILY_FAILURE_PROBABILITY,
    HORIZON,
    MAXIMUM_CANDIDATES_PER_GATE,
    MAXIMUM_SEEDS,
    POWER,
    PROBES,
    SWEEPS,
    THRESHOLDS,
    Candidate,
)

GREEN_PROTOCOL_VERSION = 2
GREEN_OPERATOR = 3
OUTPUT_JACOBIAN = 1


def maximum_operator_count() -> dict:
    candidates = MAXIMUM_SEEDS * len(THRESHOLDS) * MAXIMUM_CANDIDATES_PER_GATE
    per_candidate = (HORIZON + 1) + 1
    return {
        "maximum_candidates": candidates,
        "output_jacobian_states_per_candidate": HORIZON + 1,
        "green_operators_per_candidate": 1,
        "operators_per_candidate": per_candidate,
        "maximum_probabilistic_operators": candidates * per_candidate,
    }


def probe_config() -> ProbeConfig:
    count = maximum_operator_count()["maximum_probabilistic_operators"]
    return ProbeConfig(PROBES, POWER, FAMILY_FAILURE_PROBABILITY / count)


def output_identity(candidate: Candidate, step: int) -> tuple[int, ...]:
    if not 0 <= step <= HORIZON:
        raise ValueError("output step is outside the maximum window")
    return (
        GREEN_PROTOCOL_VERSION,
        candidate.seed,
        candidate.gate_index,
        candidate.anchor,
        step,
        SWEEPS,
        OUTPUT_JACOBIAN,
    )


def green_identity(candidate: Candidate, horizon: int) -> tuple[int, ...]:
    if not 1 <= horizon <= HORIZON:
        raise ValueError("Green horizon is outside 1..H")
    return (
        GREEN_PROTOCOL_VERSION,
        candidate.seed,
        candidate.gate_index,
        candidate.anchor,
        horizon,
        SWEEPS,
        GREEN_OPERATOR,
    )


def candidate_universe(
    candidates: tuple[Candidate, ...], horizons: dict[Candidate, int]
) -> frozenset[tuple[int, ...]]:
    rows = set()
    for candidate in candidates:
        horizon = horizons[candidate]
        rows.add(green_identity(candidate, horizon))
        for step in range(horizon + 1):
            rows.add(output_identity(candidate, step))
    return frozenset(rows)


def make_registry(
    candidates: tuple[Candidate, ...],
    horizons: dict[Candidate, int],
    master_nonce: str,
) -> ProbeRegistry:
    return ProbeRegistry(candidate_universe(candidates, horizons), master_nonce)
