#!/usr/bin/env python3
"""Frozen constants and random-operator universe for fresh Green confirmation."""
from __future__ import annotations

from probe_jacobian_bound import ProbeConfig, ProbeRegistry
from transformer_certificate_protocol import Candidate

PROTOCOL_VERSION = 3
SEEDS = tuple(range(331, 355))
THRESHOLDS = (0.70, 0.80, 0.90)
MAXIMUM_CANDIDATES_PER_GATE = 1
HORIZON = 300
SWEEPS = 4
PERSISTENCE = 25
SCAN_STEPS = 1_200
CHECKPOINT_SPACING = 40
MAX_DEFICIT = 3
PROBES = 16
POWER = 8
FAMILY_FAILURE_PROBABILITY = 1.0e-6
MASTER_NONCE = "c0b81a6cb799088f0679c5b5ad39cb25e5eac84a2bca01762bf4a8f07f529ab7"

OUTPUT_JACOBIAN = 1
GREEN_OPERATOR = 3


def maximum_operator_count() -> dict:
    candidates = len(SEEDS) * len(THRESHOLDS) * MAXIMUM_CANDIDATES_PER_GATE
    per_candidate = (HORIZON + 1) + 1
    return {
        "maximum_seeds": len(SEEDS),
        "gates_per_seed": len(THRESHOLDS),
        "maximum_candidates_per_gate": MAXIMUM_CANDIDATES_PER_GATE,
        "maximum_candidates": candidates,
        "probabilistic_screen_operators": 0,
        "output_jacobian_states_per_candidate": HORIZON + 1,
        "green_operators_per_candidate": 1,
        "operators_per_candidate": per_candidate,
        "maximum_probabilistic_operators": candidates * per_candidate,
    }


def probe_config() -> ProbeConfig:
    count = maximum_operator_count()["maximum_probabilistic_operators"]
    return ProbeConfig(PROBES, POWER, FAMILY_FAILURE_PROBABILITY / count)


def output_identity(candidate: Candidate, step: int) -> tuple[int, ...]:
    if candidate.seed not in SEEDS:
        raise ValueError("candidate seed is outside the frozen population")
    if not 0 <= step <= HORIZON:
        raise ValueError("output step is outside 0..H")
    return (
        PROTOCOL_VERSION,
        candidate.seed,
        candidate.gate_index,
        candidate.anchor,
        step,
        SWEEPS,
        OUTPUT_JACOBIAN,
    )


def green_identity(candidate: Candidate, horizon: int) -> tuple[int, ...]:
    if candidate.seed not in SEEDS:
        raise ValueError("candidate seed is outside the frozen population")
    if not 1 <= horizon <= HORIZON:
        raise ValueError("Green horizon is outside 1..H")
    return (
        PROTOCOL_VERSION,
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
    seen: set[tuple[int, int]] = set()
    rows: set[tuple[int, ...]] = set()
    for candidate in candidates:
        pair = (candidate.seed, candidate.gate_index)
        if pair in seen:
            raise ValueError(f"multiple candidates for one frozen seed/gate: {pair}")
        seen.add(pair)
        if candidate.anchor < 0 or candidate.anchor % CHECKPOINT_SPACING:
            raise ValueError(f"candidate anchor is off the frozen grid: {candidate}")
        horizon = horizons[candidate]
        rows.add(green_identity(candidate, horizon))
        for step in range(horizon + 1):
            rows.add(output_identity(candidate, step))
    return frozenset(rows)


def make_registry(
    candidates: tuple[Candidate, ...],
    horizons: dict[Candidate, int],
) -> ProbeRegistry:
    return ProbeRegistry(candidate_universe(candidates, horizons), MASTER_NONCE)
