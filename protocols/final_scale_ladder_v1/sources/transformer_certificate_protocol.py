#!/usr/bin/env python3
"""Frozen-shape constants and operator identities for Transformer certification.

Candidate selection is a deterministic phase that never calls a probabilistic
operator.  After the candidate file is sealed, its coordinates instantiate a
subset of this maximum family.  The uniform failure budget is nevertheless
divided by the maximum permitted family, not by the realised issuance count.
"""
from __future__ import annotations

from dataclasses import dataclass

from probe_jacobian_bound import ProbeConfig, ProbeRegistry

PROTOCOL_VERSION = 1
THRESHOLDS = (0.70, 0.80, 0.90)
MAXIMUM_SEEDS = 8
MAXIMUM_CANDIDATES_PER_GATE = 1
HORIZON = 300
SWEEPS = 4
PERSISTENCE = 25
PROBES = 16
POWER = 8
FAMILY_FAILURE_PROBABILITY = 1.0e-6
SCAN_STEPS = 1_200
CHECKPOINT_SPACING = 40

OUTPUT_JACOBIAN = 1
OPTIMIZER_JACOBIAN = 2


@dataclass(frozen=True, order=True)
class Candidate:
    seed: int
    threshold: float
    anchor: int

    @property
    def gate_index(self) -> int:
        try:
            return THRESHOLDS.index(round(float(self.threshold), 2))
        except ValueError as exc:
            raise ValueError(f"threshold is outside the frozen gate set: {self.threshold}") from exc


def scan_anchor_count() -> int:
    """The inclusive frozen scan visits offsets 0,...,1200: 31 anchors."""
    if SCAN_STEPS % CHECKPOINT_SPACING:
        raise ValueError("scan extent must be divisible by checkpoint spacing")
    return SCAN_STEPS // CHECKPOINT_SPACING + 1


def operators_per_candidate() -> dict:
    # State/output envelopes may query states 0..H.  Optimizer-map Jacobians
    # govern transitions 0..H-1.  Keeping both complete ranges makes runtime
    # subset checking simple and removes endpoint ambiguity.
    output = HORIZON + 1
    optimizer = HORIZON
    return {
        "output_jacobian_states": output,
        "optimizer_jacobian_transitions": optimizer,
        "total": output + optimizer,
    }


def maximum_operator_count() -> dict:
    per = operators_per_candidate()
    candidates = MAXIMUM_SEEDS * len(THRESHOLDS) * MAXIMUM_CANDIDATES_PER_GATE
    total = candidates * per["total"]
    return {
        "maximum_seeds": MAXIMUM_SEEDS,
        "gates_per_seed": len(THRESHOLDS),
        "maximum_candidates_per_gate": MAXIMUM_CANDIDATES_PER_GATE,
        "maximum_candidates": candidates,
        "inclusive_screen_anchors_per_gate": scan_anchor_count(),
        "probabilistic_screen_operators": 0,
        **per,
        "maximum_probabilistic_operators": total,
    }


def per_operator_failure_probability() -> float:
    return FAMILY_FAILURE_PROBABILITY / maximum_operator_count()[
        "maximum_probabilistic_operators"
    ]


def probe_config() -> ProbeConfig:
    return ProbeConfig(
        probes=PROBES,
        power=POWER,
        delta=per_operator_failure_probability(),
    )


def operator_identity(candidate: Candidate, step: int, kind: int) -> tuple[int, ...]:
    if kind == OUTPUT_JACOBIAN:
        if not 0 <= step <= HORIZON:
            raise ValueError("output state step is outside 0..H")
    elif kind == OPTIMIZER_JACOBIAN:
        if not 0 <= step < HORIZON:
            raise ValueError("optimizer transition step is outside 0..H-1")
    else:
        raise ValueError(f"unknown probabilistic operator kind: {kind}")
    return (
        PROTOCOL_VERSION,
        int(candidate.seed),
        int(candidate.gate_index),
        int(candidate.anchor),
        int(step),
        SWEEPS,
        int(kind),
    )


def candidate_universe(candidates: tuple[Candidate, ...]) -> frozenset[tuple[int, ...]]:
    seen_pairs: set[tuple[int, int]] = set()
    identities: set[tuple[int, ...]] = set()
    for candidate in candidates:
        pair = (candidate.seed, candidate.gate_index)
        if pair in seen_pairs:
            raise ValueError(f"more than one candidate for frozen seed/gate pair: {pair}")
        seen_pairs.add(pair)
        if candidate.anchor < 0 or candidate.anchor % CHECKPOINT_SPACING:
            raise ValueError(f"candidate anchor is not a frozen checkpoint: {candidate}")
        for step in range(HORIZON + 1):
            identities.add(operator_identity(candidate, step, OUTPUT_JACOBIAN))
        for step in range(HORIZON):
            identities.add(operator_identity(candidate, step, OPTIMIZER_JACOBIAN))
    return frozenset(identities)


def make_registry(
    candidates: tuple[Candidate, ...], master_nonce: str
) -> ProbeRegistry:
    return ProbeRegistry(candidate_universe(candidates), master_nonce)
