#!/usr/bin/env python3
"""Failure-budget schedules for adaptive matrix-free operator queries."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RoleBudget:
    """One candidate's predeclared Green/output failure allocation."""

    candidate_delta: float
    horizon: int
    green_delta: float
    output_delta: float

    @property
    def total(self) -> float:
        return self.green_delta + self.horizon * self.output_delta


def p_series_delta(total_delta: float, query_index: int) -> float:
    """Return the 6/(pi^2 n^2) predictable allocation for query n >= 1."""

    total_delta = float(total_delta)
    if not math.isfinite(total_delta) or not 0.0 < total_delta < 1.0:
        raise ValueError("total_delta must lie strictly between zero and one")
    if int(query_index) != query_index or query_index < 1:
        raise ValueError("query_index must be a positive integer")
    return 6.0 * total_delta / (math.pi**2 * int(query_index) ** 2)


def role_stratified_budget(
    *, total_delta: float, candidate_count: int, horizon: int
) -> RoleBudget:
    """Split each candidate budget equally between Green and output roles."""

    total_delta = float(total_delta)
    if not math.isfinite(total_delta) or not 0.0 < total_delta < 1.0:
        raise ValueError("total_delta must lie strictly between zero and one")
    if int(candidate_count) != candidate_count or candidate_count < 1:
        raise ValueError("candidate_count must be a positive integer")
    if int(horizon) != horizon or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    candidate_delta = total_delta / int(candidate_count)
    return RoleBudget(
        candidate_delta=candidate_delta,
        horizon=int(horizon),
        green_delta=candidate_delta / 2.0,
        output_delta=candidate_delta / (2.0 * int(horizon)),
    )

