#!/usr/bin/env python3
"""Predictable acquisition of sparse first-passage witnesses.

The policy uses only centerline margins and results of operators already
queried.  It never reads a future trajectory.  A caller may therefore assign a
fresh, domain-separated random block to every adaptively requested operator and
apply the predictable-family failure budget from the anytime Green theorem.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping


@dataclass(frozen=True)
class WitnessQuery:
    time: int
    guarantee: bool
    exclusion: bool


@dataclass(frozen=True)
class WitnessPolicyResult:
    issued: bool
    success_times: tuple[int, ...]
    failure_witnesses: tuple[int, ...]
    query_order: tuple[int, ...]
    reason: str


def uncovered_starts(
    failures: set[int], *, event: int, persistence: int
) -> tuple[int, ...]:
    """Earlier persistence-window starts not hit by a certified failure."""

    if event < 0 or persistence < 1:
        raise ValueError("event must be nonnegative and persistence positive")
    return tuple(
        start
        for start in range(event)
        if not any(start <= point < start + persistence for point in failures)
    )


def _next_candidate(
    *,
    raw_exclusion_slacks: Mapping[int, float],
    queried: set[int],
    uncovered: tuple[int, ...],
    persistence: int,
) -> int | None:
    """Choose a predictable high-coverage, high-slack output query.

    Coverage is the primary score because one certified failure can eliminate
    several candidate event starts.  Raw centerline exclusion slack is a
    deterministic secondary score, and the latest time breaks remaining ties.
    """

    choices: list[tuple[int, float, int]] = []
    for point, slack in raw_exclusion_slacks.items():
        point = int(point)
        if point in queried or float(slack) <= 0.0:
            continue
        coverage = sum(
            start <= point < start + persistence for start in uncovered
        )
        if coverage:
            choices.append((coverage, float(slack), point))
    if not choices:
        return None
    return max(choices)[2]


def acquire_witnesses(
    *,
    event: int,
    persistence: int,
    horizon: int,
    raw_exclusion_slacks: Mapping[int, float],
    query: Callable[[int], WitnessQuery],
    exact_failures: set[int] | None = None,
) -> WitnessPolicyResult:
    """Acquire a persistent success window and an earlier failure hitting set.

    ``query`` is called at most once per time.  Its implementation may evaluate
    a deterministic enclosure or a fresh randomized output operator.  The
    selection of the next time depends only on deterministic centerline slacks
    and prior query results, so it is predictable with respect to fresh blocks.
    """

    if persistence < 1 or event < 0:
        raise ValueError("invalid event or persistence")
    if event + persistence - 1 > horizon:
        raise ValueError("success window exceeds the horizon")
    failures = set() if exact_failures is None else set(exact_failures)
    if any(point < 0 or point > horizon for point in failures):
        raise ValueError("exact failure lies outside the horizon")

    success_times = tuple(range(event, event + persistence))
    queried: set[int] = set()
    order: list[int] = []

    for point in success_times:
        if point in failures:
            return WitnessPolicyResult(
                False, success_times, tuple(sorted(failures)), tuple(order),
                "an exact failure intersects the proposed success window",
            )
        row = query(point)
        if row.time != point:
            raise ValueError("query returned the wrong time")
        queried.add(point)
        order.append(point)
        if not row.guarantee:
            return WitnessPolicyResult(
                False, success_times, tuple(sorted(failures)), tuple(order),
                "success-window output margin did not certify",
            )

    while True:
        uncovered = uncovered_starts(
            failures, event=event, persistence=persistence
        )
        if not uncovered:
            return WitnessPolicyResult(
                True,
                success_times,
                tuple(sorted(failures)),
                tuple(order),
                "success window and every earlier failure witness certified",
            )
        point = _next_candidate(
            raw_exclusion_slacks=raw_exclusion_slacks,
            queried=queried,
            uncovered=uncovered,
            persistence=persistence,
        )
        if point is None:
            return WitnessPolicyResult(
                False,
                success_times,
                tuple(sorted(failures)),
                tuple(order),
                "no unqueried positive-centerline failure candidate hits an uncovered window",
            )
        row = query(point)
        if row.time != point:
            raise ValueError("query returned the wrong time")
        queried.add(point)
        order.append(point)
        if row.exclusion:
            failures.add(point)

