#!/usr/bin/env python3
"""Regression and exhaustive small-case tests for adaptive witnesses."""
from __future__ import annotations

import itertools
import random

from adaptive_witness_policy import WitnessQuery, acquire_witnesses, uncovered_starts


def exhaustive_logic() -> int:
    checked = 0
    for horizon in range(2, 11):
        for persistence in range(1, min(5, horizon + 1)):
            # GREENCERT's exact anchor is below the gate, so a certified event
            # cannot begin at time zero.
            for event in range(1, horizon - persistence + 2):
                success = set(range(event, event + persistence))
                earlier = [point for point in range(1, horizon + 1) if point not in success]
                for bits in itertools.product((False, True), repeat=len(earlier)):
                    certified = {point for point, bit in zip(earlier, bits) if bit}
                    raw = {point: (2.0 if point in certified else 1.0) for point in earlier}

                    def query(point: int) -> WitnessQuery:
                        return WitnessQuery(
                            point,
                            point in success,
                            point in certified,
                        )

                    result = acquire_witnesses(
                        event=event,
                        persistence=persistence,
                        horizon=horizon,
                        raw_exclusion_slacks=raw,
                        query=query,
                        exact_failures={0},
                    )
                    feasible = not uncovered_starts(
                        certified | {0}, event=event, persistence=persistence
                    )
                    if result.issued != feasible:
                        raise AssertionError((horizon, persistence, event, certified, result))
                    if result.issued and uncovered_starts(
                        set(result.failure_witnesses),
                        event=event,
                        persistence=persistence,
                    ):
                        raise AssertionError("issued policy left an earlier window uncovered")
                    checked += 1
    return checked


def randomized_failed_queries() -> int:
    rng = random.Random(20260826)
    checked = 0
    for _ in range(2_000):
        horizon = rng.randint(10, 80)
        persistence = rng.randint(2, min(10, horizon))
        event = rng.randint(1, horizon - persistence + 1)
        success = set(range(event, event + persistence))
        truth = {
            point
            for point in range(1, horizon + 1)
            if point not in success and rng.random() < 0.55
        }
        raw = {
            point: rng.uniform(0.01, 3.0)
            for point in range(1, horizon + 1)
            if point not in success
        }

        def query(point: int) -> WitnessQuery:
            return WitnessQuery(point, point in success, point in truth)

        result = acquire_witnesses(
            event=event,
            persistence=persistence,
            horizon=horizon,
            raw_exclusion_slacks=raw,
            query=query,
            exact_failures={0},
        )
        feasible = not uncovered_starts(
            truth | {0}, event=event, persistence=persistence
        )
        if result.issued != feasible:
            raise AssertionError((horizon, persistence, event, result.reason))
        if len(result.query_order) != len(set(result.query_order)):
            raise AssertionError("policy queried one time more than once")
        checked += 1
    return checked


def main() -> None:
    exact = exhaustive_logic()
    random_cases = randomized_failed_queries()
    print(f"adaptive witness tests passed: {exact} exhaustive, {random_cases} randomized")


if __name__ == "__main__":
    main()
