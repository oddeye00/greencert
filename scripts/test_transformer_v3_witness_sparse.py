#!/usr/bin/env python3
"""Brute-force checks for the interval witness construction."""
from __future__ import annotations

from itertools import combinations
import random

from audit_transformer_v3_witness_sparse import _minimum_interval_witnesses


def _covers(points: tuple[int, ...] | list[int], starts: range, persistence: int) -> bool:
    return all(
        any(start <= point < start + persistence for point in points)
        for start in starts
    )


def _brute_minimum(available: set[int], starts: range, persistence: int) -> int | None:
    ordered = sorted(available)
    for size in range(len(ordered) + 1):
        for choice in combinations(ordered, size):
            if _covers(choice, starts, persistence):
                return size
    return None


def main() -> None:
    rng = random.Random(20260826)
    checked = 0
    for horizon in range(1, 13):
        for persistence in range(1, min(6, horizon + 1)):
            for event in range(1, horizon + 1):
                universe = set(range(event + persistence))
                for _ in range(100):
                    available = {point for point in universe if rng.random() < 0.55}
                    starts = range(event)
                    optimum = _brute_minimum(available, starts, persistence)
                    if optimum is None:
                        try:
                            _minimum_interval_witnesses(
                                available, starts=starts, persistence=persistence
                            )
                        except AssertionError:
                            pass
                        else:
                            raise AssertionError("greedy found a witness set for an infeasible case")
                    else:
                        greedy = _minimum_interval_witnesses(
                            available, starts=starts, persistence=persistence
                        )
                        if len(greedy) != optimum or not _covers(
                            greedy, starts, persistence
                        ):
                            raise AssertionError("greedy interval witnesses are not minimum")
                    checked += 1
    print(f"witness-sparse tests passed: {checked} brute-force instances")


if __name__ == "__main__":
    main()
