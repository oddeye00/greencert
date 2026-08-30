#!/usr/bin/env python3
"""Exhaustive finite-horizon audit of the persistent first-passage bracket.

The audit enumerates valid pointwise envelopes ``n_minus <= n <= n_plus`` and
checks the theorem against the independent WDBC and Transformer implementations.
It includes trajectories with no persistent event in the certified horizon.
"""

from __future__ import annotations

from itertools import product

import numpy as np

import transformer_v3_certificate as transformer
from real_dataset_greencert import persistent_bracket as wdbc_bracket


def first_persistent(values: tuple[int, ...], required: int, persistence: int) -> int | None:
    for start in range(len(values) - persistence + 1):
        if all(value >= required for value in values[start : start + persistence]):
            return start
    return None


def transformer_bracket(
    guaranteed: tuple[int, ...],
    possible: tuple[int, ...],
    required: int,
    persistence: int,
) -> list[int] | None:
    guarantee_slacks = [1.0 if value >= required else -1.0 for value in guaranteed]
    # Positive exclusion slack certifies failure; nonpositive means the gate is possible.
    exclusion_slacks = [-1.0 if value >= required else 1.0 for value in possible]
    original = transformer.PERSISTENCE
    try:
        transformer.PERSISTENCE = persistence
        return transformer._persistent_bracket(guarantee_slacks, exclusion_slacks)
    finally:
        transformer.PERSISTENCE = original


def audit_configuration(population: int, length: int, required: int, persistence: int) -> int:
    states = tuple(
        (lower, truth, upper)
        for lower in range(population + 1)
        for truth in range(lower, population + 1)
        for upper in range(truth, population + 1)
    )
    checked = 0
    for path in product(states, repeat=length):
        guaranteed = tuple(row[0] for row in path)
        truth = tuple(row[1] for row in path)
        possible = tuple(row[2] for row in path)

        lower = first_persistent(possible, required, persistence)
        upper = first_persistent(guaranteed, required, persistence)
        expected = None if lower is None or upper is None or lower > upper else [lower, upper]
        actual = first_persistent(truth, required, persistence)

        wdbc = wdbc_bracket(
            np.asarray(guaranteed, dtype=np.int64),
            np.asarray(possible, dtype=np.int64),
            required,
            persistence,
        )
        transformed = transformer_bracket(guaranteed, possible, required, persistence)
        assert wdbc == expected
        assert transformed == expected

        if expected is not None:
            assert actual is not None
            assert expected[0] <= actual <= expected[1]
            assert 0 <= expected[0] <= length - persistence
            assert 0 <= expected[1] <= length - persistence
        elif actual is None:
            # No guaranteed finite upper endpoint may exist when the true event is absent.
            assert upper is None
        checked += 1
    return checked


def main() -> None:
    total = 0
    for persistence in (1, 2, 3):
        total += audit_configuration(1, 7, 1, persistence)
    for required in (1, 2):
        for persistence in (1, 2, 3):
            total += audit_configuration(2, 4, required, persistence)
    assert total == 109_152
    print(
        "PASS: 109,152 exhaustive lower/true/upper paths satisfy the finite-window "
        "persistent first-passage theorem in both implementations"
    )


if __name__ == "__main__":
    main()
