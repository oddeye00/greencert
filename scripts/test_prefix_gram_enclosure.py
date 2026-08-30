#!/usr/bin/env python3
"""Tests for nested-prefix Gaussian Gram calibration."""
from __future__ import annotations

import math

import numpy as np

from prefix_gram_enclosure import (
    equal_family_stage_delta,
    family_failure_upper,
    prefix_increment,
    prefix_gram_rows,
)


def main() -> None:
    assert prefix_increment(initial_count=0, final_count=0, target=4) == 4
    assert prefix_increment(initial_count=4, final_count=4, target=8) == 4
    assert prefix_increment(initial_count=8, final_count=8, target=16) == 8
    try:
        prefix_increment(initial_count=4, final_count=2, target=8)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched paired-probe counts were accepted")

    prefixes = (4, 8, 16)
    delta = equal_family_stage_delta(
        family_failure=1.0e-6, operators=15, prefixes=prefixes
    )
    assert math.isclose(delta, 1.0e-6 / 45.0)
    assert math.isclose(
        family_failure_upper(stage_delta=delta, operators=15, prefixes=prefixes),
        1.0e-6,
    )

    rows = prefix_gram_rows(
        final_norms=tuple(float(value) for value in range(1, 17)),
        initial_norms=(1.0,) * 16,
        prefixes=prefixes,
        power=1,
        stage_delta=delta,
    )
    assert [row["Y"] for row in rows] == [4.0, 8.0, 16.0]
    assert [row["gram_applications"] for row in rows] == [4, 8, 16]
    assert all(row["operator_norm_upper_bound"] > 0.0 for row in rows)

    # Monte Carlo checks the stated union guarantee at a deliberately visible
    # failure level. For T=diag(3,1), every bound is computed from A g exactly.
    rng = np.random.default_rng(20260829)
    trials = 40_000
    family_failure = 0.03
    stage = family_failure / len(prefixes)
    misses = 0
    for _ in range(trials):
        probes = rng.standard_normal((16, 2))
        initial = np.linalg.norm(probes, axis=1)
        final = np.linalg.norm(probes @ np.diag([9.0, 1.0]), axis=1)
        trial_rows = prefix_gram_rows(
            final_norms=final,
            initial_norms=initial,
            prefixes=prefixes,
            power=1,
            stage_delta=stage,
        )
        if any(row["operator_norm_upper_bound"] < 3.0 for row in trial_rows):
            misses += 1
    observed = misses / trials
    assert observed < family_failure + 0.004, (observed, family_failure)
    print(
        {
            "status": "nested-prefix Gram tests passed",
            "trials": trials,
            "observed_any_prefix_failure": observed,
            "nominal_union_upper": family_failure,
        }
    )


if __name__ == "__main__":
    main()
