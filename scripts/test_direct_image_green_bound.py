#!/usr/bin/env python3
"""Deterministic and Monte Carlo tests for staged direct-image Green bounds."""
from __future__ import annotations

import numpy as np

from direct_image_green_bound import direct_image_rows
from prefix_gram_enclosure import prefix_gram_rows


def main() -> None:
    rng = np.random.default_rng(20260830)
    prefixes = (4, 8, 16)
    stage_delta = 0.01
    trials = 30_000
    any_failure = 0
    simultaneous_checks = 0
    for _ in range(trials):
        left, _ = np.linalg.qr(rng.normal(size=(7, 7)))
        right, _ = np.linalg.qr(rng.normal(size=(7, 7)))
        singular = np.asarray((4.0, 1.7, 0.8, 0.4, 0.2, 0.1, 0.03))
        operator = left @ np.diag(singular) @ right.T
        probes = rng.normal(size=(16, 7))
        images = probes @ operator.T
        grams = images @ operator
        initial_norms = np.linalg.norm(probes, axis=1)
        image_norms = np.linalg.norm(images, axis=1)
        gram_norms = np.linalg.norm(grams, axis=1)
        direct = direct_image_rows(
            image_norms=image_norms,
            initial_norms=initial_norms,
            prefixes=prefixes,
            stage_delta=stage_delta,
        )
        gram = prefix_gram_rows(
            final_norms=gram_norms,
            initial_norms=initial_norms,
            prefixes=prefixes,
            power=1,
            stage_delta=stage_delta,
        )
        failed = False
        for direct_row, gram_row in zip(direct, gram):
            failed |= direct_row["operator_norm_upper_bound"] < singular[0]
            failed |= gram_row["operator_norm_upper_bound"] < singular[0]
            simultaneous_checks += 2
        any_failure += failed
    observed = any_failure / trials
    nominal = len(prefixes) * stage_delta
    assert observed <= nominal + 0.004
    print(
        {
            "status": "direct-image Green tests passed",
            "trials": trials,
            "simultaneous_bound_checks": simultaneous_checks,
            "observed_any_bound_failure": observed,
            "nominal_union_upper": nominal,
        }
    )


if __name__ == "__main__":
    main()
