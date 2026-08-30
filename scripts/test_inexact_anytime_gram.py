#!/usr/bin/env python3
"""Deterministic and randomized tests for residual-corrected Gram roots."""
from __future__ import annotations

import math

import numpy as np

from inexact_anytime_gram import (
    gram_root_polynomial,
    inexact_gram_operator_upper_bound,
    q1_relative_terminal_residual_upper,
)


def main() -> None:
    deterministic = 0
    for q in range(1, 9):
        for y in (0.0, 1.0e-12, 0.2, 3.0, 1.0e5):
            if y == 0.0:
                continue
            c = 0.37
            observed = inexact_gram_operator_upper_bound(
                terminal_norm=y,
                calibration=c,
                residual_norms=(0.0,) * q,
            )
            expected = (y / c) ** (1.0 / (2.0 * q))
            if not math.isclose(observed, expected, rel_tol=4.0e-14, abs_tol=1.0e-14):
                raise AssertionError((q, y, observed, expected))
            deterministic += 1
    q1 = q1_relative_terminal_residual_upper(
        terminal_norm=7.0,
        calibration=0.4,
        relative_residual=0.25,
    )
    if not math.isclose(q1, math.sqrt(8.75 / 0.4), rel_tol=4.0e-14):
        raise AssertionError("q=1 specialization is wrong")
    deterministic += 1
    degenerate = inexact_gram_operator_upper_bound(
        terminal_norm=0.0,
        calibration=0.5,
        residual_norms=(2.0, 0.0),
    )
    if not math.isclose(degenerate, math.sqrt(4.0), rel_tol=4.0e-14):
        raise AssertionError("positive root with an additional zero root is wrong")
    deterministic += 1

    generator = np.random.default_rng(20260826)
    randomized = 0
    for dimension in (2, 3, 5, 9):
        for q in range(1, 7):
            for _ in range(250):
                basis, _ = np.linalg.qr(generator.normal(size=(dimension, dimension)))
                eigenvalues = np.sort(generator.uniform(0.0, 4.0, size=dimension))[::-1]
                matrix = basis @ np.diag(eigenvalues) @ basis.T
                top = basis[:, 0]
                probes = generator.normal(size=(8, dimension))
                projection = float(np.max(np.abs(probes @ top)))
                calibration = max(projection * 0.8, 1.0e-12)
                vectors = probes.copy()
                residuals = []
                for _power in range(q):
                    noise = generator.normal(size=vectors.shape)
                    noise *= generator.uniform(0.0, 1.0e-4) / max(
                        np.linalg.norm(noise, axis=1).max(), 1.0e-300
                    )
                    next_vectors = vectors @ matrix.T + noise
                    exact_residual = next_vectors - vectors @ matrix.T
                    residuals.append(
                        float(np.linalg.norm(exact_residual, axis=1).max())
                    )
                    vectors = next_vectors
                terminal = float(np.linalg.norm(vectors, axis=1).max())
                upper = inexact_gram_operator_upper_bound(
                    terminal_norm=terminal,
                    calibration=calibration,
                    residual_norms=residuals,
                )
                true_t_norm = math.sqrt(float(eigenvalues[0]))
                if upper + 1.0e-12 < true_t_norm:
                    raise AssertionError((dimension, q, upper, true_t_norm))
                lam_upper = upper * upper
                if gram_root_polynomial(
                    lam_upper,
                    terminal_norm=terminal,
                    calibration=calibration,
                    residual_norms=residuals,
                ) < -1.0e-8:
                    raise AssertionError("returned root is not an upper supersolution")
                randomized += 1

    # Monotonicity in every residual coordinate.
    base = inexact_gram_operator_upper_bound(
        terminal_norm=2.0,
        calibration=0.3,
        residual_norms=(0.1, 0.2, 0.3),
    )
    larger = inexact_gram_operator_upper_bound(
        terminal_norm=2.0,
        calibration=0.3,
        residual_norms=(0.2, 0.4, 0.6),
    )
    if larger < base:
        raise AssertionError("root must be monotone in residual budgets")
    deterministic += 1
    print(
        {
            "status": "inexact anytime Gram tests passed",
            "deterministic_cases": deterministic,
            "randomized_cases": randomized,
        }
    )


if __name__ == "__main__":
    main()
