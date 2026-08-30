#!/usr/bin/env python3
"""Numerical theorem tests for chi-calibrated block Gram bounds."""
from __future__ import annotations

import math

import numpy as np

from chi_block_gram import chi_block_operator_upper_bound, chi_lower_calibration


def main() -> None:
    for probes in (1, 2, 4, 8, 16, 32):
        for delta in (1.0e-12, 1.0e-6, 0.05):
            value = chi_lower_calibration(probes=probes, delta=delta)
            if not math.isfinite(value) or value <= 0.0:
                raise AssertionError("invalid chi calibration")

    generator = np.random.default_rng(260826)
    cases = 0
    for dimension in (2, 5, 11):
        for probes in (2, 5, 16):
            for power in range(1, 7):
                for _ in range(200):
                    basis, _ = np.linalg.qr(generator.normal(size=(dimension, dimension)))
                    eigenvalues = np.sort(generator.uniform(0.0, 5.0, size=dimension))[::-1]
                    matrix = basis @ np.diag(eigenvalues) @ basis.T
                    block = generator.normal(size=(dimension, probes))
                    top_projection = float(np.linalg.norm(basis[:, 0] @ block))
                    calibration = max(0.8 * top_projection, 1.0e-12)
                    terminal = block.copy()
                    for _q in range(power):
                        terminal = matrix @ terminal
                    upper = chi_block_operator_upper_bound(
                        terminal_frobenius_norm=float(np.linalg.norm(terminal, ord="fro")),
                        calibration=calibration,
                        power=power,
                    )
                    true_t_norm = math.sqrt(float(eigenvalues[0]))
                    if upper + 1.0e-12 < true_t_norm:
                        raise AssertionError((dimension, probes, power, upper, true_t_norm))
                    cases += 1
    print({"status": "chi block Gram tests passed", "randomized_cases": cases})


if __name__ == "__main__":
    main()
