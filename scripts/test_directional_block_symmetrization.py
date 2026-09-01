#!/usr/bin/env python3
"""Exact finite-dimensional checks for the v2 block symmetrization lemma."""
from __future__ import annotations

import itertools
import math
import random

import numpy as np


BLOCKS = 4
SLOTS = tuple(itertools.permutations(range(4)))
INDICES = tuple(itertools.product(range(BLOCKS), repeat=4))


def apply(coefficients: np.ndarray, vectors: tuple[np.ndarray, ...]) -> float:
    total = 0.0
    for index in INDICES:
        term = float(coefficients[index])
        for slot, block in enumerate(index):
            term *= float(vectors[slot][block])
        total += term
    return total


def symmetrize(coefficients: np.ndarray) -> np.ndarray:
    out = np.zeros_like(coefficients)
    for index in INDICES:
        out[index] = sum(
            float(coefficients[tuple(index[slot] for slot in permutation)])
            for permutation in SLOTS
        ) / math.factorial(4)
    return out


def diagonal_gradient(coefficients: np.ndarray, radius: np.ndarray) -> np.ndarray:
    gradient = np.zeros(BLOCKS, dtype=np.float64)
    for index in INDICES:
        coefficient = float(coefficients[index])
        for free_slot in range(4):
            term = coefficient
            for slot, block in enumerate(index):
                if slot != free_slot:
                    term *= float(radius[block])
            gradient[index[free_slot]] += term
    return gradient


def close(left: float, right: float, label: str) -> None:
    if not math.isclose(left, right, rel_tol=2.0e-12, abs_tol=2.0e-12):
        raise AssertionError(f"{label}: {left} != {right}")


def main() -> None:
    rng = random.Random(20260901)
    cases = 0
    for _ in range(200):
        raw = np.array(
            [rng.random() for _ in range(BLOCKS**4)], dtype=np.float64
        ).reshape((BLOCKS,) * 4)
        symmetric = symmetrize(raw)
        radius = np.array([rng.random() for _ in range(BLOCKS)])
        if cases % 7 == 0:
            radius[cases % BLOCKS] = 0.0
        free = np.array([rng.random() for _ in range(BLOCKS)])
        free /= np.linalg.norm(free)

        raw_diagonal = apply(raw, (radius,) * 4)
        symmetric_diagonal = apply(symmetric, (radius,) * 4)
        close(raw_diagonal, symmetric_diagonal, "diagonal preservation")

        gradient = diagonal_gradient(symmetric, radius)
        mixed = apply(symmetric, (radius, radius, radius, free))
        close(mixed, float(gradient @ free) / 4.0, "polarized gradient")
        close(
            float(gradient @ radius),
            4.0 * symmetric_diagonal,
            "Euler homogeneity",
        )

        for permutation in SLOTS:
            permuted = tuple((radius, radius, radius, free)[slot] for slot in permutation)
            close(apply(symmetric, permuted), mixed, "slot symmetry")
        cases += 1

    print(
        {
            "status": "directional block symmetrization checks passed",
            "random_asymmetric_majorants": cases,
            "slot_permutations_per_case": len(SLOTS),
        }
    )


if __name__ == "__main__":
    main()
