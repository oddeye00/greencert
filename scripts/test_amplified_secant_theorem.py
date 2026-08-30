#!/usr/bin/env python3
"""Finite-dimensional tests of the amplified-secant response corollary."""

from __future__ import annotations

import numpy as np


def green_matrix(jacobians: list[np.ndarray]) -> np.ndarray:
    horizon = len(jacobians)
    dimension = jacobians[0].shape[0]
    matrix = np.zeros((horizon * dimension, horizon * dimension))
    for column in range(horizon * dimension):
        injection = np.zeros((horizon, dimension))
        injection.reshape(-1)[column] = 1.0
        state = np.zeros(dimension)
        rows = []
        for step in range(horizon):
            state = jacobians[step] @ state + injection[step]
            rows.append(state.copy())
        matrix[:, column] = np.asarray(rows).reshape(-1)
    return matrix


def main() -> None:
    rng = np.random.default_rng(20260829)
    cases = 2000
    worst_ratio = 0.0
    for _ in range(cases):
        dimension = int(rng.integers(1, 5))
        horizon = int(rng.integers(1, 7))
        jacobians = [
            rng.normal(scale=0.25, size=(dimension, dimension))
            for _ in range(horizon)
        ]
        green = green_matrix(jacobians)
        kappa = float(np.linalg.norm(green, ord=2))
        q_rows = []
        secant_rows = []
        bounds = []
        for _step in range(horizon):
            raw_q = rng.normal(scale=0.3, size=(dimension,) * 3)
            quadratic = 0.5 * (raw_q + raw_q.swapaxes(1, 2))
            raw_c = rng.normal(scale=0.2, size=(dimension,) * 4)
            cubic = sum(
                raw_c.transpose((0,) + axes)
                for axes in (
                    (1, 2, 3),
                    (1, 3, 2),
                    (2, 1, 3),
                    (2, 3, 1),
                    (3, 1, 2),
                    (3, 2, 1),
                )
            ) / 6.0
            z = rng.normal(scale=0.04, size=dimension)
            lam = float(2.0 ** rng.integers(0, 8))
            q2 = 0.5 * np.einsum("iab,a,b->i", quadratic, z, z)
            q3 = np.einsum("iabc,a,b,c->i", cubic, z, z, z) / 6.0
            q_rows.append(q2 + q3)
            secant_rows.append(q2 + lam * q3)
            derivative_upper = float(np.linalg.norm(cubic))
            bounds.append(
                abs(lam - 1.0)
                * derivative_upper
                * float(np.linalg.norm(z)) ** 3
                / 6.0
            )
        q = np.asarray(q_rows).reshape(-1)
        secant = np.asarray(secant_rows).reshape(-1)
        sigma_secant = float(np.linalg.norm(bounds))
        arithmetic = rng.normal(scale=1.0e-9, size=q.shape)
        q_tilde = secant + arithmetic
        y_exact_for_tilde = (green @ q_tilde).reshape(horizon, dimension)
        response_noise = rng.normal(
            scale=1.0e-9, size=(horizon, dimension)
        )
        y_tilde = y_exact_for_tilde + response_noise
        previous = np.zeros(dimension)
        recurrence_rows = []
        for step in range(horizon):
            recurrence_rows.append(
                y_tilde[step]
                - jacobians[step] @ previous
                - q_tilde.reshape(horizon, dimension)[step]
            )
            previous = y_tilde[step]
        tau_y = float(np.linalg.norm(recurrence_rows))
        sigma_ar = float(np.linalg.norm(arithmetic))
        beta = float(np.linalg.norm(y_tilde)) + kappa * (
            sigma_secant + sigma_ar + tau_y
        )
        target = float(np.linalg.norm(green @ q))
        if target > beta * (1.0 + 5.0e-13) + 1.0e-14:
            raise AssertionError((target, beta))
        if beta > 0.0:
            worst_ratio = max(worst_ratio, target / beta)
    print(
        {
            "status": "amplified-secant response theorem tests passed",
            "cases": cases,
            "worst_target_to_beta_ratio": worst_ratio,
        }
    )


if __name__ == "__main__":
    main()
