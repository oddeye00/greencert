#!/usr/bin/env python3
"""Deterministic regression checks for constants in the analytic jet appendix.

The appendix contains the proof.  These checks independently evaluate every
one-dimensional GELU extremum and stress the closed-form softmax derivative
identities used to obtain the global (1/2, 2, 6) constants.
"""
from __future__ import annotations

import math

import mpmath as mp
import torch


def gelu_gate() -> dict:
    mp.mp.dps = 80
    sqrt = mp.sqrt
    pi = mp.pi

    def phi(x):
        return mp.e ** (-x * x / 2) / sqrt(2 * pi)

    def Phi(x):
        return (1 + mp.erf(x / sqrt(2))) / 2

    def first(x):
        return Phi(x) + x * phi(x)

    def second(x):
        return (2 - x * x) * phi(x)

    def third(x):
        return (x**3 - 4 * x) * phi(x)

    first_candidates = [mp.mpf("0"), sqrt(2), -sqrt(2)]
    second_candidates = [mp.mpf("0"), mp.mpf("2"), mp.mpf("-2")]
    y1 = (7 - sqrt(33)) / 2
    y2 = (7 + sqrt(33)) / 2
    third_candidates = [sqrt(y1), -sqrt(y1), sqrt(y2), -sqrt(y2)]
    maxima = {
        "gelu_first": max(abs(first(x)) for x in first_candidates),
        "gelu_second": max(abs(second(x)) for x in second_candidates),
        "gelu_third": max(abs(third(x)) for x in third_candidates),
    }
    assert maxima["gelu_first"] < mp.mpf("1.13")
    assert maxima["gelu_second"] < mp.mpf("0.80")
    assert maxima["gelu_third"] < mp.mpf("2.0")
    return {key: float(value) for key, value in maxima.items()}


def covariance(p: torch.Tensor) -> torch.Tensor:
    return torch.diag(p) - torch.outer(p, p)


def covariance_first_derivative(p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    q = covariance(p) @ h
    return torch.diag(q) - torch.outer(q, p) - torch.outer(p, q)


def covariance_second_derivative(
    p: torch.Tensor, h: torch.Tensor, k: torch.Tensor
) -> torch.Tensor:
    c = covariance(p)
    q = c @ h
    r = c @ k
    q_prime = covariance_first_derivative(p, k) @ h
    return (
        torch.diag(q_prime)
        - torch.outer(q_prime, p)
        - torch.outer(q, r)
        - torch.outer(r, q)
        - torch.outer(p, q_prime)
    )


def softmax_gate() -> dict:
    generator = torch.Generator().manual_seed(20260824)
    worst = {"first": 0.0, "second": 0.0, "third": 0.0}
    for dimension in (2, 3, 5, 17, 31):
        for _ in range(300):
            logits = torch.randn(dimension, generator=generator, dtype=torch.float64)
            p = torch.softmax(logits, dim=0)
            h = torch.randn(dimension, generator=generator, dtype=torch.float64)
            k = torch.randn(dimension, generator=generator, dtype=torch.float64)
            h /= torch.linalg.vector_norm(h)
            k /= torch.linalg.vector_norm(k)
            c = covariance(p)
            dc = covariance_first_derivative(p, h)
            d2c = covariance_second_derivative(p, h, k)
            worst["first"] = max(worst["first"], float(torch.linalg.matrix_norm(c, 2)))
            worst["second"] = max(worst["second"], float(torch.linalg.matrix_norm(dc, 2)))
            worst["third"] = max(worst["third"], float(torch.linalg.matrix_norm(d2c, 2)))
    assert worst["first"] <= 0.5 * (1 + 1e-12)
    assert worst["second"] < 2.0
    assert worst["third"] < 6.0
    return worst


def finite_difference_identity_gate() -> float:
    """Check the analytic D^2 softmax formula against central differences."""
    generator = torch.Generator().manual_seed(991)
    worst = 0.0
    epsilon = 1.0e-5
    for dimension in (3, 7, 17):
        for _ in range(25):
            x = torch.randn(dimension, generator=generator, dtype=torch.float64)
            h = torch.randn(dimension, generator=generator, dtype=torch.float64)
            k = torch.randn(dimension, generator=generator, dtype=torch.float64)
            h /= torch.linalg.vector_norm(h)
            k /= torch.linalg.vector_norm(k)
            p = torch.softmax(x, dim=0)
            analytic = covariance_second_derivative(p, h, k)
            plus_p = torch.softmax(x + epsilon * k, dim=0)
            minus_p = torch.softmax(x - epsilon * k, dim=0)
            numeric = (
                covariance_first_derivative(plus_p, h)
                - covariance_first_derivative(minus_p, h)
            ) / (2 * epsilon)
            error = float(torch.linalg.matrix_norm(analytic - numeric, 2))
            worst = max(worst, error)
    assert worst < 2.0e-9
    return worst


def main() -> None:
    gelu = gelu_gate()
    softmax = softmax_gate()
    fd_error = finite_difference_identity_gate()
    assert math.isfinite(fd_error)
    print(
        "PASS: GELU extrema "
        f"({gelu['gelu_first']:.6f}, {gelu['gelu_second']:.6f}, "
        f"{gelu['gelu_third']:.6f}); softmax observed maxima "
        f"({softmax['first']:.6f}, {softmax['second']:.6f}, "
        f"{softmax['third']:.6f}); D2 identity error {fd_error:.3e}."
    )


if __name__ == "__main__":
    main()
