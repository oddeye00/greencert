#!/usr/bin/env python3
"""Property tests for response-centered observable transport."""

from __future__ import annotations

import math

import numpy as np

from response_centered_event_transport import (
    classification_margin_origin_radius,
    classification_margin_remainder_radius,
    observable_remainder_radius,
    response_centering_dominates_origin_radius,
)


def main() -> None:
    rng = np.random.default_rng(20260827)
    trials = 4_000
    for _ in range(trials):
        dimension = int(rng.integers(1, 18))
        raw = rng.normal(size=(dimension, dimension))
        hessian = 0.5 * (raw + raw.T)
        hessian_norm = float(np.linalg.norm(hessian, ord=2))
        gradient = rng.normal(size=dimension)
        response = rng.normal(size=dimension)
        remainder = rng.normal(size=dimension)
        response *= float(rng.uniform(0.0, 2.0)) / max(np.linalg.norm(response), 1e-300)
        remainder *= float(rng.uniform(0.0, 0.7)) / max(np.linalg.norm(remainder), 1e-300)

        # m(x) = g^T x + x^T H x / 2, centered at c=0.
        before = float(gradient @ response + 0.5 * response @ hessian @ response)
        moved = response + remainder
        after = float(gradient @ moved + 0.5 * moved @ hessian @ moved)
        bound = observable_remainder_radius(
            center_gradient_upper=float(np.linalg.norm(gradient)),
            hessian_upper=hessian_norm,
            response_norm=float(np.linalg.norm(response)),
            remainder_radius=float(np.linalg.norm(remainder)),
        )
        assert abs(after - before) <= bound + 2.0e-13 * max(1.0, bound)

        jacobian = float(rng.uniform(0.0, 20.0))
        second = float(rng.uniform(0.0, 30.0))
        d = float(rng.uniform(0.0, 4.0))
        p = d + float(rng.uniform(0.0, 4.0))
        e = float(rng.uniform(0.0, 2.0))
        assert response_centering_dominates_origin_radius(
            output_jacobian_upper=jacobian,
            output_hessian_upper=second,
            response_norm=d,
            response_max_norm=p,
            remainder_radius=e,
        )
        new = classification_margin_remainder_radius(
            output_jacobian_upper=jacobian,
            output_hessian_upper=second,
            response_norm=d,
            remainder_radius=e,
        )
        old = classification_margin_origin_radius(
            output_jacobian_upper=jacobian,
            output_hessian_upper=second,
            total_radius=p + e,
        )
        assert new <= old + 1.0e-13 * max(1.0, old)

    assert classification_margin_remainder_radius(
        output_jacobian_upper=3.0,
        output_hessian_upper=5.0,
        response_norm=2.0,
        remainder_radius=0.0,
    ) == 0.0
    expected = math.sqrt(2.0) * 3.0
    assert math.isclose(
        classification_margin_origin_radius(
            output_jacobian_upper=2.0,
            output_hessian_upper=2.0,
            total_radius=1.0,
        ),
        expected,
    )
    print(f"PASS: {trials:,} quadratic observable and radius-dominance cases")


if __name__ == "__main__":
    main()
