#!/usr/bin/env python3
"""Deterministic checks for the residual-aware forcing release."""
from __future__ import annotations

import numpy as np

from cost_aware_forcing import cost_aware_forcing_upper


def main() -> None:
    rng = np.random.default_rng(20260829)
    improved = 0
    for _ in range(20_000):
        dimension = 9
        matrix = rng.normal(size=(dimension, dimension))
        kappa = float(np.linalg.norm(matrix, ord=2))
        surrogate = rng.normal(size=dimension)
        error = 1.0e-3 * rng.normal(size=dimension)
        forcing = surrogate + error
        response_error = 1.0e-5 * rng.normal(size=dimension)
        computed_response = matrix @ surrogate + response_error
        result = cost_aware_forcing_upper(
            kappa=kappa,
            surrogate_injection_norm=float(np.linalg.norm(surrogate)),
            surrogate_error_norm=float(np.linalg.norm(error)),
            direct_response_norm=float(np.linalg.norm(computed_response)),
            direct_response_recurrence_residual_norm=(
                float(np.linalg.norm(response_error)) / max(kappa, 1.0e-300)
            ),
        )
        truth = float(np.linalg.norm(matrix @ forcing))
        assert truth <= result.norm_only_response_upper * (1.0 + 2.0e-14)
        assert truth <= result.response_aware_upper * (1.0 + 2.0e-14)
        assert truth <= result.selected_response_upper * (1.0 + 2.0e-14)
        improved += result.selected_method == "direct_response"

    plain = cost_aware_forcing_upper(
        kappa=3.0,
        surrogate_injection_norm=2.0,
        surrogate_error_norm=0.25,
    )
    assert plain.selected_method == "norm_only"
    assert plain.selected_response_upper == 6.75
    print(
        {
            "status": "cost-aware forcing tests passed",
            "random_trials": 20_000,
            "direct_response_tighter": improved,
        }
    )


if __name__ == "__main__":
    main()
