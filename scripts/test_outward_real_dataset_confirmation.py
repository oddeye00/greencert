#!/usr/bin/env python3
"""Containment tests for the WDBC outward arithmetic."""
from __future__ import annotations

import numpy as np
import torch

from outward_interval_certificate import _interval_matmul
from outward_real_dataset_confirmation import (
    _dense_hessian_interval,
    _gradient_interval,
    _hessian_beta,
    _network_intervals,
)
from real_dataset_mlp import (
    RealMLPConfig,
    analytic_gradient,
    analytic_objective_hvp,
    initialize,
    logits,
    make_split,
    parameter_spec,
)


def main() -> None:
    config = RealMLPConfig(
        width=8,
        learning_rate=0.005,
        weight_decay=1e-3,
        seed=101,
        threads=1,
        dtype="float64",
    )
    torch.set_num_threads(1)
    data = make_split(config)
    spec = parameter_spec(config)
    parameter = initialize(config)
    array = parameter.numpy()
    network = _network_intervals(
        array, data["train_x"], data["train_y"], spec
    )
    float_logits = logits(parameter, data["train_x"], spec).numpy()
    assert np.all(network["logits_lower"] <= float_logits)
    assert np.all(float_logits <= network["logits_upper"])

    gradient_lower, gradient_upper = _gradient_interval(
        array,
        data["train_x"],
        data["train_y"],
        spec,
        config,
        network,
    )
    float_gradient = analytic_gradient(
        parameter, data["train_x"], data["train_y"], spec, config
    ).numpy()
    assert np.all(gradient_lower <= float_gradient)
    assert np.all(float_gradient <= gradient_upper)

    hessian_lower, hessian_upper = _dense_hessian_interval(
        array,
        data["train_x"],
        data["train_y"],
        spec,
        config,
        network,
    )
    generator = torch.Generator().manual_seed(991)
    for _ in range(4):
        vector = torch.randn(spec.size, generator=generator, dtype=torch.float64)
        product = analytic_objective_hvp(
            parameter,
            vector,
            data["train_x"],
            data["train_y"],
            spec,
            config,
        ).numpy()
        lower, upper = _interval_matmul(
            hessian_lower,
            hessian_upper,
            vector.numpy()[:, None],
            vector.numpy()[:, None],
        )
        assert np.all(lower[:, 0] <= product)
        assert np.all(product <= upper[:, 0])

    beta, diagnostics = _hessian_beta(
        array,
        data["train_x"],
        data["train_y"],
        spec,
        config,
        network,
    )
    assert beta >= 0.0
    assert diagnostics["hessian_interval_row_radius"] >= 0.0
    print(
        "PASS WDBC outward intervals; "
        f"max gradient width={np.max(gradient_upper-gradient_lower):.3e}, "
        f"max Hessian width={np.max(hessian_upper-hessian_lower):.3e}, "
        f"beta={beta:.9f}"
    )


if __name__ == "__main__":
    main()
