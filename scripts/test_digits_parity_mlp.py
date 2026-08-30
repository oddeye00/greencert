#!/usr/bin/env python3
from __future__ import annotations

import torch

from digits_parity_mlp import (
    initialize,
    make_selection_split,
    make_split,
    parameter_spec,
    raw_data_sha256,
)
from real_dataset_mlp import RealMLPConfig, analytic_gradient, autograd_objective_hvp, objective_hvp


def main() -> None:
    config = RealMLPConfig(width=8, learning_rate=0.03, seed=501, threads=1)
    selection = make_selection_split(config)
    full = make_split(config)
    assert "certificate_x" not in selection and "certificate_y" not in selection
    assert len(selection["train_y"]) + len(selection["trigger_y"]) + len(full["certificate_y"]) == 1797
    assert selection["metadata"]["raw_data_sha256"] == raw_data_sha256()
    assert torch.equal(selection["train_x"], full["train_x"])
    assert torch.equal(selection["trigger_y"], full["trigger_y"])
    spec = parameter_spec(config)
    parameter = initialize(config)
    assert parameter.numel() == 538 == spec.size
    direction = torch.randn_like(parameter, generator=torch.Generator().manual_seed(44))
    analytic = objective_hvp(
        parameter, direction, full["train_x"], full["train_y"], spec, config
    )
    reference = autograd_objective_hvp(
        parameter, direction, full["train_x"], full["train_y"], spec, config
    )
    assert torch.allclose(analytic, reference, atol=2e-11, rtol=2e-10)
    gradient = analytic_gradient(
        parameter, full["train_x"], full["train_y"], spec, config
    )
    assert torch.isfinite(gradient).all()
    print("PASS: digits loader enforces the split barrier and reuses the validated analytic HVP.")


if __name__ == "__main__":
    main()
