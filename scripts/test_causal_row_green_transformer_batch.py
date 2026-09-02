#!/usr/bin/env python3
"""Regression test for probe/forcing row independence in a Green batch."""
from __future__ import annotations

import torch

from batched_green_operator import make_batched_transformer_green_products, relative_error
from transformer_hvp_grokking import (
    TransformerConfig,
    flat_spec,
    flatten_parameters,
    make_disjoint_split,
    make_template,
)


def main() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = TransformerConfig(
        modulus=5,
        model_dim=8,
        hidden_dim=16,
        heads=2,
        depth=1,
        train_fraction=0.60,
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=0.01,
        steps=1,
        seed=20260901,
        threads=1,
        dtype="float64",
        loss="cross_entropy",
        normalization="none",
    )
    template = make_template(config)
    spec = flat_spec(template)
    parameter = flatten_parameters(template)
    train_pairs, train_labels, *_ = make_disjoint_split(config)
    generator = torch.Generator().manual_seed(92_611)
    path = torch.stack(
        (
            parameter,
            parameter
            + 1.0e-4
            * torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype),
            parameter
            + 1.0e-4
            * torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype),
        )
    )
    apply, _ = make_batched_transformer_green_products(
        path, train_pairs, train_labels, template, spec, config
    )
    probes = torch.randn(
        4,
        len(path) * 2 * parameter.numel(),
        generator=generator,
        dtype=parameter.dtype,
    )
    forcing = torch.randn(
        1,
        probes.shape[1],
        generator=generator,
        dtype=parameter.dtype,
    )
    probe_only = apply(probes)
    combined = apply(torch.cat((probes, forcing), dim=0))
    error = relative_error(combined[: len(probes)], probe_only)
    if error > 3.0e-12:
        raise AssertionError(f"appended forcing row perturbed Gaussian images: {error}")
    print(
        "PASS: appended deterministic forcing row preserves Gaussian Green "
        f"images (relative error {error:.3e})."
    )


if __name__ == "__main__":
    main()
