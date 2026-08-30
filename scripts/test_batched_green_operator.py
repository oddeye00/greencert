#!/usr/bin/env python3
"""Regression gates for batched GreenCert probe products."""
from __future__ import annotations

import torch

from batched_green_operator import (
    batched_gram_norm_bound,
    make_batched_output_gram_operator,
    make_batched_scaled_optimizer_products,
    make_batched_transformer_green_products,
    objective_hvp_batch,
    relative_error,
)
from probe_jacobian_bound import ProbeConfig, gram_norm_bound, make_gram_operator, probe_seed
from transformer_green_operator import make_transformer_green_products
from transformer_hvp_grokking import (
    TransformerConfig,
    flat_spec,
    flatten_parameters,
    make_disjoint_split,
    make_template,
    objective_hvp,
)
from transformer_optimizer_probe import make_scaled_optimizer_jvp_vjp


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
        seed=20260824,
        threads=1,
        dtype="float64",
        loss="cross_entropy",
        normalization="none",
    )
    template = make_template(config)
    spec = flat_spec(template)
    parameter = flatten_parameters(template)
    train_pairs, train_labels, _, _, cert_pairs, _ = make_disjoint_split(config)
    generator = torch.Generator().manual_seed(81)
    vectors = torch.randn(3, parameter.numel(), generator=generator, dtype=parameter.dtype)

    batched_hvp = objective_hvp_batch(
        parameter, vectors, train_pairs, train_labels, template, spec, config
    )
    scalar_hvp = torch.stack(
        [
            objective_hvp(
                parameter,
                vector,
                train_pairs,
                train_labels,
                template,
                spec,
                config,
            )
            for vector in vectors
        ]
    )
    hvp_error = relative_error(batched_hvp, scalar_hvp)
    assert hvp_error < 2e-12, hvp_error

    scalar_jvp, scalar_vjp = make_scaled_optimizer_jvp_vjp(
        parameter, train_pairs, train_labels, template, spec, config
    )
    batch_jvp, batch_vjp = make_batched_scaled_optimizer_products(
        parameter, train_pairs, train_labels, template, spec, config
    )
    states = torch.randn(
        3, 2 * parameter.numel(), generator=generator, dtype=parameter.dtype
    )
    jvp_error = relative_error(batch_jvp(states), torch.stack([scalar_jvp(v) for v in states]))
    vjp_error = relative_error(batch_vjp(states), torch.stack([scalar_vjp(v) for v in states]))
    assert jvp_error < 2e-12, jvp_error
    assert vjp_error < 2e-12, vjp_error

    path = torch.stack((parameter, parameter + 1e-4 * vectors[0], parameter - 1e-4 * vectors[1]))
    scalar_green, scalar_green_t = make_transformer_green_products(
        path, train_pairs, train_labels, template, spec, config
    )
    batch_green, batch_green_t = make_batched_transformer_green_products(
        path, train_pairs, train_labels, template, spec, config
    )
    injections = torch.randn(
        3,
        len(path) * 2 * parameter.numel(),
        generator=generator,
        dtype=parameter.dtype,
    )
    green_error = relative_error(
        batch_green(injections), torch.stack([scalar_green(v) for v in injections])
    )
    green_t_error = relative_error(
        batch_green_t(injections), torch.stack([scalar_green_t(v) for v in injections])
    )
    assert green_error < 3e-12, green_error
    assert green_t_error < 3e-12, green_t_error

    scalar_output, _ = make_gram_operator(parameter, cert_pairs, template, spec)
    batch_output = make_batched_output_gram_operator(parameter, cert_pairs, template, spec)
    output_error = relative_error(
        batch_output(vectors), torch.stack([scalar_output(v) for v in vectors])
    )
    assert output_error < 3e-12, output_error

    probe = ProbeConfig(probes=3, power=2, delta=1e-3)
    identity = (9, 8, 7)
    scalar_bound = gram_norm_bound(
        scalar_output,
        dimension=parameter.numel(),
        dtype=parameter.dtype,
        device=parameter.device,
        config=probe,
        identity=identity,
    )
    batch_bound = batched_gram_norm_bound(
        batch_output,
        dimension=parameter.numel(),
        dtype=parameter.dtype,
        device=parameter.device,
        config=probe,
        seed=probe_seed(*identity),
    )
    y_error = abs(batch_bound["Y"] - scalar_bound["Y"]) / scalar_bound["Y"]
    bound_error = abs(
        batch_bound["operator_norm_upper_bound"]
        - scalar_bound["operator_norm_upper_bound"]
    ) / scalar_bound["operator_norm_upper_bound"]
    assert y_error < 3e-12, y_error
    assert bound_error < 3e-12, bound_error
    assert batch_bound["batched_gram_calls"] == probe.power
    assert batch_bound["logical_gram_applications"] == probe.probes * probe.power

    print(
        "PASS: batched HVP/JVP/VJP/Green/output products reproduce scalar "
        f"operators (max relative error {max(hvp_error, jvp_error, vjp_error, green_error, green_t_error, output_error, y_error, bound_error):.3e})."
    )


if __name__ == "__main__":
    main()
