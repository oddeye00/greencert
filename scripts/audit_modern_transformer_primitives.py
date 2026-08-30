#!/usr/bin/env python3
"""Post-seal derivative audit on a deeper LayerNorm + AdamW Transformer."""
from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import torch

from adamw_optimizer_probe import (
    AdamWSettings,
    adamw_step_from_gradient,
    make_adamw_jvp_vjp,
)
from batched_green_operator import make_batched_output_gram_operator
from benchmark_transformer_scaling import peak_rss_bytes
from transformer_green_operator import make_causal_green_products
from transformer_hvp_grokking import (
    TransformerConfig,
    flat_spec,
    flatten_parameters,
    gradient,
    make_disjoint_split,
    make_template,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "modern_transformer_primitive_audit.json"
OUTPUT_MD = ROOT / "results" / "modern_transformer_primitive_audit.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    config = TransformerConfig(
        modulus=17,
        model_dim=64,
        hidden_dim=256,
        heads=4,
        depth=2,
        train_fraction=0.60,
        learning_rate=1e-3,
        momentum=0.0,
        weight_decay=0.0,
        steps=1,
        seed=20260825,
        threads=4,
        dtype="float64",
        loss="cross_entropy",
        normalization="layernorm",
    )
    template = make_template(config)
    spec = flat_spec(template)
    parameter = flatten_parameters(template)
    train_pairs, train_labels, _, _, cert_pairs, _ = make_disjoint_split(config)
    first_moment = torch.zeros_like(parameter)
    # A tiny positive floor models a reachable noninitial checkpoint and keeps
    # the standard sqrt(v_hat)+epsilon derivative away from its singularity.
    second_moment = torch.full_like(parameter, 1e-12)
    settings_base = dict(
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=1e-2,
    )

    products = []
    training_started = time.perf_counter()
    for step in range(1, 6):
        settings = AdamWSettings(step=step, **settings_base)
        if step >= 3:
            products.append(
                make_adamw_jvp_vjp(
                    parameter,
                    first_moment,
                    second_moment,
                    train_pairs,
                    train_labels,
                    template,
                    spec,
                    config,
                    settings,
                )
            )
        objective_gradient = gradient(
            parameter, train_pairs, train_labels, template, spec, config
        )
        parameter, first_moment, second_moment = adamw_step_from_gradient(
            parameter,
            first_moment,
            second_moment,
            objective_gradient,
            settings,
        )
    checkpoint_construction_seconds = time.perf_counter() - training_started
    settings = AdamWSettings(step=6, **settings_base)
    jvp, vjp = make_adamw_jvp_vjp(
        parameter,
        first_moment,
        second_moment,
        train_pairs,
        train_labels,
        template,
        spec,
        config,
        settings,
    )
    dimension = parameter.numel()
    state_dimension = 3 * dimension
    generator = torch.Generator().manual_seed(1107)
    direction = torch.randn(state_dimension, generator=generator, dtype=parameter.dtype)
    cotangent = torch.randn(state_dimension, generator=generator, dtype=parameter.dtype)
    started = time.perf_counter()
    jv = jvp(direction)
    jvp_seconds = time.perf_counter() - started
    started = time.perf_counter()
    jtw = vjp(cotangent)
    vjp_seconds = time.perf_counter() - started
    adjoint_left = float(torch.dot(jv, cotangent))
    adjoint_right = float(torch.dot(direction, jtw))
    optimizer_adjoint_relative_error = abs(adjoint_left - adjoint_right) / max(
        abs(adjoint_left), abs(adjoint_right), 1.0
    )

    green_apply, green_transpose = make_causal_green_products(
        [row[0] for row in products],
        [row[1] for row in products],
        state_dimension,
    )
    green_dimension = len(products) * state_dimension
    injection = torch.randn(
        green_dimension, generator=generator, dtype=parameter.dtype
    )
    green_cotangent = torch.randn(
        green_dimension, generator=generator, dtype=parameter.dtype
    )
    started = time.perf_counter()
    green_value = green_apply(injection)
    green_apply_seconds = time.perf_counter() - started
    started = time.perf_counter()
    green_transposed = green_transpose(green_cotangent)
    green_transpose_seconds = time.perf_counter() - started
    green_left = float(torch.dot(green_value, green_cotangent))
    green_right = float(torch.dot(injection, green_transposed))
    green_adjoint_relative_error = abs(green_left - green_right) / max(
        abs(green_left), abs(green_right), 1.0
    )

    output_gram = make_batched_output_gram_operator(
        parameter, cert_pairs, template, spec
    )
    output_vectors = torch.randn(
        2, dimension, generator=generator, dtype=parameter.dtype
    )
    started = time.perf_counter()
    output_values = output_gram(output_vectors)
    output_gram_batch2_seconds = time.perf_counter() - started
    finite = all(
        bool(torch.isfinite(value).all())
        for value in (parameter, first_moment, second_moment, jv, jtw, green_value, green_transposed, output_values)
    )
    if not finite:
        raise RuntimeError("nonfinite modern-architecture primitive")
    if optimizer_adjoint_relative_error >= 2e-11:
        raise RuntimeError("AdamW optimizer adjoint identity failed")
    if green_adjoint_relative_error >= 2e-11:
        raise RuntimeError("AdamW Green adjoint identity failed")

    result = {
        "status": "post-seal modern-architecture matrix-free primitive audit passed",
        "scope": (
            "This validates exact matrix-free optimizer/Green products on a two-block "
            "pre-LayerNorm AdamW Transformer. It is not a first-passage certificate; "
            "LayerNorm/AdamW global derivative envelopes remain unimplemented."
        ),
        "config": config.__dict__,
        "adamw": settings.__dict__,
        "parameter_count": dimension,
        "optimizer_state_dimension": state_dimension,
        "green_horizon": len(products),
        "strictly_positive_second_moment_minimum": float(second_moment.min()),
        "finite": finite,
        "optimizer_adjoint_relative_error": optimizer_adjoint_relative_error,
        "green_adjoint_relative_error": green_adjoint_relative_error,
        "timings_seconds": {
            "five_step_checkpoint_construction": checkpoint_construction_seconds,
            "adamw_jvp": jvp_seconds,
            "adamw_vjp": vjp_seconds,
            "green_apply_horizon_3": green_apply_seconds,
            "green_transpose_horizon_3": green_transpose_seconds,
            "output_gram_batch_2": output_gram_batch2_seconds,
        },
        "observed_process_peak_rss_bytes": peak_rss_bytes(),
        "source_sha256": {
            "adamw_optimizer_probe.py": sha256(
                ROOT / "scripts" / "adamw_optimizer_probe.py"
            ),
            "batched_green_operator.py": sha256(
                ROOT / "scripts" / "batched_green_operator.py"
            ),
            "audit_modern_transformer_primitives.py": sha256(Path(__file__)),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "torch_threads": torch.get_num_threads(),
            "dtype": "float64",
            "device": "cpu",
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Modern Transformer matrix-free primitive audit",
        "",
        f"- Architecture: two-block pre-LayerNorm Transformer, {dimension:,} parameters.",
        f"- Optimizer: bias-corrected AdamW, {state_dimension:,}-coordinate state.",
        f"- AdamW JVP/VJP adjoint relative error: `{optimizer_adjoint_relative_error:.3e}`.",
        f"- Horizon-3 Green adjoint relative error: `{green_adjoint_relative_error:.3e}`.",
        f"- Peak RSS: `{result['observed_process_peak_rss_bytes']/2**30:.2f} GiB`.",
        "- Scope: exact matrix-free primitives only; no LayerNorm/AdamW jet envelope or event certificate is claimed.",
        "",
    ]
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "parameters": dimension,
        "optimizer_state_dimension": state_dimension,
        "optimizer_adjoint_relative_error": optimizer_adjoint_relative_error,
        "green_adjoint_relative_error": green_adjoint_relative_error,
        "peak_gib": result["observed_process_peak_rss_bytes"] / 2**30,
    }, indent=2))


if __name__ == "__main__":
    main()
