#!/usr/bin/env python3
"""Outward Arb evaluation of the smooth no-LayerNorm Transformer objective.

This is a scalar verifier, not an interval tensor training implementation.  It
reconstructs the exact forward objective from dyadic parameters using Arb ball
arithmetic.  Directional derivatives can then be enclosed by high-precision
finite differences plus explicit derivative-remainder bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from flint import arb, arb_mat

from transformer_hvp_grokking import FlatSpec, TransformerConfig


@dataclass(frozen=True)
class ArbParameterMap:
    names: tuple[str, ...]
    values: dict[str, arb_mat]
    flat: tuple[arb, ...]


def _as_arb(value) -> arb:
    return value if isinstance(value, arb) else arb(float(value))


def arb_line_point(
    base: Sequence[float],
    terms: Iterable[tuple[float | arb, Sequence[float]]],
) -> tuple[arb, ...]:
    """Return the exact Arb point ``base + sum(scale * direction)``."""

    base_array = np.asarray(base, dtype=np.float64).reshape(-1)
    term_rows = [(_as_arb(scale), np.asarray(row, dtype=np.float64).reshape(-1)) for scale, row in terms]
    if any(row.size != base_array.size for _, row in term_rows):
        raise ValueError("all line directions must match the base vector")
    output = []
    for index, value in enumerate(base_array):
        point = arb(float(value))
        for scale, row in term_rows:
            point += scale * arb(float(row[index]))
        output.append(point)
    return tuple(output)


def _matrix(rows: int, cols: int, entries: Sequence[arb]) -> arb_mat:
    if len(entries) != rows * cols:
        raise ValueError("matrix entry count mismatch")
    return arb_mat(rows, cols, list(entries))


def unflatten_arb(parameter: Sequence[arb], spec: FlatSpec) -> ArbParameterMap:
    if len(parameter) != sum(spec.sizes):
        raise ValueError("flat parameter has the wrong dimension")
    values: dict[str, arb_mat] = {}
    offset = 0
    for name, shape, size in zip(spec.names, spec.shapes, spec.sizes):
        dims = tuple(int(value) for value in shape)
        if len(dims) == 1:
            rows, cols = 1, dims[0]
        elif len(dims) == 2:
            rows, cols = dims
        else:
            raise ValueError(f"unsupported parameter rank for {name}: {dims}")
        values[name] = _matrix(rows, cols, parameter[offset : offset + size])
        offset += size
    return ArbParameterMap(tuple(spec.names), values, tuple(parameter))


def _linear(hidden: arb_mat, weight: arb_mat, bias: arb_mat | None) -> arb_mat:
    output = hidden * weight.transpose()
    if bias is not None:
        if bias.nrows() != 1 or bias.ncols() != output.ncols():
            raise ValueError("linear bias shape mismatch")
        for row in range(output.nrows()):
            for col in range(output.ncols()):
                output[row, col] += bias[0, col]
    return output


def _gelu(value: arb) -> arb:
    return value * (1 + (value / arb(2).sqrt()).erf()) / 2


def _softmax_row(values: Sequence[arb]) -> list[arb]:
    # Subtracting an arbitrary point constant is an exact identity and keeps
    # exponentials compact; no interval ordering decision enters the proof.
    shift = max(float(value.mid()) for value in values)
    exponentials = [(value - shift).exp() for value in values]
    total = sum(exponentials, arb(0))
    return [value / total for value in exponentials]


def _attention(
    hidden: arb_mat,
    parameter: ArbParameterMap,
    *,
    examples: int,
    model_dim: int,
    heads: int,
) -> arb_mat:
    values = parameter.values
    qkv = _linear(
        hidden,
        values["blocks.0.attention.in_proj_weight"],
        values["blocks.0.attention.in_proj_bias"],
    )
    head_dim = model_dim // heads
    if head_dim * heads != model_dim:
        raise ValueError("model dimension must divide the number of heads")
    joined = arb_mat(examples * 3, model_dim)
    scale = arb(head_dim).sqrt()
    for example in range(examples):
        base_row = 3 * example
        for head in range(heads):
            q = arb_mat(3, head_dim)
            k = arb_mat(3, head_dim)
            v = arb_mat(3, head_dim)
            base_col = head * head_dim
            for token in range(3):
                for col in range(head_dim):
                    q[token, col] = qkv[base_row + token, base_col + col]
                    k[token, col] = qkv[
                        base_row + token, model_dim + base_col + col
                    ]
                    v[token, col] = qkv[
                        base_row + token, 2 * model_dim + base_col + col
                    ]
            scores = (q * k.transpose()) / scale
            probabilities = arb_mat(3, 3)
            for query in range(3):
                allowed = [scores[query, key] for key in range(query + 1)]
                row = _softmax_row(allowed)
                for key, probability in enumerate(row):
                    probabilities[query, key] = probability
            attended = probabilities * v
            for token in range(3):
                for col in range(head_dim):
                    joined[base_row + token, base_col + col] = attended[token, col]
    return _linear(
        joined,
        values["blocks.0.attention.out_proj.weight"],
        values["blocks.0.attention.out_proj.bias"],
    )


def arb_transformer_objective(
    parameter: Sequence[arb] | Sequence[float] | np.ndarray,
    pairs: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    spec: FlatSpec,
    config: TransformerConfig,
) -> arb:
    """Outward-enclose the full-batch objective at one exact parameter point."""

    if config.normalization != "none" or config.depth != 1:
        raise ValueError("the outward scalar evaluator currently supports one no-norm block")
    if config.loss != "cross_entropy":
        raise ValueError("the outward scalar evaluator currently supports cross entropy")
    flat = tuple(_as_arb(value) for value in parameter)
    mapped = unflatten_arb(flat, spec)
    values = mapped.values
    pair_array = np.asarray(
        pairs.detach().cpu().numpy() if isinstance(pairs, torch.Tensor) else pairs,
        dtype=np.int64,
    )
    label_array = np.asarray(
        labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else labels,
        dtype=np.int64,
    ).reshape(-1)
    examples = int(pair_array.shape[0])
    if pair_array.shape != (examples, 2) or label_array.size != examples:
        raise ValueError("training data shape mismatch")
    d = int(config.model_dim)
    embedding = values["token_embedding.weight"]
    position = values["position_embedding"]
    hidden = arb_mat(3 * examples, d)
    for example, (left, right) in enumerate(pair_array):
        tokens = (int(left), int(right), int(config.modulus))
        for token_index, token in enumerate(tokens):
            for col in range(d):
                hidden[3 * example + token_index, col] = (
                    embedding[token, col] + position[token_index, col]
                )

    attended = _attention(
        hidden,
        mapped,
        examples=examples,
        model_dim=d,
        heads=int(config.heads),
    )
    hidden = hidden + attended
    feedforward = _linear(
        hidden,
        values["blocks.0.linear1.weight"],
        values["blocks.0.linear1.bias"],
    )
    feedforward = arb_mat(
        feedforward.nrows(),
        feedforward.ncols(),
        [_gelu(value) for value in feedforward.entries()],
    )
    feedforward = _linear(
        feedforward,
        values["blocks.0.linear2.weight"],
        values["blocks.0.linear2.bias"],
    )
    hidden = hidden + feedforward
    last = arb_mat(examples, d)
    for example in range(examples):
        for col in range(d):
            last[example, col] = hidden[3 * example + 2, col]
    logits = _linear(last, values["readout.weight"], None)
    loss = arb(0)
    for example in range(examples):
        row = [logits[example, col] for col in range(int(config.modulus))]
        shift = max(float(value.mid()) for value in row)
        log_normalizer = arb(shift) + sum(
            ((value - shift).exp() for value in row), arb(0)
        ).log()
        loss += log_normalizer - row[int(label_array[example])]
    loss /= examples
    regularizer = sum((value * value for value in mapped.flat), arb(0))
    return loss + arb(config.weight_decay) * regularizer / 2
