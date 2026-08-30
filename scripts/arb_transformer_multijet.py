#!/usr/bin/env python3
"""Batched outward first/mixed directional jets for the smooth Transformer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from flint import arb, arb_mat

from transformer_hvp_grokking import FlatSpec, TransformerConfig


@dataclass
class JetScalar:
    value: arb
    y: list[arb]
    x: arb | None = None
    xy: list[arb] | None = None

    @property
    def probes(self) -> int:
        return len(self.y)

    @property
    def mixed(self) -> bool:
        return self.x is not None


@dataclass
class JetMatrix:
    value: arb_mat
    y: list[arb_mat]
    x: arb_mat | None = None
    xy: list[arb_mat] | None = None

    @property
    def probes(self) -> int:
        return len(self.y)

    @property
    def mixed(self) -> bool:
        return self.x is not None

    def transpose(self) -> "JetMatrix":
        return JetMatrix(
            self.value.transpose(),
            [row.transpose() for row in self.y],
            None if self.x is None else self.x.transpose(),
            None if self.xy is None else [row.transpose() for row in self.xy],
        )

    def get(self, row: int, col: int) -> JetScalar:
        return JetScalar(
            self.value[row, col],
            [item[row, col] for item in self.y],
            None if self.x is None else self.x[row, col],
            None if self.xy is None else [item[row, col] for item in self.xy],
        )

    def set(self, row: int, col: int, value: JetScalar) -> None:
        _compatible_scalar_matrix(value, self)
        self.value[row, col] = value.value
        for index, item in enumerate(value.y):
            self.y[index][row, col] = item
        if self.mixed:
            assert self.x is not None and self.xy is not None
            assert value.x is not None and value.xy is not None
            self.x[row, col] = value.x
            for index, item in enumerate(value.xy):
                self.xy[index][row, col] = item


def _compatible(left: JetScalar, right: JetScalar) -> None:
    if left.probes != right.probes or left.mixed != right.mixed:
        raise ValueError("incompatible scalar jets")


def _compatible_matrix(left: JetMatrix, right: JetMatrix) -> None:
    if left.probes != right.probes or left.mixed != right.mixed:
        raise ValueError("incompatible matrix jets")


def _compatible_scalar_matrix(left: JetScalar, right: JetMatrix) -> None:
    if left.probes != right.probes or left.mixed != right.mixed:
        raise ValueError("incompatible scalar/matrix jets")


def constant_like(value, template: JetScalar) -> JetScalar:
    zero = arb(0)
    return JetScalar(
        arb(value),
        [zero for _ in range(template.probes)],
        zero if template.mixed else None,
        [zero for _ in range(template.probes)] if template.mixed else None,
    )


def jadd(left: JetScalar, right: JetScalar) -> JetScalar:
    _compatible(left, right)
    return JetScalar(
        left.value + right.value,
        [a + b for a, b in zip(left.y, right.y)],
        None if not left.mixed else left.x + right.x,  # type: ignore[operator]
        None
        if not left.mixed
        else [a + b for a, b in zip(left.xy or [], right.xy or [])],
    )


def jneg(value: JetScalar) -> JetScalar:
    return JetScalar(
        -value.value,
        [-item for item in value.y],
        None if value.x is None else -value.x,
        None if value.xy is None else [-item for item in value.xy],
    )


def jsub(left: JetScalar, right: JetScalar) -> JetScalar:
    return jadd(left, jneg(right))


def jmul(left: JetScalar, right: JetScalar) -> JetScalar:
    _compatible(left, right)
    y = [
        a * right.value + left.value * b
        for a, b in zip(left.y, right.y)
    ]
    if not left.mixed:
        return JetScalar(left.value * right.value, y)
    assert left.x is not None and right.x is not None
    assert left.xy is not None and right.xy is not None
    xy = [
        axy * right.value
        + left.x * by
        + ay * right.x
        + left.value * bxy
        for axy, by, ay, bxy in zip(left.xy, right.y, left.y, right.xy)
    ]
    return JetScalar(
        left.value * right.value,
        y,
        left.x * right.value + left.value * right.x,
        xy,
    )


def jinv(value: JetScalar) -> JetScalar:
    inverse = 1 / value.value
    inverse2 = inverse * inverse
    y = [-item * inverse2 for item in value.y]
    if not value.mixed:
        return JetScalar(inverse, y)
    assert value.x is not None and value.xy is not None
    inverse3 = inverse2 * inverse
    return JetScalar(
        inverse,
        y,
        -value.x * inverse2,
        [
            2 * value.x * item * inverse3 - mixed * inverse2
            for item, mixed in zip(value.y, value.xy)
        ],
    )


def jdiv(left: JetScalar, right: JetScalar) -> JetScalar:
    return jmul(left, jinv(right))


def jexp(value: JetScalar) -> JetScalar:
    base = value.value.exp()
    y = [base * item for item in value.y]
    if not value.mixed:
        return JetScalar(base, y)
    assert value.x is not None and value.xy is not None
    return JetScalar(
        base,
        y,
        base * value.x,
        [
            base * (mixed + value.x * item)
            for item, mixed in zip(value.y, value.xy)
        ],
    )


def jlog(value: JetScalar) -> JetScalar:
    inverse = 1 / value.value
    y = [item * inverse for item in value.y]
    if not value.mixed:
        return JetScalar(value.value.log(), y)
    assert value.x is not None and value.xy is not None
    return JetScalar(
        value.value.log(),
        y,
        value.x * inverse,
        [
            mixed * inverse - value.x * item * inverse * inverse
            for item, mixed in zip(value.y, value.xy)
        ],
    )


def jgelu(value: JetScalar) -> JetScalar:
    root_two = arb(2).sqrt()
    root_two_pi = (2 * arb.pi()).sqrt()
    phi = (-value.value * value.value / 2).exp() / root_two_pi
    cdf = (1 + (value.value / root_two).erf()) / 2
    base = value.value * cdf
    first = cdf + value.value * phi
    second = (2 - value.value * value.value) * phi
    y = [first * item for item in value.y]
    if not value.mixed:
        return JetScalar(base, y)
    assert value.x is not None and value.xy is not None
    return JetScalar(
        base,
        y,
        first * value.x,
        [
            second * value.x * item + first * mixed
            for item, mixed in zip(value.y, value.xy)
        ],
    )


def zero_matrix(rows: int, cols: int, probes: int, *, mixed: bool) -> JetMatrix:
    return JetMatrix(
        arb_mat(rows, cols),
        [arb_mat(rows, cols) for _ in range(probes)],
        arb_mat(rows, cols) if mixed else None,
        [arb_mat(rows, cols) for _ in range(probes)] if mixed else None,
    )


def madd(left: JetMatrix, right: JetMatrix) -> JetMatrix:
    _compatible_matrix(left, right)
    return JetMatrix(
        left.value + right.value,
        [a + b for a, b in zip(left.y, right.y)],
        None if not left.mixed else left.x + right.x,  # type: ignore[operator]
        None
        if not left.mixed
        else [a + b for a, b in zip(left.xy or [], right.xy or [])],
    )


def mmatmul(left: JetMatrix, right: JetMatrix) -> JetMatrix:
    _compatible_matrix(left, right)
    value = left.value * right.value
    y = [
        ay * right.value + left.value * by
        for ay, by in zip(left.y, right.y)
    ]
    if not left.mixed:
        return JetMatrix(value, y)
    assert left.x is not None and right.x is not None
    assert left.xy is not None and right.xy is not None
    x = left.x * right.value + left.value * right.x
    xy = [
        axy * right.value
        + left.x * by
        + ay * right.x
        + left.value * bxy
        for axy, by, ay, bxy in zip(left.xy, right.y, left.y, right.xy)
    ]
    return JetMatrix(value, y, x, xy)


def mscale(value: JetMatrix, scale: arb) -> JetMatrix:
    return JetMatrix(
        value.value / scale,
        [item / scale for item in value.y],
        None if value.x is None else value.x / scale,
        None if value.xy is None else [item / scale for item in value.xy],
    )


def _flat_matrix(entries: Sequence[arb], shape: tuple[int, ...]) -> arb_mat:
    if len(shape) == 1:
        rows, cols = 1, shape[0]
    elif len(shape) == 2:
        rows, cols = shape
    else:
        raise ValueError(f"unsupported parameter rank: {shape}")
    return arb_mat(rows, cols, list(entries))


@dataclass
class JetParameterMap:
    values: dict[str, JetMatrix]
    flat: list[JetScalar]


def make_parameter_jet(
    base: Sequence[float],
    y_directions: Sequence[Sequence[float]],
    spec: FlatSpec,
    *,
    x_direction: Sequence[float] | None = None,
    base_terms: Sequence[tuple[float | arb, Sequence[float]]] = (),
    y_direction_terms: Sequence[
        Sequence[tuple[float | arb, Sequence[float]]]
    ] = (),
) -> JetParameterMap:
    base_array = np.asarray(base, dtype=np.float64).reshape(-1)
    ys = [np.asarray(row, dtype=np.float64).reshape(-1) for row in y_directions]
    x = None if x_direction is None else np.asarray(x_direction, dtype=np.float64).reshape(-1)
    offsets = [
        (scale if isinstance(scale, arb) else arb(float(scale)), np.asarray(row, dtype=np.float64).reshape(-1))
        for scale, row in base_terms
    ]
    expanded_y_terms = [
        [
            (
                scale if isinstance(scale, arb) else arb(float(scale)),
                np.asarray(row, dtype=np.float64).reshape(-1),
            )
            for scale, row in probe_terms
        ]
        for probe_terms in y_direction_terms
    ]
    if expanded_y_terms and ys:
        raise ValueError("provide direct y directions or exact y-direction terms, not both")
    if expanded_y_terms:
        ys = [np.zeros_like(base_array) for _ in expanded_y_terms]
    if base_array.size != sum(spec.sizes):
        raise ValueError("base parameter has the wrong dimension")
    if any(row.size != base_array.size for row in ys):
        raise ValueError("probe direction has the wrong dimension")
    if x is not None and x.size != base_array.size:
        raise ValueError("mixed direction has the wrong dimension")
    if any(row.size != base_array.size for _, row in offsets):
        raise ValueError("base offset has the wrong dimension")
    if any(
        row.size != base_array.size
        for probe_terms in expanded_y_terms
        for _, row in probe_terms
    ):
        raise ValueError("y-direction term has the wrong dimension")
    values: dict[str, JetMatrix] = {}
    flat: list[JetScalar] = []
    offset = 0
    for name, shape, size in zip(spec.names, spec.shapes, spec.sizes):
        dims = tuple(int(value) for value in shape)
        base_entries = []
        for local, value in enumerate(base_array[offset : offset + size]):
            entry = arb(float(value))
            for scale, row in offsets:
                entry += scale * arb(float(row[offset + local]))
            base_entries.append(entry)
        if expanded_y_terms:
            y_entries = []
            for probe_terms in expanded_y_terms:
                entries = []
                for local in range(size):
                    entry = arb(0)
                    for scale, row in probe_terms:
                        entry += scale * arb(float(row[offset + local]))
                    entries.append(entry)
                y_entries.append(entries)
        else:
            y_entries = [
                [arb(float(value)) for value in row[offset : offset + size]] for row in ys
            ]
        x_entries = (
            None
            if x is None
            else [arb(float(value)) for value in x[offset : offset + size]]
        )
        matrix = JetMatrix(
            _flat_matrix(base_entries, dims),
            [_flat_matrix(row, dims) for row in y_entries],
            None if x_entries is None else _flat_matrix(x_entries, dims),
            None
            if x_entries is None
            else [_flat_matrix([arb(0) for _ in range(size)], dims) for _ in ys],
        )
        values[name] = matrix
        for index in range(size):
            flat.append(
                JetScalar(
                    base_entries[index],
                    [row[index] for row in y_entries],
                    None if x_entries is None else x_entries[index],
                    None if x_entries is None else [arb(0) for _ in ys],
                )
            )
        offset += size
    return JetParameterMap(values, flat)


def _linear(hidden: JetMatrix, weight: JetMatrix, bias: JetMatrix | None) -> JetMatrix:
    output = mmatmul(hidden, weight.transpose())
    if bias is not None:
        for row in range(output.value.nrows()):
            for col in range(output.value.ncols()):
                output.set(row, col, jadd(output.get(row, col), bias.get(0, col)))
    return output


def _softmax(values: Sequence[JetScalar]) -> list[JetScalar]:
    shift = max(float(value.value.mid()) for value in values)
    exponentials = []
    for value in values:
        shifted = JetScalar(
            value.value - shift,
            value.y,
            value.x,
            value.xy,
        )
        exponentials.append(jexp(shifted))
    total = constant_like(0, exponentials[0])
    for value in exponentials:
        total = jadd(total, value)
    return [jdiv(value, total) for value in exponentials]


def _attention(
    hidden: JetMatrix,
    parameter: JetParameterMap,
    *,
    examples: int,
    model_dim: int,
    heads: int,
) -> JetMatrix:
    values = parameter.values
    qkv = _linear(
        hidden,
        values["blocks.0.attention.in_proj_weight"],
        values["blocks.0.attention.in_proj_bias"],
    )
    head_dim = model_dim // heads
    joined = zero_matrix(examples * 3, model_dim, hidden.probes, mixed=hidden.mixed)
    scale = arb(head_dim).sqrt()
    for example in range(examples):
        base_row = 3 * example
        for head in range(heads):
            q = zero_matrix(3, head_dim, hidden.probes, mixed=hidden.mixed)
            k = zero_matrix(3, head_dim, hidden.probes, mixed=hidden.mixed)
            v = zero_matrix(3, head_dim, hidden.probes, mixed=hidden.mixed)
            base_col = head * head_dim
            for token in range(3):
                for col in range(head_dim):
                    q.set(token, col, qkv.get(base_row + token, base_col + col))
                    k.set(
                        token,
                        col,
                        qkv.get(base_row + token, model_dim + base_col + col),
                    )
                    v.set(
                        token,
                        col,
                        qkv.get(base_row + token, 2 * model_dim + base_col + col),
                    )
            scores = mscale(mmatmul(q, k.transpose()), scale)
            probabilities = zero_matrix(3, 3, hidden.probes, mixed=hidden.mixed)
            for query in range(3):
                row = _softmax([scores.get(query, key) for key in range(query + 1)])
                for key, probability in enumerate(row):
                    probabilities.set(query, key, probability)
            attended = mmatmul(probabilities, v)
            for token in range(3):
                for col in range(head_dim):
                    joined.set(
                        base_row + token,
                        base_col + col,
                        attended.get(token, col),
                    )
    return _linear(
        joined,
        values["blocks.0.attention.out_proj.weight"],
        values["blocks.0.attention.out_proj.bias"],
    )


def arb_transformer_objective_jet(
    parameter: JetParameterMap,
    pairs: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    config: TransformerConfig,
) -> JetScalar:
    if config.normalization != "none" or config.depth != 1 or config.loss != "cross_entropy":
        raise ValueError("jet evaluator supports one no-norm cross-entropy block")
    pair_array = np.asarray(
        pairs.detach().cpu().numpy() if isinstance(pairs, torch.Tensor) else pairs,
        dtype=np.int64,
    )
    label_array = np.asarray(
        labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else labels,
        dtype=np.int64,
    ).reshape(-1)
    examples = int(pair_array.shape[0])
    probes = len(parameter.flat[0].y)
    mixed = parameter.flat[0].mixed
    d = int(config.model_dim)
    embedding = parameter.values["token_embedding.weight"]
    position = parameter.values["position_embedding"]
    hidden = zero_matrix(3 * examples, d, probes, mixed=mixed)
    for example, (left, right) in enumerate(pair_array):
        for token_index, token in enumerate((int(left), int(right), int(config.modulus))):
            for col in range(d):
                hidden.set(
                    3 * example + token_index,
                    col,
                    jadd(embedding.get(token, col), position.get(token_index, col)),
                )
    hidden = madd(
        hidden,
        _attention(
            hidden,
            parameter,
            examples=examples,
            model_dim=d,
            heads=int(config.heads),
        ),
    )
    feedforward = _linear(
        hidden,
        parameter.values["blocks.0.linear1.weight"],
        parameter.values["blocks.0.linear1.bias"],
    )
    activated = zero_matrix(
        feedforward.value.nrows(),
        feedforward.value.ncols(),
        probes,
        mixed=mixed,
    )
    for row in range(feedforward.value.nrows()):
        for col in range(feedforward.value.ncols()):
            activated.set(row, col, jgelu(feedforward.get(row, col)))
    hidden = madd(
        hidden,
        _linear(
            activated,
            parameter.values["blocks.0.linear2.weight"],
            parameter.values["blocks.0.linear2.bias"],
        ),
    )
    last = zero_matrix(examples, d, probes, mixed=mixed)
    for example in range(examples):
        for col in range(d):
            last.set(example, col, hidden.get(3 * example + 2, col))
    logits = _linear(last, parameter.values["readout.weight"], None)
    loss = constant_like(0, parameter.flat[0])
    for example in range(examples):
        row = [logits.get(example, col) for col in range(int(config.modulus))]
        shift = max(float(value.value.mid()) for value in row)
        total = constant_like(0, row[0])
        for value in row:
            shifted = JetScalar(value.value - shift, value.y, value.x, value.xy)
            total = jadd(total, jexp(shifted))
        normalizer = jadd(constant_like(shift, total), jlog(total))
        loss = jadd(loss, jsub(normalizer, row[int(label_array[example])]))
    loss = jdiv(loss, constant_like(examples, loss))
    regularizer = constant_like(0, loss)
    for value in parameter.flat:
        regularizer = jadd(regularizer, jmul(value, value))
    regularizer = jmul(
        constant_like(float(config.weight_decay) / 2.0, loss), regularizer
    )
    return jadd(loss, regularizer)
