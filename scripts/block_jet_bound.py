#!/usr/bin/env python3
"""Block-aware polynomial jet bounds for the smooth Transformer.

The shipped scalar jet (``transformer_jet_bound``) represents each parameter
block's sensitivity by the constant ``1.0``, which allows the *entire* unit
perturbation budget to be spent in every block simultaneously.  A unit
perturbation ``u`` satisfies ``sum_b ||u_b||^2 = 1``, so that is strictly
pessimistic.

This module carries the same jet, but each derivative order is a *polynomial*
in the per-block perturbation radii ``s_b`` instead of a scalar:

    first  = sum_b        c_b   s_b
    second = sum_{b<=b'}  c_bb' s_b s_b'
    third  = sum_{b<=b'<=b''} ... s_b s_b' s_b''

Setting every ``s_b = 1`` recovers the shipped scalar jet exactly, so the
polynomial bound is valid whenever the scalar one is.  The bound is then
maximised over the sphere ``sum_b s_b^2 <= 1`` monomial by monomial:

    sup_{||s||=1} prod_b s_b^{a_b}  =  prod_b (a_b/d)^{a_b/2},     d = sum_b a_b

which is exact for a single monomial (Lagrange), and

    sup (sum of monomials)  <=  sum of (monomial suprema)

is a valid relaxation.  Because every monomial supremum is at most 1, the
resulting bound never exceeds the shipped scalar jet.  ``test_block_jet_bound.py``
asserts both that domination and a random-direction stress test.

No probabilistic argument appears in this module; it is entirely deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Dict, Tuple

Monomial = Tuple[int, ...]


def _merge(target: Dict[Monomial, float], source: Dict[Monomial, float], scale: float) -> None:
    for key, coefficient in source.items():
        if coefficient == 0.0:
            continue
        target[key] = target.get(key, 0.0) + scale * coefficient


def _convolve(
    left: Dict[Monomial, float], right: Dict[Monomial, float], scale: float
) -> Dict[Monomial, float]:
    out: Dict[Monomial, float] = {}
    for lk, lv in left.items():
        if lv == 0.0:
            continue
        for rk, rv in right.items():
            if rv == 0.0:
                continue
            key = tuple(sorted(lk + rk))
            out[key] = out.get(key, 0.0) + scale * lv * rv
    return out


def monomial_sphere_supremum(monomial: Monomial) -> float:
    """sup of prod_b s_b^{a_b} on the unit sphere sum_b s_b^2 = 1."""
    degree = len(monomial)
    if degree == 0:
        return 1.0
    exponents: Dict[int, int] = {}
    for block in monomial:
        exponents[block] = exponents.get(block, 0) + 1
    value = 1.0
    for exponent in exponents.values():
        value *= (exponent / degree) ** (exponent / 2.0)
    return value


@dataclass
class BlockJet:
    """value plus degree-1/2/3 polynomials in the per-block radii."""

    value: float
    p1: Dict[Monomial, float] = field(default_factory=dict)
    p2: Dict[Monomial, float] = field(default_factory=dict)
    p3: Dict[Monomial, float] = field(default_factory=dict)

    # --- reductions ----------------------------------------------------
    def scalar(self, order: int) -> float:
        """The shipped scalar jet: every s_b = 1."""
        table = {1: self.p1, 2: self.p2, 3: self.p3}[order]
        return sum(table.values())

    def sphere(self, order: int) -> float:
        """Valid maximum over sum_b s_b^2 <= 1, monomial by monomial."""
        table = {1: self.p1, 2: self.p2, 3: self.p3}[order]
        return sum(c * monomial_sphere_supremum(m) for m, c in table.items())

    def evaluate(self, order: int, radii: Dict[int, float]) -> float:
        """Evaluate the polynomial at an explicit radius assignment."""
        table = {1: self.p1, 2: self.p2, 3: self.p3}[order]
        total = 0.0
        for monomial, coefficient in table.items():
            term = coefficient
            for block in monomial:
                term *= radii.get(block, 0.0)
            total += term
        return total


def constant(value: float) -> BlockJet:
    return BlockJet(value)


def block_linear(value: float, block: int, coefficient: float = 1.0) -> BlockJet:
    return BlockJet(value, {(block,): coefficient})


def add(left: BlockJet, right: BlockJet) -> BlockJet:
    out = BlockJet(left.value + right.value)
    for order, table in ((1, out.p1), (2, out.p2), (3, out.p3)):
        _merge(table, {1: left.p1, 2: left.p2, 3: left.p3}[order], 1.0)
        _merge(table, {1: right.p1, 2: right.p2, 3: right.p3}[order], 1.0)
    return out


def product(left: BlockJet, right: BlockJet, *, scale: float = 1.0) -> BlockJet:
    out = BlockJet(scale * left.value * right.value)
    _merge(out.p1, left.p1, scale * right.value)
    _merge(out.p1, right.p1, scale * left.value)

    _merge(out.p2, left.p2, scale * right.value)
    _merge(out.p2, right.p2, scale * left.value)
    _merge(out.p2, _convolve(left.p1, right.p1, 1.0), 2.0 * scale)

    _merge(out.p3, left.p3, scale * right.value)
    _merge(out.p3, right.p3, scale * left.value)
    _merge(out.p3, _convolve(left.p2, right.p1, 1.0), 3.0 * scale)
    _merge(out.p3, _convolve(left.p1, right.p2, 1.0), 3.0 * scale)
    return out


def smooth_map(
    source: BlockJet, *, value: float, first: float, second: float, third: float
) -> BlockJet:
    out = BlockJet(value)
    _merge(out.p1, source.p1, first)

    _merge(out.p2, source.p2, first)
    _merge(out.p2, _convolve(source.p1, source.p1, 1.0), second)

    _merge(out.p3, source.p3, first)
    _merge(out.p3, _convolve(source.p1, source.p2, 1.0), 3.0 * second)
    _merge(
        out.p3,
        _convolve(_convolve(source.p1, source.p1, 1.0), source.p1, 1.0),
        third,
    )
    return out


def affine_parameter(
    source: BlockJet,
    *,
    weight_operator: float,
    weight_block: int,
    bias_norm: float,
    bias_block: int,
    bias_repetitions: int,
) -> BlockJet:
    """Bound ``source @ W.T + b`` when source, W and b all vary."""
    bias_scale = sqrt(float(bias_repetitions))
    weight = block_linear(weight_operator, weight_block, 1.0)
    bias = block_linear(bias_scale * bias_norm, bias_block, bias_scale)
    return add(product(source, weight), bias)
