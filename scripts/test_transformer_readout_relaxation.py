#!/usr/bin/env python3
"""Check the sealed readout superset against the exact bias-free graph."""
from __future__ import annotations

import math

from block_jet_bound import (
    BlockJet,
    affine_parameter as block_affine,
    block_linear,
    product as block_product,
)
from transformer_jet_bound import (
    JetBound,
    affine_parameter as scalar_affine,
    product as scalar_product,
)


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-14, abs_tol=1e-14)


def dict_close(left: dict, right: dict) -> bool:
    keys = set(left) | set(right)
    return all(close(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)


def main() -> None:
    hidden = JetBound(2.0, 3.0, 4.0, 5.0)
    weight = JetBound(1.7, 1.0, 0.0, 0.0)
    exact = scalar_product(hidden, weight)
    relaxed = scalar_affine(
        hidden, weight_operator=1.7, bias_norm=0.0, bias_repetitions=1
    )
    assert close(relaxed.value, exact.value)
    assert close(relaxed.first, exact.first + 1.0)
    assert close(relaxed.second, exact.second)
    assert close(relaxed.third, exact.third)

    block = BlockJet(
        2.0,
        {(1,): 3.0},
        {(1, 2): 4.0},
        {(1, 2, 3): 5.0},
    )
    exact_block = block_product(block, block_linear(1.7, 12, 1.0))
    relaxed_block = block_affine(
        block,
        weight_operator=1.7,
        weight_block=12,
        bias_norm=0.0,
        bias_block=12,
        bias_repetitions=1,
    )
    difference = dict(relaxed_block.p1)
    for key, value in exact_block.p1.items():
        difference[key] = difference.get(key, 0.0) - value
    assert dict_close(difference, {(12,): 1.0})
    assert dict_close(relaxed_block.p2, exact_block.p2)
    assert dict_close(relaxed_block.p3, exact_block.p3)
    assert close(relaxed_block.value, exact_block.value)

    # Cross-entropy and margin compositions used by the paper have nonnegative
    # coefficients in the jet arguments, so removing the auxiliary direction
    # can only tighten them.
    exact_ce = 2.0 * exact.first**3 + 1.5 * exact.first * exact.second + math.sqrt(2.0) * exact.third
    relaxed_ce = (
        2.0 * relaxed.first**3
        + 1.5 * relaxed.first * relaxed.second
        + math.sqrt(2.0) * relaxed.third
    )
    assert relaxed_ce >= exact_ce
    print(
        "PASS: the sealed bias-free-readout relaxation adds exactly one "
        "degree-one monomial and no higher-order term."
    )


if __name__ == "__main__":
    main()
