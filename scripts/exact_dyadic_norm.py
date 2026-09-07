"""Exact squared norms of binary64 vectors using bounded uint64 bins.

Return S as a Python integer such that sum(x_i**2) = S * 2**(-2148).
No floating multiplication, BLAS, tolerance, or underflow assumption is
used in the accumulation or in comparison with a saved dyadic bound.
This is a separate audit primitive; frozen producers are not modified.
"""
import math

import numpy as np

MAX_CHUNK = 1 << 20
DEFAULT_CHUNK = 1 << 16
SQUARE_UNIT_EXPONENT = -2148
MASK32 = np.uint64((1 << 32)-1)
MASK52 = np.uint64((1 << 52)-1)


def squared_integer(values, *, chunk_size=DEFAULT_CHUNK):
    """Compute an exact integer, with O(chunk_size + 2048) working storage."""
    if not isinstance(values, np.ndarray) or values.dtype != np.dtype(np.float64) or values.ndim != 1:
        raise ValueError("one-dimensional native binary64 array required; no implicit conversion")
    if type(chunk_size) is not int or not 1 <= chunk_size <= MAX_CHUNK:
        raise ValueError("chunk size outside the proved accumulator bound")
    if np.array([1.], dtype=np.float64).view(np.uint64)[0] != np.uint64(0x3ff0000000000000):
        raise ValueError("IEEE binary64 representation required")
    total = 0
    for start in range(0, values.size, chunk_size):
        words = values[start:start+chunk_size].view(np.uint64)
        exponent = (words >> np.uint64(52)) & np.uint64(0x7ff)
        if np.any(exponent == 0x7ff):
            raise ValueError("nonfinite coordinate")
        mantissa = (words & MASK52) | ((exponent != 0).astype(np.uint64) << np.uint64(52))
        # x = +/- m * 2**(effective_exponent - 1075); subnormals use e=1.
        bins = np.maximum(exponent, np.uint64(1)).astype(np.intp)-1
        low, high = mantissa & MASK32, mantissa >> np.uint64(32)
        low_square = low*low                  # < 2**64, no overflow.
        cross = np.uint64(2)*low*high         # < 2**54.
        middle = (low_square >> np.uint64(32)) + (cross & MASK32)  # < 2**33.
        upper = high*high + (cross >> np.uint64(32)) + (middle >> np.uint64(32))  # < 2**43.
        limbs = (low_square & MASK32, middle & MASK32, upper & MASK32, upper >> np.uint64(32))
        for limb_index, limb in enumerate(limbs):
            accum = np.zeros(2046, dtype=np.uint64)
            # Every term <2**32; a chunk has <=2**20 terms. Every partial
            # bin sum is <2**52, so uint64 addition cannot wrap.
            np.add.at(accum, bins, limb)
            for exponent_index in np.flatnonzero(accum):
                total += int(accum[exponent_index]) << (2*int(exponent_index)+32*limb_index)
    return total


def encloses_norm(squared, bound):
    """Decide bound >= ||x|| by exact integer arithmetic, including subnormals."""
    if type(squared) is not int or squared < 0 or type(bound) not in (int, float):
        raise ValueError("nonnegative exact square and real dyadic bound required")
    if bound < 0 or (type(bound) is float and not math.isfinite(bound)):
        raise ValueError("finite nonnegative bound required")
    numerator, denominator = bound.as_integer_ratio() if type(bound) is float else (bound, 1)
    return (numerator*numerator << 2148) >= squared*denominator*denominator


def norm_upper(values, *, chunk_size=DEFAULT_CHUNK, precision_bits=256):
    """Optional outward float64 upper bound; comparison above needs no Arb."""
    from flint import arb, ctx
    if type(precision_bits) is not int or precision_bits < 64:
        raise ValueError("at least 64 Arb bits required")
    squared = squared_integer(values, chunk_size=chunk_size)
    if squared == 0:
        return 0.
    previous = ctx.prec
    ctx.prec = precision_bits
    try:
        root = arb(squared).sqrt() * (arb(2)**(-1074))
        value = math.nextafter(float(root.upper()), math.inf)
    finally:
        ctx.prec = previous
    if not math.isfinite(value) or not encloses_norm(squared, value):
        raise ValueError("no verified finite binary64 norm upper bound")
    return value
