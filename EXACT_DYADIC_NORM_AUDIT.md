# Exact-integer norm auditing

This is an implementation-level verification primitive, not a change to the
GREENCERT theorem, a new certificate, or a claim of a new summation method.
Its intended use is to check a saved binary64 norm bound without looping
through one arbitrary-precision floating operation per coordinate.

Fixed-point superaccumulation is established; see
[Neal (2015), *Fast exact summation using small and large superaccumulators*](https://arxiv.org/abs/1505.05571).
Here the accumulator receives exact significand products, rather than
binary64-rounded squares, and checks an already recorded norm inequality.

## Reproduce the fixed-vector benchmark

The public archive contains six predetermined, evenly spaced norm records
from a 1,408-binding development population. Each vector has 902,016
coordinates. The original saved vectors, bounds, binding summaries, and audit
record are included with byte hashes. They are reference/Green-product data,
not observed future optimizer states. This does not add any empirical event
to the manuscript or distribute the full larger-model certificate graph.

With Python 3.12, NumPy 2.5.2, and python-flint 0.9.0, run from the repo root:

```text
python scripts/test_exact_dyadic_norm.py
python scripts/extract_saved_norm_benchmark.py --extract output/saved_norm_benchmark
python output/saved_norm_benchmark/scripts/benchmark_saved_dyadic_norms.py --manifest-sha256 5817bf337c156f7055e23108dc0d15e31935b0bff09a93e5d3899ac23baf2ea1
```

Extraction requires a new output directory and verifies the archive before
exposing its code. The archive SHA256 is
`1943fbfbda015e9b6907a55703a9d286e5ede568512f78714e5856a6235b076d`.
The comparator is the unchanged 256-bit Arb coordinate-loop implementation.
Each kernel runs three times on the same loaded array, alternating order;
file reads and hash checks are outside those kernel timings.

| Execution | Median of six pairwise speedups |
| --- | ---: |
| Initial Windows audit driver | 18.26x |
| Standalone package, Windows/AMD64 | 17.89x |
| Same package, Linux/aarch64 | 20.83x |

All original norm bounds are retained. The initial and standalone Windows
records are both reported, not selected by speed. Raw per-repetition timings
are in `results/exact_dyadic_norm_benchmark_windows_20260907.json`,
`results/exact_dyadic_norm_portable_windows_20260907.json`, and
`results/exact_dyadic_norm_benchmark_arm_20260907.json`.
The machines were not dedicated performance hosts; these are kernel
measurements on six fixed arrays, not a broad hardware benchmark or an
end-to-end certificate-construction speedup.

## Exact representation

For a finite IEEE binary64 value, let `e` be the 11-bit exponent field and
`f` its 52-bit fraction. Set `a = max(e,1)` and
`m = f + 2^52` for normal values, `m = f` for subnormals and zero. Then

```text
x = sign * m * 2^(a - 1075)
x^2 = m^2 * 2^(2(a-1)) * 2^(-2148).
```

Thus the squared norm is exactly `S * 2^(-2148)` for a nonnegative integer
`S`. Sign bits do not affect this sum. In particular, the square of the
smallest subnormal contributes `1` to `S`; it cannot disappear by underflow.

Write `m = l + 2^32 h`, where `l < 2^32` and `h < 2^21`. Compute
`l^2`, `2lh`, and `h^2` in unsigned 64-bit arithmetic. Their respective
upper bounds are `2^64`, `2^54`, and `2^42`, strictly. Splitting the products
and carrying through the two middle words gives four base-`2^32` limbs of
the exact integer `m^2`. The intermediate middle sum is below `2^33`, and
the final upper sum is below `2^43`, so no multiplication or carry wraps.

The implementation groups each limb by `a-1`. Each chunk has at most
`2^20` coordinates, and each limb is below `2^32`. Consequently every
intermediate unsigned bin sum is below `2^52`, well within 64 bits.
After each chunk, the bins are shifted and added to an arbitrary-precision
Python integer. Chunk boundaries and accumulation order do not change `S`.

For a finite nonnegative saved bound `b = p/q`, the norm test is exactly

```text
(p^2 << 2148) >= S q^2.
```

No square root or floating arithmetic is required for this comparison.
The optional `norm_upper` function computes an outward Arb square root and
then checks the resulting binary64 upper bound using the integer test.
A norm with no representable finite upper bound is refused.

## Implementation and scope

The primitive uses explicitly sized NumPy unsigned integers and unbuffered
`add.at` accumulation, including repeated indices. These semantics are
documented in the [NumPy type reference](https://numpy.org/doc/stable/user/basics.types.html)
and [`ufunc.at` reference](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html).
The mathematical argument assumes those integer operations and the host's
IEEE binary64 representation behave as specified. It is not a proof of the
NumPy compiler or hardware. The runtime checks the binary64 representation
and refuses nonfinite coordinates, implicit dtype conversion, and chunk
sizes outside the proved bound.

`scripts/test_exact_dyadic_norm.py` compares against a separate oracle based
on Python's exact `float.as_integer_ratio`, including every finite exponent
field, subnormal boundaries, seeded bit patterns, carry-heavy maximal chunks,
strided arrays, overflow, and a deliberately one-ulp-inward norm bound.

The original Arb norm producers and sealed artifacts remain unchanged.
Passing a saved norm bound verifies that inequality only; it does not replay
the neural HVP that produced a vector, prove an event, or validate a PRNG.
Any speedup must be measured on matched arrays and reported as a norm-audit
speedup, not as end-to-end certificate construction speed.
