# Exact-input scalar and event assembly

The numerical input contract is part of a certificate. An integer bound
must not become a smaller bound merely because it was converted to a float.
The separately versioned `exact_input_scalar_closure.py` and
`exact_input_window_event_assembly.py` preserve Python integers and Fractions,
and interpret finite Python floats as their exact dyadic values.

This is implementation hardening, not a new theorem or an additional neural
experiment. The frozen scalar solver and event assembler remain unchanged
for historical replay.

## Reproduce the tests

In the repository's pinned Python environment:

```text
python scripts/test_exact_input_scalar_closure.py
python scripts/test_exact_input_window_event_assembly.py
```

Both suites pass on Windows/AMD64 and Linux/aarch64. The event test reports
are byte-identical across the two hosts:
`results/exact_input_event_assembly_{windows,arm}_20260907_v1.json`.
Their SHA256 is
`353ce7959d9ac3e7e16eda658307c74991b38768f9f1afa60abddcd3b5cd031e`.

The event suite preserves 50 existing contract refusals and 32 count-level
configurations (192 admissible count paths). It adds 13 exact-input
refusals and 200 rational output-transport cases. An independent rational
oracle checks the scalar supersolutions and output signs; exhaustive
enumeration checks the 297 compatible count paths. The resulting 114
conditional brackets are synthetic test cases, not empirical certificates.

## Input and output contract

Validation preserves exact values in response norms, derivative bounds,
domain comparisons, interval ordering, Green gains, and failure
probabilities. Rational extrema are computed before Arb enclosure. Output
transport uses outward Arb operations, without an intermediate float sum.
Booleans, implicit numeric strings, nonfinite numbers and negative bounds
are rejected.

For example, a positive injection `Fraction(1, 2**1100)` cannot fit a zero
domain, and a response norm `2**53+1` cannot fit the domain `2**53`.
Tiny positive margins remain positive rather than becoming ties.

The scalar result returns a finite binary64 radius only after checking its
supersolution and domain inequalities against the original exact inputs.
It can abstain when the chosen precision or binary64 radius representation
does not resolve a valid bound. Exact input handling does not guarantee
closure.

Failure probabilities are always encoded exactly as hexadecimal rational
numerator/denominator fields. The legacy-shaped `failure_probability`
field is a float only when that float represents the probability exactly;
otherwise it is null. `failure_probability_upper` is an outward binary64
upper bound. A positive rational probability is never relabeled as zero
or as deterministic.

The supported geometry remains scaled-momentum Euclidean state with an
exact parameter anchor and an explicit first-injection error. The
assembler checks identity, population, complete-window coverage, outward
arithmetic scope and probe completeness. It returns a conditional
first-passage bracket or abstention. It does not authenticate files, prove
supplied neural bounds, or issue a certificate.

## Recorded 451,008-parameter check

A read-only audit reassembled the recorded development window using the
new interface. Both count paths, the `[44,44]` bracket, probability scope,
and radius `7.285063634286043e-16` are unchanged. The radius passes an
independent exact-rational supersolution check.

The audit rechecked 14,899 dependency files and checked 3,717,802 numeric
visits before using legacy bound extractors. All visited numeric inputs
were exactly representable by the conversions those extractors perform.
This is a check of this recorded input set, not every possible caller.
The original neural/Green premises remain those of the completed recorded
replay. No future outcome was read and no certificate was newly issued.

Report: `results/recorded_exact_input_assembly_arm_20260907_v1.json`.
Driver: `scripts/audit_recorded_exact_input_assembly.py`. It requires the
complete external evidence graph and a hash-pinned completed replay.
The component source dependencies are in the version-3 archive described
in [RECORDED_WINDOW_REPLAY.md](RECORDED_WINDOW_REPLAY.md); put its extracted
`scripts` directory on `PYTHONPATH`, then use the driver's `--help` for
the graph, manifest and replay-report arguments. Run without Python's
assertion-disabling `-O` option; the driver rejects that mode.

The full graph is not included in the public repository. This check is
separate from the self-contained tests above, and cannot establish the
realized future crossing while that outcome remains sealed.
