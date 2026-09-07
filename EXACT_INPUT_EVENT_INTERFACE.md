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
python scripts/test_roundtrip_evidence_reader.py
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

## Authenticate the serialized numbers, too

An exact assembler cannot recover a number already changed by a JSON parser.
For example, the legacy reader decodes `1e-400` as zero. The separate
`RoundtripEvidenceReader` authenticates the byte string before parsing and
requires a caller-selected `shortest_roundtrip_binary64_json_v1` contract.
Integer tokens remain exact integers. Fractional/exponent tokens must have
the same decimal value as a spelling of Python's shortest round-trip
representation of the decoded finite float. Equivalent formatting such as
`0.1000` is accepted; hidden precision such as `1.00000000000000001` is not.

This declares binary64 semantics, not exact-decimal semantics. Rational
inputs remain supported by the in-memory exact assembler; this reader is
for the binary64-producing archived protocols. It cannot establish that a
producer computed its original bounds correctly.

The reader tests pass on Windows and ARM: 9,996 finite random binary64
round-trips, 11 invalid numeric spellings, and all 31 inherited path/hash/
malformed-record refusals. The two host reports are byte-identical:
`results/roundtrip_evidence_reader_{windows,arm}_20260907_v2.json`.

The complete recorded graph also passes the stricter numeric audit:
16,109 JSON files, 2,904,077 floating-point visits and 152,740 integer visits,
with no rejected numeric spelling. The 1,604 non-JSON files are outside this
parsing audit. See `scripts/audit_recorded_json_contract.py` and
`results/recorded_json_numeric_contract_arm_20260907_v1.json`.

The reassembly driver now requires
`--numeric-contract shortest_roundtrip_binary64_json_v1`. It reparses the
context under that contract, guards legacy numeric conversions, and then
uses the exact-input assembler. The integrated ARM check retains the same
`[44,44]` bracket, radius and count paths, with source hashes in
`results/recorded_exact_input_assembly_arm_20260907_v3.json`.
The prior report and all frozen producers remain unchanged.
