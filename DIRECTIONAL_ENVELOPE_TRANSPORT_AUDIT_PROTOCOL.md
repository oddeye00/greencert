# Frozen directional-envelope transport audit protocol

Frozen: 2026-09-02, before executing the four-case cohort audit.

## Purpose

Test whether the directionally transported neural-envelope theorem can replace
fresh corrected-center envelope construction on every previously issued
three-sweep directional Transformer certificate.  This is a post-release,
outcome-blind equivalence and implementation audit, not new prospective model
evidence.

## Cohort

Use exactly the four rows marked `closure_passed` in
`results/transformer_directional_block_remainder_diagnostic.json`.  Seed 366,
80% at anchor 1120 remains the development row; the other three rows are the
fixed nondevelopment audit set.  No future outcome file may be read.

## Frozen construction

For every row:

1. Rebuild the three-sweep reference with fused map/JVP evaluations and a
   replayable anchor Hessian tape.
2. Rebuild its signed correction with the fused scaled map/JVP primitive.
3. At each checkpoint evaluate exact stage values and parameter geometry once
   at the uncorrected three-sweep center.
4. Run the mixed three-known/one-free directional jet with those shared inputs.
5. Transport stage values and parameter geometry along the known correction.
6. Expand a ball of the already frozen corrected-path domain radius from the
   transported majorants.
7. Reuse the already audited directional forcing constants and Green norm;
   make no new randomized query.
8. Recompute corrected-path closure and persistent-event margins around the
   corrected center.

## Required gates

The audit passes only if all of the following hold for all four rows:

- the three-sweep centerline and corrected-path hashes equal their frozen
  parents;
- the mixed directional sequence agrees with the frozen directional sequence
  to relative tolerance (3\times10^{-12});
- transported stage and parameter-geometry majorants dominate direct float64
  evaluations at the corrected center within a one-sided (3\times10^{-13})
  diagnostic tolerance;
- transported first-, second-, and third-derivative envelopes dominate the
  separately evaluated corrected-center envelopes within the same tolerance;
- every transported fixed point is self-consistent;
- all four nonlinear closures remain issued; and
- every persistent bracket equals its frozen four-sweep bracket.

The dominance comparisons are regression tests of the implementation.  The
validity argument is the exact monotone-majorant theorem, not the float64
comparison itself.

## Reporting

Report all four rows, the three nondevelopment rows separately, maximum
one-sided majorant inflation, timing by phase, source hashes, and
`outcome_files_read = 0`.  Do not promote a speed claim from this cohort;
timing is reported separately by process-isolated paired benchmarks.
