# Frozen causal row-Green composition audit

Status: frozen after one disclosed development replay and before execution of
the other 14 cases.

## Question and evidence boundary

The released 15-case Transformer panel already has sealed four-sweep brackets.
This is a post-release systems audit, not a new coverage experiment. It asks
whether composing the released four-sweep reference with the directional mixed
jet and chronological row closure can retain the entire bracket set using only
one direct-image Green pass per case, including the nine cases that previously
required a transpose/Gram fallback.

Development row: seed 360, threshold 0.70, anchor 3480. Its direct replay was
used to select this composition. The remaining 14 rows are held out from the
promotion decision. Future outcome files are forbidden; comparison is only to
the already sealed brackets.

## Frozen construction

For every row in
`results/transformer_fully_recentered_three_sweep_audit.json`:

1. Rebuild exactly four anchor-fixed variational sweeps.
2. Propagate one signed first-response correction. The resulting path must
   match the path hash in
   `results/anchor_fixed_structured_parameter_green_transformer_audit.json`.
3. Construct the signed directional quadratic forcing and mixed fourth-order
   error with `transformer_mixed_directional_jet_v2.py`.
4. Linearize at the corrected path and evaluate one batch containing four
   standard-Gaussian sequence probes plus the deterministic signed forcing.
5. Divide a family failure budget of `1e-6` equally over all 15 candidate
   operators, then divide each candidate budget equally over its chronological
   rows.
6. Apply the causal row-Green recursion, unchanged neural-output margins, and
   unchanged persistent-event logic. No transpose Green pass or Gram power is
   allowed.

The probe namespace, candidate coordinates, sweep count, and probe count are
hashed by `diagnose_transformer_causal_row_green.py`; all 15 streams must be
distinct. The ideal-PRNG interpretation is the same as in the released
Transformer study.

## Promotion gate

Promotion requires all conditions below:

- 14/14 held-out cases issue;
- 15/15 total brackets exactly equal their sealed comparators;
- 15/15 corrected-path hashes match the released paths;
- every issued row remains in its derivative domain;
- no outcome file is read;
- the complete 15-case denominator is reported;
- total logical Green applications are exactly 60 (four direct images per
  case) with zero transpose applications.

Runtime and every failure are reported whether or not the gate passes. The
protocol will not be changed in response to the cohort result.
