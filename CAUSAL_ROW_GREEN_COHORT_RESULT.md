# Causal row-Green cohort result

Status: completed negative promotion audit. This route is not a manuscript
claim.

The protocol and implementation were committed and pushed at
`bde1c9ee69189afb8e2cdfe1320400bf21c286bf` before the holdout cohort ran.
The disclosed development case was seed 366, threshold 0.80, anchor 1120; the
other 14 cases were held out from method selection.

## Frozen result

- Declared cases: 15.
- Holdout cases: 14.
- Causal row certificates issued: 1/15.
- Holdout certificates issued: 0/14.
- Issued bracket retained: 1/1, the disclosed development bracket `[2,2]`.
- Matched old global scalar closures: 0/15.
- Matched signed global scalar closures: 0/15.
- Outcome files read: 0.
- Wall time with three workers: 1,926.36 seconds.
- Sum of per-case end-to-end times: 5,464.43 seconds.

The prespecified requirement of at least three issuing holdouts therefore
failed. The theorem and its randomized tests remain valid, but the two-sweep
neural specialization is not promoted.

## Failure localization

Every holdout causal radius eventually left its predeclared derivative domain.
The median first failing checkpoint was 47.5. In all 14 holdouts, the affine
response bound itself eventually exceeded the domain even before adding later
quadratic feedback. Thus a sharper solution of the same scalar discriminant is
not the missing ingredient: the analytic unresolved-forcing envelope after two
reference sweeps is too large. Direct defect evaluation or an additional
signed response would be required.

The complete denominator, per-case arrays, timing records, probe identities,
and source hashes are stored in
`results/transformer_causal_row_green_cohort_audit.json` (SHA-256
`3FE402850068AF790E371ADB8A2325E017A4577D48EF4AFAE58EFD137894C059`).
