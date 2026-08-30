# Independent fresh Transformer confirmation audit

Status: **PASS**.

This checker independently reloaded all 23 sealed candidate certificates and
their post-seal outcome audits. It verified certificate and audit hashes,
recomputed the signed-radius and nonlinear-closure inequalities, reconstructed
every 25-step persistent bracket from the guaranteed/possible count paths, and
recomputed all aggregate statistics without importing the experiment runner.

## Claim-bearing result

- 24 untouched Transformer seeds; 72 seed-threshold cases.
- 23 prospectively frozen candidates across 12 seeds.
- 9 certificates issued across 6 distinct seeds; all 9 contained the observed
  first passage.
- All 9 brackets are singletons. Certified leads range from 28 to
  274 updates, with median 192.
- Zero observed issued sequence-tube violations and zero observed issued
  pointwise state-tube violations.
- Minimum nonlinear closure slack: 0.144549.
- Minimum strict output-logic slack: 2.67167e-05.
- Largest observed issued sequence-error/radius ratio:
  0.502766.
- The raw four-sweep centerline hit the exact persistent-event offset in all
  23/23 frozen candidates; this is a
  secondary timing diagnostic, not a coverage guarantee.

## Denominators and clustering

- Candidate rate: 23/72 (31.9%).
- Issuance over all prespecified seed-threshold cases: 9/72
  (12.5%).
- Issuance conditional on a frozen candidate: 9/23
  (39.1%).
- Issuing seeds: 6/24 (25.0%).
- Event-level 9/9 observations are correlated within seed. If one nevertheless
  treats issued events as independent, the two-sided 95% Clopper--Pearson lower
  endpoint is 0.664. At the stricter
  issuing-seed level, 6/6 clusters have no failure and the analogous endpoint
  is 0.541. These are sensitivity
  summaries, not unconditional population-coverage claims.

## Randomized verification budget

- Queried operators: 4,961 of the predeclared maximum
  21,744.
- Realized union bound: 2.28155e-07.
- Frozen family-wise ceiling: 1e-06.

## Numerical boundary

The artifact chain and scalar inequalities pass exactly as stored. The neural
jet, HVP/VJP, and randomized power computations were performed in float64, so
this is a high-confidence probabilistic numerical certificate, not an
outward-rounded computer-assisted proof of the PyTorch execution. The margins
are far from scalar rounding ties, but the paper must preserve that boundary.
