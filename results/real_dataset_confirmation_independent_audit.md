# Independent WDBC confirmation audit

Status: **PASS, with an explicit float64 rounding diagnostic**.

This checker imports neither the experiment runner nor the model code. It
verified the three-stage SHA-256 chain and all 16
sealed source/data files; independently reconstructed all 72 trigger-only
selection decisions; rechecked all 71 certificate hashes; recomputed every
Gaussian-Gram bound, signed and unsigned radii-polynomial calculation, stored
first-passage bracket, post-seal event time, and headline aggregate.

## Confirmatory result

- Candidates: 71/72 (98.6%).
- Issued: 56/72 overall and 56/71 among candidates.
- Containment: 56/56 issued brackets, across
  22 distinct issuing seeds; all brackets are singletons.
- Lead: median 79 updates, maximum
  256 updates.
- Minimum closure slack: 1; minimum strict
  output slack: 2.37531e-05.
- The raw four-sweep clock exactly matches all
  59 comparable post-seal event offsets.
- The matched unsigned right-inverse baseline also issues 56
  cases on this numerically easy, highly contractive transfer task; this batch
  demonstrates transfer and information isolation, not a signed-vs-unsigned
  advantage.

- 0.900: 21/24 issued, 21/21 covered, median lead 43.0, maximum lead 168.
- 0.925: 21/24 issued, 21/21 covered, median lead 79.0, maximum lead 200.
- 0.950: 14/24 issued, 14/14 covered, median lead 166.0, maximum lead 256.

## Finite-precision diagnostic

The runner's declared tolerance-adjusted audit records zero sequence or
pointwise state-tube violations. Under a literal no-tolerance float64 ratio,
however, 23/56 accumulated sequence norms
slightly exceed the extremely small analytic radius; the maximum ratio is
1.329 and the largest absolute sequence discrepancy is
1.261e-14. Every pointwise state
error remains strictly inside its radius (maximum ratio
0.540). The strict margin slack is at least
2.375e-05, over nine orders of magnitude larger
than the largest observed absolute trajectory discrepancy. This is benign for
the observed event classifications but must be described as high-confidence
float64 evidence unless an outward-rounded computer-error budget is added.

Thresholds within a seed are correlated. The seed-cluster descriptive result
is 22/22 issuing seeds with all issued
events covered. The two-sided 95% exact lower endpoint would be
0.846
if those issuing-seed indicators were treated as independent Bernoulli trials;
this is not a population-generalization guarantee.
