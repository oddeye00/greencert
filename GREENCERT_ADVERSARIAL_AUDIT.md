# GREENCERT adversarial submission audit

Date: 2026-08-29

Recommended NeurIPS 2026 contribution type: **Theory**. The paper's primary
contribution is the response-centered finite-window theorem and executable
first-passage verifier. The experiments test theorem issuance, implementation
integrity, numerical layers, and computational realization rather than compete
with large-model predictive baselines.

## Current evidence ledger

- WDBC: 72 prespecified cases, 71 candidates, 56 issued, 56/56 covered across
  22 issuing seeds; all 56 independently retained by direct 192-bit outward
  continuation.
- Digits: 24 prespecified cases, 7 signed certificates, 7/7 covered across six
  seeds; all seven independently outward-retained. The signed method uniquely
  retains a 147-step event. Both inaccurate finite centerline forecasts in the
  frozen study abstain before randomized probing.
- Fixed-radius Transformer: 72 prespecified cases, 23 candidates, 9 issued,
  9/9 covered across six seeds. Matched scalar-Euclidean unsigned propagation
  retains 1/9 signed certificates.
- Response-centered Transformer: 72 prespecified cases, 19 candidates, 11
  issued, 11/11 covered across seven seeds. The matched fixed-radius rule issues
  10. First issuing powers are q=1--4 (median 2).
- Total: 83 issued brackets, all covered; 63 small-network brackets have an
  independent exact-real continuation audit.

## Hostile-review matrix

| Attack | Current evidence or repair | Status |
|---|---|---|
| The response-centered fixed-point argument hides an indexing error. | State and injection spaces are now distinct (`X` and `U`); `q` is defined only on transition inputs; the exact anchor removes `j=0` drift and the terminal state is not a transition input. | Closed. |
| Quadratic sweep contraction uses the wrong operator norm. | The contraction now explicitly uses the induced max-injection to max-state gain `kappa_{infty,H}`, distinct from the sequence-L2 Green gain used for certification. | Closed. |
| Approximate response arithmetic is not represented in the theorem. | The local recurrence identity is `K s-z_tilde=K(s-s_tilde-d)`, so verified defect and response residuals give `alpha=kappa(sigma+tau)` without outward-enclosing a monolithic solve. A second identity gives `beta=||y_tilde||+kappa(sigma_q+tau_q)` for the corrected-defect response, and the same accounting covers inexact variational sweeps. The amplified-secant branch now has a 204-interval outward scalar forcing audit. | Closed mathematically; upstream Green/derivative residual production remains open. |
| Observable transport still pays for the complete known response as uncertainty. | The response-centered observable corollary evaluates margins at `c+z` and charges only `(B1+B2 d)E+B2 E^2/2`. Its center-Jacobian versus ball-Hessian assumptions are explicit. Four thousand analytic tests pass; an outcome-blind post-seal reconstruction retains all 11 sealed v3 brackets at identical powers while reducing maximum output radius to 9.1--60.2% of the old value (median 36.5%) with no randomized query. | Closed mathematically and implemented; the audit is method-development evidence, not prospective issuance. |
| Adaptive Gram powers incur an unreported multiple-testing penalty. | One Gaussian top-eigenvector event is simultaneous over every inspected power. A predictable-operator theorem covers adaptively created operators receiving fresh independent blocks and summable failure allocations. | Closed. |
| The operator theorem is stated only for the Green map but used for outputs. | Proposition 1 is now stated for an arbitrary fixed linear operator `T`; Green and output Jacobians are specializations. | Closed. |
| V3 is merely a retrospective tuning. | Method, candidate, and certificate seals precede fresh training, randomized queries, and future rollout respectively. One pre-outcome constructor mismatch is retained as an abstention. | Closed, subject to the ordinary limitation that local seals are not external timestamps. |
| “Untouched Transformer” contradicts a selector reading current certification count. | All Transformer wording is prospective/outcome-sealed. The current count is disclosed; future outcomes remain inaccessible. | Closed. |
| Conditional 83/83 coverage is statistically overstated. | The paper reports every denominator, says coverage is conditional on issuance, omits event-level binomial intervals, and reports distinct issuing-seed counts. | Closed. |
| Abstention only separates proved from unproved correct forecasts. | The prospective digits study contains two inaccurate finite predictions (1 and 3 updates early); both fail deterministic closure before randomized probing. | Closed as an existence/selectivity demonstration; only two errors are observed. |
| Signed propagation is cosmetic. | It retains the 147-step digits event and 8/9 fixed-radius Transformer certificates that the matched unsigned construction loses; median zero-order inflation is about 320x. | Closed for the matched scalar-Euclidean comparator. Do not generalize to every unsigned validator. |
| Classical shadowing makes the work unoriginal. | The paper differentiates the delivered object: an anchor-preserved realized optimizer orbit, endogenous signed defect, neural derivative transport, and persistent first-passage bracket/abstention. The current sweep also covers certified learned dynamics, Certified Grokking, 2026 barrier robustness for poisoned training trajectories, and time-uniform stopping for strongly convex SGD. No “first known” claim is used. | Mitigated; synthesis novelty remains judge-dependent. |
| The theorem is too conservative to issue. | Response centering previously increased median rigorous horizon about 5.6x. The cancellation-safe corollary closes all 15 Green-evaluable sealed records rather than 11, converts four abstentions, and advances five closures. Corrected-path relinearization removes the mixed term. A frozen cohort-wide 4/8/16 release then preserves all 15 directional brackets: 14 stop at four probes and one at eight, reducing Green Gram applications 560 to 64 (8.75x) with minimum forcing headroom 2.293x. Frozen prospective counts remain unchanged. | Strongly mitigated mathematically and across the full evaluable cohort; fresh prospective use of the sharpened branch remains open. |
| Corrected-path relinearization silently assumes more than the original drift condition. | The corollary explicitly assumes pairwise Jacobian Lipschitzness on the local ball, which follows from the bounded-Hessian envelopes used in the implementation. Its proof no longer attempts to infer a pairwise bound from center-relative drift. | Closed in the theorem statement and proof. |
| Literal corrected-path defects are numerically meaningless at the relevant scale. | The first frozen audit is retained as a failed interface: direct float64 subtraction yields norm `3.10e-15` and a negative discriminant. The replacement uses the exact identity `bar_s=N(v)-r^v` and an amplified secant, never subtracting nearly equal optimizer maps. | Closed at the scalar-forcing layer; outward response recurrence remains open. |
| HVP accounting hides the cost of 1,797 directional third products. | A four-repeat alternating-order benchmark on immutable seed 366, gate 70%, H=52 first gives 2.91x versus an extra Gram power. The stronger matched three-way audit includes every branch-specific fourth-order envelope: the amplified secant takes 6.56 s median, third-order AD takes 8.10 s, and the next 16-probe power takes 26.01 s. The secant is 1.19x faster than third-order AD and 4.04x faster than the replaced power; every power/secant pair is 3.82--4.37x. Independent hash/arithmetic auditors read no outcomes. | Closed for the demonstrated incremental choice; absolute latency and full-cohort timing remain machine/load dependent. |
| Matrix-free still means impractically slow. | The implementation improvements are now composed in one stopwatch rather than multiplied on paper. Three separately launched outcome-blind replays of the sealed H=26 seed-366 case reproduce `[2,2]` in 9.01--11.71 s (median 9.21), using four forward Green probes, no transpose, and no randomized output operator. This is 378x below the historical fixed-q8 cross-batch median. A same-machine control is still faster: 0.298 s for 26 direct updates and 3.718 s for 300, so certification remains 30.9x/2.48x slower. The arithmetic and provenance are independently replayed. | Closed for the demonstrated small issued case; open for long horizons and modern end-to-end models. |
| Adaptive prefix stopping or direct screening quietly spends extra failure probability. | The nested-prefix corollary allocates a summable family budget before queries. Direct-image and Gram inequalities use the same top-singular-vector Gaussian event, so inspecting `K g` before applying `K^T` to cached images adds no probability. The new Green family is `1e-6`; inherited output plus Green is explicitly at most `2e-6`. | Closed mathematically and independently replayed over 64 unique probes. |
| The cohort-wide speedup hides protocol failures or post-hoc repair. | The supplement retains a burned smoke nonce that failed before any statistic/output/cache and a conservative v2 short-circuit whose latent tighter bound differs by 1.11e-8 relatively. Neither changes a prefix, disposition, or bracket. Independent auditors regenerate all probes and hashes. | Closed by complete execution provenance. |
| “15/15 Green-evaluable” cherry-picks successful operators from 19 candidates. | The other four frozen candidates all have the original `early_abstention_before_green_probe=true` flag and no sealed Green trace, so no operator exists on which a probe-count replay could be run. The panel contains every pre-existing Green operator, including four old certificate abstentions that the sharpened closure converts. | Closed by an explicit denominator and machine-checked provenance. |
| Witness sparsity is only hindsight intuition. | A formal interval-stabbing proposition is proved; 37,000 brute-force instances verify minimum cardinality. A predictable acquisition corollary uses only centerline slacks and prior query results; 15,875 exhaustive and 2,000 randomized tests pass, and all 11 issued v3 brackets reconstruct. Candidate-generic independent audits reproduce four sealed timing replays with zero trace discrepancy and no fallback; a monolithic combined-Green replay also issues without fallback. | Closed logically and executably across H=26,52,94,142; a full-cohort repeated timing study remains optional. |
| Frozen Transformer value inflation is not literally post-fixed. | The shortest issued case has 338 positive binary64 deficits, but maximum deficit is only 1.90e-23 absolute / 5.15e-14 relative. A strict `nextafter` wrapper removes every deficit, leaves rounded jets unchanged, and reproduces the bracket. The appendix now states the frozen 1e-9 tolerance explicitly. | Quantified and hardened in binary64; not outward real arithmetic. |
| Folded-normal calibration, scalar roots, and approximate products are evaluated in ordinary float64. | A 256-bit Arb audit encloses exact-dyadic calibration and roots. The inexact-Gram theorem turns local residuals into an outward root; a float32 audit lies 1.79 million-fold inside its admissible threshold. The amplified branch now evaluates 204 full-sequence scalar Transformer jets in 192-bit Arb and independently re-sums them to an exact forcing bound of 1.31e-29. | Closed for scalar roots and amplified-secant jets; upstream HVP/JVP/VJP residuals, norm accumulation, derivative envelopes, and margins remain open. |
| “Exact dyadic probe” hides a rounded subtraction. | The first Arb execution formed `g_w-g_theta` in binary64 before interval evaluation. The hostile audit identified this before promotion. V2 preserves the same frozen probes and all scientific choices, forms the difference inside Arb, reruns all 204 intervals, and reproduces the forcing bound and bracket. V1 is retained as superseded provenance. | Closed, disclosed, and regression-tested. |
| The WDBC timing denominator is internally inconsistent. | A manuscript claim checker resolves the edge case explicitly: 59 clocks have comparable timing, 56 are future issued events, and three gates are already present at the anchor. All 15 candidate abstentions have no strictly future centerline event, including those three step-zero cases. | Closed and machine-checked. |
| A new time-weighted norm is presented as a practical improvement without cost evidence. | The weighted theorem and adjoint were implemented and tested. It tightens the state radius on one immutable replay but needs q=2 where the unweighted method issues at q=1. | Rejected as a headline method; retained only as a documented negative preconditioning branch. |
| Transformer 9/9 and 11/11 are called rigorous computer-assisted proofs. | The paper consistently calls them predeclared high-confidence numerical certificates under the ideal-Gaussian/PRNG model and distinguishes the 63 direct outward proofs. | Closed by claim discipline. |
| The evaluation event is statistical generalization. | The paper repeatedly defines a deterministic event on a fixed finite evaluation set and explicitly excludes population-generalization claims. | Closed. |
| Modern architecture evidence is weak. | Full closure is one-block/13,792 parameters; separate two-block LayerNorm/AdamW tests verify only matrix-free derivative transport. | Open and correctly scoped. |
| The submission violates the nine-page limit. | Blind main content now ends on page 9; references start on page 10. Moved protocol detail and secondary figures/tables to appendices without shrinking official style. | Closed. |
| The supplement leaks identity or invalidates cited seals after sanitization. | The packager scans identity/path tokens and records both original source hashes and sanitized packaged hashes. The verifier checks every packaged file and confirms the cited v3 source-seal prefix. | Closed for the current package. |
| Related work misses the closest current training-dynamics certificates. | Added Wicker et al. and Mathiesen et al. for learned dynamics, Naughton for a fixed grokking checkpoint, Taheri et al. for barrier robustness over poisoned training trajectories, and Aolaritei et al. for adaptive strongly-convex SGD stopping. The object-level table distinguishes each delivered guarantee from a realized nonconvex persistent first passage. | Closed for the 2026-08-28 sweep; literature can always evolve. |

## Strongest remaining reject arguments

1. **End-to-end modern scale.** The complete event certificate is still a
   shallow 13,792-parameter Transformer, while the million-parameter result is
   operator accounting and the 102,400-parameter LayerNorm/AdamW result is
   derivative transport only. The optimized H=26 proof now takes 9.21 seconds
   median, but matched continuation is still 30.9x faster over that window and
   the result is only one small issued case; nevertheless
   end-to-end closure with modern jets, normalization, and adaptive optimizers
   is still unproved.
2. **Transformer computer-assisted rigor.** Scalar calibration and exact- or
   inexact-Gram roots are now outward, but HVP/VJP residuals, norm
   accumulation, jets, and margins are not fully outward-enclosed. The strict
   post-fixed wrapper and mixed-precision audit expose headroom without closing
   this larger gap.
3. **Compositional novelty.** Green operators, pseudo-orbit correction,
   randomized power bounds, and Taylor jets have antecedents. The defensible
   novelty is the response-centered neural-training event verifier and its
   theorem/experiment package.
4. **Breadth.** The evidence is unusually deep but covers three small task
   families. A reviewer who values architecture breadth over a new theoretical
   object may still score significance at 3/4.

## Highest-value next research actions

1. Build verified local residual contracts for Transformer HVP/VJP and norm
   accumulation, then feed them through the now-executable inexact Gram,
   operator-cap precision controller, and causal-response roots. The float32
   audit shows six orders of magnitude of
   case-specific residual headroom, so this is the shortest path to a complete
   outward Transformer result.
2. Compose the now-verified streaming centerline and staged direct/Gram release
   with outward residual contracts in one end-to-end timing replay. Report
   derivative-envelope reuse explicitly; do not compare a partial replay with
   full certificate construction.
3. Attack the remaining measured cost: long-horizon Green transport and neural
   jet/envelope construction. Use the inexact-sweep identity to certify
   reduced-precision or iterative recentering, and benchmark it under a fixed
   post-seal protocol. The next theorem should target one of these measured
   bottlenecks rather than another scalar closure refinement.
4. Close global LayerNorm/AdamW derivative envelopes on the existing
   102,400-parameter map before attempting deeper or stochastic architectures.
5. If compute permits, freeze one additional architecture/task cohort only
   after the role-separated implementation is fixed. Do not tune on individual
   outcomes.

## Submission posture

The manuscript now has a strong 5/6 Theory-paper case: a new certification
object, a sharpened theorem that materially changes issuance, four frozen
cohorts, independent exact-real audits where feasible, selective rejection,
directional ablations, local residual interfaces across every expensive linear
stage, a cancellation-safe theorem that converts four abstentions, a
corrected-path theorem that removes the mixed term, a full-sequence 192-bit
scalar Transformer primitive, and a cohort-wide same-bracket 8.75x Green-work
reduction with exact prefix streaming. A 6/6 claim
would still require either end-to-end closure at materially larger/modern scale
or full computer-assisted Transformer arithmetic; wording should remain
ambitious about the framework while exact about those two boundaries.
