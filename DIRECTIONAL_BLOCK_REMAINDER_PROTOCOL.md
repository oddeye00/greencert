# Frozen directional-block remainder diagnostic

Frozen before the first cohort evaluation of the directional fourth-order
bound.

## Parent and firewall

- Parent: `results/transformer_fully_recentered_three_sweep_audit.json`.
- Cases: all 15 parent cases; `(366, 0.8, 1120)` remains the disclosed
  development row and the other 14 are reported separately.
- No future-outcome or revealed-event file may be read.
- No new randomized Green query is made.  The diagnostic reuses each parent's
  already recorded familywise-calibrated Green upper bound.
- Centerline, three-sweep correction, quadratic surrogate, domain radius,
  drift, and closure equation are unchanged.

## Deterministic replacement

At every correction step, replace

\[
 \|D^4F\|\|z\|^3/6
\]

by the segment-valid directional block remainder

\[
 \|\nabla P_4(r(z))\|_2/24.
\]

The scaled momentum contribution remains multiplied by `sqrt(2) * eta`, and
stepwise contributions remain aggregated in Euclidean sequence norm.

## Validity gates

1. Fourth-order block product and chain rules collapse exactly to the scalar
   fourth-order identities.
2. The fourteen analytic blocks exactly partition every tested flat direction.
3. The polynomial gradient satisfies Euler's homogeneous identity, including
   zero block radii.
4. Direct fourth-order autodiff stress tests on points throughout random
   realized segments show no violation.
5. Every cohort stage-value fixed point closes and is checked after iteration.

## Prespecified promotion gate

The method is eligible for integration into the next paper version only if,
among the 14 nondevelopment cases:

- at least three additional corrected-path closures pass under the unchanged
  recorded Green bounds; and
- the directional Taylor sequence bound is no larger than the scalar bound in
  every evaluated step.

Failure of either condition is reported as a negative result.  No event
outcome is needed to apply this gate.

