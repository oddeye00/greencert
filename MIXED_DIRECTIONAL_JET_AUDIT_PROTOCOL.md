# Frozen mixed-directional jet audit

Frozen before evaluating the linear-cost mixed jet on any cohort correction.

## Objective

Independently reproduce the directional block-polynomial remainder without
constructing or differentiating its degree-four polynomial.  The mixed jet
propagates only `D_t^k` and `D_t^k D_epsilon`, `k <= 3`, where `t` is the known
correction and `epsilon` is a free block direction.

## Parent and firewall

- Parent:
  `results/transformer_directional_block_remainder_diagnostic.json`.
- Recompute all 15 three-sweep corrections from frozen checkpoints.
- Read no future-outcome or revealed-event file.
- Make no randomized Green query.
- Require the corrected-path hash to match the parent before comparing any
  remainder.

## Equivalence gates

1. Every local mixed-jet gradient remainder agrees with the polynomial result
   to relative tolerance `3e-12` (with a `1e-300` absolute floor).
2. Every sequence remainder, injection, and closure decision agrees with the
   parent.
3. All independent algebra/autodiff tests in
   `test_transformer_mixed_directional_jet.py` pass.
4. The median end-to-end per-case runtime is at least twice as fast as the
   polynomial parent on the same machine and replay path.

Failure is reported; tolerances and the speed gate are not changed after the
run.

