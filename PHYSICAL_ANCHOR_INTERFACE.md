# Physical checkpoints and rounded coordinate encodings

The preprint's physical-anchor corollary makes explicit how a stored
reference can differ from the physical checkpoint after a rounded coordinate
conversion. This is an application of causal variation of constants, not a
new shadowing theorem. It changes no frozen experiment or issuance count.

For a reference `c` and physical checkpoint `x_a`, use the exact source
`a0 = G(x_a) - G(c0)` in the first transition. The usual zero-initial-state
Green operator then propagates `s + (a0,0,...,0)` on times `1,...,H`.
The virtual zero at time zero is only an indexing convention: output tests
at that time must be evaluated or enclosed at `x_a` itself. The first
Jacobian does not affect the zero-initial-state Green operator. Later
derivative envelopes and nonlinear closure conditions are unchanged.

If the stored defect, injection and response have verified errors bounded
by `sigma`, `gamma` and `tau`, respectively, the response discrepancy is
bounded by `kappa*(sigma + gamma + tau)`. A known injection can instead be
propagated directionally in the signed sweep; its computation must still
account for the recurrence residual. This reuses the same Green-norm event
and does not grant a new probability budget or a free additional sweep.

For scaled momentum, `w = eta*v`, the exact encoding discrepancy is
`delta = eta*v0 - fl(eta*v0)`. Because the optimizer is affine in `w`, its
exact first-step source is `(-mu*delta, mu*delta)`. No HVP is needed for this
identity. In a general nonlinear map, `J0*(x_a-c0)` alone is insufficient:
the nonlinear initial-displacement residual must also be enclosed.

## Reproduction

Run the standard-library-only exact-rational audit:

```text
python scripts/audit_physical_anchor_injection.py
```

The audit checks 48 nonlinear path identities, 48 initial-Jacobian
independence cases, 48 perturbed-source/response identities, and 32 momentum
identities. It also checks a closing nonlinear tube, the one-transition
boundary, and counterexamples to dropping the nonlinear initial source or
using the wrong time-zero gate. Both Windows/Python 3.12.10 and DGX
Linux-ARM/Python 3.12.3 pass. These are finite regression examples; the
corollary's proof, not the examples, establishes the general statement.

## Relation to the existing Transformer results

The archived small-Transformer constructors start from the original raw
`(theta,v)` checkpoint and evaluate a rounded scaled reference. Their
floating-point implementation does not outward-enclose every conversion,
neural operation, or response residual. Those results retain their stated
float64/ideal-PRNG numerical scope; this corollary does not retroactively
promote them to end-to-end computer-assisted proofs. Their future outcomes
and reported counts are unchanged.

The separate larger-model development implementation already transports
its encoding source and verifies the computed response residual. The new
corollary supplies an explicit paper-level interface for that existing
calculation. No larger-model outcome has been added to the paper's
83-event population by this clarification.
