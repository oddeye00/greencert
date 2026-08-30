# Cost-aware corrected-defect forcing release

## Corollary (one-sweep directional fallback)

Let `K_bar` be the causal Green operator on a corrected reference path.  Let
`s_bar` be that path's exact defect and let `q_tilde` be a cancellation-safe
surrogate satisfying

```text
||s_bar - q_tilde|| <= sigma.
```

For any certified `kappa >= ||K_bar||`, the norm-only release is

```text
||K_bar s_bar|| <= kappa (||q_tilde|| + sigma).                 (1)
```

Alternatively, propagate `q_tilde` once through the corrected variational
recurrence and obtain a computed response `z_tilde`.  If its stacked recurrence
residual has norm `tau`, then

```text
||K_bar q_tilde - z_tilde|| <= kappa tau
```

and hence

```text
||K_bar s_bar|| <= ||z_tilde|| + kappa (sigma + tau).          (2)
```

The minimum of (1) and (2) is therefore a valid corrected-defect response
bound.

### Proof

The first inequality is submultiplicativity.  For the second, replaying the
computed response recurrence shows that its error from `K_bar q_tilde` is
`K_bar` applied to the signed recurrence residual (up to an immaterial sign),
so its norm is at most `kappa tau`.  Add and subtract `K_bar q_tilde` in
`K_bar s_bar` and apply the triangle inequality.  Taking the smaller of two
simultaneously valid upper bounds preserves validity.

## Cost-aware release policy

The cohort audit first tries (1) after the four-probe Green prefix.  Only if the
resulting nonlinear closure and output event do not issue does it pay for the
single forward recurrence used in (2), which costs one causal Jacobian sweep
versus two sweeps for another Gram application.  The same computed response is
reused at later `(8,16)` prefixes.  This deterministic fallback does not spend
additional probability and does not inspect a future event outcome.

The float64 audit additionally replays each computed recurrence once to expose
its numerical residual.  That audit-only replay makes the measured execution
cost two sweeps, equal to one Gram application and still below the four Gram
applications in the next probe prefix.  An outward implementation can replace
the replay by a certified arithmetic-error envelope.

If the surrogate error `sigma` includes both the analytic Taylor remainder and
the first correction's recurrence residual, the release also covers inexact
construction of the corrected path.
