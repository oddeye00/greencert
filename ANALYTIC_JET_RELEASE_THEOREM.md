# Deterministic analytic-jet release

## Statement

Let `b_j = c_j + v_j` be an anchor-fixed corrected reference and suppose a
Green bound `kappa >= ||K_b||` and a corrected-defect response bound `Y` are
available.  Assume the complete ball `B(c_j,rho)` carries verified output jets

```text
||Df_j|| <= A_j,   ||D2 f_j|| <= B_j,   ||D3 f_j|| <= C_j.
```

For mean cross entropy plus quadratic weight decay, define

```text
L_j = 2 A_j^3 + (3/2) A_j B_j + sqrt(2) C_j,
M   = max_{1 <= j < H} sqrt(2) eta L_j.
```

If

```text
Y + (kappa M / 2) E^2 <= E,
max_j ||v_j|| + E <= rho,
```

then the realized optimizer path lies within `E` of `b`.  Moreover, the
true-class-versus-competitor margin at time `j` differs from its value at the
original reference `c_j` by at most

```text
sqrt(2) A_j (max_i ||v_i|| + E).
```

Consequently, strict first-passage logic may be evaluated using only the
deterministic jets and raw reference logits.  No randomized output-Jacobian
operator is required.

## Proof

The cross-entropy derivative bounds used by the shipped neural jet are
`||D ell|| <= sqrt(2)`, `||D2 ell|| <= 1/2`, and `||D3 ell|| <= 2`.  The
third-order chain rule therefore gives

```text
||D3(ell o f_j)||
  <= 2 A_j^3 + 3(1/2) A_j B_j + sqrt(2) C_j = L_j.
```

Averaging examples cannot increase this norm, and quadratic weight decay has
zero third derivative.  In scaled momentum coordinates, the change of the map
Jacobian is `sqrt(2) eta` times the objective-Hessian change, hence it is at
most `M ||u||`.  The corrected-path radii argument then gives the stated
quadratic closure.  Every segment from `c_j` to a point in the certified tube
stays in `B(c_j,rho)`.  The mean-value theorem bounds each logit change by
`A_j(max_i ||v_i||+E)`; subtracting two logits introduces the factor `sqrt(2)`.
Applying the persistent first-passage proposition to the resulting strict
margin signs completes the claim.

## Staged policy and probability accounting

The verifier first attempts this deterministic release.  Only if its closure
or event margins fail does it instantiate the sharper randomized output
operator.  The screen uses no random output block, so it consumes no output
failure probability.  Predictable fallback preserves the preallocated output
family budget.  The Green event is unchanged.

This is a systems-relevant corollary, not a new prospective experiment: it can
be audited on every pre-existing operator record without reading a revealed
future trajectory or changing any frozen issuance count.
