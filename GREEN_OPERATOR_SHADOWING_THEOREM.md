# Finite-window Green-operator shadowing certificate

## Setting

Let `x_{j+1}=G(x_j)` and let `c_0,...,c_H` be a computable reference with
`c_0=x_0`.  Define

```text
s_j = G(c_j) - c_{j+1},
J_j = DG(c_j),
h_j = x_j - c_j.
```

Then

```text
h_{j+1} = J_j h_j + s_j + n_j(h_j),
n_j(u) = G(c_j+u) - G(c_j) - J_j u.
```

Equip the finite sequence space `X_H=(R^d)^H` with

```text
||h||_X^2 = sum_{j=1}^H ||h_j||_2^2.
```

Define the causal Green operator `K_H:X_H->X_H` by: for an injection sequence
`u_0,...,u_{H-1}`, set `z_0=0` and

```text
z_{j+1}=J_j z_j+u_j,
K_H u=(z_1,...,z_H).
```

This operator contains the ordered, time-varying propagators exactly.  It is
not the product of the individual bounds `||J_j||`.

## Theorem

Assume `G` is continuously differentiable on every ball used below and that,
for a radius `R`,

```text
||DG(c_j+u)-J_j||_2 <= M ||u||_2       whenever ||u||_2 <= R
```

for every `0<=j<H`.  Let

```text
kappa >= ||K_H||_2,
S = (sum_{j=0}^{H-1} ||s_j||_2^2)^(1/2).
```

If

```text
kappa * (S + 0.5*M*R^2) <= R,
```

then

```text
(sum_{j=1}^H ||x_j-c_j||_2^2)^(1/2) <= R,
```

and consequently `||x_j-c_j||_2<=R` at every state in the window.

### Proof

For `h in X_H`, define the nonlinear sequence

```text
N(h) = (n_0(0), n_1(h_1), ..., n_{H-1}(h_{H-1})).
```

The integral Taylor remainder gives

```text
||n_j(h_j)||_2 <= 0.5*M*||h_j||_2^2.
```

Therefore, on `||h||_X<=R`,

```text
||N(h)||_X
 <= 0.5*M*(sum_j ||h_j||_2^4)^(1/2)
 <= 0.5*M*sum_j ||h_j||_2^2
 <= 0.5*M*R^2.
```

The actual error sequence is the unique causal fixed point of

```text
T(h)=K_H(s+N(h)).
```

The displayed condition implies `T` maps the closed radius-`R` ball in the
finite-dimensional sequence space into itself.  Brouwer gives a fixed point in
that ball.  A fixed point is unique because its coordinates are determined
successively from `h_0=0` by the causal recurrence.  Hence that fixed point is
the actual error sequence, proving the claim.

## Two-times-radius corollary

Set

```text
R = 2*kappa*S.
```

If the derivative bound is valid on radius `R` and

```text
2*kappa^2*M*S <= 1,
```

then the theorem's closure condition holds.  This is the form used by the
implementation: the Green norm and defect norm determine `R` before any event
margin is inspected.

## Signed-response theorem

The preceding corollary still discards the direction of the known defect by
replacing `||K_H s||` with `||K_H|| ||s||`. That loss is unnecessary because
both `s` and its causal linear response can be computed before any event margin
is inspected. Define

```text
z = K_H s,
Z = ||z||_X.
```

If, for a radius `R`, the same derivative-drift bound holds and

```text
Z + 0.5*kappa*M*R^2 <= R,
```

then

```text
||x-c||_X <= R.
```

Indeed, the exact error is the unique causal fixed point of

```text
T(h) = z + K_H N(h).
```

On the closed radius-`R` ball,

```text
||T(h)||_X <= Z + 0.5*kappa*M*R^2 <= R.
```

Brouwer and causal uniqueness then give the claimed enclosure exactly as in
the first theorem. In particular, choosing the protocol-fixed radius

```text
R = 2*Z
```

is valid whenever

```text
2*kappa*M*Z <= 1.
```

This is strictly sharper than the norm-only sufficient condition because
`Z <= kappa*S`, often by orders of magnitude when the signed defect is poorly
aligned with the most amplified input direction. The four-sweep centreline is
unchanged: `z` is used only to certify the unknown error around that frozen
centreline, so this is not an additional recentering sweep.

### Safe pre-probe abstention

The block matrix of `K_H` has identity operators on its diagonal. Equivalently,
an injection supported only at the final transition is copied unchanged to the
final state. Hence

```text
||K_H|| >= 1.
```

For the fixed rule `R=2Z`, if a computed derivative-drift upper bound already
satisfies

```text
2*M*Z > 1,
```

then no valid Green upper bound can make the sufficient condition pass. The
implementation may therefore abstain before querying the expensive Green
operator. This changes neither the candidate nor the issuance rule; it is a
proof-preserving early exit for a certificate that is already impossible
under the frozen envelope.

## Matrix-free enclosure

`K_H` is generally non-symmetric.  Apply the PSD-Gram theorem to

```text
A = K_H^T K_H.
```

One `K_H` application is a forward sweep of optimizer JVPs.  One `K_H^T`
application is the corresponding reverse sweep of optimizer VJPs.  No dense
Hessian, Jacobian, or propagator is formed.  A family-wise union bound combines
this probabilistic enclosure with the output-Jacobian enclosures used for the
neural derivative and margin bounds.

## Why this theorem is needed

For the burned Transformer candidates, the scaled momentum map has one-step
norm about `2.188` although its eigenmodes and observed finite-window dynamics
are stable.  Multiplying one-step norms destroys the tube by updates 26--27.
`K_H` certifies the gain of the actual ordered finite-window linear response,
which is precisely the quantity denoted `kappa_H` in the earlier quadratic
defect-contraction corollary.
