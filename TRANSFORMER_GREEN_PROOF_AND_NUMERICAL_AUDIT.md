# Independent proof-to-implementation audit: signed Green certificates

Status: **PASS, with an explicit numerical qualification**.

The abstract finite-window theorem, the scaled-momentum specialization, the
matrix-free adjoint, the persistent first-passage logic, and the frozen fresh
artifact chain are internally consistent. The fresh Transformer result is a
family-wise high-confidence numerical certificate evaluated in float64. It is
not an outward-rounded computer-assisted proof of the exact-real neural map or
of PyTorch's floating execution.

## 1. Sequence-space theorem

For a reference path `c_0,...,c_H`, define

```text
s_j = G(c_j) - c_{j+1},
J_j = DG(c_j),
h_j = x_j - c_j,
N_j(u) = G(c_j+u) - G(c_j) - J_j u.
```

On the sequence Hilbert space with
`||h||_X^2 = sum_{j=1}^H ||h_j||_2^2`, the exact causal error satisfies

```text
h = K_H s + K_H N(h).
```

If `||DG(c_j+u)-J_j|| <= M ||u||` on every radius-`R` ball, then

```text
||N(h)||_X <= (M/2) ||h||_X^2 <= (M/2) R^2.
```

Thus `Z + kappa*M*R^2/2 <= R`, where `Z=||K_H s||_X` and
`kappa>=||K_H||`, makes the continuous causal fixed-point map preserve the
closed radius-`R` ball. Brouwer supplies a fixed point, and causal forward
recursion makes it unique. With the frozen choice `R=2Z`, this becomes exactly

```text
2*kappa*M*Z <= 1.
```

The signs, factor `1/2`, norm conversion, and factor `2` in the implemented
closure statistic are correct.

## 2. Robust computational-error extension

The theorem remains valid when the defect and signed response are computed
approximately. Let `s_tilde` approximate `s`, define the exact-real recurrence
residual

```text
d_j = z_tilde[j+1] - J_j z_tilde[j] - s_tilde[j],
||s-s_tilde||_U <= sigma,
||d||_U <= tau.
```

The causal identity is

```text
K_H s-z_tilde = K_H(s-s_tilde-d).
```

Hence `Z_bar = ||z_tilde||_X + kappa*(sigma+tau)` is an upper bound on
`||K_H s||_X`, and the sufficient condition is

```text
Z_bar + (kappa*M/2) R^2 <= R.
```

This version accepts verified local defect and recurrence errors without a
monolithic outward response solve. The current Transformer implementation does
not yet supply nonzero outward `sigma,tau` for the upstream Green/HVP/JVP/VJP
transport; it therefore does not discharge all finite-precision neural-kernel
error in the exact-real theorem. The later amplified-secant branch does
outward-enclose its 204 scalar forcing jets conditional on stored dyadic
inputs, and a separate residual-corrected Gram theorem and 256-bit scalar
solver provide the local interface needed to extend that treatment upstream.

## 3. Safe early abstention

The last injection block of `K_H` is copied identically to the last state, so
`||K_H||>=1`. For `R=2Z`, `2MZ>1` implies that no admissible
`kappa>=||K_H||` can pass. Skipping the expensive Green probe in this case is
proof-preserving. It can only abstain early; it cannot manufacture issuance.

## 4. Randomized matrix-free Green bound

For `A=K_H^T K_H` and independent standard Gaussian probes `g_i`, set

```text
Y = max_i ||A^q g_i||,
c_delta = Phi^{-1}((1+delta^(1/m))/2).
```

Selecting a top eigenvector gives
`||A^q g_i|| >= ||A||^q |v^T g_i|`; the probability that all `m`
projections are below `c_delta` is exactly `delta`. Hence, with probability at
least `1-delta`,

```text
||K_H|| <= (Y/c_delta)^(1/(2q)).
```

The implementation uses the required exponent `1/(2q)`. Direct explicit
matrix tests verify `K_H`, `K_H^T`, and their Gram product. The fresh protocol
predeclares 21,744 possible operators, assigns each a domain-separated probe
stream from an OS-random nonce, rejects duplicate/unknown identities, and uses
a union bound of `1e-6`. Only 4,961 operators were queried, yielding a realized
union bound `2.28155e-7` under the stated Gaussian/pseudorandom model.

## 5. Scaled momentum specialization

In coordinates `(theta,w)` with `w=eta*v`, the optimizer map is

```text
r = mu*w + eta*grad F(theta),
(theta,w) -> (theta-r,r).
```

Its products are

```text
J(dtheta,dw) = (dtheta-dr, dr),
dr = mu*dw + eta*H*dtheta,

J^T(a,b) = (a + eta*H*(b-a), mu*(b-a)).
```

These match the code and pass finite-difference/adjoint tests. If the objective
Hessian is `L`-Lipschitz in `theta`, then the full optimizer Jacobian changes by
at most `sqrt(2)*eta*L` times the scaled-state displacement. This is the exact
factor used for `M`.

The neural jet bound controls the first three parameter derivatives of all
logits on the radius-`R` ball. Cross-entropy composition uses the valid global
bounds `||D ell||<=sqrt(2)`, `||D^2 ell||<=1/2`, and
`||D^3 ell||<=2`, giving

```text
L <= 2 B1^3 + 1.5 B1 B2 + sqrt(2) B3.
```

Weight decay contributes no Hessian drift. The code maximizes this quantity at
`c_1,...,c_{H-1}`. The anchor transition needs no nonlinear envelope because
`h_0=0`; the terminal state needs none because no transition leaves `c_H`.

## 6. Output and persistent-event logic

The parameter component of the scaled-state error is at most the complete
state norm. If `B1` and `B2` enclose the logit Jacobian and Hessian on the ball,
the output-vector error is at most `B1 R + B2 R^2/2`. A true-versus-competitor
margin has row norm `sqrt(2)`, yielding the implemented margin radius

```text
sqrt(2) * (B1 R + B2 R^2/2).
```

Strict positive lower margins certify correctness; one strict negative upper
margin certifies incorrectness. The earliest all-possible persistent block is
a lower event bound, and the earliest all-guaranteed block is an upper event
bound. The independent audit reconstructed every issued 25-step bracket from
the stored count paths and strict order-statistic slacks.

## 7. Fresh result and numerical boundary

The independently recomputed fresh result is:

- 24 prospective/outcome-sealed seeds and 72 prespecified seed-threshold cases;
- 23 frozen candidates across 12 seeds;
- 9 issued singleton brackets across 6 seeds, all 9 covered;
- median lead 192 updates and maximum lead 274;
- minimum closure slack `0.144549`;
- minimum strict output-logic slack `2.67167e-5`;
- maximum observed issued sequence-error/radius ratio `0.502766`;
- no observed issued state- or sequence-tube violation; and
- exact raw timing in all 23 frozen candidates.

The last quantity is a prospective timing diagnostic, not a theorem or a
population-coverage guarantee. Gates within a seed are correlated, issuance
is conditional, and one frozen construction failure was conservatively counted
as an abstention. Two transient OneDrive cache-write failures were retried with
the unchanged sealed executable before the outcome join; both retries and the
amendment itself were sealed.

## 8. Required manuscript language

Safe object-level headline:

> GREENCERT couples an anchor-preserved signed Green response, neural derivative
> envelopes, and strict output margins into a prospective bracket-or-abstain
> construction for persistent neural-training first passages in a causal
> Transformer.

Required qualification:

> The Transformer certificates have a predeclared family-wise randomized
> failure budget and comfortable strict slacks, but their neural and operator
> computations are float64 rather than outward-rounded. Exact-real numerical
> continuation is independently demonstrated for all 63 issued WDBC/digits
> brackets from their stored dyadic checkpoints.

Claims to avoid:

- a new general shadowing lemma or a new Green-operator construction;
- unconditional prediction from early training;
- independent 9/9 population coverage;
- a statistical generalization guarantee; or
- a formal proof of the PyTorch execution.
