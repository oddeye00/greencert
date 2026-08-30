# Time-resolved response-centered Green theorem

## Statement

Let the anchor-fixed error satisfy

\[
h=K_Hs+K_HN(h),\qquad z=K_Hs,\qquad e=h-z.
\]

Because the realized and reference trajectories share their anchor,
\(h_0=z_0=e_0=0\).  Consequently \(N_0(h_0)=N_0(z_0)=0\), and no
Jacobian-drift envelope is required at the anchor.  The terminal state is not
an input to a transition either.  It is enough to have, for
\(j=1,\ldots,H-1\),

\[
\|DG_j(c_j+u)-J_j\|\le M_j\|u\|
\quad\text{on }\|u\|\le\rho.
\]

Let \(d_j=\|z_j\|\), \(p=\max_{1\le j\le H}\|z_j\|\), and suppose
\(\kappa\ge\|K_H\|\).  Define

\[
Q=\frac12\left(\sum_{j=1}^{H-1}M_j^2d_j^4\right)^{1/2},
\qquad
A=\max_{1\le j<H}M_jd_j,
\qquad
B=\max_{1\le j<H}M_j,
\]

with empty maxima and sums equal to zero.  If a computed response
\(\widetilde z\) has \(\|K_Hs-\widetilde z\|_X\le\alpha\), use the
corresponding computed state norms and include \(\alpha\) below.  If some
\(E\ge0\) satisfies

\[
\boxed{
\alpha+\kappa Q+\kappa A E+\frac{\kappa B}{2}E^2\le E,
\qquad p+E\le\rho,
}
\]

then

\[
\|h-z\|_X\le E,
\qquad
\max_j\|h_j\|\le p+E.
\]

For a genuinely computed \(\widetilde z\), the displayed \(Q,A\) require
outward accommodation of response error; the implementation currently uses
the exact-mathematical-response form plus a separate \(\alpha\) term.  A fully
outward implementation should bound the aligned state norms before evaluating
\(Q\) and \(A\).

## Proof

For each transition input,

\[
\|N_j(z_j)\|\le\frac{M_j}{2}\|z_j\|^2
\]

and

\[
\|N_j(z_j+e_j)-N_j(z_j)\|
\le M_j\|z_j\|\|e_j\|+\frac{M_j}{2}\|e_j\|^2.
\]

Taking the sequence \(\ell_2\) norm gives

\[
\|N(z)\|_X\le Q,
\]

\[
\|N(z+e)-N(z)\|_X
\le A\|e\|_X+\frac B2\|e\|_X^2,
\]

where the quadratic term uses
\(\|(\|e_j\|^2)_j\|_2\le\|e\|_X^2\).  Applying \(K_H\), adding the
response-computation error, and using the displayed scalar inequality shows
that the exact causal fixed-point map sends the radius-\(E\) sequence ball to
itself.  Brouwer gives a fixed point, and causal forward recurrence identifies
it uniquely with the realized trajectory.  The domain inequality validates
every local derivative envelope.

## Strict improvement over the scalar-drift corollary

With \(M=\max_jM_j\), \(Z=\|z\|_X\), and \(p=\|z\|_{\infty,2}\),

\[
Q\le\frac M2pZ,
\qquad
A\le Mp,
\qquad
B=M.
\]

Thus the time-resolved closure is never weaker than the zero-extra-operator
scalar closure.  It is strictly sharper when the largest drift and largest
signed response occur at different checkpoints or when the response is
temporally concentrated away from high-drift states.

## Practical consequence

The refinement adds no HVP, VJP, neural-jet, or random-probe call when
checkpointwise drift bounds are already produced.  It can only reduce the
required Green/output power and the certified state radius.  The executable
implementation is `scripts/heterogeneous_recenter_closure.py`; deterministic
tests compare it with exact nonlinear trajectories and the scalar corollary.

