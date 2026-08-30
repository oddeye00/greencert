# Amplified-secant response theorem

## 1. Motivation

The exact corrected defect

\[
q_j=N_j(z_j)=G_j(c_j+z_j)-G_j(c_j)-J_jz_j
\]

is quadratically small.  Evaluating this expression at the realized response
can therefore lose every meaningful bit even when the surrounding optimizer
map is accurate.  A third-derivative contraction avoids that subtraction, but
it introduces a new high-order automatic-differentiation primitive.

The amplified secant uses only the map and Jacobian products already required
by GREENCERT.  For a chosen \(\lambda_j>0\), define

\[
q_j^{[\lambda_j]}
=\frac{G_j(c_j+\lambda_j z_j)-G_j(c_j)-\lambda_jJ_jz_j}
       {\lambda_j^2}.
\]

Before division, its nonlinear signal is amplified by \(\lambda_j^2\).

## 2. Ray-secants track the realized defect

**Lemma 1 (amplified-secant discrepancy).**  Suppose \(G_j\) is \(C^3\) on
the ray \(c_j+t z_j\),
\(0\le t\le\max\{1,\lambda_j\}\), and

\[
\sup_t\|D^3G_j(c_j+t z_j)\|\le L_j.
\]

Then

\[
\boxed{
\|N_j(z_j)-q_j^{[\lambda_j]}\|
\le \frac{|\lambda_j-1|}{6}L_j\|z_j\|^3 .
}
\]

**Proof.**  Taylor's integral identity gives

\[
\frac{N_j(tz_j)}{t^2}
=\int_0^1(1-s)D^2G_j(c_j+stz_j)[z_j,z_j],ds.
\]

Subtract the identities at \(t=1\) and \(t=\lambda_j\).  Along the common
ray, the two Hessians are separated by
\(s|\lambda_j-1|\|z_j\|\), so the integrand difference is at most
\(s(1-s)|\lambda_j-1|L_j\|z_j\|^3\).  Integrating
\(\int_0^1s(1-s)ds=1/6\) proves the claim.  \(\square\)

The factor vanishes at \(\lambda_j=1\), where the secant is the exact defect;
larger amplification trades a linear analytic penalty for a quadratic increase
in the pre-division signal.

## 3. Certified amplified response

Let

\[
\sigma_{\rm sec}
=\left(\sum_{j=0}^{H-1}
 \left[
  \frac{|\lambda_j-1|}{6}L_j\|\widetilde z_j\|^3
 \right]^2\right)^{1/2}.
\]

Suppose computed secants satisfy

\[
\|\widetilde q^{[\lambda]}-q^{[\lambda]}\|_U
\le\sigma_{\rm ar},
\]

and an anchor-fixed computed response has recurrence residual

\[
d^y_j=\widetilde y_{j+1}-J_j\widetilde y_j
       -\widetilde q_j^{[\lambda]},
\qquad \|d^y\|_U\le\tau_y.
\]

Then the response-centered theorem may use

\[
\boxed{
\beta=\|\widetilde y\|_X
+\widehat\kappa(\sigma_{\rm sec}+\sigma_{\rm ar}+\tau_y).
}
\]

This follows immediately from Lemma 1 and the residualized two-response
identity.  The amplification may be checkpoint-specific and deterministic.

## 4. Adaptive choice

For uniform \(\lambda\ge1\), write

\[
B=\left(\sum_j(L_j\|z_j\|^3/6)^2\right)^{1/2}.
\]

An analytic secant budget \(S\) permits

\[
\lambda\le1+S/B.
\]

The certified derivative domain additionally requires
\(\lambda\|z_j\|\le\rho_j\) at every checkpoint.  Thus a safe deterministic
choice is

\[
\lambda=\min\left\{1+S/B,\ \min_j\rho_j/\|z_j\|\right\},
\]

possibly rounded downward to a prespecified power-of-two ladder.  A power of
two makes scaling exact in binary arithmetic and makes the policy easy to seal.

If an outward evaluator bounds the arithmetic error in the unscaled numerator
by \(\varepsilon_j\), then checkpoint \(j\)'s complete local error is

\[
E_j(\lambda_j)
=A_j|\lambda_j-1|+\frac{\varepsilon_j}{\lambda_j^2},
\qquad A_j=L_j\|z_j\|^3/6.
\]

For \(\lambda_j\ge1\) and a \(\lambda\)-independent numerator budget, the
unconstrained minimizer is explicit:

\[
\boxed{
\lambda_j^\star
=\max\left\{1,\left(\frac{2\varepsilon_j}{A_j}\right)^{1/3}\right\}.
}
\]

It is then clipped by the derivative domain and the reserved global error
budget, or rounded to the nearest admissible power of two.  Thus amplification
is not a tuning knob: a verified local arithmetic contract determines it.

## 5. Optimizer specialization

For scaled momentum, with \(z_j=(a_j,b_j)\), the nonlinear part is

\[
q_j^{[\lambda]}
=\eta\left(-r_j^{[\lambda]},r_j^{[\lambda]}\right),
\quad
r_j^{[\lambda]}
=\frac{\nabla F(\theta_j+\lambda a_j)-\nabla F(\theta_j)
       -\lambda\nabla^2F(\theta_j)a_j}{\lambda^2}.
\]

If \(\|D^4F\|\le L_{F,4,j}\), then
\(L_j=\sqrt2\eta L_{F,4,j}\).  The implementation therefore uses one shifted
gradient, one existing HVP primitive, and a causal response.  It needs neither
a third-order AD graph nor subtraction at the unamplified \(10^{-14}\)-scale
response.

## 6. Evidence boundary

The theorem is exact.  A computer-assisted use must still bound the arithmetic
error of the shifted gradient, HVP, vector combination, and causal recurrence.
Its practical advantage is that these are the same first/second-order
primitives already covered by GREENCERT's inexact-product interfaces, while
the nonlinear signal can be amplified quadratically before subtraction.
