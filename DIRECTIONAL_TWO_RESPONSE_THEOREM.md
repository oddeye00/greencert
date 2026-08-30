# Cancellation-safe directional two-response corollary

## Purpose

The response-centered theorem permits a second causal response

\[
y=K_Hq,\qquad q_j=N_j(z_j),
\]

instead of the zero-extra-operator estimate
\(\|K_Hq\|_X\le \kappa MpZ/2\).  Directly forming
\(G(c+z)-G(c)-DG(c)z\) is ill-conditioned after several variational sweeps,
because the exact quantity is quadratic in a response that can be close to
machine precision.  The following corollary evaluates the same signed forcing
through a directional derivative and exposes an explicit remainder that can be
outward enclosed.

## Corollary (directional quadratic second response)

Use the state and injection sequence spaces \(X\) and \(U\) from the main
finite-window theorem.  Let

\[
q_j=G_j(c_j+z_j)-G_j(c_j)-J_jz_j,
\qquad J_j=DG_j(c_j),
\]

and suppose \(G_j\) is three times continuously differentiable on the segment
\(c_j+t z_j\), \(0\le t\le1\), with

\[
\sup_{0\le t\le1}\|D^3G_j(c_j+t z_j)\|\le L_j.
\]

Define the cancellation-safe quadratic injection

\[
q^{(2)}_j=\frac12D^2G_j(c_j)[z_j,z_j]
\]

and

\[
\sigma_2=
\left(\sum_{j=0}^{H-1}
\left(\frac{L_j}{6}\|z_j\|^3\right)^2\right)^{1/2}.
\]

If a computed \(\widetilde q^{(2)}\) and anchor-fixed response
\(\widetilde y\) satisfy

\[
\|q^{(2)}-\widetilde q^{(2)}\|_U\le\sigma_{\rm arith},
\qquad
\|d^y\|_U\le\tau_y,
\qquad
d^y_j=\widetilde y_{j+1}-J_j\widetilde y_j-
\widetilde q^{(2)}_j,
\]

then the main theorem may take

\[
\boxed{
\beta=\|\widetilde y\|_X+\widehat\kappa\,
(\sigma_2+\sigma_{\rm arith}+\tau_y)
}
\]

for any \(\widehat\kappa\ge\|K_H\|\).

### Proof

Third-order Taylor expansion with integral remainder gives, checkpointwise,

\[
q_j=q^{(2)}_j+r^{(3)}_j,
\qquad
\|r^{(3)}_j\|\le \frac{L_j}{6}\|z_j\|^3.
\]

Taking the sequence \(U\)-norm yields
\(\|q-q^{(2)}\|_U\le\sigma_2\).  The recurrence residual identity gives

\[
\widetilde y=K_H(\widetilde q^{(2)}+d^y).
\]

Therefore

\[
\begin{aligned}
\|K_Hq\|_X
&\le \|\widetilde y\|_X+
\|K_H(q-\widetilde q^{(2)}-d^y)\|_X\\
&\le \|\widetilde y\|_X+\widehat\kappa\,
(\sigma_2+\sigma_{\rm arith}+\tau_y),
\end{aligned}
\]

which is the required \(\beta\) interface.  No subtraction of nearly equal
optimizer maps is needed.

## Scaled momentum specialization

For

\[
G(\theta,w)=(\theta-r,r),\qquad
r=\mu w+\eta\nabla F(\theta),
\]

write \(z=(a,b)\).  Then

\[
q^{(2)}=
\frac{\eta}{2}
\left(-D^3F(\theta)[a,a,\cdot],
       D^3F(\theta)[a,a,\cdot]\right).
\]

If \(\|D^4F\|\le L_{F,4}\) on the parameter segment, then

\[
L_j\le\sqrt2\,\eta L_{F,4,j},
\qquad
\sigma_2\le
\left(\sum_j
\left(\frac{\sqrt2\eta L_{F,4,j}}6\|a_j\|^3\right)^2
\right)^{1/2}.
\]

Thus one third-objective directional contraction per checkpoint constructs the
quadratic injection, and one additional causal Green sweep constructs its
signed response.  Both are deterministic.  With \(r\) randomized probes, one
additional Gram power costs \(2rH\) objective HVPs; the second response costs
\(H\) HVPs, so eliminating one power saves \((2r-1)H\) HVPs before accounting
for the directional third-derivative products.  A proof-producing Transformer
implementation still needs a fourth-order objective envelope and outward local
arithmetic bounds.

## Measured incremental cost

A post-seal benchmark holds the centerline, first signed response, and first
Gram power fixed on sealed seed 366 at the 70% gate ((H=52)).  Four
alternating-order pairs compare the exact implementation choice: construct all
directional third products and the scalar second response, or execute one more
16-probe Green Gram power.  Median times are 78.58 and 233.45 seconds,
respectively, for a (2.91\times) paired speedup; every pair lies between
(2.77\times) and (3.28\times).  Checksums agree across repeats, an
independent arithmetic/hash audit reproduces the aggregates, and no future
outcome is read.  Absolute times were measured under concurrent workspace load,
so the portable observation is the within-pair advantage, not the latency.

## Evidence boundary

The outcome-blind post-seal Transformer audit evaluates the center quadratic
response in float64 and adds a checkpoint-local analytic fourth-objective
envelope for the Taylor remainder.  The adaptive policy invokes the response
in nine sealed Green-evaluable cases; all nine pass the fourth-order gate, with
minimum residual headroom (55.3\times).  A separate full-graph test compares
the envelope with fourth directional derivatives obtained by automatic
differentiation through attention, softmax, GELU, and cross entropy.

This remains method-development evidence, not a replacement for any frozen
prospective certificate.  The analytic Taylor term is explicit, but the
directional-product arithmetic \(\sigma_{\rm arith}\) and response residual
\(\tau_y\) are not yet outward-enclosed.  Any fresh computer-assisted issuance
claim requires those budgets and the policy to be frozen before outcomes are
opened.  The later 204-interval Arb audit closes the alternative
amplified-secant scalar forcing calculation conditional on stored dyadic
inputs; it does not retroactively outward-enclose this third-product response
branch or its upstream Green solve.
