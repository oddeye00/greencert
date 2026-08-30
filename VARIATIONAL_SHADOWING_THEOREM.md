# Defect-corrected variational shadowing

## 1. Why the original majorant is conservative

The moving-window theorem freezes a modal reference path and bounds every
centerline defect by its norm.  Two avoidable losses result:

1. a Hessian at the anchor is used to transport error even after the reference
   path has moved; and
2. known defect vectors are converted to scalar radii before propagation, so
   directional cancellation is discarded.

For the smooth tanh MLP, the first loss appears as a global third-derivative
bound multiplied by the full anchor-to-center path length.  It ended a tube at
step 33 even when the local model predicted the step-35 event exactly.

The improved theorem linearizes the update map at every point of the *known
reference path* and propagates the signed defect response as a vector.  Only
the nonlinear remainder is enclosed by a scalar tube.

## 2. Arbitrary-reference dynamics

All statements in this note are in exact real arithmetic.  All vector norms
are Euclidean norms and all matrix norms are the induced operator norms.  Let
(U\subset\mathbb R^d) be open, let (G:U\to\mathbb R^d) be continuously
differentiable on the neighborhoods used below, and let

\[
x_{j+1}=G(x_j),\qquad x_j\in U,
\]

and let \(\bar x_0,\ldots,\bar x_H\) be any checkpoint-computable reference
path with \(\bar x_0=x_0\).  Define its one-step defect and centerline Jacobian

\[
r_j=G(\bar x_j)-\bar x_{j+1},
\qquad
J_j=DG(\bar x_j).
\]

The true reference error \(e_j=x_j-\bar x_j\) obeys

\[
e_{j+1}=J_je_j+r_j+q_j(e_j),
\]

where

\[
q_j(u)=G(\bar x_j+u)-G(\bar x_j)-J_ju.
\]

Suppose the Jacobian has a local Lipschitz modulus

\[
\|DG(\bar x_j+u)-J_j\|_2
\le M_j(R)\|u\|_2
\quad\text{for }\|u\|_2\le R.
\]

Then

\[
\boxed{
\|q_j(u)\|_2\le\frac12M_j(R)\|u\|_2^2.
}
\]

## 3. Vector defect correction

Define the first variational correction

\[
z_0=0,
\qquad
\boxed{z_{j+1}=J_jz_j+r_j.}
\]

Thus

\[
z_j=\sum_{i<j}\Phi_{j,i+1}r_i,
\qquad
\Phi_{j,i}=J_{j-1}\cdots J_i.
\]

Unlike a sum of defect norms, this expression preserves the directions and
cancellations of all known centerline defects.  The corrected reference is

\[
\widetilde x_j=\bar x_j+z_j.
\]

Write the remaining uncertainty as

\[
w_j=x_j-\widetilde x_j=e_j-z_j.
\]

Subtracting the correction recursion gives the exact dynamics

\[
\boxed{w_{j+1}=J_jw_j+q_j(z_j+w_j).}
\]

The known linear forcing has disappeared.

## 4. Main theorem

### Theorem 4 (defect-corrected nonautonomous shadowing)

Let \(\beta_j\ge\|J_j\|_2\).  Set

\[
\omega_0=0,
\qquad
R_j=\|z_j\|_2+\omega_j,
\]

and recursively define

\[
\boxed{
\omega_{j+1}
=\beta_j\omega_j+\frac12M_j(R_j)R_j^2.
}
\]

If each Jacobian-Lipschitz bound is valid on the ball
\(B(\bar x_j,R_j)\), then for all \(j\le H\),

\[
\boxed{
\|x_j-\widetilde x_j\|_2\le\omega_j.
}
\]

**Proof.**  The claim is true at \(j=0\).  Assume
\(\|w_j\|\le\omega_j\).  Then

\[
\|z_j+w_j\|\le R_j.
\]

Using the exact recurrence for \(w_j\) and the integral Taylor bound,

\[
\begin{aligned}
\|w_{j+1}\|
&\le \|J_j\|\,\|w_j\|+
\|q_j(z_j+w_j)\|\\
&\le \beta_j\omega_j+
\frac12M_j(R_j)R_j^2
=\omega_{j+1}.
\end{aligned}
\]

Induction completes the proof.  QED.

### Corollary 4.1 (exactness for affine dynamics)

If \(G(x)=Ax+b\), then \(M_j=0\) and

\[
\omega_j=0
\]

for every reference path.  The single vector defect correction recovers the
true trajectory exactly.  A norm-only defect tube generally remains nonzero,
so this is a strict improvement.

### Corollary 4.2 (ordered-product target projection)

For any row target \(v_j^\top\), the exact Duhamel representation of the
remaining error gives

\[
|v_j^\top w_j|
\le
\frac12\sum_{i<j}
\|v_j^\top\Phi_{j,i+1}\|_2
M_i(R_i)R_i^2.
\]

This can replace \(\|v_j\|\omega_j\) when a deployment direction is nearly
orthogonal to the amplified uncertainty.  The current implementation uses the
scalar \(\omega_j\); this projected form is a further theorem-level tightening.

### Theorem 4.3 (one-sweep recentered shadowing)

Theorem 4 still spends a global Taylor budget on the *known* correction
\(z_j\): its nonlinear injection contains
\(M_j(\|z_j\|+\omega_j)(\|z_j\|+\omega_j)^2/2\).  That term can be evaluated
before taking a norm.

Define the corrected centerline

\[
c_j=\bar x_j+z_j
\]

and its exact one-step defect

\[
s_j=G(c_j)-c_{j+1}.
\]

Because

\[
c_{j+1}=G(\bar x_j)+J_jz_j,
\]

the new defect is exactly the old nonlinear residual:

\[
\boxed{
s_j=G(\bar x_j+z_j)-G(\bar x_j)-J_jz_j=q_j(z_j).
}
\]

For (0\le j<H), let

\[
\widehat J_j=DG(c_j),\qquad
\widehat\beta_j\ge\|\widehat J_j\|_2,
\]

where \(\widehat\beta_j\) is finite and nonnegative.  Suppose that every
certified ball (B(c_j,\epsilon_j)) lies in (U), every Taylor segment stays
in (U), and

\[
\|DG(c_j+u)-\widehat J_j\|_2
\le \widehat M_j(R)\|u\|_2
\quad (\|u\|_2\le R).
\]

Set \(\epsilon_0=0\) and

\[
\boxed{
\epsilon_{j+1}
=\widehat\beta_j\epsilon_j+\|s_j\|_2
+\frac12\widehat M_j(\epsilon_j)\epsilon_j^2.
}
\]

Then, wherever the displayed local bounds are valid,

\[
\boxed{\|x_j-c_j\|_2\le\epsilon_j.}
\]

**Proof.**  Write \(h_j=x_j-c_j\).  Direct subtraction gives

\[
h_{j+1}=\widehat J_jh_j+s_j+\widehat q_j(h_j),
\]

where

\[
\|\widehat q_j(h_j)\|_2
\le\frac12\widehat M_j(\epsilon_j)\epsilon_j^2
\]

under the inductive hypothesis.  Taking norms yields the recursion.  QED.

This is not an empirical correction: \(c_j\), \(s_j\), and all derivative
bounds are computable from the anchor, the known update map, and the proposed
modal path.  It uses no future trained parameter.  Relative to Theorem 4, it
replaces a bound on \(q_j(z_j+h_j)\) by the exact vector \(q_j(z_j)\) plus a
Taylor bound only in the genuinely unknown error \(h_j\).

### Corollary 4.4 (quadratic contraction of path defect)

The recentering can be iterated without changing the argument.  Given a
reference \(c^{(k)}\), define

\[
z^{(k)}_0=0,\qquad
z^{(k)}_{j+1}
=DG(c^{(k)}_j)z^{(k)}_j
+G(c^{(k)}_j)-c^{(k)}_{j+1},
\]

and \(c^{(k+1)}=c^{(k)}+z^{(k)}\).  Its residual satisfies the exact identity

\[
r^{(k+1)}_j
=G(c^{(k)}_j+z^{(k)}_j)-G(c^{(k)}_j)
-DG(c^{(k)}_j)z^{(k)}_j.
\]

Consequently, if the Jacobian is uniformly \(M\)-Lipschitz and the finite
window variational gain obeys

\[
\max_j\|z^{(k)}_j\|_2\le
\kappa_H\max_i\|r^{(k)}_i\|_2,
\]

then

\[
\boxed{
\max_j\|r^{(k+1)}_j\|_2
\le\frac12M\kappa_H^2
\left(\max_i\|r^{(k)}_i\|_2\right)^2.
}
\]

Put

\[
\delta_k=\max_{0\le j<H}\|r_j^{(k)}\|_2,
\qquad
C=\frac12M\kappa_H^2.
\]

The conclusion is \(\delta_{k+1}\le C\delta_k^2\), so the new residual has
quadratic order in the old residual.  It is strictly smaller only when
\(C\delta_k<1\).  Repeated convergence additionally requires the same gain
and regularity bounds to remain valid on a common region; under those uniform
assumptions and \(C\delta_0<1\), the residuals converge to zero.  Without the
small-defect condition, the displayed inequality alone does not imply
contraction.  The empirical protocol fixes exactly one sweep, preserving the
interpretation that the event clock originates in the anchor modal model.

## 5. Full-batch gradient descent specialization

For

\[
G(x)=x-\eta\nabla F(x),
\]

the centerline quantities are

\[
J_j=I-\eta\nabla^2F(\bar x_j),
\]

and, if the objective Hessian is \(L_{H,j}(R)\)-Lipschitz on the reference
ball,

\[
M_j(R)=\eta L_{H,j}(R).
\]

The certified recursion becomes

\[
\boxed{
\omega_{j+1}
=\|I-\eta\nabla^2F(\bar x_j)\|_2\omega_j
+\frac{\eta}{2}L_{H,j}(R_j)R_j^2.
}
\]

There is no anchor-to-center Hessian-mismatch injection.  Curvature drift
enters only through the nonautonomous propagator and a term quadratic in the
remaining total reference error.

For the one-sweep recentered path (c_j), Theorem 4.3 instead gives

\[
\boxed{
\epsilon_{j+1}
=\|I-\eta\nabla^2F(c_j)\|_2\epsilon_j
+\|c_j-\eta\nabla F(c_j)-c_{j+1}\|_2
+\frac{\eta}{2}\widehat L_{H,j}(\epsilon_j)\epsilon_j^2.
}
\]

The middle term is evaluated exactly before taking its norm.  It is the
second-order residual left after the signed variational correction, rather
than a global upper bound on nonlinear behavior over a ball containing the
entire correction.

## 6. Exact-reference deployment certificate

Let \(P\) be any scalar deployment quantity evaluated exactly on the corrected
reference.  If

\[
K_{P,j}(r)
\ge
\sup_{\|u\|\le r}
\|\nabla P(\widetilde x_j+u)\|_2,
\]

then

\[
\boxed{
|P(x_j)-P(\widetilde x_j)|
\le K_{P,j}(\omega_j)\omega_j.
}
\]

For a true-vs-competitor logit margin \(m_{i,c}\), this produces lower and
upper margin envelopes without a separate anchor-to-reference Taylor tail.
Guaranteed and possible correct counts then yield the same first-passage
accuracy bracket as Corollary 3.3 of `MOVING_WINDOW_THEOREM.md`.

The same statement applies to the recentered path after replacing
\((\widetilde x_j,\omega_j)\) by \((c_j,\epsilon_j)\).  Because the deployment
model is evaluated exactly at (c_j), the only output uncertainty is the
certified state radius around that path.

## 7. Development result

The theorem was derived after the frozen replication audit and therefore does
not replace it.  On the diagnosed smooth-MLP development case (seed 2, anchor
73,750), the old global tanh tube ended at step 33.  The improved theorem:

- remains finite through the requested 100 steps;
- predicts the 90% event at +35;
- certifies the singleton bracket `[35,35]`;
- contains the exact +35 crossing;
- has zero state-tube violations over 101 inspected states; and
- has maximum observed-error/bound ratio 0.079.

This is a post-audit development result, not confirmatory evidence.  A fresh
v1 protocol was frozen separately in `VARIATIONAL_THEOREM_PROTOCOL.md`.

That protocol then trained untouched seeds 5--8.  Sixteen of the twenty
seed-threshold events occur, twelve satisfy the fixed trigger rule, and v1
issues no bracket.  There are zero state-tube violations.  Nine triggers are
horizon-limited and three are margin-limited.  The only event inside the fixed
250-step window is seed 5's 60% transition at +20; v1 reaches +15.

Theorem 4.3 was motivated and implemented on seed 6 only.  At anchor 37,250,
it extends the rigorous horizon from 29 to 222 steps, a 7.66-fold increase,
without a violation.  Before evaluating v2 tube geometry on seeds 5, 7, or 8,
the one-sweep method and constants were frozen in
`RECENTERED_VARIATIONAL_PROTOCOL.md`.  On those nine sequential-holdout
triggers:

- median rigorous horizon increases from 23 to 148 steps;
- median pairwise horizon gain is 5.53-fold;
- both methods retain three full 250-step tubes;
- v2 has zero audited state-tube violations; and
- v2 issues one new bracket, `[20,26]`, for the seed-5 60% event whose exact
  first crossing is +20.  V1 abstains on the same anchor.

For the issued bracket, the recentered centerline predicts +22.  The minimum
pre-event exclusion slack is `3.80e-5`, the upper-endpoint guarantee slack is
`5.17e-6`, and the event-time state radius is `1.18e-5` versus observed error
`1.06e-5`.

This is a genuine cross-seed theorem improvement, but not a fully blinded
replication: training outcomes and trigger locations were known before the v2
holdout calculation.  The v2 conditional-coverage sample size is one.

### Fully untouched confirmatory audit

The theorem and one-sweep implementation were then frozen again before any
training artifact existed for the complete seed population 9--16.  The
protocol hash, code manifest, model, optimizer, five thresholds, 250-step
window, derivative bounds, trigger rules, and issuance logic were fixed in
advance.  All hashes matched after 52 paired anchor calculations.

Under the prespecified operational trigger, 27 seed-threshold cases trigger
and 19 realized events lie inside the fixed window.  Frozen v1 issues three
brackets; recentered v2 issues ten, retains all three v1 issuances, and covers
all ten.  The exact event-level 95% Clopper--Pearson interval is
`[0.692,1.000]`.  Median reached horizon rises from 25 to 179 steps, the median
pairwise gain is 5.60-fold, and no audited state trajectory exits its v2 tube.

The ten brackets occur on five distinct seeds and span every threshold from
60% to 95%.  Nine are singletons; the remaining bracket has width two.  Median
lead is 51.5 steps and maximum lead is 235.  In the sole seed meeting the
frozen natural-grokking delay criterion, v2 certifies the 60%, 80%, and 90%
events at leads 235, 92, and 186, while abstaining at a margin-limited 70%
event and a 95% event outside the window.

Because exact singleton brackets merit a numerical check, a separate post-hoc
audit rebuilds all ten issued tubes and recomputes the certifying count
inequalities.  Every bracket is reproduced exactly; the minimum strict output
slack is `5.7623e-7`.  This rules out float64 zero ties in the issued events,
but it is not directed-rounding interval arithmetic.

The statistical claim remains conditional and clustered: ten threshold
events arise from five issuing seeds, and 17/27 operational triggers abstain.
The audit establishes reproducible certificate availability under this fixed
near-event trigger, not unconditional long-range prediction.

## 8. Prior-art boundary

The recentering identity is a forward, fixed-initial-condition instance of a
classical idea.  [Chow, Lin, and Palmer
(1989)](https://xblin.math.ncsu.edu/preprint/chow-lin-palmer-1.pdf) formulate
shadowing of pseudo-orbits through Newton's method on a Banach space of
sequences, and [Chow and Palmer
(1992)](https://doi.org/10.1016/0885-064X(92)90004-U) give computable finite-time
shadowing bounds for higher-dimensional maps, including roundoff analysis.
Residual-based a-posteriori trajectory bounds and quadratic Newton residual
reduction are therefore established numerical-analysis ideas.

The theorem-level claim here is narrower.  The anchor is fixed rather than
adjusted to locate some nearby shadowing orbit; the correction is the causal
forward variational response of an anchor-modal forecast; the remaining state
tube uses neural-objective derivative bounds; and the result is coupled to
strict deployment-risk or classification first-passage brackets.  The paper
should claim that synthesis and its grokking application, not priority for
Newton correction or pseudo-orbit shadowing themselves.

## 9. Implementation

The generic theorem is implemented in `scripts/variational_shadowing.py`.
The v1 and recentered smooth-MLP specializations are in
`scripts/variational_mlp_certificate.py` and
`scripts/recentered_variational_mlp_certificate.py`.  Aggregate and numerical
slack audits are in `scripts/recentered_variational_audit.py` and
`scripts/recentered_key_event_diagnostics.py`.  The fully untouched driver and
its independent output-margin audit are
`scripts/fresh_v2_confirmatory.py` and
`scripts/fresh_v2_margin_audit.py`.  Tests include affine
exactness, literal nonlinear containment, quadratic residual contraction,
exact-Hessian agreement, exact-reference margin counts, and summary-accounting
checks.
