# Residual-corrected inexact-operator GreenCert theorem

## 1. Why this interface matters

The exact-arithmetic Gaussian Gram theorem is not invalidated by an approximate
HVP/JVP/VJP implementation, but its observed terminal norm must not be treated
as exact.  The correct object to verify is the residual of each computed Gram
power step.  The theorem below converts those local residual bounds into an
explicit spectral-norm enclosure without assuming the norm being estimated.

## 2. Inexact anytime Gram enclosure

Let \(T:\mathbb R^d\to\mathbb R^k\), \(A=T^*T\succeq0\), and let
\(g_1,\ldots,g_m\) be one committed iid standard-Gaussian block.  An arbitrary
implementation returns iterates

\[
\widetilde v_{i,0}=g_i,\qquad \widetilde v_{i,1},\ldots,
\widetilde v_{i,q}.
\]

Suppose verified numbers \(\xi_0,\ldots,\xi_{q-1}\ge0\) satisfy

\[
\max_i\|\widetilde v_{i,\ell+1}-A\widetilde v_{i,\ell}\|_2
\le \xi_\ell,
\qquad 0\le\ell<q,
\]

and write \(\widetilde Y_q=\max_i\|\widetilde v_{i,q}\|_2\).  For the same
folded-normal calibration

\[
c_{\delta,m}=\Phi^{-1}\!\left(\frac{1+\delta^{1/m}}2\right),
\]

define \(R_q\) as the unique positive root of

\[
\boxed{
c_{\delta,m}R_q^q
=\widetilde Y_q+
\sum_{\ell=0}^{q-1}\xi_\ell R_q^{q-1-\ell}.}
\tag{1}
\]

If \(\widetilde Y_q\) and every \(\xi_\ell\) vanish, set \(R_q=0\).

**Theorem (residual-corrected anytime Gram bound).** With probability at least
\(1-\delta\), simultaneously for every inspected power \(q\ge1\),

\[
\boxed{\|T\|_2\le \sqrt{R_q}.}
\tag{2}
\]

The same one-event argument supports adaptive stopping over powers.  A union
bound or predictable confidence spending supports a family of inexact
operators exactly as in the exact-product theorem, provided every reported
\(\xi_\ell\) is a deterministic or otherwise valid simultaneous upper bound.

### Proof

Let \(\lambda=\|A\|_2=\|T\|_2^2\) and let \(v\) be a unit top eigenvector.
On the standard Gaussian event, some committed probe \(g_i\) obeys

\[
|\langle v,g_i\rangle|\ge c_{\delta,m}.
\]

Define residuals
\(r_{i,\ell}=\widetilde v_{i,\ell+1}-A\widetilde v_{i,\ell}\).  Telescoping
the inexact recurrence gives

\[
A^qg_i=\widetilde v_{i,q}
-\sum_{\ell=0}^{q-1}A^{q-1-\ell}r_{i,\ell}.
\]

The top-eigenvector projection and the triangle inequality therefore imply

\[
c_{\delta,m}\lambda^q
\le \|A^qg_i\|_2
\le \widetilde Y_q+
\sum_{\ell=0}^{q-1}\xi_\ell\lambda^{q-1-\ell}.
\tag{3}
\]

The polynomial obtained by moving the right side of (1) to the left has one
positive leading coefficient and only nonpositive lower coefficients.  Unless
all data vanish, Descartes' rule of signs gives exactly one positive root; the
polynomial is nonpositive at zero and positive at infinity.  (Zero can also be
a root when the trailing coefficients vanish, which does not affect the unique
positive root.)  Inequality (3) forces
\(\lambda\le R_q\), proving (2).  The Gaussian projection event is the same
for every power, so the conclusion is simultaneous over adaptively inspected
powers. \(\square\)

## 3. Cheap special cases

At \(q=1\), no polynomial solve is needed:

\[
\boxed{
\|T\|_2\le
\left(\frac{\widetilde Y_1+\xi_0}{c_{\delta,m}}\right)^{1/2}.}
\tag{4}
\]

More generally, any outward supersolution \(\overline R\) satisfying

\[
c_{\delta,m}\overline R^q\ge
\widetilde Y_q+
\sum_{\ell=0}^{q-1}\xi_\ell\overline R^{q-1-\ell}
\]

can replace the exact root.  Thus a proof-producing implementation needs only
verified residual norms and a one-dimensional outward root solve; it need not
outward-enclose the entire power iteration as one monolithic computation.

### Certificate-aware operator-cap controller

The supersolution form gives an explicit local precision policy.  Fix a desired
squared operator cap \(\overline R>0\), a power \(q\), and nonnegative budgets
\(\overline Y,b_0,\ldots,b_{q-1}\) satisfying

\[
\overline Y+\sum_{\ell=0}^{q-1}b_\ell
\le c_{\delta,m}\overline R^q.
\]

If the computed block obeys

\[
\widetilde Y_q\le\overline Y,
\qquad
\xi_\ell\overline R^{q-1-\ell}\le b_\ell
\quad(0\le\ell<q),
\]

then \(\overline R\) is a supersolution of (1), and therefore

\[
\boxed{\|T\|_2\le\sqrt{\overline R}.}
\]

Each Gram step now has a local residual allowance
\(b_\ell/\overline R^{q-1-\ell}\).  A verifier can try the cheapest arithmetic,
recompute only a failed step at higher precision, and retain the same committed
probe event.  If the terminal allowance fails, it refines or abstains.  This
turns a downstream closure limit on \(\|T\|\) directly into an executable
precision contract; no root solve is needed when the only question is whether
the declared cap holds.  `scripts/precision_budget_controller.py` implements
this policy, and its property tests cover 2,405 deterministic/randomized cases.

The arithmetic precision and implementation may also be selected or refined
after inspecting preceding iterates.  Conditional on the same Gaussian event,
the telescoping inequality is deterministic for every iterate sequence with
valid residual bounds, so precision adaptation consumes no additional failure
budget.

If a kernel supplies a uniform exact-real defect inequality

\[
\|\widetilde A w-Aw\|\le\varepsilon\|w\|,
\]

then the executable iterates themselves give

\[
\xi_\ell=\varepsilon\|\widetilde v_{\ell}\|.
\]

This turns a verified JVP/VJP or fused Gram-kernel error contract into the
operator bound consumed by GreenCert.  The nonlinear closure then remains
unchanged except that every exact-product randomized bound is replaced by its
residual-corrected counterpart.

### Causal response residual

The signed response itself has an equally local a posteriori interface.  Let
\(\widetilde s\) approximate the exact defect \(s\), let
\(\widetilde z_0=0\), and define the exact-real recurrence residual of the
computed response by

\[
d_j=\widetilde z_{j+1}-J_j\widetilde z_j-\widetilde s_j.
\]

By the definition of the causal Green operator,

\[
\widetilde z=K_H(\widetilde s+d),
\qquad
K_Hs-\widetilde z=K_H(s-\widetilde s-d).
\]

Therefore, if \(\|s-\widetilde s\|_U\le\sigma\),
\(\|d\|_U\le\tau\), and \(\kappa\ge\|K_H\|\), then

\[
\boxed{\|K_Hs-\widetilde z\|_X\le\kappa(\sigma+\tau).}
\tag{5}
\]

This replaces a monolithic outward enclosure of the full response solve by
verified local defect and recurrence-residual norms.  The same HVP/JVP kernel
contracts used for the inexact Gram theorem can supply the needed local terms.

The same identity residualizes the second-order response used by the sharper
closure.  If \(q=N(\widetilde z)\), \(\widetilde q\) approximates \(q\), and
an anchor-fixed \(\widetilde y\) has residual

\[
d^q_j=\widetilde y_{j+1}-J_j\widetilde y_j-\widetilde q_j,
\]

then verified bounds \(\|q-\widetilde q\|_U\le\sigma_q\) and
\(\|d^q\|_U\le\tau_q\) give

\[
\boxed{
\|K_Hq\|_X\le
\|\widetilde y\|_X+\kappa(\sigma_q+\tau_q).}
\tag{6}
\]

Together, (5) and (6) supply both numerical quantities \(\alpha\) and
\(\beta\) in the response-centered closure from local residual certificates.

### Inexact variational sweep

The same local accounting applies while constructing the reference path.  Let
\(r_j=G(c_j)-c_{j+1}\), \(J_j=DG(c_j)\), let \(\widetilde r_j\) approximate
\(r_j\), and let an anchor-fixed correction \(\widetilde z_0=0\) have
recurrence residual

\[
d_j=\widetilde z_{j+1}-J_j\widetilde z_j-\widetilde r_j.
\]

For the recentered path \(c'=c+\widetilde z\), direct substitution gives the
exact identity

\[
\boxed{
r'_j=N_j(\widetilde z_j)+(r_j-\widetilde r_j)-d_j.}
\tag{7}
\]

Thus, if \(DG\) is \(M_j\)-Lipschitz on the connecting segment and verified
local bounds satisfy \(\|r_j-\widetilde r_j\|\le\sigma_j\) and
\(\|d_j\|\le\tau_j\), then

\[
\boxed{
\|r'_j\|\le \frac12M_j\|\widetilde z_j\|^2+\sigma_j+\tau_j.}
\tag{8}
\]

The exact-sweep quadratic contraction is the special case
\(\sigma_j=\tau_j=0\).  Equation (8) is the practical version: reduced
precision or an iterative response solve does not destroy the second-order
mechanism; its verified local residual enters additively and can be checked
after the sweep.

## 4. Block-Frobenius calibration

The maximum-over-probes statistic is not the only valid aggregation of a
committed Gaussian block.  Write \(G=[g_1\ \cdots\ g_m]\) and let

\[
c^{\chi}_{\delta,m}=F^{-1}_{\chi_m}(\delta),
\]

the lower \(\delta\)-quantile of a chi random variable with \(m\) degrees of
freedom.  If computed block iterates satisfy

\[
\widetilde V_0=G,\qquad
\|\widetilde V_{\ell+1}-A\widetilde V_\ell\|_F\le\xi^F_\ell,
\]

define \(R^F_q\) by

\[
\boxed{
c^{\chi}_{\delta,m}(R^F_q)^q
=\|\widetilde V_q\|_F+
\sum_{\ell=0}^{q-1}\xi^F_\ell(R^F_q)^{q-1-\ell}.}
\tag{9}
\]

**Corollary (inexact chi-block bound).** With probability at least
\(1-\delta\), simultaneously over inspected powers,

\[
\boxed{\|T\|_2\le\sqrt{R^F_q}.}
\tag{10}
\]

Indeed, for a unit top eigenvector \(v\), the row \(v^TG\) consists of \(m\)
iid standard normals, so \(\|v^TG\|_2\sim\chi_m\).  Furthermore,
\(\|A^qG\|_F\ge\|v^TA^qG\|_2=\lambda^q\|v^TG\|_2\).  The block residuals
then telescope exactly as in the theorem, with Frobenius norms replacing the
maximum column norm.

The max-column and chi-Frobenius bounds consume different summaries of the
same computed block.  If their failure probabilities are preallocated as
\(\delta_1+\delta_2\le\delta\), their minimum is valid with probability at
least \(1-\delta\) and costs no additional Gram application.  This hybrid is
not uniformly tighter because splitting confidence changes both calibrations;
it should be selected by a frozen protocol rather than by outcome.

## 5. Outward scalar realization and mixed-precision audit

`scripts/outward_inexact_anytime_gram.py` implements the one-dimensional root
with 256-bit Arb arithmetic.  Binary64 terminal and residual bounds are treated
as exact dyadics, the folded-normal calibration is rounded downward, and the
returned binary64 operator bound is checked as an exact-dyadic polynomial
supersolution.  Tests cover 1,002 roots and 15 calibrations; the largest
outward-to-binary64 root ratio is \(1+4.4\times10^{-16}\).

The binary64 sensitivity boundary on the immutable \(H=26,q=1\) replay is
numerically critical: the endpoint \(r=0.5999285448884594\) sits at the
closure discriminant.  A fixed 1% interior factor,
\(r=0.5939292594395748\), survives the complete 256-bit outward scalar replay
with the same `[2,2]` bracket, discriminant \(8.58\times10^{-3}\), and minimum
event-logic slack \(1.87\times10^{-3}\).  The proof-engineering lesson is
general: do not promote a floating-point threshold that rests on a zero
discriminant.

An actual post-seal float32 audit evaluates every committed q=1 output and
Green Gram product against its binary64 target.  Its largest measured residual
ratio is \(3.36\times10^{-7}\), about \(1.79\times10^6\) below the common
admissible threshold.  Residual-corrected outward scalar roots preserve
`[2,2]`, Green-bound inflation is only \(1+2.75\times10^{-8}\), and paired q=1
kernel time improves in all 20 pairs across four separately launched,
five-repeat alternating-order invocations.  The pooled median is
\(1.83\times\), the range is \(1.04\)--\(2.38\times\), and the invocation medians
are \(1.66\times\), \(1.86\times\), \(1.89\times\), and \(1.78\times\).
Independent auditors recheck every scalar
supersolution, closure, margin, hash, and timing ratio; the aggregate preserves
the full wall-time variability rather than filtering it.  The measured
binary64 discrepancy is not an outward exact-real neural-kernel residual, so
this establishes a concrete mixed-precision path and headroom---not the final
computer-assisted kernel proof.

Records:

- `results/transformer_v3_outward_inexact_root_postseal_audit.json`;
- `results/transformer_v3_mixed_precision_residual_postseal_audit.json`;
- `results/transformer_v3_mixed_precision_residual_independent_audit.json`;
- `results/transformer_v3_mixed_precision_residual_replication1_postseal_audit.json`;
- `results/transformer_v3_mixed_precision_residual_replication1_independent_audit.json`;
- `results/transformer_v3_mixed_precision_residual_replication2_postseal_audit.json`;
- `results/transformer_v3_mixed_precision_residual_replication2_independent_audit.json`;
- `results/transformer_v3_mixed_precision_residual_replication3_postseal_audit.json`;
- `results/transformer_v3_mixed_precision_residual_replication3_independent_audit.json`;
- `results/transformer_v3_mixed_precision_timing_aggregate_audit.json`.

## 6. Claim boundary

This theorem supplies the missing deterministic interface; it does not by
itself verify the current Transformer kernels.  A complete computer-assisted
Transformer certificate must still produce outward residual bounds for every
queried Gram step and outward bounds for neural jets and output margins.  The
post-seal sensitivity audit measures admissible residual budgets conditional
on those remaining quantities; it is not an empirical estimate of the actual
floating-point residual.

On the immutable \(H=26,q=1\) Transformer replay, a 50--50 Bonferroni hybrid
of max-column and chi-Frobenius summaries preserves the same singleton event
and decreases the certified state radius from \(2.1331\times10^{-15}\) to
\(2.0705\times10^{-15}\), with no additional operator application.  The
single-case replay is implementation evidence, not a replacement for a frozen
multi-case comparison.

The diagonal time-weighted extension in `WEIGHTED_GREENCERT_THEOREM.md` is a
valid preconditioning theorem but not a practical win on this replay: it
reduces the final state radius from \(1.8262\times10^{-15}\) to
\(1.6737\times10^{-15}\) only after increasing the Green depth from q=1 to
q=2.  It remains method-development evidence and is not part of the headline
algorithm.
