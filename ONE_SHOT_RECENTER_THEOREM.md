# One-shot response-centered Green certification

Status: theorem and implementation audit complete; Transformer and digits
results below are post-seal analyses, not prospective confirmations.

## 1. Setup

For a finite sequence of differentiable maps (the autonomous case is a
special case), let

\[
x_{j+1}=G_j(x_j),\qquad c_0=x_0,
\]

and define the reference-path defect and Jacobian

\[
s_j=G_j(c_j)-c_{j+1},\qquad J_j=DG_j(c_j).
\]

For an injection sequence \(u=(u_0,\ldots,u_{H-1})\), the anchor-fixed causal
Green operator is

\[
(K_Hu)_0=0,\qquad (K_Hu)_{j+1}=J_j(K_Hu)_j+u_j.
\]

Write

\[
\|v\|_X^2=\sum_{j=1}^H\|v_j\|^2,
\qquad
\|v\|_{\infty,2}=\max_{1\le j\le H}\|v_j\|.
\]

Let

\[
N_j(u)=G_j(c_j+u)-G_j(c_j)-J_ju.
\]

The exact state error \(h_j=x_j-c_j\) satisfies

\[
h=K_Hs+K_HN(h).
\]

## 2. Response-centered theorem with numerical error

**Theorem (one-shot response-centered closure).** Suppose

\[
\|DG_j(c_j+u)-J_j\|\le M\|u\|
\]

for every \(j<H\) and every \(\|u\|\le\rho\), and let
\(\kappa\ge\|K_H\|\). Let a computed signed response \(\widetilde z\), with
\(\widetilde z_0=0\), satisfy

\[
\|K_Hs-\widetilde z\|_X\le\alpha.
\]

Set

\[
Z=\|\widetilde z\|_X,
\qquad
p=\|\widetilde z\|_{\infty,2},
\qquad
q=N(\widetilde z).
\]

Suppose \(\beta\ge\|K_Hq\|_X\), and define

\[
b=\kappa M,\qquad Y=\alpha+\beta.
\]

If some \(E\ge0\) obeys

\[
p+E\le\rho,
\qquad
Y+bpE+\frac b2E^2\le E,
\tag{1}
\]

then the realized trajectory satisfies

\[
\|h-\widetilde z\|_X\le E,
\qquad
\|h_j\|\le\|\widetilde z_j\|+E\le p+E
\quad(1\le j\le H).
\]

**Proof.** Put \(e=h-\widetilde z\). The exact recurrence gives

\[
e=(K_Hs-\widetilde z)+K_HN(\widetilde z+e).
\]

For each state, the integral Taylor identity implies

\[
\|N_j(\widetilde z_j+e_j)-N_j(\widetilde z_j)\|
\le M\|\widetilde z_j\|\|e_j\|+\frac M2\|e_j\|^2.
\]

Taking the sequence norm and using
\(\|(\|e_j\|^2)_j\|_2\le\|e\|_X^2\) yields

\[
\|N(\widetilde z+e)-N(\widetilde z)\|_X
\le Mp\|e\|_X+\frac M2\|e\|_X^2.
\]

Thus the exact fixed-point map for \(e\) maps the closed radius-\(E\) sequence
ball into itself under (1). The domain condition makes every derivative bound
valid. Brouwer supplies a fixed point, and causal forward recurrence makes that
fixed point unique and equal to the realized trajectory error. QED.

### Computable forcing bound

If \(\widetilde q\) approximates \(q=N(\widetilde z)\), with

\[
\|q-\widetilde q\|_X\le\tau,
\qquad
\|K_H\widetilde q-\widetilde y\|_X\le\xi,
\]

then one may use

\[
\beta=\|\widetilde y\|_X+\kappa\tau+\xi.
\]

This is the outward-rounded route. It is unattractive in Transformer float64
because \(q\) is second order and can be smaller than subtraction roundoff.

### Zero-extra-operator corollary

Taylor's theorem also gives

\[
\|q\|_X
\le\frac M2
\left(\sum_{j=0}^{H-1}\|\widetilde z_j\|^4\right)^{1/2}
\le\frac M2pZ.
\]

Therefore the fully analytic choice

\[
\boxed{\beta=\frac12\kappa MpZ=\frac12bpZ}
\tag{2}
\]

requires no new model evaluation, HVP, VJP, or randomized probe. With
\(Y=\alpha+bpZ/2\), closure exists when

\[
bp<1,
\qquad
D=(1-bp)^2-2bY\ge0.
\]

The smaller root is evaluated without cancellation as

\[
\boxed{
E=\frac{2Y}{1-bp+\sqrt D}
}
\tag{3}
\]

(with \(E=0\) when \(Y=0\)). The certified pointwise state radius is
\(p+E\), not the old sequence-ball radius \(2Z\).

The improvement is controlled by temporal concentration. When the signed
response is spread across many checkpoints, \(p/Z\) is small and (2) can be
far tighter than charging \(Z\) itself as an unexplained displacement.

## 3. Free response-aware abstention

In exact arithmetic, \(z=K_Hs\) gives

\[
\|K_H\|\ge\frac{\|z\|_X}{\|s\|_X}.
\]

Together with the last-transition injection bound \(\|K_H\|\ge1\), define

\[
\underline\kappa
=\max\left\{1,\frac{\|z\|_X}{\|s\|_X}\right\}.
\tag{4}
\]

For every fixed \(E\), the left side of (1) is nondecreasing in \(\kappa\).
Hence if no admissible root exists at \(\underline\kappa\), no valid Green
upper bound can close the theorem. The verifier may abstain before a randomized
Green query. With response error \(\alpha\), replace the numerator in (4) by
\((\|\widetilde z\|_X-\alpha)_+\).

On the stored Transformer population this rule preserves all four original
no-Green early abstentions, including the one for which the weaker
\(\|K_H\|\ge1\) test alone would no longer suffice.

## 4. Anytime Gaussian power enclosure

Let \(A=K_H^\top K_H\succeq0\), let \(v\) be a unit top eigenvector, and draw
\(m\) independent standard Gaussian vectors \(g_i\). For

\[
c_\delta=\Phi^{-1}\!\left(\frac{1+\delta^{1/m}}2\right),
\]

the event

\[
\max_i|v^\top g_i|\ge c_\delta
\]

has probability \(1-\delta\). On this *single event*, simultaneously for every
integer \(q\ge1\),

\[
\boxed{
\|K_H\|
\le
\left(
\frac{\max_i\|A^qg_i\|}{c_\delta}
\right)^{1/(2q)}.
}
\tag{5}
\]

Indeed, \(\|A^qg_i\|\ge\|A\|^q|v^\top g_i|\), and
\(\|A\|=\|K_H\|^2\). Because the event does not depend on \(q\), a verifier
may inspect powers \(1,2,\ldots,q_{\max}\) and stop at the first downstream
closure without a union bound over powers. This is optional stopping on a
simultaneous deterministic consequence of one precommitted random event, not
fresh adaptive sampling.

The same argument applies to every output-Jacobian Gram probe. Numerical
roundoff in the Gram products must still be outward enclosed for a
computer-assisted exact-real claim.

## 5. Audited empirical consequences

The conservative corollary was applied to all 23 sealed Transformer candidate
records without changing any stored quantity:

- 18 records had a Green query and were evaluable;
- closure/issuance changes from 9 to 13;
- all 13 retrospective brackets contain the revealed event;
- the four conversions are seeds 342 (70%), 348 (90%), 352 (80%), and 354
  (70%);
- the new radius is at most 14.92% of the old \(2Z\) radius among passing
  records;
- all 13 observed state errors lie inside the new radius;
- all four old no-Green abstentions remain safely rejectable using (4).

This is a post-seal theorem audit. It is evidence that the theorem is useful,
not confirmatory coverage evidence for a method chosen before outcomes.

The same audit on digits preserves 7/7 issuance and adds no case. This is an
important negative control: the gain depends on temporal response geometry and
is not a relabeling that mechanically turns every old certificate into a new
one.

Two complete Transformer replays audited (5):

| Seed/gate | Horizon | First issuing power | Probe speedup | Projected end-to-end speedup | Bracket |
|---|---:|---:|---:|---:|---:|
| 333 / 70% | 52 | 1 | 7.77x | 3.27x | [28, 28] |
| 342 / 70% | 93 | 5 | 1.57x | 1.45x | [69, 69] |

Issuance persists from the first passing power through \(q=8\) in both
records. The timings are measured post-seal implementation audits, not a broad
hardware-independent scaling claim.

## 6. Prior-art boundary

Newton correction of pseudo-orbits and radii-polynomial recentering are
classical. Chow, Lin, and Palmer formulate shadowing through Newton's method
and an invertible sequence operator (SIAM J. Math. Anal. 20(3), 1989,
doi:10.1137/0520038). Radii-polynomial methods likewise center a contraction
argument at a refined numerical approximation (Lessard and Reinhardt, SIAM J.
Numer. Anal. 52(1), 2014, doi:10.1137/13090883X). Random-start power analysis
is also classical (Kuczyński and Woźniakowski, SIAM J. Matrix Anal. Appl.
13(4), 1992, doi:10.1137/0613066).

The defensible contribution is narrower and more concrete: an anchor-fixed
causal neural-training specialization whose already-computed signed Green
response becomes a zero-extra-operator center; a mixed temporal
\(\ell_{\infty,2}\)-by-\(\ell_2\) remainder bound; transport to persistent
first-passage output events; a response-aware no-query rejection rule; and a
same-random-event stopping rule that materially cuts measured verifier cost.

## 7. Files and reproduction

- Scalar theorem implementation: `scripts/one_shot_recenter_closure.py`
- Transformer post-seal theorem audit:
  `scripts/audit_one_shot_signed_recenter.py`
- Exact nonlinear and stored-record tests:
  `scripts/test_one_shot_signed_recenter.py`
- Digits negative-control audit:
  `scripts/audit_digits_one_shot_recenter.py`
- Progressive Gram implementation: `scripts/batched_green_operator.py`
- Progressive replay: `scripts/audit_progressive_probe_replay.py`
- Progressive aggregate/test: `scripts/audit_progressive_probe_results.py`,
  `scripts/test_progressive_probe_results.py`
- Audit outputs:
  `results/one_shot_signed_recenter_postseal_audit.json`,
  `results/digits_signed_confirmation/one_shot_recenter_postseal_audit.json`,
  `results/progressive_probe_replay_audit.json`
