# Causal row-Green closure

Status: v1.4 theorem development. This note is not yet a paper claim.

## 1. Rowwise finite-window closure

Let a differentiable map generate the realized anchored trajectory

\[
x_{a+j+1}=G_j(x_{a+j}),\qquad e_0=0,
\]

and let `b_0,...,b_H` be any anchor-fixed reference. Put

\[
J_j=DG_j(b_j),\qquad
s_j=G_j(b_j)-b_{j+1},
\]

and let the causal Green operator `K` solve

\[
z_0=0,\qquad z_{j+1}=J_jz_j+u_j.
\]

Write `K_i` for its row operator from the whole injection sequence to state
`z_{i+1}`. Causality means that `K_i` ignores every injection after `i`.

### Theorem 1 (causal row-Green envelope)

Suppose

\[
\kappa_i\ge\|K_i\|_{2\to2},\qquad
Y_{i+1}\ge\|(Ks)_{i+1}\|,
\]

and, on the declared domain at transition input `k`,

\[
\|G_k(b_k+u)-G_k(b_k)-J_ku\|
\le \frac{M_k}{2}\|u\|^2.
\]

Define `r_0=0` and, chronologically for `0 <= i < H`,

\[
\boxed{
r_{i+1}=Y_{i+1}+\kappa_i
\left\{\sum_{k=1}^{i}
\left(\frac{M_kr_k^2}{2}\right)^2\right\}^{1/2}.
}
\]

If every ball used in the recursion lies in its declared derivative domain,
then

\[
\boxed{\|x_{a+k}-b_k\|\le r_k\quad(1\le k\le H).}
\]

The recursion has no global discriminant or contraction condition. It may
still abstain when a local derivative domain fails or a numerical enclosure is
nonfinite.

### Proof

Let

\[
N_k(u)=G_k(b_k+u)-G_k(b_k)-J_ku.
\]

The exact error satisfies

\[
e_{i+1}=(Ks)_{i+1}+K_iN(e),
\]

where the input to `K_i` is the prefix `N_0(e_0),...,N_i(e_i)` padded by
zeros. Since `e_0=0`, the first nonlinear injection vanishes. Assume
inductively that `||e_k|| <= r_k` for `1 <= k <= i`. Then

\[
\begin{aligned}
\|e_{i+1}\|
&\le Y_{i+1}+\kappa_i
\left\|\bigl(N_1(e_1),\ldots,N_i(e_i)\bigr)\right\|_2\\
&\le Y_{i+1}+\kappa_i
\left\{\sum_{k=1}^{i}
\left(\frac{M_kr_k^2}{2}\right)^2\right\}^{1/2}
=r_{i+1}.
\end{aligned}
\]

Chronological induction proves the claim. The domain checks validate each
Taylor bound before it is used. ∎

## 2. Signed second response

Suppose `s = q + h`, a computed signed response `y` satisfies

\[
y_{i+1}=J_iy_i+q_i+d_i,\qquad y_0=0,
\]

and pointwise enclosures `epsilon_i >= ||h_i-d_i||` are available. Then the
same row bounds give

\[
\boxed{
Y_{i+1}=\|y_{i+1}\|+\kappa_i
\left(\sum_{k=0}^{i}\epsilon_k^2\right)^{1/2}.
}
\]

This retains the direction of the known forcing `q`; only the unresolved
forcing is converted to a norm. For cancellation-safe optimizer certification,
`q` is the directional quadratic Taylor term and `epsilon` contains the
fourth-order directional remainder plus verified arithmetic residuals.

## 3. Simultaneous row norms from one Green pass

Let `g_1,...,g_m` be independent standard Gaussian vectors in the full
injection-sequence space. For positive budgets `delta_i` satisfying
`sum_i delta_i <= delta`, define

\[
c_i=\Phi^{-1}\!\left(\frac{1+\delta_i^{1/m}}2\right),\qquad
\widehat\kappa_i=\frac{\max_{\ell\le m}\|K_ig_\ell\|}{c_i}.
\]

### Proposition 2 (one-pass simultaneous row enclosure)

With probability at least `1-delta`, every row bound holds simultaneously:

\[
\boxed{\|K_i\|\le\widehat\kappa_i\quad(0\le i<H).}
\]

### Proof

Fix a unit top right singular vector `v_i` of `K_i`, embedded in the full
sequence space. For every probe,

\[
\|K_ig_\ell\|\ge\|K_i\|\,|v_i^Tg_\ell|.
\]

The probability that all `m` absolute standard-normal projections are below
`c_i` is exactly `delta_i`. A union bound over rows proves the simultaneous
claim. ∎

All vectors `K_i g_l` are already present as the checkpoint rows of the usual
batched direct-image Green evaluation. Thus Proposition 2 adds scalar norm
reductions and a stricter calibration, but no JVP or VJP pass. A known signed
forcing can be appended as one additional batch row, producing every `y_i` in
the same sequential pass.

## 4. Relation to the scalar closure

The response-centered scalar theorem compresses the whole output sequence to
one Green norm, one curvature maximum, and one quadratic root. The row-Green
theorem instead preserves when uncertainty reaches each checkpoint. It is
especially useful when curvature or gain is concentrated late in the window:
late constants cannot invalidate earlier event margins. It is not guaranteed
to be numerically tighter after finite-probe calibration; the relevant test is
certificate issuance and measured runtime under a frozen protocol.
