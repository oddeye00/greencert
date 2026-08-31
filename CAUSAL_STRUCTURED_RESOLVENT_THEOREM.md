# Causal structured resolvent theorem

Status: post-v1.1.0 theorem development. This theorem is intended to attack
the present full-horizon HVP cost. It has not yet changed a released
certificate or manuscript claim.

## 1. Exact and approximate causal dynamics

Let \(E\) be the optimizer-state space, \(Q\) the parameter space, and
\(P:E\to Q\) the parameter projection. For a horizon \(H\), define the
anchor-fixed difference operator

\[
(\mathcal L_Jy)_j=y_{j+1}-J_jy_j,\qquad y_0=0,\qquad 0\le j<H.
\]

Because this operator is block lower triangular with identity diagonal, it is
invertible on every finite horizon. Its inverse is the causal Green operator
\(K_J=\mathcal L_J^{-1}\).

For scaled momentum, let

\[
Bq=(-\eta q,\eta q),\qquad \eta>0,
\]

and apply \(B\) blockwise as \(\mathcal B:Q^H\to E^H\). Let
\(\mathcal P:E^H\to Q^H\) apply \(P\) blockwise, and define the causal shift

\[
\mathcal S(p_1,\ldots,p_H)=(0,p_1,\ldots,p_{H-1}).
\]

Suppose a cheap approximate Jacobian sequence has the same optimizer skeleton
and differs only through a parameter Hessian residual:

\[
J_j=\widetilde J_j+B\Delta_jP.
\]

Write

\[
\widetilde K=\mathcal L_{\widetilde J}^{-1},\qquad
T_0=\mathcal P\widetilde K\mathcal B,\qquad
\mathcal A=\mathcal D_\Delta\mathcal S T_0,
\]

where
\(\mathcal D_\Delta=\operatorname{diag}(\Delta_0,\ldots,\Delta_{H-1})\).

## 2. Exact structured resolvent identity

### Theorem 1 (finite causal resolvent)

The mismatch operator \(\mathcal A:Q^H\to Q^H\) is strictly block lower
triangular, hence \(\mathcal A^H=0\), and

\[
\boxed{
\mathcal P K_J\mathcal B
=T_0(I-\mathcal A)^{-1}
=T_0\sum_{n=0}^{H-1}\mathcal A^n.
}
\]

This identity requires no asymptotic stability assumption and no smallness
condition on \(\mathcal A\).

### Proof

For \(w\in Q^H\), set \(y=\widetilde K\mathcal Bw\). Then

\[
\begin{aligned}
\mathcal L_Jy
&=\mathcal L_{\widetilde J}y
  -\mathcal B\mathcal D_\Delta\mathcal S\mathcal Py\\
&=\mathcal Bw-\mathcal B\mathcal D_\Delta
  \mathcal S\mathcal Py\\
&=\mathcal B(I-\mathcal A)w.
\end{aligned}
\]

Causality makes \(T_0\) block lower triangular from forcing time to later
state time; the shift makes \(\mathcal A\) strictly lower triangular. Thus
\(\mathcal A^H=0\) and
\((I-\mathcal A)^{-1}=\sum_{n=0}^{H-1}\mathcal A^n\). Taking
\(w=(I-\mathcal A)^{-1}q\), uniqueness of the anchored recurrence gives
\(K_J\mathcal Bq=\widetilde K\mathcal Bw\). Applying \(\mathcal P\)
proves the result. \(\square\)

## 3. Certified gain and truncated-response bounds

For \(\alpha\ge0\), define

\[
g_H(\alpha)=\sum_{n=0}^{H-1}\alpha^n,\qquad
t_{H,m}(\alpha)=\sum_{n=m+1}^{H-1}\alpha^n,\quad 0\le m<H.
\]

These are finite polynomials; in particular, they remain valid when
\(\alpha\ge1\).

### Corollary 1 (preconditioned structured gain)

If

\[
\kappa_0\ge\|T_0\|,\qquad \alpha\ge\|\mathcal A\|,
\]

then

\[
\boxed{
\|\mathcal P K_J\mathcal B\|
\le\kappa_0g_H(\alpha).
}
\]

When \(\alpha<1\), the familiar bound
\(\kappa_0/(1-\alpha)\) is valid but slightly weaker than the finite-horizon
polynomial.

### Corollary 2 (cheap signed response with a certified tail)

Let \(q\) be an exact structured forcing and let
\(\widetilde q\) satisfy \(\|q-\widetilde q\|\le\sigma_q\). For
\(0\le m<H\), define

\[
w_m=\sum_{n=0}^{m}\mathcal A^n\widetilde q,\qquad
\widetilde y_m=T_0w_m.
\]

If the computed parameter response has numerical error at most
\(\delta_y\), then

\[
\boxed{
\|\mathcal P K_J\mathcal Bq-\widetilde y_m^{\rm comp}\|
\le
\delta_y+
\kappa_0\left[g_H(\alpha)\sigma_q
+t_{H,m}(\alpha)\|\widetilde q\|\right].
}
\]

Thus \(m=0\) uses one response through the cheap approximate dynamics and
certifies every omitted exact-Jacobian correction as a finite causal tail.
If \(\alpha\ll1\), one or two terms replace a full exact-HVP Green sweep.

### Proof

Insert the finite Neumann expansion from Theorem 1, separate the forcing
error and the terms of degree greater than \(m\), and bound each power by
\(\|\mathcal A^n\|\le\alpha^n\). \(\square\)

## 4. Feeding the nonlinear closure

The gain in Corollary 1 may be substituted directly for \(\kappa_B\) in the
forcing-subspace directional theorem. A scalar-curvature closure may take

\[
\kappa_{L,0}
\le \kappa_0g_H(\alpha)
\max_{1\le j<H}L_j.
\]

A sharper profiled version follows from

\[
\mathcal P K_J\mathcal B\mathcal D_LQ_0
=T_0\sum_{n=0}^{H-1}\mathcal A^n\mathcal D_LQ_0.
\]

For example, if
\(\kappa_{0,L}\ge\|T_0\mathcal D_LQ_0\|\), then

\[
\boxed{
\kappa_{L,0}
\le \kappa_{0,L}
+\kappa_0\|\mathcal D_LQ_0\|
\sum_{n=1}^{H-1}\alpha^n.
}
\]

Any directly certified norm of the displayed finite sum can replace this
triangle bound.

## 5. Segmented low-rank specialization

Partition the checkpoint window into segments and choose a cheap symmetric
approximation \(\widehat H_s\) at each segment anchor. For checkpoint \(j\) in
segment \(s(j)\), set \(\widetilde J_j\) equal to the scaled-momentum Jacobian
with Hessian \(\widehat H_{s(j)}\). If

\[
\|H_{a_s}-\widehat H_s\|\le\varepsilon_s,\qquad
\|H_j-H_{a_s}\|\le L_{H,s}\|\theta_j-\theta_{a_s}\|,
\]

then

\[
\|\Delta_j\|\le
\delta_j:=\varepsilon_s+L_{H,s}\|\theta_j-\theta_{a_s}\|.
\]

Let \(\mathcal D_\delta\) multiply block \(j\) by \(\delta_j\). The mismatch
gain obeys

\[
\boxed{
\|\mathcal A\|
\le\|\mathcal D_\delta\mathcal S T_0\|.
}
\]

The operator on the right contains only the approximate causal solver and
scalar weights. If each \(\widehat H_s\) is rank \(r\), its products require
low-rank matrix-vector operations rather than objective HVPs. Building and
certifying \(S\) segment sketches costs roughly \(O(Sr)\) expensive HVPs;
subsequent Green probes cost \(O(Hrd)\) arithmetic and no exact checkpoint
HVPs. The intended regime is \(S\ll H\).

This cost statement includes neither the problem-dependent construction of a
residual spectral certificate \(\varepsilon_s\) nor derivative-envelope work;
both must be counted in an empirical benchmark.

## 6. Causal block-majorant closure

The scalar norm \(\alpha\) discards where in time each mismatch occurs. A
strictly sharper causal interface keeps an \(H\times H\) matrix of block
bounds. Write \((T_0)_{ik}:Q\to Q\) for the block mapping forcing at update
\(k\) to the parameter state at update \(i+1\), and suppose

\[
N_{ik}\ge\|(T_0)_{ik}\|,
\qquad
\delta_j\ge\|\Delta_j\|.
\]

The matrix \(N\) is lower triangular. Define the strictly lower triangular
majorant

\[
M_{jk}=
\begin{cases}
\delta_jN_{j-1,k},&1\le j<H,\ k<j,\\
0,&\text{otherwise}.
\end{cases}
\]

### Theorem 2 (finite causal block majorant)

Every block of the exact structured Green operator is majorized by

\[
\boxed{
C=N(I-M)^{-1}=N\sum_{n=0}^{H-1}M^n.
}
\]

Consequently,

\[
\boxed{
\|\mathcal P K_J\mathcal B\|_{2\to2}\le\|C\|_{2\to2}.
}
\]

More specifically, if \(v_k=\|q_k\|\), then the exact response has block-norm
vector bounded componentwise by \(Cv\), and hence

\[
\|\mathcal P K_J\mathcal Bq\|_2\le\|Cv\|_2.
\]

### Proof

For every block vector \(q\), the block-norm vector of \(\mathcal Aq\) is at
most \(Mv\) componentwise. Induction gives the corresponding bound
\(M^nv\) for \(\mathcal A^nq\). Apply the finite resolvent identity and the
block majorant \(N\) for \(T_0\). Finally,
\(\|v\|_2=\|q\|_2\), so the scalar matrix spectral norm gives the displayed
operator bound. \(\square\)

For local curvature profiles, let \(D_{L,0}\) be the scalar
\(H\times(H-1)\) matrix that prepends the zero update-zero block and weights
the remaining forcing blocks by \(L_j\). Then

\[
\boxed{
\|\mathcal P K_J\mathcal B\mathcal D_LQ_0\|
\le\|C D_{L,0}\|_2.
}
\]

This is a finite calculation even when \(\|\mathcal A\|\ge1\). For a scalar
Hessian baseline \(\widetilde H_j=\lambda_jI\), the blocks of \(T_0\) are
scalar multiples of the identity, so \(N\) is obtained exactly from an
\(H\times H\) temporal recurrence with no neural-network HVP and no random
probe. If deterministic residual bounds \(\delta_j\) are available, both the
Green gain and the profiled nonlinear coefficient are therefore reduced to
small dense linear algebra.

## 7. Relation to established validation machinery

Approximate inverses, inverse-defect bounds, and radii-polynomial validation
are classical; see, for example, the \(Y_0,Z_0,Z_1,Z_2\) framework in
Lessard and Matsue (2023, DOI 10.1007/s00332-023-09900-6). Finite-time
shadowing via computable right-inverse bounds also predates this work; see
Chow and Van Vleck (1994, DOI 10.1137/0915058), and componentwise shadowing
bounds appear in Van Vleck (2002, DOI 10.1137/S1064827599353452).

The optimizer-specific step here is the exact causal factorization

\[
\mathcal P K_J\mathcal B
=T_0(I-\mathcal D_\Delta\mathcal S T_0)^{-1},
\]

whose inverse is a finite nilpotent polynomial and remains entirely in the
scaled-momentum parameter-forcing channel. Its research value must be judged
by whether segmented low-rank approximations make \(\alpha\) small enough to
reduce measured HVP work while preserving neural first-passage certificates.
