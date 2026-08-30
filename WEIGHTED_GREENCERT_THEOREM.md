# Diagonally time-scaled response-centered Green theorem

## 1. Statement

Let the anchor-fixed trajectory error satisfy

\[
h=K_Hs+K_HN(h),
\]

and let \(\widetilde z\) be a computed signed response.  Choose positive state
weights \(w_1,\ldots,w_H\) and injection weights
\(v_0,\ldots,v_{H-1}\), and define

\[
\|x\|_{X,w}^2=\sum_{j=1}^H w_j^2\|x_j\|_2^2,
\qquad
\|u\|_{U,v}^2=\sum_{j=0}^{H-1}v_j^2\|u_j\|_2^2.
\]

The corresponding causal gain is

\[
\kappa_{w,v}\ge
\|D_wK_HD_v^{-1}\|_2,
\]

which is queried by exactly the same matrix-free Green products as the
unweighted operator, with only diagonal scaling before and after each call.

Assume

\[
\|K_Hs-\widetilde z\|_{X,w}\le\alpha,
\]

write \(p_j=\|\widetilde z_j\|_2\), and suppose that for
\(j=1,\ldots,H-1\),

\[
\|DG_j(c_j+u)-J_j\|_2\le M_j\|u\|_2
\]

on the required domain.  Define

\[
\begin{aligned}
Q_{w,v}
 &=\frac12\left(\sum_{j=1}^{H-1}
       v_j^2M_j^2p_j^4\right)^{1/2},\\
A_{w,v}
 &=\max_{1\le j<H}\frac{v_jM_jp_j}{w_j},\\
B_{w,v}
 &=\max_{1\le j<H}\frac{v_jM_j}{w_j^2}.
\end{aligned}
\]

Empty sums and maxima are zero.  If some \(E\ge0\) obeys

\[
\boxed{
\alpha+\kappa_{w,v}Q_{w,v}
+\kappa_{w,v}A_{w,v}E
+\frac12\kappa_{w,v}B_{w,v}E^2\le E,
}
\tag{1}
\]

and

\[
\boxed{p_j+E/w_j\le\rho_j\quad(1\le j\le H),}
\tag{2}
\]

then the realized trajectory is enclosed checkpointwise by

\[
\boxed{
\|x_j-c_j-\widetilde z_j\|_2\le E/w_j,
\qquad
\|x_j-c_j\|_2\le p_j+E/w_j.}
\tag{3}
\]

Consequently every output margin may use its own physical radius
\(p_j+E/w_j\), rather than a common worst-case radius.

## 2. Proof

Put \(e=h-\widetilde z\).  The response error
\(a=K_Hs-\widetilde z\) satisfies \(\|a\|_{X,w}\le\alpha\), and

\[
e=a+K_HN(\widetilde z+e).
\]

For \(\|e\|_{X,w}\le E\), each component obeys
\(\|e_j\|\le E/w_j\).  Taylor's theorem gives

\[
\|N_j(\widetilde z_j)\|
 \le \tfrac12M_jp_j^2
\]

and

\[
\|N_j(\widetilde z_j+e_j)-N_j(\widetilde z_j)\|
 \le M_jp_j\|e_j\|+\tfrac12M_j\|e_j\|^2.
\]

The first terms have weighted injection norm at most \(Q_{w,v}\).  The
linear multiplication map from \((w_j e_j)_j\) to
\((v_jM_jp_je_j)_j\) has norm \(A_{w,v}\).  For the quadratic terms,

\[
\left\|\left(
 \frac{v_jM_j}{w_j^2}(w_j\|e_j\|)^2
\right)_j\right\|_2
\le B_{w,v}\|e\|_{X,w}^2,
\]

because \(\|(y_j^2)_j\|_2\le\|y\|_2^2\).  Therefore

\[
\|N(\widetilde z+e)\|_{U,v}
\le Q_{w,v}+A_{w,v}E+\tfrac12B_{w,v}E^2.
\]

Applying the weighted Green gain gives exactly the right side of (1).
Thus the causal fixed-point map preserves the weighted radius-\(E\) ball.
Brouwer supplies a fixed point; causal forward uniqueness identifies it with
the realized orbit.  Equation (2) validates every derivative domain, and
\(\|e_j\|\le E/w_j\) proves (3).

## 3. Relation to the existing theorem

Setting every \(w_j=v_j=1\) recovers the time-resolved response-centered
closure in `HETEROGENEOUS_RECENTER_THEOREM.md`.  Multiplying all state and
injection weights by the same positive constant leaves every physical radius
unchanged, so implementations should normalize the metric and cap its
condition ratio.

The metric may be selected from deterministic centerline geometry or from any
history available before the Green block is drawn.  The predictable-family
theorem then applies to the scaled operator.  Selecting a metric after reading
the same block's probe values is invalid unless the candidate metrics receive
an explicit simultaneous failure allocation.

## 4. Executable audit and decision

The implementation is in:

- `scripts/weighted_recenter_closure.py`;
- `scripts/weighted_green_operator.py`;
- `scripts/test_weighted_recenter_closure.py`; and
- `scripts/test_weighted_green_operator.py`.

The tests cover 1,000 exact unweighted-equivalence cases, 900 certified
nonlinear trajectories, global scaling invariance, and 720 explicit-matrix and
adjoint checks.

On the immutable horizon-26 Transformer case, a metric chosen from the signed
response, drift bounds, and a scalar directional-gain proxy was validated with
a fresh independent Green block.  It retained the same `[2,2]` bracket and
reduced the maximum time-resolved radius from
\(1.8262\times10^{-15}\) to \(1.6737\times10^{-15}\), but required power two
instead of the unweighted power one.  The branch therefore **does not establish
a practical speedup** and is not promoted into the headline method.  Its value
is a proved preconditioning interface for longer or more heterogeneous windows;
a future frozen protocol must show lower operator work, not merely a smaller
radius.

Audit record:
`results/transformer_v3_weighted_green_postseal_audit.json`.
