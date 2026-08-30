# Projected HVP variational certificates

## Exact matrix-free local clock

For gradient descent anchored at `theta[0]`, write `g[0]=grad F(theta[0])`
and let `H[0]` denote the anchor Hessian only as a linear operator.  The
full frozen-quadratic displacement is

\[
d_0=0,\qquad
d_{j+1}=d_j-\eta\{g_0+H_0d_j\}.
\]

This recurrence is algebraically identical to propagating every eigenmode of
the dense Hessian.  It needs one HVP per step, linear state memory, and no
eigendecomposition.  The active subspace below is therefore used only to
factor the *uncertainty*; it never truncates the predictive clock.

## Iterated signed recentering

Let `c^(k)` be any anchor-fixed reference, with

\[
s_j^{(k)}=G(c_j^{(k)})-c_{j+1}^{(k)}.
\]

Define the signed variational correction and next reference by

\[
z_0^{(k)}=0,\qquad
z_{j+1}^{(k)}=DG(c_j^{(k)})z_j^{(k)}+s_j^{(k)},
\qquad c_j^{(k+1)}=c_j^{(k)}+z_j^{(k)}.
\]

Then the new defect obeys the exact identity

\[
\boxed{
s_j^{(k+1)}
=G(c_j^{(k)}+z_j^{(k)})-G(c_j^{(k)})
-DG(c_j^{(k)})z_j^{(k)}.}
\]

Consequently, if `DG` is `M`-Lipschitz on the correction segment,

\[
\|s_j^{(k+1)}\|_2\le \tfrac12M\|z_j^{(k)}\|_2^2.
\]

If the finite-window variational gain satisfies
`max_j ||z_j^(k)|| <= kappa_H max_j ||s_j^(k)||`, then

\[
\boxed{
\max_j\|s_j^{(k+1)}\|_2
\le \tfrac12M\kappa_H^2
\left(\max_j\|s_j^{(k)}\|_2\right)^2.}
\]

Thus every prespecified successful sweep squares the known path-defect scale;
the theorem does not require stopping after one sweep.  The larger-model
protocol fixes two sweeps before its prospective seeds.

## Fixed-subspace theorem

Let `x[j+1] = G(x[j])`, let `c[0],...,c[H]` be any computable reference with
`c[0]=x[0]`, and put

\[
s_j=G(c_j)-c_{j+1},\qquad J_j=DG(c_j).
\]

Fix an orthonormal matrix `U` and the orthogonal projectors
`P=UU^T`, `Q=I-P`.  For the reference error `h_j=x_j-c_j`, write

\[
a_j\ge\|Ph_j\|_2,\qquad b_j\ge\|Qh_j\|_2,
\qquad \rho_j=(a_j^2+b_j^2)^{1/2}.
\]

Assume the four Jacobian blocks satisfy

\[
\begin{aligned}
\|P J_j P\|&\le A_j,&\quad \|P J_j Q\|&\le B_j,\\
\|Q J_j P\|&\le C_j,&\quad \|Q J_j Q\|&\le D_j,
\end{aligned}
\]

and on the ball of radius `rho` around `c_j`,

\[
\|DG(c_j+u)-J_j\|_2\le M_j(\rho)\|u\|_2.
\]

Set `a_0=b_0=0`, decompose the known defect as

\[
p_j=\|Ps_j\|_2,\qquad q_j=\|Qs_j\|_2,
\]

and propagate

\[
\boxed{
\begin{aligned}
a_{j+1}&=A_ja_j+B_jb_j+p_j+\nu_j,\\
b_{j+1}&=C_ja_j+D_jb_j+q_j+\nu_j,\\
\nu_j&=\tfrac12M_j(\rho_j)\rho_j^2.
\end{aligned}}
\]

Then, while the displayed bounds remain valid,

\[
\boxed{\|P(x_j-c_j)\|_2\le a_j,\qquad
       \|Q(x_j-c_j)\|_2\le b_j.}
\]

The proof is a direct projected induction from

\[
h_{j+1}=J_jh_j+s_j+
\bigl(G(c_j+h_j)-G(c_j)-J_jh_j\bigr).
\]

For full-batch gradient descent `J_j=I-eta H_j` is symmetric, so `B_j=C_j`.
All active and cross blocks can be computed from the block HVP `J_j U`.  Only
the complement block needs a separate enclosure.

## Target-projected output theorem

For a scalar output `m` with gradient `g_j=nabla m(c_j)` and Hessian norm at
most `K_j(rho_j)` on the certified ball,

\[
\boxed{
|m(x_j)-m(c_j)|
\le \|U^Tg_j\|_2a_j+\|Qg_j\|_2b_j
+\tfrac12K_j(\rho_j)\rho_j^2.}
\]

This replaces a spherical first-derivative charge by the actual alignment of
the event margin with the active and complementary errors.

## Matrix-free probabilistic complement enclosure

Let `A=QJQ` be fixed and symmetric.  Draw `m` independent standard Gaussian
vectors and project them into `range(Q)`.  For integer `q>=1`, let

\[
Y=\max_{1\le i\le m}\|A^qQg_i\|_2.
\]

If

\[
c=\Phi^{-1}\!\left(\frac{1+\delta^{1/m}}2\right),
\]

then

\[
\boxed{\|QJQ\|_2\le(Y/c)^{1/q}}
\]

with probability at least `1-delta`.  This follows because for a top unit
eigenvector `v`, `||A^q Qg_i|| >= ||A||^q |v^Tg_i|`, and the maximum of the
folded-normal coordinates exceeds `c` with probability `1-delta`.

For gradient descent with coupled weight decay `lambda`, probe the data
curvature `Q(H-lambda I)Q` instead of the near-identity Jacobian.  If its norm
is at most `sigma`, then

\[
\|QJQ\|_2\le |1-\eta\lambda|+\eta\sigma.
\]

Independent probe-point failures are combined by a union bound.  Between
probe points, a deterministic Hessian-Lipschitz envelope transfers the block
bounds.  The resulting certificate is matrix-free and carries an explicit
total failure probability; it is not silently treated as deterministic.

## Persistent first-passage corollary

Suppose the state/output certificate gives integer paths

\[
n_j^-\le n_j\le n_j^+,
\]

where `n_j` is the actual number of correct examples.  For a required count
`r` and persistence length `K`, define

\[
T_K=\min\{j:\min_{0\le \ell<K}n_{j+\ell}\ge r\}.
\]

Let

\[
L_K=\min\{j:\min_{0\le\ell<K}n_{j+\ell}^+\ge r\},\qquad
U_K=\min\{j:\min_{0\le\ell<K}n_{j+\ell}^-\ge r\}.
\]

Whenever `U_K` exists and `L_K <= U_K`,

\[
\boxed{L_K\le T_K\le U_K.}
\]

Indeed, every start before `L_K` contains a step whose upper count is below
`r`, while every step in the block beginning at `U_K` is guaranteed above
`r`.  This rules out the parity-flip artifact exposed by the first
larger-model development run without assuming monotone accuracy.
