# Forcing-subspace Green closure with local curvature profiles

Status: post-v1.0.1 theorem development.  This extension was written after the
sealed v2 structured-parameter audit and therefore does not alter that audit or
any prospective count in the released paper.

The scalar-curvature theorem in `STRUCTURED_PARAMETER_GREEN_THEOREM.md` is a
special case.  The two additions here are (i) arbitrary structured plus
unstructured path defect and (ii) a time-local curvature profile inside the
Green operator rather than a worst-window maximum outside it.

## Setting

For a horizon (H), let (K_H) be the anchor-fixed causal Green operator of
the Jacobian sequence (J_0,ldots,J_{H-1}).  Let (P) project a state onto
the parameters that determine both the nonlinear remainder and the event.  At
each step suppose

\[
G_j(b_j+u)-G_j(b_j)-J_j u=B_jR_j(Pu).
\]

Write \(\mathcal B=\operatorname{diag}(B_0,\ldots,B_{H-1})\).  The Green
output is indexed as

\[
p=(Pe_1,\ldots,Pe_H),
\]

whereas the nonlinear forcing at update (j) depends on (Pe_j).  Define the
anchor-fixed causal shift

\[
\mathcal S(p_1,\ldots,p_H)=(0,p_1,\ldots,p_{H-1})
\]

and let \(Q_0\) inject an \((H-1)\)-block forcing sequence into an (H)-block
sequence by prepending zero.  Thus \(\lVert\mathcal S\rVert=\lVert Q_0\rVert=1\)
and the first nonlinear forcing is exactly zero, not merely small.  Let

\[
\mathcal D_L=\operatorname{diag}(L_0I,\ldots,L_{H-1}I),
\qquad
T_{L,0}=P K_H\mathcal B\mathcal D_LQ_0.
\]

For scaled momentum, (B_j=B) with

\[
Bq=(-\eta q,\eta q).
\]

## Theorem (profiled forcing-subspace closure)

Let the reference defect be

\[
s_j=G_j(b_j)-b_{j+1},
\]

and allow any decomposition

\[
s=\mathcal B r+d,
\]

where (d) is an arbitrary full-state residual.  Assume that the declared
parameter domains satisfy

\[
\lVert R_j(v)\rVert\le {L_j\over2}\lVert v\rVert^2,
\qquad L_j\ge0.
\]

Define

\[
a=P K_Hd+P K_H\mathcal B r=P K_Hs.
\]

Suppose certified bounds satisfy

\[
Y\ge\lVert a\rVert_2,
\qquad
\kappa_{L,0}\ge\lVert T_{L,0}\rVert_{2\to2}.
\]

If

\[
D=1-2\kappa_{L,0}Y\ge0,
\qquad
E={2Y\over1+\sqrt D},
\]

and every pointwise parameter ball of radius (E) remains in its declared
domain, then the realized trajectory obeys

\[
\left(\sum_{j=1}^{H}\lVert P(x_{a+j}-b_j)\rVert^2\right)^{1/2}\le E.
\]

The bound may be evaluated without forming (P K_Hs).  For example,

\[
Y\ge \lVert P K_H\mathcal B r\rVert_2+
       \lVert P K_Hd\rVert_2
 =\lVert T r\rVert_2+\lVert P K_Hd\rVert_2
\]

is valid, where (T=P K_H\mathcal B).  This is useful when (r) is the
known signed nonlinear forcing and (d) is a small numerical recurrence
residual.

### Proof

For (L_j>0), set

\[
\widehat R_j(v)=R_j(v)/L_j.
\]

If (L_j=0), the assumed bound forces (R_j=0) on the declared domain, and
set \(\widehat R_j=0\).  Then

\[
R(\mathcal Sp)=\mathcal D_LQ_0\widehat R_+(\mathcal Sp),
\]

where \(\widehat R_+\) retains updates (1,\ldots,H-1); the update-zero
remainder is exactly zero because the anchor error is zero.  Then

\[
\begin{aligned}
\lVert\widehat R_+(\mathcal Sp)\rVert_2
&\le {1\over2}
  \left(\sum_{j=1}^{H-1}\lVert p_j\rVert^4\right)^{1/2}\\
&\le {1\over2}\sum_j\lVert p_j\rVert^2
 ={1\over2}\lVert p\rVert_2^2.
\end{aligned}
\]

The parameter error satisfies the closed finite-dimensional equation

\[
p=a+T_{L,0}\widehat R_+(\mathcal Sp).
\]

On the radius-(E) ball, its right-hand side has norm at most

\[
Y+{1\over2}\kappa_{L,0}E^2.
\]

The displayed root is exactly the smaller nonnegative solution of

\[
Y+{1\over2}\kappa_{L,0}E^2\le E.
\]

Continuity and Brouwer's theorem therefore provide a fixed point in the ball.
Lifting it by

\[
e=K_H\{s+\mathcal B\mathcal D_LQ_0
       \widehat R_+(\mathcal Sp)\}
\]

solves the full causal error recurrence and has (Pe=p).  The deterministic
forward recurrence from the realized anchor has only one trajectory, so this
lift is its error sequence.

## Exact dominance

The profiled gain is never larger than the scalar-window coefficient:

\[
\lVert T_{L,0}\rVert
\le \lVert T\rVert\max_{1\le j<H}L_j
\le \lVert P\rVert\,\lVert K_H\rVert\,
    \lVert\mathcal B\rVert\max_{1\le j<H}L_j.
\]

Thus replacing \\(\lVert T\rVert\max_jL_j\\) by
\\(\lVert T_{L,0}\rVert\\) cannot worsen the exact quadratic closure.  It is
strictly better whenever high-curvature forcing directions are attenuated by
the causal parameter response.  As in the v1 theorem, independently randomized
upper bounds do not inherit samplewise dominance; adaptive route selection
must use a predeclared family-wise probability budget.

## Inexact numerical interfaces

The theorem needs bounds, not exact arithmetic.  If a computed response
\(\widetilde a\) has certified error \(\delta_a\), use

\[
Y=\lVert\widetilde a\rVert_2+\delta_a.
\]

If a numerical norm routine returns \(\widetilde\kappa_{L,0}\) with certified
operator error \(\delta_T\), use

\[
\kappa_{L,0}=\widetilde\kappa_{L,0}+\delta_T.
\]

This keeps the structural nonlinear coefficient even when an approximate
variational correction leaves an unstructured recurrence residual.  Only its
response contribution enters (Y); it does not revert the nonlinear gain to
the full optimizer-state norm.

## Relation to established validation machinery

The self-map/radii-polynomial step is an instance of classical
Newton--Kantorovich and radii-polynomial validation, and product-space or
componentwise enclosures are established tools in computer-assisted analysis.
The optimizer-specific content is the exact forcing factorization, the closed
parameter-sequence equation for the anchored optimizer orbit, and its direct
transport to a persistent neural-output event.  Novelty claims should be made
at that object-and-construction level, not for the quadratic root itself.
