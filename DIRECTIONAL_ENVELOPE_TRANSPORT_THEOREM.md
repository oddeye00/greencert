# Directionally transported neural-envelope theorem

## 1. Setting

Let the optimizer state be (x_j=(\theta_j,w_j)) in the scaled state norm,
and let a finite-window reference be corrected from (c_j) to

\[
\bar c_j=c_j+z_j .
\]

Write (P_\theta x=\theta).  The scaled Euclidean norm used by GREENCERT
satisfies (\|P_\theta x\|_2\leq\|x\|_2).  Suppose the remaining unknown
shadowing error obeys

\[
x_j^\star=\bar c_j+e_j,\qquad \|e_j\|_2\leq E\leq\rho .
\]

The usual implementation evaluates a neural derivative envelope afresh at
(\bar c_j).  The result below transports the inputs of that envelope from
(c_j) along the *known* signed displacement (z_j), then expands only the
unknown radius (\rho).

## 2. Directional transport of stage values

Let (h_s(\theta)) denote any intermediate network stage, measured in the
stage norm used by the compositional neural jet.  Suppose a directional jet
on the segment (\theta(t)=\theta+t a), (0\leq t\leq1), certifies

\[
\sup_{0\leq t\leq1}\left\|\frac{d}{dt}h_s(\theta(t))\right\|
\leq A_s .
\]

Then the fundamental theorem of calculus gives

\[
\boxed{\|h_s(\theta+a)\|\leq \|h_s(\theta)\|+A_s.}
\]

This is the same fixed-point stage inflation already computed by the mixed
three-known/one-free jet used for the directional fourth-order remainder.  No
new neural evaluation is required.

## 3. Directional transport of parameter geometry

Partition parameters into the disjoint analytic blocks used by the neural
jet.  If a matrix block changes from (W_b) to (W_b+A_b), then

\[
\boxed{\|W_b+A_b\|_2\leq\|W_b\|_2+\|A_b\|_F.}
\]

For a bias block the identical statement holds with Euclidean norms.  Hence
the block radii already computed for the mixed jet transform every parameter
operator/vector norm at (c_j) into a valid majorant at (c_j+z_j).  Only
the embedding block has no separate operator norm; its effect is already
included in the transported stage values.

## 4. Monotone envelope transport

The block neural envelope is a composition of nonnegative majorants.  It is
coordinatewise monotone in (i) its stage-value inputs, (ii) parameter
operator/vector norms, and (iii) the unknown ball radius.  Starting its
stage-value iteration from certified upper bounds rather than exact center
values therefore preserves every derivative inequality and its post-fixed
point argument.

**Lemma (transported corrected-center envelope).**  Let
(\widehat V_{s,j}\) be the stage-value majorants from Section 2 and
(\widehat G_{b,j}\) the geometry majorants from Section 3.  Running the
ordinary ball-valid block jet with inputs ((\widehat V_j,\widehat G_j)) and
radius (\rho) yields valid bounds for all network derivatives throughout

\[
B(P_\theta\bar c_j,\rho).
\]

**Proof.**  Sections 2 and 3 show that the supplied inputs dominate their
exact values at (P_\theta\bar c_j).  For any
(u\in B(P_\theta\bar c_j,\rho)), every parameter block changes by at most
(\rho), and every stage change is bounded by the same monotone fixed-point
inequality as in the ordinary ball envelope.  Coordinatewise monotonicity
then dominates the envelope initialized from the exact corrected-center
values.  A returned post-fixed point is therefore valid on the stated ball.
(\square)

## 5. Fused corrected-path certificate

Let (L_{1,j},L_{2,j},L_{3,j}) be the first three output-derivative bounds
returned by the transported envelope.  For cross-entropy with scaled momentum,
set

\[
M_j=\sqrt2\,\eta\left(2L_{1,j}^3+
\frac32L_{1,j}L_{2,j}+\sqrt2L_{3,j}\right),
\qquad M=\max_{j<H}M_j.
\]

Let (\widehat\kappa) bound the corrected-path causal Green operator and let
(B) bound its response to the corrected defect.  If

\[
B+\frac12\widehat\kappa M E^2\leq E,
\qquad E\leq\rho,
\]

then the moving-window shadowing theorem applies around (\bar c) exactly as
with a freshly evaluated corrected-center envelope.  In particular, the
smaller quadratic root is a valid state-tube radius.

For a true-versus-competitor logit margin at checkpoint (j), the uncertainty
charge is

\[
\boxed{\sqrt2\,L_{1,j}E,}
\]

not (\sqrt2L_{1,j}(\|z_j\|+E)).  The known signed correction has already
been incorporated into the evaluated center (\bar c_j); it is used only to
transport the derivative envelope.  Therefore every persistent first-passage
bracket obtained from these margins has the same deterministic certificate
semantics as the original corrected-center construction.

## 6. Computational consequence

At each checkpoint the mixed directional remainder and corrected-center
derivative envelope can share:

1. one exact stage-value evaluation at (c_j);
2. one collection of parameter spectral/vector norms at (c_j); and
3. one directional stage-inflation/block-radius calculation.

The corrected center still supplies the event logits, but it needs neither a
second full stage audit nor a second set of spectral decompositions.  The
theorem changes no random event, Green query, closure equation, or
first-passage logic.

## 7. Numerical boundary

The statements above are exact-real inequalities.  A computer-assisted proof
must outward-enclose the center values, block radii, parameter norms, mixed
jet, and final fixed point.  A float64 implementation is an identity and
performance audit of the construction, not by itself an outward-rounded proof.
