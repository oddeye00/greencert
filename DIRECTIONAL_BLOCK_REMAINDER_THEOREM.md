# Directional block remainder theorem

## Purpose

The fully recentered three-sweep audit is limited by the scalar replacement

\[
 \|D^4F\|\,\|z\|^3/6
\]

for the third-order remainder of the objective gradient.  Three of the four
directions in this contraction are not unknown: all three equal the realized
variational correction `z`.  This note retains those directions and takes a
worst case only over the remaining dual direction.

## Block-majorant definition

Let the Euclidean parameter space be an orthogonal direct sum
\(\Theta=\bigoplus_{b=1}^B\Theta_b\), with projections \(\pi_b\).  On a set
\(S\), suppose the symmetric fourth derivative of \(F\) has a nonnegative
polarized block majorant \(c_{b_1b_2b_3b_4}\): for every \(x\in S\) and every
\(h_1,\ldots,h_4\),

\[
 |D^4F(x)[h_1,h_2,h_3,h_4]|
 \le
 \sum_{b_1,\ldots,b_4}
 c_{b_1b_2b_3b_4}
 \prod_{i=1}^4\|\pi_{b_i}h_i\|_2.
\]

The associated homogeneous polynomial is

\[
 P_4(s)=\sum_{b_1,\ldots,b_4}
 c_{b_1b_2b_3b_4}\prod_{i=1}^4s_{b_i}.
\]

The block-jet rules used here preserve this polarized property
coefficientwise under sums, products, affine parameter maps, and smooth scalar
composition.  The constants are the same deterministic softmax, GELU, and
cross-entropy derivative constants used by the scalar fourth-order envelope.

## Theorem 1: three-known, one-free contraction

Let \(r_b=\|\pi_bz\|_2\).  Under the block-majorant assumption,

\[
 \sup_{x\in S}\|D^4F(x)[z,z,z,\cdot]\|_{2\to\mathbb R}
 \le \frac14\|\nabla P_4(r)\|_2.
\]

### Proof

For a unit dual direction \(u\), put \(t_b=\|\pi_bu\|_2\), so
\(t_b\ge0\) and \(\|t\|_2=1\).  Applying the polarized majorant with
\((h_1,h_2,h_3,h_4)=(z,z,z,u)\), then using symmetry of the coefficient
array, gives

\[
 |D^4F(x)[z,z,z,u]|
 \le \frac14\nabla P_4(r)^\top t
 \le \frac14\|\nabla P_4(r)\|_2.
\]

The first equality follows because differentiating the diagonal fourth form
inserts the free block direction in each of four symmetric slots.  Taking the
supremum over unit `u` proves the result.  ∎

## Corollary 1: directional gradient Taylor remainder

Let the segment \([\theta,\theta+z]\) lie in `S`.  Then

\[
\begin{aligned}
 \nabla F(\theta+z)
 &=\nabla F(\theta)+\nabla^2F(\theta)z
   +\tfrac12D^3F(\theta)[z,z]+R_3,\\
 \|R_3\|_2
 &\le \frac1{24}\|\nabla P_4(r)\|_2.
\end{aligned}
\]

### Proof

The integral Taylor remainder is

\[
 R_3=\frac12\int_0^1(1-t)^2
 D^4F(\theta+tz)[z,z,z,\cdot]\,dt.
\]

Theorem 1 and \(\frac12\int_0^1(1-t)^2dt=1/6\) give the claim.  ∎

For the scaled momentum state used by GREENCERT, the corresponding local
forcing contribution is at most

\[
 \sqrt2\,\eta\,\|\nabla P_4(r)\|_2/24.
\]

## Segment-valid neural envelope

For each realized correction `z`, the implementation bounds the entire segment
\(\theta+tz\), not a Euclidean ball that permits unrelated block directions.
Parameter operator norms are inflated by their realized block radii.  Stage
values satisfy the monotone simultaneous inequalities

\[
 V_k\le V_k(\theta)+J_{k,1}(V,r),
\]

where `J_{k,1}` is the nonnegative first-order block polynomial evaluated at
`r`.  Iteration from zero inflation is monotone.  A returned assignment is used
only after a final self-consistency check.  The fourth-order polynomial is then
built once at that assignment.

## Scope

This theorem changes only the deterministic Taylor enclosure.  It does not
alter the Green calibration, familywise probability budget, centerline,
event logic, or future-outcome firewall.  It is currently implemented for the
one-block normalization-free Transformer covered by the existing analytic
jets.  A cohort result is not claimed until the frozen diagnostic in
`DIRECTIONAL_BLOCK_REMAINDER_PROTOCOL.md` passes.
