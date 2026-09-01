# Directional block theorem source supersession

The frozen cohort diagnostic records the exact v1 theorem source hash:

- `DIRECTIONAL_BLOCK_REMAINDER_THEOREM.md`
- SHA-256 `F9C680DD7E6C47EFA8AE91753612464DF8622456A0F22179BF5174A6134B6AAF`

That file remains immutable. A hostile proof audit found that v1 used slot
symmetry in the identity

\[
 \mathcal P(r,r,r,t)=\tfrac14\nabla P(r)^\top t
\]

without explicitly requiring the polarized majorant itself to be symmetric.
The numerical implementation already uses sorted monomials and the symmetric
polarization, but the theorem statement should not ask the reader to infer
that convention.

The maintained statement is therefore:

- `DIRECTIONAL_BLOCK_REMAINDER_THEOREM_V2.md`
- SHA-256 `6F59C5F579BC3CE882E456DDC6FBE083633774F2D3B153ED54974F9C92F48CBE`

Version 2 proves that any valid nonnegative four-linear block majorant can be
averaged over all 24 slot permutations. Symmetry of \(D^4F\) makes the average
a valid majorant, and diagonal evaluation is unchanged. The gradient identity
then follows from the explicitly symmetric majorant. This is a statement and
proof clarification: no polynomial coefficient, closure, bracket, timing, or
promotion result changes.

The independent regression test
`scripts/test_directional_block_symmetrization.py` (SHA-256
`11476070F0BF25F9FD582BF8B953C7F337B4B534DD3CAA6118B3C7626089FDD4`)
checks 200 asymmetric nonnegative coefficient arrays, all 24 slot
permutations, diagonal preservation, the polarized gradient identity, Euler
homogeneity, and zero block radii.
