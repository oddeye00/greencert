# Indexing note for the sealed v1 structured-parameter theorem

The file `STRUCTURED_PARAMETER_GREEN_THEOREM.md` is hash-sealed by
`STRUCTURED_PARAMETER_GREEN_AUDIT_PROTOCOL_V2.md` and is therefore preserved
byte-for-byte.

Its norm bound and the completed 15-case audit remain valid, but the displayed
fixed-point equation suppresses a causal index shift.  With the implemented
Green convention, (K_H f=(e_1,\ldots,e_H)), while the nonlinear remainder at
update (j) depends on (e_j).  The explicit equation is

\[
p=P K_Hs+P K_H\mathcal B R(\mathcal Sp),
\qquad
\mathcal S(p_1,\ldots,p_H)=(0,p_1,\ldots,p_{H-1}).
\]

Because \(\lVert\mathcal S\rVert_{2\to2}=1\), every scalar inequality used in
the v1 theorem and audit is unchanged.  In fact, the anchor-fixed first
remainder block is identically zero, so restricting the structured operator to
that forcing subspace can only tighten the gain.  The corrected and stronger
statement is `STRUCTURED_PARAMETER_GREEN_THEOREM_V2.md`; it must be used in any
manuscript integration.
