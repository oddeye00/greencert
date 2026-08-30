# Response-centered event transport

## Statement

Suppose the response-centered Green theorem gives a known signed response
\(\widetilde z_j\) and an unknown remainder enclosure

\[
\|x_{a+j}-(c_j+\widetilde z_j)\|\le E.
\]

Let \(f\) denote the vector of logits. Assume that on the containing ball

\[
\|Df(c_j)\|\le B_{1j},\qquad \|D^2f\|\le B_{2j},
\]

and write \(d_j=\|\Pi_\theta\widetilde z_j\|\). For any
true-class-minus-competitor
margin \(m_{iq}\),

\[
\boxed{
|m_{iq}(x_{a+j})-m_{iq}(c_j+\widetilde z_j)|
\le
\sqrt 2\left[(B_{1j}+B_{2j}d_j)E+\frac12B_{2j}E^2\right].}
\]

The corrected-center logits require one deterministic forward evaluation and
no new Green, Gram, HVP, VJP, or randomized query.

## Proof

The Hessian bound and the fundamental theorem of calculus give

\[
\|Df(c_j+\widetilde z_j)\|\le B_{1j}+B_{2j}d_j.
\]

Taylor-expand from the corrected center to the realized state. The linear
term is bounded by \((B_{1j}+B_{2j}d_j)E\), the second-order remainder by
\(B_{2j}E^2/2\), and a difference of two output coordinates has Euclidean
operator norm \(\sqrt2\). This proves the claim.

For comparison, the origin-centered transport charges

\[
\Delta_j=\sqrt2\left[B_{1j}(p+E)+\frac12B_{2j}(p+E)^2\right],
\qquad p=\max_j d_j.
\]

Since \(d_j\le p\), subtracting the response-centered radius with \(d_j\)
replaced by \(p\) leaves

\[
\sqrt2\left(B_{1j}p+\frac12B_{2j}p^2\right)\ge0.
\]

Thus the new uncertainty radius is never larger. Intersecting its sign proofs
with the original valid enclosure makes the executable event rule
proof-preserving even when the two Taylor intervals have different centers.

## Outcome-blind sealed-record audit

`scripts/audit_transformer_v3_output_recentering.py` reconstructs the 19
sealed response-centered candidates without reading any outcome file. Fifteen
have a stored Green trace. The proof-preserving hybrid retains all 11 issued
certificates at exactly the same powers and brackets. At issuance, the maximum
output-radius ratio is 0.091--0.602 (median 0.365), i.e. a 39.8--90.9% reduction
with median 63.5%. No abstention converts and no power decreases on this
cohort. The audit adds no randomized query; its reconstruction performs 2,951
corrected-center forward evaluations because large centerline tensors are
regenerated rather than archived.

An independent arithmetic and hash audit is recorded in
`results/transformer_v3_output_recentering_independent_audit.json`.
