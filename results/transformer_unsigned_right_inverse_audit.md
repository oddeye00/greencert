# Matched unsigned finite-window baseline audit

This post-seal audit replaces the signed Green response $\lVert K_Hs\rVert$ by the
standard norm-only upper bound $\kappa\lVert s\rVert$, while retaining the **same
finite-window right inverse**. It also gives the baseline the smallest valid
radii-polynomial root, so this is stronger than the common product-of-local-
norms comparison.

## Result

- Frozen candidates with a queried Green operator: **18**.
- Frozen signed certificates in that matched set: **9**.
- Strong unsigned certificates: **1**.
- Signed certificates destroyed solely by discarding the defect direction:
  **8/9**.
- Unsigned/signed zero-order inflation: median **319.70x**,
  range **83.02x--411.54x**.
- 17 cases fail closure even
  under the favourable smaller signed-ball derivative envelope.

The sole survivor of that favourable screen was seed
345 at threshold
0.80. Its own outer-ball closure statistic
is **0.816393** and its strongest unsigned
bracket is **[30, 30]**.

## Interpretation

The finite-window inverse itself is not enough. The gain comes from applying
that inverse to the **signed, time-ordered defect** before taking a norm. On
the main Transformer confirmation, this cancellation is the difference
between 9 issued certificates and 1
under an otherwise matched validated-dynamics construction.

This audit is post-seal and does not modify the frozen candidate, certificate,
or outcome artifacts. Cases where the frozen protocol never queried a Green
operator are excluded from the matched-operator denominator.
