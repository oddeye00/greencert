# Corrected-path prefix-panel execution deviations

This ledger is additive.  It does not modify the frozen protocol whose SHA-256
is `6740E5D32B5A5841E81AE0B25F17FDE316322CA4B279F9A2A1EB8F0C55BE1358`.

## Burned v1 prefix-counter attempt

The first v1 worker instantiated and applied two vectors, then stopped before a
prefix statistic because the pending vectors were double-counted.  It produced
no bound, closure, bracket, cache, or result.  Its nonce was burned.  The exact
repair, fresh v2 nonce, and superseding source hashes were frozen in amendment 1
of the protocol before any v2 vector was generated.

## Conservative v2 response-release short circuit

For `(seed 373, threshold .7, anchor 1280)`, the four-probe norm-only attempt
failed and triggered the predeclared direct forcing response.  At prefix eight,
the implementation evaluated the norm-only branch first; because that branch
already issued `[247,247]`, it stopped without re-evaluating the cached direct
response under the new `kappa`.

This differs from the protocol sentence saying that later prefixes use the
tighter of both releases.  It cannot invalidate or improve the reported
certificate: the stored norm-only upper bound is itself theorem-valid and is
slightly *larger* than the latent response-aware bound.  Independent arithmetic
gives

```text
stored norm-only response upper = 8.959740869759852e-18
latent response-aware upper     = 8.959740770707588e-18
latent / stored                  = 0.9999999889447401
```

Thus the deviation is conservative by about `1.11e-8` relative and changes no
issuance count, prefix, or bracket.  The paper must describe the executed rule:
try norm-only at each prefix; compute the directional fallback after a failed
norm-only attempt; and stop immediately if a later norm-only attempt issues.

## Independent audit

The independent auditor regenerated all 64 ideal-PRNG vectors and recomputed
every prefix calibration, forcing release, closure root, domain check, and
headroom.  It also verified exact cache/result equality and all immutable
artifact hashes.  Result SHA-256:
`08E501B51FEAC3D96FFE02BE0B5D84E0E682C2E73CB906C083ED0FEF7E75E12B`.

No future outcome file was read by either execution or audit.
