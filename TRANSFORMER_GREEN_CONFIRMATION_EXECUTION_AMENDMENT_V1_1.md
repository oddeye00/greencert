# Transformer signed-Green confirmation: execution amendment v1.1

## Status and scope

This amendment is frozen **before any exact future trajectory is rolled out or
joined**.  It does not alter the candidate manifest, centerline equation,
number of recentering sweeps, signed radius, derivative envelope, randomized
probe family, failure budget, closure inequality, output margin, persistence
rule, or issuance rule in the original method seal.

The original outcome-blind certificate pass encountered three pre-outcome
execution failures:

1. seed 333, threshold 0.80, anchor 3080: transient Windows/OneDrive
   `PermissionError` while overwriting its nonce-bound cache;
2. seed 350, threshold 0.70, anchor 1440: the same transient cache-write
   failure; and
3. seed 335, threshold 0.70, anchor 2440: the second frozen recentering sweep
   truncated, so the required reference path does not exist.

The first and second failures are retried sequentially with the **unchanged
hash-sealed certificate executable**.  Retries are allowed only when the
process log ends in `PermissionError`, use the same deterministic nonce-bound
cache, and are capped at three attempts.  A retry cannot change a probe,
candidate, radius, or inequality.

The third failure is mapped conservatively to `certificate_issued = false`.
It remains in every denominator.  No replacement centerline, shorter horizon,
extra sweep, or tuned constant is permitted.  This amendment can therefore
only preserve or reduce issuance relative to a hypothetical successful
execution of the original construction.

## Information barrier

Before the amended certificate seal is written, the recovery program may read
only:

- the original method, candidate, and candidate-manifest seals;
- outcome-blind training/checkpoint artifacts;
- outcome-blind certificate caches and certificate files; and
- the three failed certificate process logs.

It must not read files ending in `.outcomes.json` or `.sealed.log`, and it must
reject any certificate containing a joined outcome.  The exact optimizer
rollout is allowed only after all 23 certificate-or-abstain records and their
hashes are sealed.

## Post-seal audit

The 22 normally constructed records use the original post-seal audit code.
The construction-abstention case is audited by an exact optimizer rollout from
the sealed checkpoint without reconstructing the failed centerline.  It has no
state tube, no Green bound, and no coverage claim.  It contributes only its raw
modal timing error and the fact of abstention.

All amendment files, original seals, failure-log hashes, and the final
certificate files are retained for traceability.
