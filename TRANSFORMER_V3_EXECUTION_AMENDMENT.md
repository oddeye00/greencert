# Transformer v3 execution amendment

## Scope

This amendment records one outcome-blind execution failure discovered after
the Transformer v3 method and candidate seals, but before the certificate seal
or any exact future rollout.

The frozen selector retained seed 365, gate 70%, anchor 1480 because the
truncated four-sweep modal path contained a persistent predicted event at
offset 127.  The selected event horizon was 151.  During certificate
construction, the frozen centerline builder enforced its stronger invariant
that every recentering sweep reach the full 300-step protocol horizon.  Sweep
2 reached only step 244, so construction stopped with
`RuntimeError: recentring sweep 2 truncated` before output or Green probing.

## Frozen disposition

The candidate remains in the prespecified 19-candidate denominator.  It is
serialized as a deterministic, outcome-blind abstention with zero randomized
operator queries.  The amendment does not:

- change the selector, centerline, theorem, constants, or any sealed method
  file;
- shorten the required 300-step centerline;
- raise the numerical cap or retry with altered arithmetic;
- remove the candidate from the denominator; or
- inspect its future certification outcome before the abstention is hashed.

The failed process log, blind modal record, abstention record, amendment code,
and this note are hash-bound in the amendment seal.  After the ordinary
certificate seal is written, the candidate's future event is recomputed from
the anchor exactly like the other post-seal audits, but state-tube diagnostics
are marked unavailable because no valid frozen reference path exists.

## Prospective interpretation

This is an execution-level abstention, not an issued certificate and not a
post-hoc exclusion.  It exposes a selector/constructor invariant mismatch in
the frozen implementation.  Future protocols should require all prescribed
recentring sweeps to reach the full selection horizon before a candidate can
be sealed.  That correction is not retroactively applied to v3.
