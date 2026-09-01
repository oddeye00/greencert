# Outcome-blind adaptive-sweep cohort protocol

Frozen on 2026-08-31 before any cohort-level reduced-sweep computation.

## Fixed cohort and evidence boundary

The cohort is exactly the 15 Green-evaluable rows in
`results/transformer_v3_relinearized_prefix_panel_audit.json`, whose SHA-256 is
`08E501B51FEAC3D96FFE02BE0B5D84E0E682C2E73CB906C083ED0FEF7E75E12B`.
The sweep-minimization idea was developed on `(366, 0.8, 1120)`; that row is
therefore reported separately from the other 14 rows.  All computations are
post-release and outcome-blind.  Revealed trajectories and event times are
forbidden inputs.

## Fixed method family

For every cohort member, construct the one-, two-, and three-sweep references
in a single causal pipeline.  For each reference:

1. append one signed variational correction;
2. reject deterministically if the correction leaves the row's pre-existing
   derivative domain;
3. otherwise form the signed quadratic defect and fourth-jet Taylor enclosure;
4. rebuild the corrected-path Green operator;
5. evaluate the same analytic neural-jet closure and persistent-event logic;
6. compare the issued bracket with the row's sealed four-sweep bracket.

The parent four-sweep records are immutable comparators.  Constants, horizons,
domains, thresholds, and persistence are inherited without modification.

## Randomized operator family

- Declared operators: `15 candidates x 3 sweep counts = 45`.
- Probes per operator: `4` independent standard Gaussian vectors.
- Total new Green-family failure upper: `1e-6`.
- Per-operator stage failure: `1e-6 / 45`.
- Fresh master nonce:
  `3631e2479441793c5bc31596f3697f5eab60bec95e53984c5aadbffaf0aa4460`.
- Probe streams are SHA-256-domain-separated by the full candidate identity,
  horizon, and sweep count.  Every identity is registered before any query.

An operator that fails a deterministic pre-query screen consumes no random
vectors, but remains charged in the declared union bound.

## Fixed reporting

Report, for each sweep count and separately for the 14 non-development rows:

- deterministic domain passes;
- closures and certificates issued;
- exact retention of the sealed bracket;
- strict logic slack;
- HVP-equivalent sequential-vector sweeps and batched operator passes;
- wall-clock phase timings; and
- the smallest successful sweep count, using four as the fallback.

A universal three-sweep replacement requires exact bracket retention on all 15
rows.  Otherwise, the result may support an a-posteriori adaptive stopping rule
only for rows whose reduced-sweep closure independently passes.  No failed or
different-bracket row may be counted as a saving.
