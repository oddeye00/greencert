# Cohort-wide corrected-path prefix result

The frozen outcome-blind post-seal panel completed on all 15 pre-existing
Green-evaluable Transformer records.

## Primary result

```text
issued / evaluated                    15 / 15
same bracket as directional baseline  15 / 15
prefix 4 / 8 / 16                     14 / 1 / 0
old / new Green Gram applications     560 / 64
aggregate Green reduction             8.75x
median pairwise Green reduction        8.00x
pairwise range                         4.00x--16.00x
old / new theoretical linear sweeps   1150 / 144
aggregate theoretical sweep reduction 7.986x
minimum forcing headroom               2.293x
minimum strict output logic slack      1.4466e-6
combined output+Green failure upper    2e-6
outcome files read                     0
```

Fourteen cases certified with one batched four-probe Gram call.  One case used
the eight-probe prefix.  That case computed the cost-aware direct forcing
fallback after its failed four-probe attempt, but the eventual eight-probe
certificate used the valid norm-only release; therefore the panel does not
claim an issuance gain from the fallback.

The 15 horizons range from 26 to 299 steps.  Immutable directional baselines
used 16 probes and earliest powers one, two, or four.  Every returned bracket
is identical to its baseline bracket.

## Scope

This is strong cohort-wide mechanism and operator-count evidence, not a new
prospective training cohort.  Output-Jacobian enclosures are inherited from the
original v3 records.  Neural products and margins remain float64 under the
ideal-PRNG model; the combined high-confidence failure upper is `2e-6`, not an
exact-real outward proof.

Artifacts:

- `results/transformer_v3_relinearized_prefix_panel_audit.json`
- `results/transformer_v3_relinearized_prefix_panel_independent_audit.json`
- `scripts/audit_transformer_relinearized_prefix_panel.py`
- `scripts/audit_transformer_relinearized_prefix_panel_result.py`
- `RELINEARIZED_PREFIX_PANEL_PROTOCOL.md`
- `RELINEARIZED_PREFIX_PANEL_EXECUTION_DEVIATIONS.md`
