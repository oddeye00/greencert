# Frozen structured causal-row Transformer panel protocol

Status: frozen after development on seed 373, threshold 0.70, anchor 1280 and
before any structured-row probe was drawn for the other 14 panel rows.

## Question

Can the forcing-subspace causal row theorem replace full Green Gram products in
the complete 15-case released Transformer panel while retaining every sealed
bracket and reducing elementary forward/adjoint linearized sweeps below the
current best total of 144?

This is a post-release systems/theorem audit on already sealed events, not a
new coverage experiment.

## Frozen construction

- Cohort: all 15 rows in
  `results/transformer_v3_relinearized_prefix_panel_audit.json`.
- Development row: `(373, 0.70, 1280)`.
- Holdout rows: the other 14 coordinates.
- Reference: four anchor-fixed variational sweeps followed by one signed
  first-response correction. The corrected-path hash must match the released
  path for every row.
- Defect: signed directional quadratic forcing, signed recurrence residual,
  and the mixed fourth-order remainder from
  `transformer_mixed_directional_jet_v2.py`.
- Operator: the chronological parameter channel
  `P K_i B`, where `Bq=(-eta*q,eta*q)`.
- Closure: the forcing-subspace recursion in
  `CAUSAL_ROW_GREEN_THEOREM.md`, followed by unchanged output margins and
  persistent-event logic.

## Anytime probes and probability accounting

Each candidate has one predeclared eight-probe Gaussian stream. Stage 1 uses
rows 0--3. If it abstains, stage 2 evaluates rows 4--7 and recomputes every row
bound from the maximum over all eight probes. No transpose product is allowed.

The new Green family spends `1e-6` over 15 potential candidates and two
potential stages. Thus each candidate-stage row family receives

`stage_delta = 1e-6 / (15 * 2)`.

Each stage divides its budget equally over that candidate's chronological
rows. Union bounding both stages and all potential candidates gives total
failure probability at most `1e-6`, whether or not stage 2 is reached. Stage-2
selection depends only on stage-1 abstention, never on a future outcome.

For memory safety, disjoint four-probe blocks may execute in fresh processes.
The combiner must verify their path, stream, event, curvature, and forcing
identities before taking rowwise maxima. Process isolation changes measured
wall time, not the estimator or logical operator count.

## Cost accounting and promotion gate

A four-probe stop costs four forward probe responses plus one signed response:
five elementary linearized sweeps. An eight-probe stop costs eight forward
probe responses plus one signed response: nine sweeps. There are zero transpose
sweeps. The released prefix/Gram panel costs 144 sweeps by its own accounting.

Promotion requires:

1. 15/15 total and 14/14 holdout cases issue;
2. every issued bracket equals its sealed comparator;
3. every corrected-path hash matches;
4. every radius remains inside its derivative domain;
5. at most two candidates require the eight-probe stage;
6. total logical linearized sweeps are at most 83, all forward;
7. no future outcome file is read and all 15 denominators are reported;
8. all theorem, mixed-jet, block-combination, and batch-independence tests pass.

Every failure and runtime is reported. This protocol will not be revised after
holdout execution.
