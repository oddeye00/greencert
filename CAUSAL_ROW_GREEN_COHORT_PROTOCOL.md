# Frozen causal row-Green Transformer cohort protocol

Status: frozen before any cohort execution. The disclosed development case is
seed 366, threshold 0.80, anchor 1120. Every other case in the parent panel is
held out for this theorem audit.

## Scientific question

Can chronological rowwise Green closure replace one sequential variational
sweep while retaining already sealed Transformer event brackets that the
matched global scalar closure cannot certify?

## Fixed cohort and information boundary

- Parent panel: `results/transformer_fully_recentered_three_sweep_audit.json`.
- Candidate family: all 15 rows in that file.
- Development row: `(366, 0.80, 1120)` only.
- Holdout rows: the remaining 14 rows.
- The audit may read frozen candidate artifacts, checkpoints through the
  declared anchor, prior outcome-blind certificates, and the parent panel.
- It must not read future trajectories, outcome files, or revealed event
  locations. Every output row records `outcome_files_read = 0`.

## Frozen construction

For every candidate, without adaptation:

1. Build the anchor-fixed reference with exactly two variational sweeps.
2. Propagate one signed first-response correction.
3. Form the signed directional quadratic forcing and its mixed fourth-order
   remainder using `transformer_mixed_directional_jet_v2.py`.
4. Linearize the causal Green operator along the corrected path.
5. Draw four standard-Gaussian sequence probes from the sealed identity and
   append the signed quadratic forcing as one deterministic fifth batch row.
6. Obtain all chronological row norms from the Gaussian rows in this single
   direct-image Green pass.
7. Apply the causal row recursion in `CAUSAL_ROW_GREEN_THEOREM.md`, followed by
   the unchanged pointwise neural-output margins and persistent-event logic.

The fifth row is not random, does not enter the norm estimator, and causes no
additional JVP/VJP pass. A regression test must verify that appending it leaves
the Gaussian images unchanged to the declared float64 tolerance.

## Probability accounting

- Family failure budget: `1e-6` over the 15 declared candidate Green
  operators.
- Candidate budget: `1e-6 / 15`, whether or not a candidate issues.
- Within a candidate, the candidate budget is divided equally over its
  chronological rows.
- Probe count: four per operator.
- Probe seed namespace:
  `greencert/causal-row-development-v1/6cf1373b`, candidate coordinates, two
  sweeps, and four probes.
- The probability statement uses the same ideal-PRNG interpretation as the
  released Transformer experiment.

## Frozen validity and promotion gates

Before running the cohort:

- the causal row theorem test must pass all 480 randomized nonlinear systems;
- the mixed directional third/fourth derivative audit must pass;
- the batch-row independence regression must pass;
- every claim-relevant Python file must compile.

The method is promoted to the manuscript only if all of the following hold:

1. every issued row stays inside its predeclared neural-jet domain;
2. no issued bracket differs from its sealed four-sweep bracket;
3. at least three of the 14 held-out candidates issue and retain that bracket;
4. no outcome file is read;
5. all reported denominators include every one of the 15 declared cases.

Runtime, abstention, and failures are reported regardless of promotion. The
protocol will not be revised in response to cohort results.
