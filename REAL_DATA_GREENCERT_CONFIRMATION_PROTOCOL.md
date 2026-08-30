# Frozen non-modular GreenCert confirmation protocol

## Status

This protocol is to be SHA-256 sealed, together with every claim-bearing source
file and the dataset bytes, before training any fresh seed in the range
101--124. Development used only seeds 0--7. No fresh seed may be substituted,
and no threshold, anchor, sweep, radius, probe, or issuance rule may be changed
after the method seal.

## Scientific question

Does the signed finite-window certificate transfer beyond modular arithmetic
when candidate selection is determined exclusively by training and trigger
data, while the selection API exposes no certification tensors and no
certification predictions, counts, or trajectories are computed until every
candidate coordinate is frozen?

## Data and model

- Dataset: Wisconsin Diagnostic Breast Cancer, 569 real examples with 30
  measured features and two labels. The exact CSV is vendored as
  `data/wdbc_breast_cancer.csv` and included in the method seal.
- Each seed induces a deterministic class-stratified 60/20/20 split. The
  training split supplies the mean and population standard deviation used to
  standardize every row.
- Model: one-hidden-layer width-8 tanh MLP with 266 parameters.
- Optimizer: deterministic full-batch float64 gradient descent for 1,000
  updates, learning rate 0.005, L2 coefficient 0.001.
- Parameter checkpoints are retained every five updates.
- Fresh seeds: every integer from 101 through 124, with no replacement.

This is a real-data finite-set event experiment. It is not a population
generalization guarantee and is not described as grokking.

## Events and trigger-only candidate rule

- Persistent accuracy gates: 90%, 92.5%, and 95%.
- Persistence: ten consecutive optimizer updates.
- For each seed/gate, freeze the first checkpoint on the five-update grid at
  which training accuracy is at least 80% and trigger accuracy lies in
  `[gate - 10 percentage points, gate)`.
- The anchor is frozen whether or not a local clock predicts an event.
- The candidate selector may read only training and trigger histories. It must
  not load certification features, labels, counts, trajectories, outcome
  files, or sealed process logs.
- A seed/gate with no such checkpoint is retained as `no trigger-only anchor`.

Thus candidate selection is both substantially earlier and less selective
than the prior Transformer scanner: it neither requires a certification-set
deficit of one to three examples nor conditions on a predicted future event.

## Local clock and certificate

- Maximum window: 300 updates.
- Construct the anchor-affine optimizer clock and apply exactly four causal,
  anchor-preserving signed variational sweeps.
- Four sweeps were chosen before the fresh run. On development seeds 0--7,
  timing typically stabilized after one or two sweeps while later sweeps
  continued reducing the known path defect; the full 0--4 sweep ablation is
  retained.
- After the candidate manifest is sealed, the certificate process may first
  read the finite certification set. If the four-sweep centerline has no
  future persistent certification-set event inside the window, it abstains.
- Otherwise truncate the causal window after the predicted event plus the
  nine-update persistence tail. Let `s` be the exact centerline defect,
  `K_H` the finite-window Green operator, and `Z = ||K_H s||`.
- Use deterministic analytic tanh-network derivative envelopes and the
  cross-entropy composition bound to obtain an optimizer Jacobian-drift
  constant `M`.
- If `2 M Z > 1`, abstain before probing because `||K_H|| >= 1`.
- Otherwise obtain `kappa >= ||K_H||` with the frozen randomized Gram probe.
  Nonlinear closure requires `2 kappa M Z <= 1`.
- Use the smallest admissible radii-polynomial root

  `R_min = 2 Z / (1 + sqrt(1 - 2 kappa M Z))`.

  This changes only output tightness: an admissible scalar radius exists if
  and only if `2 kappa M Z <= 1`, the same feasibility condition obtained by
  testing `R = 2Z`.
- Transport the state ball through analytic true-minus-competitor margin jets.
  Issue only when strict guaranteed/possible count paths define a persistent
  first-passage bracket; otherwise abstain.

## Strong right-inverse baseline

For every probed candidate, also evaluate the classical direction-free bound

`Z_unsigned = kappa ||s||`.

Its derivative envelope is recomputed on its own radius. This is a matched
finite-window right-inverse/radii-polynomial baseline, strictly stronger than
the prior product of one-step Jacobian norms. It isolates the contribution of
propagating the realized signed defect `K_H s`.

## Randomized operator family

- Gaussian probes per Green operator: `m = 8`.
- Gram power: `q = 4`.
- Family failure probability: `Delta = 1e-6`.
- Maximum candidate operators: `24 seeds * 3 gates = 72`.
- Uniform per-operator allocation: `1e-6 / 72`.
- A 256-bit operating-system nonce is committed in the method seal before
  fresh training. Every frozen candidate identity is domain-separated under
  that nonce. The realized candidate universe is collision-checked against a
  prespecified ceiling of 72 operators.

The theorem is an ideal independent-Gaussian statement. The implementation
uses deterministic float64 pseudorandom streams generated from the committed
nonce. Accordingly, fresh certificates are labelled high-confidence numerical
certificates under the stated PRNG model, not exact-real computer-assisted
proofs.

## Information barrier and execution

1. The protocol, data, implementation, constants, and master nonce are sealed
   before fresh training.
2. Mutable fresh artifacts are written only beneath
   `%LOCALAPPDATA%/GreenCert/wdbc_confirmation_v1`, not a synchronized folder.
3. Training writes only outcome-blind train/trigger records and checkpoints.
   The selection-only loader returns no certification tensors, and no
   certification prediction, count, or accuracy trajectory exists at this
   stage.
4. Candidate selection hard-fails if a blind record contains certification
   fields and has no code path to a certification tensor or trajectory.
5. Candidate coordinates and the complete 72-case denominator are sealed
   before the full-split loader or any certification evaluation is called.
6. Every certificate-or-abstain JSON and its hash are sealed before any exact
   certification trajectory is generated.
7. Only after the certificate seal, post-seal auditing reconstructs the exact
   certification trajectories, checks every saved checkpoint, reconstructs
   issued centerlines by hash, and evaluates timing, containment, and observed
   state/sequence tubes.
8. No execution amendment is permitted. A failed phase leaves the fresh study
   incomplete rather than changing a rule or replacing a seed.

The SHA-256 record is a local integrity seal, not an external timestamp. No
priority claim depends on the seal's wall-clock date.

## Primary reporting

Report separately and prominently:

- trigger-only candidate rate over all 72 seed/gate cases;
- overall issuance rate over all 72 cases;
- conditional issuance among frozen candidates;
- conditional containment among issued certificates;
- number of distinct issuing seeds;
- abstention dispositions;
- bracket widths and lead times;
- raw timing error only on comparable future-event cases;
- strict closure and margin boundaries;
- queried-operator count and realized family failure budget;
- observed state/sequence tube violations after the sealed join;
- strong right-inverse baseline behavior; and
- 0--4 sweep accuracy/defect/runtime ablations from development only.

No repeat-until-success, population-coverage, independent-event, or
early-plateau grokking claim is permitted.
