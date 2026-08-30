# Frozen handwritten-digits signed-Green confirmation

Status at seal: prospective, outcome-blind confirmation. Development seeds
0--2 and their certification trajectories were inspected. Fresh seeds
501--512 have not been trained or evaluated before the method seal.

## Scientific question

Does signed defect propagation produce covered first-passage certificates on
a non-modular image-classification task when the matched scalar Euclidean
right-inverse bound abstains?

## Data and information barrier

- Data: the 1,797-example, 64-feature `sklearn.datasets.load_digits` copy of
  the UCI Optical Recognition of Handwritten Digits test subset.
- Target: digit parity (even/odd), fixed before fresh training.
- Each seed creates a deterministic parity-stratified 60/20/20
  train/trigger/certification split.
- Standardization uses only the training mean and population standard
  deviation and is then applied unchanged to trigger and certification rows.
- Training and candidate selection call `make_selection_split`, which does not
  materialize certification tensors.  The full loader is called only after the
  candidate manifest is sealed.
- The realized future certification trajectory is reconstructed only after
  every certificate-or-abstain record is sealed.

## Frozen model and optimizer

- One-hidden-layer width-8 tanh MLP, 538 parameters, affine readout.
- Full-batch gradient descent, learning rate 0.03, L2 coefficient 0.001.
- 600 updates, float64, checkpoints every 5 updates.
- Fresh seeds: 501--512.

## Frozen event and trigger

- Accuracy gates: 0.90 and 0.925.
- Persistence: 10 consecutive updates.
- Candidate anchor: first checkpoint at which train accuracy is at least 0.80
  and trigger accuracy lies in `[gate-0.10, gate)`.
- Horizon: 400 updates.

## Frozen certificate

- Three causal variational sweeps.  This finite correction budget was selected
  on development seeds specifically to leave a nontrivial residual and test
  the signed propagation claim; both methods receive the identical centerline.
- Signed zero-order term: `Z = ||K_H s||_2`.
- Matched scalar unsigned comparator: `kappa ||s||_2`, retaining the same
  Euclidean norm, same finite-window right inverse, same derivative envelope,
  and same output logic while discarding defect direction.
- Gaussian Gram norm bound: 8 probes, 4 power iterations.
- Family-wise failure budget: 1e-6 over at most 24 queried operators.
- Analytic tanh/cross-entropy derivative envelopes and strict output margins.

## Primary endpoints

1. Signed certificates issued and covered after outcome reveal.
2. Signed-only certificates: signed issues while the matched unsigned
   comparator abstains.
3. Seed-level issuance, lead, bracket width, closure statistics, and the
   directional gain ratio.

No hyperparameter, threshold, anchor rule, sweep count, probe setting, or
analysis rule may change after the method seal.
