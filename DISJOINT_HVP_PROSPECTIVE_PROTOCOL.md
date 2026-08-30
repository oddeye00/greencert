# Frozen Disjoint-Set Matrix-Free Prospective Protocol

Frozen on 2026-08-22 before training or inspecting seeds 202--205.

## Scientific question

Can the recentered variational certificate issue covered, persistent
first-passage brackets in a neural network larger than the exact-Hessian
experiments, while (i) forming no dense Hessian and (ii) separating the data
that opens the audit window from the data whose event is certified?

## Untouched models

- Seeds: 202, 203, 204, 205.
- Task: addition modulo 13.
- Model: one-hidden-layer tanh MLP, width 48, 1,933 parameters.
- Split: 101 training examples, 34 trigger examples, and 34 certification
  examples. The three sets are deterministic, mutually disjoint, and exhaust
  all 169 inputs.
- Full-batch gradient descent: learning rate 1.0, weight decay 0.0001,
  initialization multiplier 1.0, 240,000 steps.
- Metrics are logged every 50 steps and parameters are stored every 250 steps.
- Training-fit eligibility requires at least 99% training accuracy.

## Event

For each accuracy threshold in 60%, 70%, 80%, 90%, and 95%, let the required
count be the ceiling of the threshold times 34. The certified event is the
first post-anchor update at which the certification-set count reaches the
required value and remains there for 25 consecutive updates. This persistence
rule excludes one-step optimizer oscillations.

## Causal audit rule

The training and trigger-set histories alone open an audit window. For a
threshold, eligibility is the first logged step at which training accuracy is
at least 99% and trigger accuracy reaches that threshold, rounded upward to
the next stored 250-step checkpoint.

From that checkpoint through 5,000 later training steps, at 250-step spacing:

1. If the certification count is already at or above threshold, record that
   the event is not a future event at eligibility and stop that threshold.
2. Otherwise construct the 250-step full-space HVP affine clock and apply
   exactly two signed variational recentering sweeps.
3. Invoke the expensive projected certificate only if that deterministic
   centerline predicts a future 25-step-persistent event in the window.
4. Issue only when the rigorous lower/upper count envelopes give a nonempty
   first-passage bracket. Otherwise abstain and continue the frozen grid.

The certification set is never used to open the audit window. Its present
margins are used by the certificate, as required to define and certify the
deployment event; future certification outcomes are not used for anchor
selection.

## Matrix-free certificate constants

- Horizon: 250 updates.
- Fixed active rank: 64.
- Active starts: training gradient and one tight certification-margin
  gradient.
- Geometry probes: every 5 centerline steps.
- Gaussian power iteration: power 12, four probes per curvature component.
- Signed recentering sweeps: two.
- Per-certificate probabilistic failure budget: 1e-9.
- Maximum number of prospective certificate tests: 420.
- Union-bound family-wise failure probability: at most 4.2e-7.
- Dense Hessian entries formed: zero by construction.

## Required reporting

Report all four seeds, all threshold dispositions, number of centerline
candidates, issuance and abstention, coverage, bracket width, certified lead,
state-tube violations, HVP counts, and distinct issuing seeds. Thresholds
within a seed are treated as correlated; seed-level results are reported
separately. No seed, threshold, checkpoint, or failed certificate may be
dropped.

## Development record excluded from confirmation

Seed 201 selected this protocol. On that development seed, the automatic rule
screened 18 checkpoints and issued the 25-step-persistent 60% bracket
`[221,221]`, which contained the actual crossing. Seed 101 used learning rate
2.0 and exposed a two-step oscillation; it is retained as a diagnostic failure
and is not part of the prospective evidence.

Any code or constant change after this seal invalidates the confirmatory label
unless the entire four-seed batch is restarted under a newly documented seal.
