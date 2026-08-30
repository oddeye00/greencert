# Corrected Frozen Smooth-Transformer HVP Transfer Protocol

Frozen before training or inspecting seeds 321--326.

This protocol is identical to `TRANSFORMER_HVP_PROSPECTIVE_PROTOCOL.md`
except for the fresh seed set and one stricter information barrier: each
training process has all terminal output redirected to a sealed log that is
not opened until the blind candidate file is written and hashed.

Seeds 311--313 are excluded from confirmation because their training CLI
printed certification summaries before blind candidate construction.  No
method constant was changed in response to those outcomes.

## Frozen constants

- Seeds: 321, 322, 323, 324, 325, 326.
- Addition modulo 17; deterministic 173/58/58 train/trigger/certification
  split.
- One normalization-free causal smooth Transformer block, four heads, model
  dimension 32, GELU feed-forward dimension 128, no dropout.
- 13,792 trainable parameters; 27,584-dimensional momentum state.
- Full-batch momentum GD: learning rate 0.01, momentum 0.9, L2 coefficient
  0.01, cross-entropy, 6,000 updates, float64.
- Metrics every 20 updates; parameter and velocity checkpoints every 40.
- Thresholds 70%, 80%, 90%; persistence 25.
- Trigger gate `max(50%, threshold - 20 percentage points)` after 99% training
  accuracy.
- Scan 1,200 steps at 40-step spacing; invoke the HVP forecast only when the
  current certification count is one to three examples below target.
- Full 300-step optimizer-state clock; exactly two signed recentering sweeps;
  freeze the first future persistent modal event or abstain.
- All seeds, thresholds, candidates, and abstentions retained.

## Information barrier

For every seed, the blind trajectory, certification outcomes, and checkpoints
are separate files.  Training stdout/stderr is redirected to
`results/transformer_hvp_seed_<seed>.sealed.log`.  Neither outcome JSON nor
sealed log may be opened before
`results/transformer_hvp_prospective_candidates_blind.json` is finalized and
hashed.  The scanner reads blind JSON plus checkpoints only and rejects a blind
trajectory containing a certification column.

The audit is an HVP-only prospective timing transfer, not a formal Transformer
state-tube certificate.  Rigorous certificate claims remain restricted to the
MLP experiments.
