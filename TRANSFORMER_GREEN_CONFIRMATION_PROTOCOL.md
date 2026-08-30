# Frozen Fresh Transformer Signed-Green Confirmation Protocol

Status: to be SHA-256 sealed before training seeds 331--354.

## Scientific question

Can the frozen four-sweep local optimizer clock issue valid persistent
first-passage certificates on genuinely untouched causal Transformers when
the scalar product-of-one-step-norms tube is replaced by a matrix-free
finite-window signed-Green certificate?

## Population and model

- Fresh seeds: every integer from 331 through 354 (24 seeds).
- Addition modulo 17 with the deterministic seed-specific 173/58/58
  train/trigger/certification split.
- One normalization-free causal Transformer block, four heads, model dimension
  32, GELU feed-forward dimension 128, no dropout, 13,792 parameters.
- Full-batch float64 momentum GD for 6,000 updates: learning rate 0.01,
  momentum 0.9, L2 coefficient 0.01, cross-entropy.
- Metrics every 20 updates and parameter/velocity checkpoints every 40.

## Candidate rule

- Gates: 70%, 80%, and 90%; persistence: 25 consecutive updates.
- Trigger eligibility: at least 99% train accuracy and trigger accuracy at
  `max(50%, gate - 20 percentage points)`.
- Starting at the first eligible checkpoint, scan an inclusive 1,200-update
  window at spacing 40.
- Invoke the deterministic HVP clock only when the current certification count
  is one to three examples below the gate.
- Propagate the complete 300-update optimizer-state clock and apply exactly
  four signed recentering sweeps.
- Freeze the first anchor whose four-sweep centreline has a future persistent
  event for that gate; otherwise abstain. At most one candidate is retained
  per seed/gate. No probabilistic probe is permitted during screening.

The certification set is disjoint from the trigger set but is a deterministic
finite event target, not an untouched population sample. Its current count and
predicted centreline event participate in candidate selection; future true
certification trajectories do not.

## Certificate rule

For a sealed candidate, let `c_0,...,c_H` be the four-sweep centreline, where
`H = predicted_event + 24`, and let

```text
s_j = G(c_j) - c_{j+1},
z = K_H s,
Z = ||z||_sequence,
R = 2 Z.
```

`K_H` is the causal finite-window Green operator for the scaled momentum state
`(theta, learning_rate * velocity)`. Neural block jets and output-Jacobian
probes give a uniform optimizer derivative-drift bound `M` on radius `R`.

Because `||K_H|| >= 1`, the process safely abstains before a Green probe if
`2 M Z > 1`. Otherwise it probes `K_H^T K_H` and obtains
`kappa >= ||K_H||`. The state certificate issues only if

```text
2 kappa M Z <= 1.
```

The certified sequence ball is transported through strict multiclass margins.
A persistent bracket issues only when both its earliest-possible and
earliest-guaranteed block starts exist. The radius rule, early-abstention rule,
and all constants are fixed here; no radius search or adaptive extra sweep is
allowed.

## Randomized operator family

- Gaussian probes per operator: `m=16`.
- Gram power: `q=8`.
- Family failure probability: `Delta=1e-6`.
- Maximum candidates: `24 seeds * 3 gates = 72`.
- Operators per candidate: `301 output Jacobians + 1 Green operator = 302`.
- Maximum family: `72 * 302 = 21,744` operators.
- Uniform per-operator failure probability: `1e-6 / 21,744`.
- Master nonce:
  `c0b81a6cb799088f0679c5b5ad39cb25e5eac84a2bca01762bf4a8f07f529ab7`.
- Probe streams are SHA-256 domain-separated by the master nonce and complete
  operator identity. Runtime identities must be a collision-free subset of
  the candidate-instantiated universe; unknown and duplicate queries hard-fail.

The master nonce was drawn from the operating system before any fresh model
was trained. Increasing the seed population changes only the mechanically
derived per-operator allocation; `m`, `q`, and family `Delta` are unchanged.

## Information barrier

1. A no-artifact audit and code manifest are written before training.
2. Training stdout/stderr are redirected to sealed logs and not opened during
   candidate selection or certification.
3. Training writes blind trigger JSON, checkpoint NPZ, and a separate outcome
   JSON. Blind code rejects any trajectory with a certification column and
   rejects reads of `.outcomes.json` or `.sealed.log`.
4. Candidate coordinates, horizons, modal-file hashes, and manifest hash are
   sealed before certification.
5. Certificate files and their hashes are sealed before exact future paths are
   rolled out for coverage auditing.
6. All 72 seed/gate records, all candidates, all abstentions, and all issued
   certificates are retained. There is no repeat-until-issuance logic.

## Primary reporting

- candidate rate and disposition counts over all 72 seed/gate cases;
- issuance among frozen candidates;
- containment among issued certificates;
- distinct issuing seeds;
- abstention rate;
- bracket widths and certified leads;
- strict output-logic slack and nonlinear closure slack;
- family-wise random-probe budget and actual query count;
- observed state/sequence tube violations after the sealed outcome join;
- seed-clustered reporting alongside event-level counts.

No unconditional population-coverage claim will be made from correlated gates
within a seed.

## Expected artifacts

- `TRANSFORMER_GREEN_CONFIRMATION_METHOD_SEAL.json`
- `results/transformer_green_confirmation_no_artifact_audit.json`
- blind seed JSON/checkpoints and separate outcome JSONs
- `results/transformer_green_confirmation_candidates_blind.json`
- `TRANSFORMER_GREEN_CONFIRMATION_CANDIDATE_SEAL.json`
- one outcome-blind certificate JSON per candidate
- `TRANSFORMER_GREEN_CONFIRMATION_CERTIFICATE_SEAL.json`
- one post-seal audit JSON per candidate
- `results/transformer_green_confirmation_audit.json`

The single frozen entry point is:

```text
python scripts/run_transformer_green_confirmation.py --all
```
