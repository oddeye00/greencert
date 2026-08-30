# Prospective primary confirmation of recentered variational certificates

Frozen on 2026-08-21 before training or inspecting any model, trajectory,
checkpoint, crossing, trigger, or certificate artifact for seeds 17--24. A
pre-freeze filesystem search found no such artifact for any of these seeds.
The code and this protocol are sealed by
`PROSPECTIVE_V2_CODE_MANIFEST.json`; every experimental entry point verifies
all recorded SHA-256 hashes and fails closed after any change.

## Objective

Evaluate the already fixed one-sweep recentered variational certificate in a
genuinely prospective setting. The operational trigger is the sole primary
trigger and is executed chronologically from information available at that
step. Event times are sealed until the float64 certificate and its outward
re-evaluation have both been completed.

No theorem, trigger, model, training hyperparameter, derivative envelope,
threshold, horizon, seed, or analysis rule may be changed in response to this
batch.

## Frozen population and stopping rule

- Seeds: exactly `{17,18,19,20,21,22,23,24}`.
- All eight seeds are retained, irrespective of fitting, final accuracy,
  grokking delay, event count, trigger availability, issuance, or coverage.
- Training stops after exactly 180,000 gradient steps.
- The population will not be extended, shortened, rerun, or selectively
  replaced in response to results.
- Earlier seeds are development or previous-confirmation evidence and are not
  pooled into this primary estimate.

## Frozen model and training process

Use `generate_smooth_mlp_seed.frozen_config(seed)` without modification:

- modular addition modulo 11;
- one-hidden-layer tanh MLP with biases and width 24 (827 parameters);
- 70% training support, selected deterministically from the seed;
- one-hot mean-squared error;
- literal deterministic full-batch gradient descent in binary64;
- learning rate 2.0 and coupled L2 coefficient `1e-4`;
- 180,000 steps, logged every 25 steps, with parameters saved every 250 steps.

Training is permitted exactly once per seed. The prospective generator refuses
to overwrite either output. The subsequent chronological replay must agree
bit-for-bit with every saved checkpoint.

## Frozen first-passage events

The five held-out-accuracy gates are

\[
\{60\%,70\%,80\%,90\%,95\%\}.
\]

There are 36 held-out examples, so the respective required correct counts are
`{22,26,29,33,35}`. A first passage is recorded at the first individual
training step whose count meets or exceeds the gate. First passages are state
variables in the chronological replay; they are not computed first and then
fed to trigger selection.

## Prospective primary trigger

At each saved checkpoint, processed in increasing training-step order, a gate
is eligible only when all of the following hold:

1. the gate has never previously been crossed;
2. no trigger has previously been selected for that seed-gate pair;
3. the current held-out count is exactly one below the required count;
4. the checkpoint is at most step 179,750, leaving its complete 250-step
   outcome window observable; and
5. the frozen raw anchor-Hessian modal path predicts that the gate will be met
   at an integer offset in `{1,...,250}`.

The raw modal path is

\[
\bar d_{j+1}=(I-\eta H_0)\bar d_j-\eta g_0,
\qquad \bar d_0=0,
\]

where the gradient and Hessian are evaluated only at the current checkpoint.
Predicted counts are obtained from exact tanh-network logits at
`theta_anchor + bar_d_j`. The trigger uses no future trained parameter,
eventual-crossing table, variational correction, tube radius, or observed
outcome. A gate that never crosses remains eligible until the final admissible
anchor. Select the first qualifying checkpoint and make at most one attempt
per seed-gate pair.

The chronological scanner writes two disjoint artifacts. The blind artifact
contains triggers, raw predictions, and checkpoint-replay checks, but no
crossing or training-outcome field. The sealed outcome artifact contains first
passages and training summaries. Certificate processes may read only the
blind artifact and saved parameters until all blind verification is complete.

## Frozen float64 candidate certificate

- Method version: `recentered-variational-v2-frozen-2026-08-21`.
- Exactly one signed variational correction sweep.
- Exact analytic gradient and full Hessian along the raw modal reference.
- Exact recentered residual and full Hessian along the corrected reference.
- The fixed global analytic tanh objective-Hessian Lipschitz envelope.
- The fixed global analytic logit-margin derivative envelope.
- Euclidean state tube, binary64 implementation, and numeric cap `1e4`.
- Maximum horizon: 250 gradient steps.
- The fixed guaranteed/possible-correct first-passage rule.
- No repeated correction, target projection, empirical derivative fit,
  seed-specific constant, post-event anchor, or outcome-dependent choice.

At every selected anchor, frozen v1 is computed as a paired baseline using the
same local Hessian work. V1 never affects v2 selection or reporting. The float
candidate stage reads no sealed event time.

## Frozen outward re-evaluation

Every issued float v2 candidate is re-evaluated before outcomes are opened.
The implementation uses `python-flint`/Arb at 192-bit precision for tanh and
analytic constants, directed binary64 endpoints, and standard IEEE-754 gamma
bounds for reductions and matrix products. It encloses:

- network values and analytic gradients at each binary reference point;
- Hessian evaluation and eigendecomposition reconstruction error;
- Jacobian operator norms, defects, Lipschitz terms, and the scalar tube
  recursion; and
- held-out logit-margin tests and the resulting first-passage bracket.

An outward result may retain a float candidate or abstain; it cannot add an
issuance. Its state statement concerns the exact-real gradient-descent map
initialized at the saved binary checkpoint and an explicitly binary reference
sequence. It does not enclose roundoff accumulated by the separately observed
binary64 training trajectory. Agreement of that trajectory with the state
tube is therefore reported as a distinct empirical audit, not as part of the
computer-assisted exact-real claim. This scope limitation must remain explicit
in every paper statement.

## Order of execution and information barrier

The mandatory order is:

1. freeze and verify protocol/code hashes;
2. train all eight seeds once;
3. run the chronological scanner, writing blind triggers and sealed outcomes;
4. construct every blind paired v1/v2 candidate;
5. outward-check every issued v2 candidate and save the blind interval result;
6. only then open and aggregate the sealed crossing outcomes;
7. join outcomes to the already fixed float and outward brackets.

No step 3 outcome field may be inspected, printed, or consumed before step 5
is complete.

## Frozen outcomes and summaries

Report all 40 seed-gate pairs and, separately, the triggered subset:

- seeds trained, fitted seeds, observed first passages, and seeds satisfying
  the prespecified natural-grokking criterion
  `exact 95% crossing / first logged 99% training-fit step > 10`;
- trigger availability and unique trigger anchors;
- v1 and v2 float issuance and abstention;
- outward candidate retention and outward issuance;
- containment conditional on issuance, always as counts `k/n`;
- exact 95% Clopper--Pearson intervals as descriptive i.i.d. event- and
  seed-level working analyses, explicitly noting within-seed dependence;
- distinct issuing seeds and whether every issuance within each such seed was
  contained;
- median bracket span and event lead, with minima and maxima where useful;
- v1/v2 median rigorous horizons and paired horizon ratio;
- state-tube violations and maximum observed error-to-bound ratio;
- all issued brackets individually; and
- failure decomposition: no trigger, event unobserved, event beyond the local
  window, state-horizon exhaustion, or output-margin ambiguity.

An issued bracket whose complete 250-step outcome window is observed but whose
event does not occur is non-contained. A gate with no trigger is an abstention,
not a coverage trial.

Thresholds from the same seed are correlated. Event-level intervals are not
presented as independent-seed evidence; the number of distinct issuing seeds
and a seed-clustered summary are primary context.

## Decision rule

Safety confirmation requires zero non-contained outward-issued brackets and
zero observed state-tube violations. Float candidate coverage is reported in
parallel for comparability. Availability has no pass/fail target: issuance,
abstention, and failure causes are reported exactly as observed. No new seed or
method variant may be added to rescue either criterion.
