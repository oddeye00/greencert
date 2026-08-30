# Frozen Transformer v3 confirmation protocol

This protocol is frozen before training seeds 355--378. Its purpose is to
confirm, without outcome-dependent tuning, the one-shot response-centered
Green theorem and the same-event progressive power rule developed after the
preceding seed-331--354 study.

## Population and model

- Seeds: 355--378 inclusive (24 fresh seeds).
- Task/model/optimizer: exactly the preceding mod-17, one-block,
  13,792-parameter causal Transformer with full-batch momentum.
- Training: 6,000 updates, learning rate 0.01, momentum 0.9, weight decay
  0.01, float64, checkpoints every 40 updates.
- Disjoint train, trigger, and 58-example certification splits are generated
  exactly as in the preceding protocol.
- Gates: 70%, 80%, 90%; persistence: 25 consecutive updates.

## Outcome barrier and candidate selection

Training writes trigger-visible and certification-outcome artifacts to
separate files. Candidate selection and certification reject outcome files.
The previous deterministic selector is unchanged: first trigger eligibility,
then the first checkpoint within 1,200 updates with certification deficit at
most three and a positive four-sweep modal event within a 300-update window.
At most one candidate is selected per seed/gate. Candidate coordinates and
modal offsets are sealed before any certification probe.

## Primary v3 certificate

The reference path uses exactly four anchor-fixed variational sweeps. Let
`s` be its defect, `z = K s`, `Z = ||z||_X`, and
`p = max_j ||z_j||`. The derivative envelope is built on the prespecified
outer domain `rho = 2 Z`.

For every Gaussian-probe power q=1,...,8, let `kappa_q` be the corresponding
simultaneous Green upper bound and `M_q` the optimizer-Jacobian drift envelope
formed from simultaneous output-Jacobian bounds at the same q. Define

    b_q = kappa_q M_q,
    Y_q = (b_q / 2) p Z.

The one-shot radius is the smaller nonnegative root E_q of

    Y_q + b_q p E + (b_q / 2) E^2 <= E,

provided `p + E_q <= rho`. Output margins are transported with pointwise
radius `p + E_q`. The primary certificate uses the first q whose nonlinear
closure and persistent output bracket both pass. If none passes, it abstains.

All q values reuse one precommitted Gaussian block per operator. Their bounds
hold simultaneously on one projection event, so the progressive stopping rule
does not divide the probability budget across powers.

## Safe pre-Green abstention

After the output envelopes are available, the construction uses

    kappa_lower = max(1, Z / ||s||_X).

If the q=8 derivative envelope cannot close even at this lower bound, the
certificate abstains without querying the Green operator. This is a fixed
logical rule, not a candidate-specific tuning choice.

## Matched baseline

The preceding Transformer rule is recomputed from the same q=8 quantities:
fixed radius `R = 2 Z`, closure `2 kappa_8 M_8 Z <= 1`, and the corresponding
output margins. This baseline is reported for every v3 candidate.

## Randomness and family accounting

- 16 independent standard-Gaussian probes per operator.
- Maximum power 8.
- Family failure probability 1e-6.
- The per-operator budget is divided by the maximum 24 x 3 candidates times
  302 possible operators (301 output states plus one Green operator), not by
  the realized candidate or issuance count.
- A fresh master nonce is generated and sealed before training; full operator
  identities derive disjoint deterministic streams from it.

## Confirmatory endpoints

Reported after the certificate seal:

- v3 and baseline issuance, coverage conditional on issuance, and distinct
  issuing seeds;
- first issuing q and its distribution;
- bracket width and lead time;
- prediction timing error, including inaccurate forecasts and abstention;
- state-radius and response-centered sequence-radius violations;
- all candidate, certificate, code, and seal hashes.

No theorem constant, selector, sweep count, threshold, persistence rule,
power ladder, or issuance rule may change after the method seal.
