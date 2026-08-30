# Independent audit of the four-sweep Transformer certificate

## Decision

**NO-GO for a fresh confirmation freeze.**

The four-sweep construction does not pass its burned-candidate development
audit once every term required by the recentered theorem is implemented.  It
issues zero certificates on the three previously frozen Transformer timing
candidates.  The construction abstains conservatively, with no observed
state-tube violation.

This result supersedes the final "all gates are discharged" status in
`TRANSFORMER_ENVELOPE_DIAGNOSIS.md`.  No fresh Transformer seed was trained.

## What the independent audit found

### 1. Gate 3's PSD-Gram theorem is correct

For a fixed rectangular output Jacobian `J`, applying the theorem to
`A = J^T J` gives

```text
||J||_2 <= (Y / c_delta)^(1/(2q)),
c_delta = Phi^{-1}((1 + delta^(1/m))/2).
```

The exponent, folded-normal quantile, and one-sided failure event are correct.
On the real Transformer, independently formed derivatives agree with the
double-backward products to relative errors

```text
Jv:       1.9e-15
J^T w:    6.4e-16
J^T J v:  1.2e-15
```

The burned anchor bound remains `6004.1045`, versus a power-iteration lower
estimate `5188.5150`.

### 2. The original RNG rule was not sufficient for a confirmation

SHA-256 of operator coordinates avoids the previously observed coordinate
collision, but coordinates alone contain no exogenous random draw.  A formal
randomized confirmation needs a master nonce independent of training and
candidate selection.  The audited implementation therefore uses

```text
committed master nonce + full operator identity -> SHA-256 -> PRNG stream
```

and checks the complete instantiated universe for seed collisions before any
query.  Unknown and duplicate runtime operator queries hard-fail.

### 3. The reported 7,920-operator count was not the certificate count

There are two independent problems with 7,920:

1. the inclusive 1,200-step scan on a 40-step grid has **31**, not 30, anchor
   positions; and
2. the count covers only an output-Jacobian operator.  The state theorem also
   requires `beta_j >= ||DG(c_j)||` for the momentum optimizer map.

Candidate selection must not use probabilistic probes, so the corrected design
assigns zero probe operators to screening.  For at most eight seeds, three
gates, one candidate per gate, a 300-update horizon, and the two operator
families, the maximum is

```text
24 candidates * (301 output states + 300 optimizer transitions) = 14,424.
```

With family-wise `Delta = 1e-6`, the uniform per-operator budget is
`6.9328896284e-11`.  Four recentering sweeps do not multiply this count because
only the final, probe-independent centerline is queried.

### 4. The missing optimizer-map bound is decisive

Use the natural scaled momentum state

```text
y = (theta, w),   w = eta v.
```

Writing `r = mu w + eta grad F(theta)`, the optimizer map is

```text
G(theta,w) = (theta-r, r).
```

Its exact products are

```text
DG[p,q]       = (p-a, a),       a = mu q + eta H p,
DG^T[u,z]     = (u + eta H(z-u), mu(z-u)).
```

Thus one `DG^T DG` application requires two objective HVPs.  JVP/VJP adjoint
tests and PSD quadratic-form tests pass on the real model.  The resulting
certified one-step optimizer norms are about `2.188`, while direct lower
estimates are about `1.504`.  This nonnormal one-step amplification was absent
from the earlier go/no-go arithmetic.

### 5. The block-aware jet needed two implementation corrections

The monomial formula

```text
sup_{sum s_b^2 <= 1} product_b s_b^(a_b)
  = product_b (a_b/d)^(a_b/2)
```

is correct.  Fuzzing confirms all degree-1/2/3 mixed product and chain-rule
coefficients.  The audit nevertheless found and fixed two ball-level issues:

- scalar reduction had only been checked at radius zero; the positive-radius
  embedding contribution was missing;
- one global output sensitivity was used to inflate every intermediate stage.
  The implementation now computes a separate monotone fixed-point inflation
  for each named stage.

After correction, scalar reduction reproduces the shipped jet at multiple
positive radii and perturbed parameter points to relative error below `1e-9`.
Twenty-four real-network directional stress tests show zero violations.

## Complete burned-candidate audit

All three candidates were fixed by the pre-existing candidate seal.  Exactly
four recentering sweeps were used.  The centerline was completed and hashed
before the probabilistic registry was instantiated.  Outcomes were used only
for the post-certificate audit columns.

| Seed | Gate | Anchor | Actual event | Raw timing error | Rigorous horizon | Max radius before stop | Output-J upper | Optimizer-J upper | Issued |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 321 | 70% | 1,440 | +211 | 0 | 26 | `2.539e4` | 6,068.64 | 2.18821 | no |
| 322 | 70% | 2,400 | +34 | 0 | 27 | `2.753` | 4,009.55 | 2.18872 | no |
| 322 | 80% | 2,640 | +87 | 0 | 27 | `6.540e-3` | 3,958.64 | 2.18851 | no |

Aggregate:

```text
candidates:                 3
certificates issued:        0
observed tube violations:   0
queried operators:          52, 54, 54
maximum family bound:       1e-6
```

The four-sweep centerlines still predict all three event times exactly.  What
fails is the scalar proof envelope: a machine-scale residual is amplified by
the probabilistic one-step optimizer bound, then the quadratic nonlinear term
causes a rapid radius jump.  For the shortest event, the radius progresses from
`1.612e-5` at step 26 to `2.753` at step 27, seven steps before the event.

## Consequence for the paper

Do not claim a formal Transformer certificate and do not spend fresh seeds on
this four-sweep scalar construction.  The already frozen Transformer result
remains valid as a prospective HVP-only **timing transfer**: 3/3 exact event
predictions, not certificates.

The no-go is itself informative.  The remaining obstruction is no longer the
network-output derivative envelope.  It is the product-of-one-step-norms
majorant for a stable but nonnormal momentum trajectory.  Any future formal
Transformer result must control finite-window propagator gain (or use an
equivalent metric-aware/projected tube) rather than multiplying scalar
`||DG(c_j)||` bounds.

## Artifacts

Machine-readable aggregate:

```text
results/transformer_four_sweep_development_audit.json
```

Per-candidate records:

```text
results/transformer_four_sweep_development_seed_321_gate_0_anchor_1440.json
results/transformer_four_sweep_development_seed_322_gate_0_anchor_2400.json
results/transformer_four_sweep_development_seed_322_gate_1_anchor_2640.json
```

Implementation and gates:

```text
scripts/transformer_four_sweep_development_audit.py
scripts/transformer_certificate_protocol.py
scripts/transformer_optimizer_probe.py
scripts/probe_jacobian_bound.py
scripts/transformer_block_envelope.py
scripts/test_probe_jacobian_bound.py
scripts/test_transformer_certificate_protocol.py
scripts/test_block_jet_bound.py
```

Aggregate SHA-256 at completion:

```text
5112180E4A576BA5987482F787C8C9C26BECCC57066F8E4B5CC43E5EF97877EF
```
