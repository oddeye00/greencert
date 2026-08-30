# Frozen 15-case corrected-path prefix-panel protocol

Status: **FROZEN BEFORE ANY NONCE-DERIVED GAUSSIAN VECTOR WAS GENERATED**.

Frozen on 2026-08-29 after implementation review, deterministic unit tests,
and artifact-hash validation.  After the first query, every case is reported;
no probe count, forcing rule, output rule, candidate, horizon, or probability
allocation may be changed in response to the panel.

## Question

Can corrected-path relinearization turn the preceding one-case four-probe
result into a cohort-wide reduction in Green work over all 15 pre-existing
Green-evaluable v3 records, without reading any revealed future outcome?

This is a post-seal method-development audit, not a new prospective training
cohort.  It may strengthen the mechanism and implementation evidence but may
not alter the frozen prospective issuance or coverage counts.

## Frozen candidate family

The candidate-set digest is
`A34AEBB6651B05C4FE18A5379D1778838B276C4709D530936285A600FE2030FB`.
Each row is `(seed, threshold, anchor, horizon, certificate SHA-256)`:

```text
(360, .7, 3480, 131, BF9A46F67BD8AD23DA7C77945B0203BB8B4CA4A99CEDF297E23ED4BF64B3CBBA)
(361, .7, 1880, 275, D26ECDAFA3568040C30790090B286CCF9925AEA2FC76ED3A1D1881C7149F6F30)
(361, .8, 2800, 289, 896292CEF8EDB70B5B2ECFB8241F737154576E99B20CCAD7745BF7F8683624A2)
(366, .7, 1040,  52, 63DA40D4989669A1D1349B25323732C6800FD05E7EF793FBDE9C624E709CE24B)
(366, .8, 1120,  26, 9F773A441C44988A876BB4A707BA7F2C5B98C407A12F474C6D314F10862D639C)
(366, .9, 1360,  94, 2AE37E3B9E914F9E38EA6EDF813887AB4C9CD41FA845DDC7E1958A2DEF947A95)
(369, .7, 4160, 181, 06EFAC13FA25E054C69EFDCCA4776124684999982C4EB88CF7ADC18BD4E723CF)
(369, .8, 4480, 142, 557661FE278E39AEB1E066B539212A1A25C33F2DE5ADBBD972718DC76DE865BF)
(369, .9, 5760, 256, AD8D4896899E4B6C719634260347BE0BCD072C7552C229C3C5F65257E122F30D)
(370, .7, 2280, 299, D1CE3675187805FCF94911432036A41792C43939918163F2AA9B6C243935594D)
(372, .7, 3440, 270, 0E0323259DFD5A8C4A87870B42CF95D9EF991660C9DA0BA5905AB4D92DE54383)
(373, .7, 1280, 271, E9A33BD6F367468238649A81D0CA91ED89AB732AAEA47FDFD4E017B7FB4782D0)
(373, .8, 1760, 238, 2C7FCD2C64990577BBAF143C716FD09F55A3FC9E3B847491AE99D9D363AEE90F)
(375, .8, 1800, 142, 43E6ADEF120EDEBE35063C83F12CD7CB2B94F07C945665DB1F42DD0B63BA3738)
(378, .7, 3640, 285, 262950FDE6B2C8E7CD4F6B4F9945A4FAF5D1FD90EE3245AF470BBEF64C62876C)
```

The source two-response record has SHA-256
`2DAF416457E016F9D9A77F2E49B4B8B4FBAC8C69C98C9A3051D9105F11A3C287`.

## Fixed corrected-path construction

1. Rebuild the frozen four-sweep centerline `c` and verify its stored hash.
2. Form its stacked defect `s` in the scaled momentum norm.
3. Propagate the signed correction `d=K_c s`, replay its recurrence, and retain
   the measured stacked float64 recurrence residual.
4. Set `b=c+d` with the anchor fixed.
5. Form the cancellation-safe quadratic surrogate
   `q_tilde_j=D^2 G_j(c_j)[d_j,d_j]/2` and add the analytic fourth-objective
   Taylor remainder and the correction-recurrence residual to its error budget.
6. Rebuild the causal Green operator `K_bar` on `b`.

The derivative-drift envelope and persistent output transport are inherited
from each immutable v3 certificate at power one.  The closure is

```text
Y + kappa_bar M E^2 / 2 <= E,
```

with `||d||_infinity+E` required to remain in the original ball-valid domain.

## Fixed cost-aware forcing policy

At every available Green prefix, first use

```text
Y_norm = kappa_bar (||q_tilde|| + sigma).
```

If and only if this does not issue a persistent-event bracket at the current
prefix, compute one direct response `z_tilde` to `q_tilde`, replay that response
once for a measured recurrence residual `tau`, and thereafter use the tighter
of

```text
Y_norm
Y_dir = ||z_tilde|| + kappa_bar (sigma + tau).
```

The direct response is computed at most once per candidate and reused at later
prefixes.  No future event outcome enters this decision.

## Frozen adaptive Gaussian query

- Corrected Green power: `q=1`.
- Nested prefixes: `(4,8,16)`.
- New vectors are evaluated in batches of four with batched JVP/VJP products.
- Stop at the first prefix for which both nonlinear closure and persistent
  output-event transport issue; never instantiate unused later vectors.
- New Green family budget: `Delta_green=1e-6`.
- Equal spending over 15 operators and three prefixes:
  `delta_stage=1e-6/(15*3)=2.2222222222222224e-8`.
- The inherited output-Jacobian family has failure upper bound `1e-6`.
- Reported combined output-plus-new-Green failure upper bound: `2e-6`.
- Ideal-PRNG master nonce, generated with Python `secrets.token_hex(32)` before
  any vector was drawn:
  `4edb04d3665b0b0ac236906410c4c620f115d192a51fca5cba079121b2054ca2`.
- Operator identity:
  `(94, seed, gate_index, anchor, horizon, 4, 1)`.

The implementation must retain every generated vector hash and its initial and
final norm, allowing independent recomputation of all prefix calibrations and
closure arithmetic.

## Evidence and numerical boundary

- No revealed future-trajectory or outcome file may be read.
- Gaussian claims use the ideal-PRNG model.
- Neural JVP/VJP products, directional derivatives, recurrence residuals,
  derivative envelopes, and output margins remain float64/high-confidence;
  this panel is not an outward exact-real computer-assisted proof.
- The cancellation-safe Taylor remainder is analytic conditional on its stored
  float64 inputs.
- Existing output bounds and the new Green family are composed by a union
  bound, hence the explicit combined `2e-6` failure upper bound.
- Every negative, abstaining, or slower case remains in the aggregate.

## Frozen primary reports

- issuance count out of 15;
- prefix distribution `(4,8,16)` and nonissuance after 16;
- norm-only versus direct-response selected route;
- total and pairwise Green Gram-application reduction against the immutable
  16-probe directional baseline at its earliest issuing power;
- theoretical causal linearized-sweep reduction, with float64 recurrence replay
  overhead reported separately;
- bracket agreement, forcing headroom, timings, and residual maxima.

## Frozen implementation hashes

```text
scripts/audit_transformer_relinearized_prefix_panel.py 07837ECDFEC8E749AD2FC78B52FC8AEF68CDA2592F8EF793B0DE9574A1CD4824
scripts/batched_green_operator.py C30C1DAF0E8A8494B518CD12E6328146B1A20DD297033192C78408C2E46F54BF
scripts/cost_aware_forcing.py 89021849BBEA4BE641BE18D80CF69542D1A87D8F78F99D0C890CE1DCD38829F3
scripts/prefix_gram_enclosure.py AA8A18A21034ABE7B896FBB9C0BA5BB47D9F1FFF29554ED68515C2BFDC06FED2
scripts/probe_jacobian_bound.py CAF1ACEC6941125FE0556ACF53DE65AEA77C8F298C5EEA5123D7F46CDE0B30DD
scripts/relinearized_green_closure.py ACFB21152D0467E6DDAAB627138325B2079C4F75C1F8BD7FF708C427F67FE5E1
scripts/transformer_four_sweep_development_audit.py B60EA89574E82116A753F0C9F166A70E388F7707694596943D82DD13E10DA5D1
scripts/transformer_fourth_jet_bound.py 32859DC43BF78BC7569F1156D3FC732C74FF23A0346549B5171F85EB2F3F313F
scripts/transformer_green_development_audit.py F91FC0213CED87DED3AC0606E4F0B560A8760554F5C23B7F8936D4C24600620D
scripts/transformer_green_operator.py F529F42D8DC6B01C94D5F5496AC60BB8F63B4BC030FF85449F106362C8CE20B4
scripts/transformer_optimizer_probe.py 3C7665274245731141F4E0F451652F486CB65D7A16B0DCF233BAA66E8BFD5C79
scripts/transformer_two_response.py 097B8B736ED4695BDB81127A4832C2AB993DCF999E3435B49BA2CAE629FB8AAB
scripts/transformer_v3_certificate.py 5050DA95BAE58B02CB15AFAAF4802F9968A70562A26934B9F5E13FD224E71CBF
NESTED_PREFIX_GRAM_THEOREM.md 11F46E2D14AF9AB513F2F88AB3DB3FE216CAE11916D26941DA46AA54DB728D42
COST_AWARE_FORCING_THEOREM.md 8BD44EDD9730E5B1F863BE2B14A1E002332A06243297A6FEB511FF3398EE6F3E
```

## Execution-only amendment 1: paired-prefix counter

The first attempted v1 execution stopped before producing a prefix statistic.
The accumulation loop counted each newly generated vector twice—once in the
initial-norm list and once in the pending block—so it instantiated and applied
two vectors for the first candidate and then asked the scalar checker for a
four-vector prefix.  The checker raised `ValueError` before any Green bound,
closure, bracket, console statistic, result file, or cache file was produced.
The two final norms existed only in the failed worker's memory and were neither
printed nor inspected.  The original nonce is permanently burned.

The execution-only repair replaces that loop by the tested invariant

```text
new_count = target_prefix - paired_final_count,
paired_initial_count == paired_final_count,
paired_initial_count == paired_final_count == target_prefix after the query.
```

It does not alter the candidate set, theorem, prefixes, power, probability
spending, forcing policy, closure, output rule, or reporting plan.  A regression
test now checks increments `0->4`, `4->8`, and `8->16` and rejects mismatched
paired counts.  Script version is incremented to 2.

The replacement ideal-PRNG nonce was generated with
`secrets.token_hex(32)` after the repair and before any v2 vector was drawn:
`b3fe3a46aafe29d1ea08c5d1e24a547932e0dff73c67be8e4899011f328fdd05`.

Superseding implementation hashes:

```text
scripts/audit_transformer_relinearized_prefix_panel.py BE35D0771CF49B53B2D0721AA4BF3035EE9A9BF2F2DFA1BABB2B9B37A47A2B58
scripts/prefix_gram_enclosure.py 758142A941D4039E72014C9352AFDE3CA01DD39EE5918D115017905E924B2D78
```

All other frozen hashes above remain unchanged.  No outcome file was read
during the failed execution or amendment.
