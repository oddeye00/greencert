# Frozen outward scalar-secant execution protocol

Frozen after the one-checkpoint Arb implementation audit and before any
full-sequence interval is computed. This is outcome-blind post-seal numerical
hardening; it cannot change prospective issuance counts.

- Candidate, horizon, power, amplification: seed 366 / 52 / 1 / 4096.
- Probe block: the four probes and `delta=1e-6` frozen in
  `AMPLIFIED_SECANT_FOUR_PROBE_PROTOCOL.md`; no new draw or selection.
- Checkpoints: every forcing row `j=1,...,51`; row zero is exactly zero.
- Arithmetic: python-flint Arb, 192-bit midpoint precision.
- Evaluator: one exact base mixed jet in the signed parameter response and all
  four objective-probe directions, plus one exact shifted first jet at
  `theta + 4096 a`. GELU uses the exact erf formula; attention, softmax,
  cross-entropy, and weight decay are evaluated as Arb balls.
- Per-checkpoint scalar identity:
  `eta/4096^2 * (D_w F(theta+4096a)-D_w F(theta)-4096 D^2_{w,a}F(theta))`.
- All stored binary64 center, response, and ideal-PRNG probe coordinates are
  treated as exact dyadic inputs. Their upstream construction is not thereby
  outward-certified.
- Parallelism: four independent worker processes; parallel order may not alter
  interval addition, which is redone serially at 192 bits from enclosing Arb
  strings.
- Calibration: compute
  `sqrt(2)*erfinv((10^-6)^(1/4))` in Arb and divide by its lower endpoint.
- Read no future trajectory or outcome file. Report all 204 checkpoint/probe
  intervals, four summed intervals, calibration enclosure, response-free norm
  bound, closure, bracket, runtime, source hashes, and outcome-file count.

Frozen source SHA-256:

- `arb_transformer_objective.py`:
  `59DC6B9889669725CD4257EE6E6F6263C46B0EA3C7D7DAF5A514AF714DF5C736`
- `arb_transformer_multijet.py`:
  `4607FE24142A95950814A85C19D9D11A2811B5DE241FC7036B3B5B30FB57AD6C`
- `test_arb_transformer_objective.py`:
  `0A04A910F4DAAFEC4D3D69FB21DA8933C756C15E96F2290C9DDD4903F3092A63`
- `test_arb_transformer_multijet.py`:
  `A8426E7F25C855E63D696AA9477C37E8BABEA915846ED62E10B7CD5724F55043`

The resulting theorem component is an outward scalar-forcing certificate
conditional on stored dyadic inputs and the ideal-PRNG model. It is not a full
computer-assisted proof of the Transformer training-event bracket because
upstream centerline/Green products, derivative envelopes, and output margins
retain their existing numerical boundary.
