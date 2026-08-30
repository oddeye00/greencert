# Frozen outward scalar-secant execution protocol v2

This v2 protocol is frozen after an adversarial audit found that v1 formed
`w = g_w - g_theta` in binary64 before entering Arb. The v1 intervals are
numerically informative but are superseded for exact-input claims. No future
trajectory or outcome was read in finding or fixing this issue.

V2 preserves every scientific and randomized choice from v1:

- candidate / horizon / power / amplification: seed 366 / 52 / 1 / 4096;
- the exact same four-probe nonce and `delta=1e-6` from
  `AMPLIFIED_SECANT_FOUR_PROBE_PROTOCOL.md`;
- all forcing rows `j=1,...,51`, 192-bit Arb, four workers;
- the same exact scalar secant identity, serial interval summation, and Arb
  folded-normal calibration.

The only change is numerical correctness: each objective probe is now formed
inside Arb as the exact difference of the two stored dyadic state-probe halves,

`w_i = arb(g_w,i) - arb(g_theta,i)`,

with no intervening binary64 subtraction. The shifted point remains the exact
Arb sum `theta + 4096 a`.

Frozen source SHA-256:

- `arb_transformer_objective.py`:
  `59DC6B9889669725CD4257EE6E6F6263C46B0EA3C7D7DAF5A514AF714DF5C736`
- `arb_transformer_multijet.py`:
  `186C6C3A7CCA06AC322778C242FA882D662EE9349F536045D50470623F92754C`
- `test_arb_transformer_objective.py`:
  `0A04A910F4DAAFEC4D3D69FB21DA8933C756C15E96F2290C9DDD4903F3092A63`
- `test_arb_transformer_multijet.py`:
  `4008532246552EF440B6854EAF1CCCF3871B93A8CBB5513E0A7AB62CDBD87F20`

The claim boundary is unchanged: v2 outward-encloses the four scalar secant
projections conditional on stored dyadic center/response/probe inputs and the
ideal-PRNG model. It does not outward-certify the upstream Green construction,
derivative envelopes, or output margins and is not a complete computer-assisted
Transformer event proof.
