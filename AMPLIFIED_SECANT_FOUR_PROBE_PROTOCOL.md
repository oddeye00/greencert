# Frozen post-seal four-probe amplified-secant audit

Frozen after the 16-probe development audit and before this fresh block is
drawn.  This is outcome-blind method-development evidence and does not alter
any prospective issuance count.

- Candidate: Transformer seed 366, threshold 0.70, anchor 1040.
- Horizon/power/amplification: 52 / 1 / 4096.
- Response-free theorem interface:
  `beta = kappa * (sigma_sec + ||q^[lambda]||_U)`.
- Probe block: exactly 4 iid standard-Gaussian forcing-sequence probes under
  the paper's ideal-PRNG model.  There is no probe-count adaptation.
- Failure allocation: `delta = 1e-6` for this block.
- Nonce:
  `5a37e5ccaf6834c438fde251d52ec1de313329314377d70cc1cb25e62fc52f2a`.
- Seed derivation: little-endian integer from the first eight bytes of
  `SHA256("greencert-response-free-secant-four-probe-v1|" + nonce)`, reduced
  modulo `2^63 - 1`.
- The executable may read only the construction/checkpoint inputs required by
  the earlier audit; it may not read outcomes or future trajectories.
- Report all four point projections, calibration, norm bound, closure, bracket,
  hashes, and the number of outcome files read.
- Float64 point projections validate the policy geometry but are not outward
  scalar intervals or a computer-assisted exact-real proof.

No quantity above may change after the block is drawn.
