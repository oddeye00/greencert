# Frozen post-seal amplified-secant probe audit

Frozen before the audit executable is run.  This is outcome-blind
method-development evidence and does not alter any prospective issuance count.

- Candidate: Transformer seed 366, threshold 0.70, anchor 1040.
- Horizon: 52 updates.
- Green power: 1.
- Amplification: 4096.
- Probe interface: response-free bound
  `beta = kappa * (sigma_sec + ||q^[lambda]||_U)`.
- Probe block: 16 iid standard-Gaussian forcing-sequence probes under the
  paper's ideal-PRNG model.
- Failure allocation: `delta = 1e-6` for this one block.
- Nonce:
  `2df178250abfd4272951e2493a1c1b93ddc7d29e73a4359246eee8998f8a0778`.
- Seed derivation: little-endian integer from the first eight bytes of
  `SHA256("greencert-response-free-secant-v1|" + nonce)`, reduced modulo
  `2^63 - 1`.
- The executable may read the sealed certificate construction, trigger/model
  artifact, and checkpoint data needed to reconstruct the centerline.  It may
  not read any outcome file or revealed future trajectory.
- It must report every projection, the calibrated norm bound, closure, bracket,
  hashes, and an explicit count of outcome files read.
- Float64 projections test the geometric interface only.  They are not outward
  scalar intervals and must not be described as a computer-assisted proof.

No parameter above may be changed after seeing the probe block.
