# Frozen four-probe corrected-path protocol

Frozen on 2026-08-29 before its Gaussian block was instantiated.

The preceding 16-probe corrected-path secant audit removed the old mixed
coefficient and measured 173.733 times injection-forcing headroom.  This audit
tests the prespecified practical reduction suggested by that slack: retain the
same candidate, corrected path, amplification, derivative envelope, power,
failure allocation, outward forcing bound, output rule, and event, but reduce
the rebuilt-Green block from 16 probes to four.

## Fixed choices

- Candidate: seed 366, 70% gate, anchor 1040.
- Horizon 52; four centerline sweeps; corrected path `b=c+K_Hs`.
- Amplified secant: `lambda=4096`.
- Corrected-path Green power: one.
- Fresh probes: four.
- Per-operator failure allocation: `4.59896983075791e-11`.
- Closure: `Y + kappa_bar*M*E^2/2 <= E` with no mixed term.
- Event and output transport: unchanged power-one 70% persistent rule.
- Master nonce:
  `611fda4bd0aa71d5a3ea2c4158a103cb32330ed279660ffe9dc35232aea14360`.
- Operator identity: `(93,366,0,1040,52,4,1)`.
- No future outcome file may be read.

## Frozen hashes

- Four-probe wrapper:
  `893CDBFDDA9D9AB1E53FE8D9F72D242E59E2F3B0FCB9631C7653CE150333CF1A`
- Base corrected-path implementation:
  `D6459D702F62AF11427FA26E197AA0C8F5D6D2EFEFC1C05C36FEA1EB0E90C8F6`
- Preceding 16-probe result:
  `78AC1D3031EC2B0EC84B2F78D26AFD2090F7AEED911ACED0314D9CF8E3CAC49E`
- Closure module:
  `ACFB21152D0467E6DDAAB627138325B2079C4F75C1F8BD7FF708C427F67FE5E1`
- Corrected-path theorem:
  `2E4928E4894B50FBD583B69596828ED34B4B48CA7A644B0960531E785C4F05F3`

This is post-seal method development.  Success would establish a measured
operator-count reduction on this sealed case, not prospective cohort-wide
issuance or modern-scale practicality.
