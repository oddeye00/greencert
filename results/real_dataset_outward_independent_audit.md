# Independent audit of the WDBC outward verification

Status: **PASS**.

The audit verified all 40 Arb cache records and all 5089
enclosed state transitions, checked cache-to-certificate hashes, recomputed the
stored radius recurrences, confirmed every requested horizon was reached, and
rejoined the outward brackets to the sealed outcomes.

- Green-issued events: 56.
- Outward-retained events: 56.
- Outward-covered events: 56 across 22 seeds.
- Brackets identical to the Green float64 brackets: 56/56, all singletons.
- Maximum outward state radius: 4.36465e-13.
- Minimum outward output-logic slack: 2.37531e-05.
- Maximum verified one-step optimizer Jacobian norm: 1.00743.
- Maximum exact-real reference-defect enclosure: 1.8864e-15.
- Arb precision: 192 bits; python-flint 0.9.0.

This is a post-seal numerical verification, not a new prospectively selected
experiment. It closes the finite-precision gap for the 56 issued real-data
events by certifying the exact-real optimizer map around the exact binary
checkpoint/reference values with a direct outward tube independent of the
probabilistic Green radius.
