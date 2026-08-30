# WDBC direct validated-continuation baseline

Status: **PASS**. This is a post-seal matched comparator; no sealed artifact changed.

The existing 192-bit Arb pass is a direct one-step validated trajectory method,
not merely a floating-point replay of the Green inequality. Starting from zero
radius at the exact dyadic checkpoint, it propagates verified optimizer defects,
Jacobian norms, Hessian-drift remainders, and output margins. The recurrence
contains no Green radius, randomized probe, or PRNG dependence.

- Matched issued events: **56**.
- Direct outward brackets: **56**, with **56/56** containment.
- Brackets identical to GreenCert: **56**.
- Unique tubes / transitions: **40 / 5,089**.
- Maximum state radius: **4.36465e-13**.
- Minimum strict output slack: **2.37531e-05**.
- Direct outward aggregate time: **7.12 h**; median **204.72 s/tube**.
- GreenCert matched event-record time: **33.88 min**.
- Aggregate direct/GreenCert runtime ratio: **12.61x**.

The timing ratio is conservative for the direct comparator because its timer
excludes construction of the shared four-sweep centerline, whereas the GreenCert
event-record timers include centerline construction. The comparator was run only
on the 56 Green-issued coordinates, so it establishes matched rigor and cost, not
availability on the 15 Green abstentions.

Consequently, the ideal-Gaussian failure budget remains part of the original
Green issuance route but is not a condition of the 56 outward-retained WDBC
brackets. Those are deterministic exact-real continuation statements from stored
binary checkpoints, not end-to-end proofs of the preceding training program.
