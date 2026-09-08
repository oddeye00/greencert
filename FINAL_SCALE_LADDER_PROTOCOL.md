# Final bounded scale experiment

This is the prespecified final scale experiment following the completed
[451,008-parameter result](451K_COMPLETED_RESULT.md). Publication of the
protocol is not a scientific result: no registered model had been constructed
and no registered future had been inspected when this package was prepared.
Execution starts only after the immutable public protocol is available and
the release/CI checks pass.

## Frozen design

| Rung | Parameters | Width | Blocks | Seed |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1,008,288 | 144 | 4 | 94101 |
| 2 | 2,095,392 | 208 | 4 | 94101 |
| 3 | 4,944,000 | 320 | 4 | 94101 |

All rungs use mod-17 addition, four attention heads, LayerNorm, GELU,
feed-forward width four times the model width, and no dropout. All parameters
are trained. The optimizer is full-batch momentum, **not AdamW**:
learning rate 0.003, momentum 0.9, weight decay 0.01, cross-entropy loss.
The 173 training, 58 trigger and 58 certification examples are disjoint.
The shared seed is fixed across widths; these are not independent seed
replications.

Every 25 updates, through at most 12,000 updates, the selector checks the
current trigger and certification counts. It evaluates a 64-update reference
clock only when the trigger count is at least 29 and the certification count
is below 35. It selects the first clock predicting a positive-offset event:
at least 35 correct certification examples for five consecutive states.
The selector reads current certification performance; the prospective
barrier applies to the selected checkpoint's future optimizer continuation,
not to all use of the certification set.

There is at most one candidate per rung, four reference-clock sweeps and one
additional float64 recentering. Construction fixes radius `1e-8`, four Green
probes, powers 1 and 2, HVP chunk size 8, and the precision choices in the
[protocol](protocols/final_scale_ladder_v1/protocol.json).
The ideal-Gaussian failure budget is 1/10,000,000 across the three rungs,
allocated equally. It is not an empirical error-rate estimate.

## Ordering and stopping

The executable order is selection, construction, independent
arithmetic replay/disposition, future observation, and output/event audit.
Construction alone cannot authorize observation. The observer first
reaudits the sealed disposition; it records every update intent before
executing that update. There is one canonical observation directory per
rung in the controller, with no resume or retry option.

Rungs proceed in ascending order. No candidate or scientific abstention
allows the next rung; an execution/resource failure, an uncovered issued
bracket, or an observed count outside its claimed bounds stops the ladder.
All dispositions and failures are retained. No seed, threshold, precision,
budget or method is changed in response to outcomes.

Execution is CPU float64 on the recorded DGX ARM runtime. Selection and
observation use four Torch threads; numerical construction uses one.
Selection is capped at two hours per rung, construction plus disposition
at 24 hours, and observation plus its audit at one hour. Subphases share
these budgets rather than resetting them. The overall cap is 96 hours.
Workers have a 48 GiB address-space/RSS bound, a 16 GiB available-memory
reserve and a 256 GiB free-disk reserve. A complete phase artifact is
limited to 512 GiB. These are upper bounds, not runtime predictions.

## Source, protocol and runtime authentication

The production package is
[`protocols/final_scale_ladder_v1/`](protocols/final_scale_ladder_v1/).
It contains 53 Python modules, an exact-byte source manifest, the protocol
and a runtime identity record. The source bundle has no test entry point
or fixture switch. Its bootstrap authenticates all bundled source bytes
before executing the production entry module under Python `-I -B`.

| Object | SHA-256 |
| --- | --- |
| Protocol | `5bd207454eaa5807e76675077157a6f01033c9338b05e25d0e4a194746995d26` |
| Source manifest | `272780f8a1c63525fb75a00eeca7d2995c000814cc729973602545ff797a8a9e` |
| Runtime | `b11c1424fd933755cf740e49866f15ea369dcbe4ba1dce55ba65050de7785a27` |
| Bootstrap | `01b25ad50ce59701f1bbf0c7768bb0c325a98de280ce9e89b716fce4a9d425ae` |

Before reserving the run directory, the controller fetches the protocol
from a specified immutable commit in the public repository and verifies
its bytes against the external protocol pin. It also requires the recorded
runtime and source identities. A JSON file claiming publication is not
sufficient without this live byte check.

The interpreter, bootstrap, installed dependency implementations, operating
system and cooperative storage remain trusted. The controls are not remote
attestation, an adversarial sandbox, or an OS-wide prohibition on manually
launching a second experiment. Runtime version metadata does not independently
attest installed package contents.

## Reproduce the engineering checks

From a clean checkout, install the locked portable environment and run:

```sh
python -m pip install -r requirements.txt
python -B scripts/test_final_scale_release_suite_v1.py
python -B scripts/test_final_scale_public_bundle_v1.py
```

On Linux x86_64 with CPython 3.12, use `requirements-linux-ci.txt` for
installation. It retains the same versions and adds the publisher-verified
Linux MarkupSafe wheel hash missing from the historical Windows lock.
This CI-only installation correction does not change the frozen production
sources, protocol or DGX runtime; see the
[provenance record](results/final_scale_linux_ci_lock_20260908_v1.json).

The release suite has 68 tests: all passed on ARM Linux; 63 passed and five
Linux-specific supervision tests were skipped on Windows. Four public-bundle
checks also passed on Windows. The pinned release-test bundle and the
production bundle are separate: test fixtures are not production sources.
See the [preseal receipt](results/final_scale_preseal_validation_20260908_v1.json)
and its linked transcripts for exact scope and hashes.

Tests use pure control fixtures and tiny seed-0 networks, never the registered
large models. Named lifecycle integration tests use a synthetic selection
gate; unmocked replay is separately tested to reject that gate. The numerical
fixture abstains. These checks are software evidence, not additional natural
certificates.

The production runtime includes a CUDA-enabled Torch build but executes
on CPU. Its strict host/runtime admission is intentionally narrower than
portable unit testing. A new replication on another environment should
retain its own registration and identity record rather than weaken the
original seal or claim bitwise equivalence.

## Numerical and scientific scope

The constructor supplies outward neural derivative bounds and captured
local arithmetic, with independent replay of the saved arithmetic graph.
This replay does not independently redifferentiate every neural operation.
The randomized operator guarantee retains its stated ideal-Gaussian
assumption. Future outcomes are observed in float64; independent audit
re-evaluates saved point logits and event counts, not an outward enclosure
of the entire exact-real optimizer trajectory.

The target is a complete finite-evaluation-set event certificate, not
population generalization or an isolated operator benchmark. A successful
rung would extend the demonstrated parameter scale for this architecture
and optimizer; no success is asserted in this preregistration.
