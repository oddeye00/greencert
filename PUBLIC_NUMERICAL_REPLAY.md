# Corrected numerical replay artifact

The public artifact contains every affected historical continuation: 63 unique
jobs supporting 79 brackets (56 WDBC, 7 digits, 16 modular-addition appendix
events). The 16 appendix events do not increase the paper's 83-event headline
population. This is an arithmetic repair of existing claims, not another seed
batch or a prospective experiment.

Archive: `artifacts/greencert_repaired_continuation_20260907.zip` (28,866,197 bytes,
253 members). SHA-256:
`77f3c2b7cfe694588de33e01f4bd949729aa2676696300846cca85eef3105cbc`.

Manifest SHA-256:
`2ea5e687f173a7f7f0f6c31bdc9b435c628b8f74ca2172d771ef33a77c3109cd`.

## Run it

`python reproduce.py outward` runs the corrected primitive regressions and
all 63 neural continuations in a fresh retained directory. For a quicker
output-only check use `python scripts/replay_corrected_continuation.py --mode outputs`.
The lower-level commands below expose the same authenticated package directly.

From the repository root, authenticate and extract the package:

```
python scripts/read_public_repair_archive.py --extract output/repaired_continuation
```

Then change to `output/repaired_continuation`. The included README describes
the arithmetic model, pinned dependencies, dataset provenance, and three
replay modes. For example:

```
python scripts/replay_public_repair.py --manifest-sha256 2ea5e687f173a7f7f0f6c31bdc9b435c628b8f74ca2172d771ef33a77c3109cd --mode outputs --report outputs.json
```

`ledger` independently reconstructs exact-rational scalar inequalities and
integer event brackets from recorded numerical premises. `outputs` also
recomputes every evaluation margin at 384-bit Arb precision. `neural` additionally
recomputes the state tube with the unchanged corrected neural kernels. Use
one CPU thread per numerical library. The neural mode is the expensive mode;
`--job` selects a fixed ordinal and `--steps` explicitly limits a neural smoke
test, which must not be reported as a full-window verification.

## What was checked on the public package

Both Windows/x86 and Linux/ARM isolated replays retain all 79 brackets,
covering 7,551 exact scalar transitions and 1,485,052 freshly recomputed output
margins per platform. Their per-job results agree exactly. Access to private
experiment directories was blocked and the guards were tested. The artifact
scan finds no machine-local paths, credential-shaped strings, blocked file
types, or oversized individual members.

The first fixed job of each study (ordinals 0, 40, 47) additionally passed full
neural-bound recomputation on ARM: three complete windows, 126 transitions,
39,149 output margins, and four retained brackets. This tests all three public
loader/kernel interfaces; it is not a second full 63-job derivative audit.
The complete derivative recomputation is the underlying arithmetic-repair
record from which the public package was exported.

Run `python scripts/summarize_public_repair_validation.py` to authenticate the
included replay reports and reproduce this summary. The figure generator
uses the same public artifact for corrected bracket coordinates; no private
ledger is needed to draw it.

`python scripts/test_public_repair_archive.py` checks exact extraction and
refusal of modified archives. `python scripts/test_public_repair_roundoff.py`
tests both packaged matrix-product backends against exact-rational interval
oracles, including eight dot lengths with accumulated subnormal rounding.
These regressions and the numerical replays are included in the CI workflow.

Per-job timings, platform labels and original strict-output-slack descriptors
are supplied separately in `results/public_repair_descriptors_20260907.json`.
Run `python scripts/audit_public_repair_statistics.py` to reconstruct the
appendix's descriptive statistics from the public arrays and these fields.
The result matches the original summary exactly. The frozen replay ZIP has
not been changed to add this metadata, and the timings are not a controlled
comparison between platforms.

## Scope and provenance

`paper/greencert_supplement.zip` is preserved as the historical anonymous
supplement; its bundled manuscript and legacy arithmetic are not the current
release. Use `paper/greencert_arxiv_source.zip` for the current paper and the
corrected numerical archive above for the repaired continuation calculations.

The correction adds an absolute subnormal-rounding allowance to the legacy
relative-only matrix-product enclosure and verifies the affected neural
bounds again. No observed historical neural bracket was disproved; all 79
survive the corrected computation. The old source and evidence are preserved
as historical records, not silently overwritten.

The new artifact schema preserves exact numerical arrays, unchanged kernel
source hashes, event coordinates, and original method/summary identities.
It omits private orchestration metadata. Its own hashes identify new public
packaging; they do not pretend to be the hashes of redacted original files.

The continuation concerns the exact-real optimizer from a stored dyadic
checkpoint, conditional on the stated Arb and ordinary-binary64 arithmetic
assumptions. It is not a proof of the preceding floating-point training
program, a population-generalization guarantee, or a numerical upgrade of
the separate Transformer study.
