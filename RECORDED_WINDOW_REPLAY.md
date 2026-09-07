# Read-only recorded-window replay

The source package in this note lets an independent reader test the
read-only replay components without downloading a neural evidence graph.
It contains 47 source files, including their unchanged reference auditors,
and generates small exact-quadratic and provenance fixtures at runtime.
It contains no future optimizer trajectory and adds no empirical event to
the manuscript.

## Reproduce the component tests

Use the repository's pinned Python environment and run:

```text
python scripts/extract_recorded_replay_sources.py --extract output/recorded_replay_tests
python output/recorded_replay_tests/scripts/test_recorded_window_components.py
python output/recorded_replay_tests/scripts/test_recorded_evidence_replay.py
```

The extractor requires a fresh destination and authenticates the archive,
its member population, and every source before exposing code. Archive:
`artifacts/greencert_recorded_replay_sources_20260907.zip`, 128,436 bytes.
SHA256: `c799935f80b06e6bfd86e71031c0b4b8bdf74264d570aea83319ef1e39c0a2de`.
The embedded manifest hash is
`c840c041d217dd0d4a76dd67184dac760c2de2477198da6da6bfaefe21113307`.

The same tests passed on Windows/AMD64 and Linux/aarch64, with Python 3.12,
NumPy 2.5.2, and python-flint 0.9.0. Raw count reports are
`results/recorded_window_component_tests_windows_20260907.json` and
`results/recorded_window_component_tests_arm_20260907.json`.

| Check | Cases per host |
| --- | ---: |
| Both integer norm implementations agree with a rational oracle | 15 |
| Mixed-runtime Green families equal the original auditor | 4 |
| Anchor/response evidence equals the original auditor | 4 |
| Homogeneous-response evidence equals the original auditor | 6 |
| Coherently rehashed semantic corruptions refused | 113 |
| Forbidden outcome-input graphs refused | 8 |
| Existing but unlisted synthetic future left unread | 4 |

The separate portable-reader suite rejects 31 malformed paths, aliases,
foreign or missing digests, unrecorded reads, and malformed JSON cases.
These are implementation regressions, not new neural certificates.

## What changes in the replay

The original construction and one-shot issuance guards are untouched.
Separate adapters retain the anchor, optimizer, source, recurrence-index,
probe, role, and residual bindings, while checking saved vector norms by
exact integer inequalities. This checks the mathematical upper bound
without requiring the particular rounded value produced by an Arb loop.
All saved scalar bounds are retained.

The repository already contained a base-2^18 integer-limb norm prototype,
`exact_dyadic_l2.py`. The newer `exact_dyadic_norm.py` uses chunked
base-2^32 bins and compares the saved inequality directly. The former sorts
the entire vector by exponent; the latter bounds its accumulator workspace
by the chunk size. Neither is a new summation principle. The cross-check
above compares both to an independent rational oracle; the published
17.9x/20.8x benchmarks compare the newer checker with the original Arb loop,
not with the earlier integer prototype.

A recorded replay may run after an outcome file exists. That file is
excluded from both the permitted input graph and the observed read set.
Replay therefore does not reestablish historical prospectivity or create a
new prospective result. It never authorizes an outcome observation.

## Full neural-window replay

The archive also includes `replay_recorded_window.py`. Given an externally
pinned, complete recorded evidence graph, it recomputes physical anchor
encoding, signed-response combination, saved norm inequalities, the Green
bound, closure, and output-event assembly. It then rehashes the entire
recorded graph. Original neural gradient/HVP/envelope production remains a
separate validated-producer obligation; this driver authenticates those
records rather than rerunning every neural kernel.

The complete 451,008-parameter development graph is about 11.75 GB and is
not included in this source-only archive. Its full replay is consequently
not reproduced by the three commands above. Distribution of that graph is
a separate outstanding release task. No larger-model outcome or new
coverage count is claimed by this component package.
