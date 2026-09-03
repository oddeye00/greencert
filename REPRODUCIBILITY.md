# Reproducibility guide

This guide distinguishes a fast claim audit from a complete regeneration. The
distinction matters because the prospective experiments were defined by an
ordering constraint: methods and candidates were committed before future
outcomes were joined. A later reader can verify that chain and reproduce the
calculation, but cannot make the historical run prospective a second time.

## Reference environment

The claim-bearing release was executed with:

- Python 3.12.10;
- PyTorch 2.13.0+cpu;
- NumPy 2.5.2, SciPy 1.18.0, pandas 3.0.5;
- scikit-learn 1.9.0 and Matplotlib 3.11.1;
- mpmath 1.3.0 and python-flint 0.9.0;
- Windows 11 on an Intel i7-1360P CPU;
- float64 unless a record explicitly identifies a mixed-precision audit;
- 192-bit Arb for WDBC/digits outward replay and 256-bit Arb for the
  conditional Transformer scalar calibration/root audit.

`requirements.txt` is generated from `requirements.in` and pins the full
Python dependency closure with hashes. `environment.yml` and `Dockerfile`
provide alternative entry points. The Docker image is useful for portability;
the Windows/Python versions above remain the reference platform for exact
timing and floating-point comparisons.

PyTorch does not promise bitwise identity across releases, platforms, or CPU
and GPU backends. The repository therefore reports three different checks:
exact file hashes, tolerance-based numerical identities, and outward interval
containment. They should not be silently substituted for one another.

## Tier 1: smoke audit

Expected time: minutes on a laptop CPU.

```bash
python scripts/check_reproduction_environment.py
python reproduce.py smoke
```

This tier checks core Green recurrences, both structured parameter-forcing
implementations, variational recentering, inexact operator interfaces, event
transport, deterministic neural-jet release, and the manuscript claim ledger.
It also checks the chronological row recursion, its scaled-momentum forcing
specialization, staged disjoint-probe composition, and the deterministic
forcing-row identity used by the Transformer panel.
It also checks the polarized directional fourth-order algebra, exact block
partition, mixed-derivative autodiff majorization, and the independent
linear-cost mixed-jet implementation. The immutable-v1/maintained-v2 source
bridge additionally verifies the explicit slot-symmetrization lemma and all
three source hashes. It also verifies the transitive Python/data dependency
closure of all three directional replay entry points and all 15 exact anchor
states in the compact deterministic anchor bundle. The v1.5 tests additionally
check directional transport of stage values and parameter geometry into the
corrected-path neural envelope, including shared-geometry cache equivalence.
Its first-passage check exhaustively enumerates
109,152 valid lower/true/upper finite-window paths, including no-event cases,
and compares both independent bracket implementations. It does not retrain the
neural networks.

The historical `scripts/test_transformer_v3_preseal.py` is intentionally a
one-time pre-freeze gate: rerunning it after the archived experiment correctly
fails because the target artifacts now exist. Use the sealed no-artifact record
and Tier 3 auditors to verify that historical ordering; do not treat that
precondition check as a post-release regression test.

## Tier 2: deterministic figures

Expected time: under two minutes on the reference machine.

```bash
python scripts/reproduce_figures.py --check-determinism
```

Every paper graph is produced by Matplotlib from tracked JSON records. The
command generates all figures twice and requires byte-identical PDF and PNG
outputs. Fixed PDF metadata removes timestamp-only differences.

## Tier 3: sealed artifact replay

Expected time: tens of minutes, depending on CPU and thread count.

```bash
python reproduce.py artifact-audit
```

This tier recomputes the study summaries, both 15-case structured-Green
mechanical replays, and the independent result-side audits. Candidate and
certificate files are read before outcome joins where
the original protocol required that order. The analytic-jet release uses a
checkpoint-free consistency replay over its sealed rows; the full derivative
replay is available after regenerating the large Transformer checkpoints.
Replaying either calculation verifies the committed computation; it does not
alter or reissue a historical seal.

The frozen chronological row panel has a fast scalar/hash verifier over the 17
shipped probe-block caches:

```bash
python scripts/verify_transformer_causal_structured_row_panel.py
```

It recomputes every Gaussian calibration, causal radius, event margin, bracket,
cache identity, and matched cost total. A complete operator replay (roughly 18
minutes on the reference machine) is available from a clean checkout with:

```bash
python scripts/audit_transformer_causal_structured_row_panel.py --workers 3
```

That replay is frozen to commit `5aff03c2515e92569be81c06f0ec7aa3d2a24b42`
and intentionally refuses a dirty claim-bearing worktree.

The optimized end-to-end Transformer timing record has a separate, fast
arithmetic/provenance audit:

```bash
python scripts/audit_transformer_v3_streaming_direct_analytic.py
```

It checks three separately launched records, the bitwise full-path-to-prefix
identity bridge, the corrected-path/direct-image source panels, deterministic
analytic-jet closure, event logic, result hashes, and matched continuation
ratios. It reads no revealed outcome. Reproducing the measured execution itself
requires the regenerated seed-366 checkpoint and then runs:

```bash
python scripts/seal_transformer_streaming_prefix_identity.py
python scripts/benchmark_transformer_matched_continuation.py
python scripts/benchmark_transformer_v3_streaming_direct_analytic.py \
  --run-label local-replication
```

The committed reference median is 9.206 seconds for the certificate, versus
0.298 seconds for 26 matched direct updates and 3.718 seconds for 300. Timing
comparisons are platform-specific; bracket and hash checks are the scientific
invariants.

The v1.3 directional-remainder chain can be recomputed separately:

```bash
python scripts/transformer_directional_anchor_bundle.py
python scripts/audit_directional_replay_dependency_closure.py
python scripts/diagnose_transformer_directional_block_remainder.py --workers 3
python scripts/audit_transformer_directional_three_sweep_events.py
python scripts/audit_transformer_mixed_directional_cohort.py --workers 3
python scripts/test_transformer_directional_envelope_transport.py
python scripts/test_transformer_envelope_geometry_cache.py
python scripts/audit_transformer_directional_envelope_transport.py
```

The 15 exact parameter/velocity anchors required by these commands are shipped
in nine timestamp-fixed sparse NumPy archives totaling 3.18 MB. A separate
aggregate archive contains the same 30 arrays. Every array is linked to its
blind training record and original full checkpoint archive by SHA-256, so this
replay does not require retraining or downloading the nine 31.9 MB trajectory
files.
These commands read no revealed outcomes and draw no new randomized Green
query. They intentionally refresh tracked diagnostic JSON, including
machine-dependent timing fields, so run them in a disposable clone when an
unchanged working tree is desired. The invariant outputs are: directional no
weaker than scalar at every checkpoint, three new nondevelopment closures,
four retained sealed brackets, and maximum mixed-jet relative discrepancy at
most 3e-12. The transported-envelope audit then performs 9,420 one-sided
dominance checks and retains those same four brackets, including all three
nondevelopment cases, without reading an outcome. The original whole-case 2x
speed gate remains failed.

The v1.5 implementation is deliberately separated from the historical v3
filenames. `DIRECTIONAL_ENVELOPE_TRANSPORT_SOURCE_ISOLATION_AMENDMENT.md`
records the release-audit finding, the four restored method-seal hashes, and
the source-isolated replay. The definitive audit calls the original
`verify_method_seal()` before doing any v1.5 work.

## Tier 4: validated WDBC/digits replay

Expected time: several CPU-hours.

```bash
python reproduce.py outward
```

These commands use python-flint/Arb to propagate the stored dyadic checkpoints
under the exact-real optimizer map. Their scope starts at the checkpoint. They
do not certify the floating-point training program that produced it.

## Paper build

The arXiv source requires a LaTeX distribution with `pdflatex` and `bibtex`,
plus Poppler for metadata/render checks.

```bash
python scripts/reproduce_figures.py --check-determinism
python scripts/audit_greencert_manuscript_claims.py
python scripts/build_arxiv_release.py
```

The builder fixes `SOURCE_DATE_EPOCH`, then recompiles a clean extraction of
the generated source bundle and requires the second PDF to be byte-identical to
the release PDF. It also fails on undefined references, overfull boxes,
incorrect author metadata, an unexpected page count, or a missing vector
figure.

The structured-parameter audits also retain the exact two protocol-sealed
source files. `scripts/structured_parameter_green_source_bridge.py` checks
their hashes, replays all 48 stored closures through both sealed and maintained
implementations, and verifies the documented post-seal binary64-only bug fix;
see `STRUCTURED_PARAMETER_GREEN_SOURCE_SUPERSESSION.md`.

## Full training regeneration

The complete training runs are intentionally not part of the default CI job.
They range from minutes to many hours and some mechanism benchmarks project to
roughly 11 CPU-hours for the million-parameter operator core alone. The frozen
protocols provide the authoritative command lines and seeds:

- `REAL_DATA_GREENCERT_CONFIRMATION_PROTOCOL.md`;
- `DIGITS_SIGNED_CONFIRMATION_PROTOCOL.md`;
- `TRANSFORMER_GREEN_CONFIRMATION_PROTOCOL.md`;
- `TRANSFORMER_V3_CONFIRMATION_PROTOCOL.md`.

Complete Transformer trajectories are regenerated rather than committed. Compact
training summaries, candidate records, certificate records, outcomes, and
independent audits are all tracked. The directional study additionally ships
its 15 exact anchor states because those are sufficient for the complete v1.3
derivative replay. Given regenerated full archives, rebuild that deterministic
bundle with:

```bash
python scripts/transformer_directional_anchor_bundle.py \
  --build-from-checkpoints /path/to/full-checkpoint-run \
  --output-root /path/to/greencert-checkout
```

For the seed-366 development audits, regenerate the complete 31.9 MB archive
directly from the committed blind trigger record. Run this in a disposable
checkout because `--force` replaces the shipped sparse archive at the same
path:

```bash
python scripts/regenerate_transformer_checkpoint.py --seed 366 --force
```

The command retrains the frozen configuration, compares every trigger-visible
trajectory value and summary field with the committed blind artifact, and
writes the checkpoint archive only after those checks pass. It does not read
the separately stored certification-outcome file.

CPU libraries can reproduce the blind trajectory within the frozen numerical
tolerance without reproducing every parameter bit. The exact candidate anchor
is therefore tracked separately as two hash-locked NumPy arrays (parameter and
momentum velocity, 13,792 doubles each). To audit the cross-platform drift and
replace the regenerated archive by the exact sealed anchor used by the
diagnostic, run:

```bash
python scripts/materialize_transformer_anchor_checkpoint.py \
  --seed 366 --anchor 1120 --force
```

This emits
`results/transformer_seed_366_anchor_1120_regeneration_bridge.json`, recording
bitwise equality, maximum and Euclidean regeneration differences, all relevant
hashes, and an explicit zero outcome-read count. The subsequent causal audit
therefore separates two reproducibility claims: tolerant regeneration of the
frozen training trace and exact replay of the sealed certification anchor.

The corrected parameter path and signed parameter correction used by the
low-rank diagnostic are also stored as hash-locked arrays. The diagnostic
recomputes both from the exact anchor, requires their maximum and Euclidean
differences to stay below frozen bridge tolerances, and then evaluates neural
HVPs and output margins on the exact stored arrays. This avoids treating
cross-platform floating-point reduction order as scientific evidence while
still auditing that the executable reconstruction reaches the sealed path.
The export utility is retained for provenance but is not part of replay; it
requires the original full checkpoint archive and refuses to overwrite the
sealed arrays by default. Its interface is documented with:

```bash
python scripts/export_transformer_corrected_parameter_path.py --help
```

## Expected invariants

The release-level audit requires:

- 56/56 issued WDBC brackets covered;
- 7/7 issued digits brackets covered;
- 9/9 fixed-radius Transformer brackets covered;
- 11/11 response-centered Transformer brackets covered;
- both inaccurate finite digits forecasts rejected before a randomized query;
- 63 WDBC/digits brackets retained by independent 192-bit continuation;
- 15/15 structured parameter-forcing brackets preserved with staged logical
  Green sweeps reduced from 112 to 96;
- 15/15 anchor-zero brackets preserved, with the prespecified strict systems
  promotion gate correctly failing at 96 to 96 sweeps;
- 15/15 chronological row brackets retained, including 14/14 holdouts, with
  prefix counts 13 at four probes and two at eight, matched full
  post-reference work reduced from 144 to 98, and zero transpose sweeps;
- the directional fourth-order bound no larger at every frozen checkpoint,
  with three nondevelopment three-sweep closures and four sealed brackets
  retained;
- the independent mixed jet reproducing all 15 closure decisions and every
  local bound to maximum relative error at most 3e-12;
- the directionally transported neural envelope dominating fresh
  corrected-center evaluations in all 9,420 audit checks and retaining all
  four three-sweep brackets with zero outcome reads;
- deterministic figure regeneration;
- no local absolute paths, credential-shaped strings, or files over GitHub's
  100 MB limit in the public repository.

## Troubleshooting

- Use CPU execution for the reference path. GPU kernels can follow different
  deterministic and numerical routes.
- Do not change thread counts in benchmark comparisons. Each benchmark record
  states its thread budget.
- On Linux, system BLAS differences may change final floating-point bits even
  when all tests remain within their stated tolerances.
- If Arb wheels are unavailable for a platform, use the Docker environment or
  install `python-flint==0.9.0` in a supported Python 3.12 environment.
