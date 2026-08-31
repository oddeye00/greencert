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

The build fails on undefined references, overfull boxes, incorrect author
metadata, an unexpected page count, or a missing vector figure.

## Full training regeneration

The complete training runs are intentionally not part of the default CI job.
They range from minutes to many hours and some mechanism benchmarks project to
roughly 11 CPU-hours for the million-parameter operator core alone. The frozen
protocols provide the authoritative command lines and seeds:

- `REAL_DATA_GREENCERT_CONFIRMATION_PROTOCOL.md`;
- `DIGITS_SIGNED_CONFIRMATION_PROTOCOL.md`;
- `TRANSFORMER_GREEN_CONFIRMATION_PROTOCOL.md`;
- `TRANSFORMER_V3_CONFIRMATION_PROTOCOL.md`.

Large Transformer checkpoints are regenerated rather than committed. Compact
training summaries, candidate records, certificate records, outcomes, and
independent audits are all tracked.

For the seed-366 development audits, regenerate the omitted 31.9 MB archive
directly from the committed blind trigger record:

```bash
python scripts/regenerate_transformer_checkpoint.py --seed 366
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
