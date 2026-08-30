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

This tier checks core Green recurrences, variational recentering, inexact
operator interfaces, event transport, deterministic neural-jet release, and
the manuscript claim ledger. Its first-passage check exhaustively enumerates
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

This tier recomputes the study summaries and the independent result-side
audits. Candidate and certificate files are read before outcome joins where
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

## Expected invariants

The release-level audit requires:

- 56/56 issued WDBC brackets covered;
- 7/7 issued digits brackets covered;
- 9/9 fixed-radius Transformer brackets covered;
- 11/11 response-centered Transformer brackets covered;
- both inaccurate finite digits forecasts rejected before a randomized query;
- 63 WDBC/digits brackets retained by independent 192-bit continuation;
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
