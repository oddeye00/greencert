# GREENCERT

[![reproducibility](https://github.com/oddeye00/greencert/actions/workflows/reproducibility.yml/badge.svg)](https://github.com/oddeye00/greencert/actions/workflows/reproducibility.yml)

This repository contains the paper, theorem records, sealed experimental
artifacts, source code, and independent audits for **GREENCERT: Signed Green
Operators for Certified Neural Training Transitions**.

GREENCERT starts from a realized training checkpoint and a local reference
path. It propagates the path's signed defect through the causal variational
dynamics, recenters on that response, and bounds the remaining nonlinear
error. The output is a persistent first-passage bracket or an abstention.

The archived studies contain 83 issued event brackets across WDBC, handwritten
digits, and two Transformer cohorts. All 83 revealed crossings fall inside the
issued brackets. The 63 WDBC/digits brackets also survive independent 192-bit
outward continuation.

## Start here

- The current preprint is [`paper/greencert_arxiv.pdf`](paper/greencert_arxiv.pdf).
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) explains the audit levels,
  expected runtimes, and exact commands.
- [`DATA.md`](DATA.md) records dataset origins, licenses, and hashes.
- [`LITERATURE_AUDIT.md`](LITERATURE_AUDIT.md) records the dated primary-source
  novelty search and object-level comparisons.
- [`FIGURES.md`](FIGURES.md) maps every paper figure to a Python/Matplotlib
  generator and its input records.
- [`SUPPLEMENT_README.md`](SUPPLEMENT_README.md) documents the full sealed
  artifact chain.

## Quick verification

The reference environment is Python 3.12 on CPU. From a clean clone:

```bash
python -m venv .venv
```

Activate the environment, then run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_reproduction_environment.py
python reproduce.py smoke
python scripts/reproduce_figures.py --check-determinism
```

Windows activation:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS activation:

```bash
source .venv/bin/activate
```

The Docker route fixes the Python and system-tooling layer:

```bash
docker build -t greencert-repro .
docker run --rm greencert-repro
```

## Reproduction levels

The repository separates four tasks that are often conflated:

1. **Artifact replay** recomputes brackets and summary claims from sealed
   records without retraining. This is the fastest and most portable audit.
2. **Validated replay** reruns the independent 192-bit WDBC/digits checks.
3. **Figure and manuscript build** regenerates all Matplotlib figures and the
   arXiv PDF from tracked inputs.
4. **Training regeneration** recreates checkpoints and candidate records. It
   is substantially more expensive and cannot retroactively recreate the
   prospective status of an already completed experiment.

Run `python reproduce.py --list` for the available entry points.

## What is and is not certified

The certificates concern deterministic persistent events on fixed evaluation
sets along realized optimizer trajectories. They are not population
generalization bounds. WDBC and digits have independent outward continuation
from stored dyadic checkpoints. Transformer certificates use the committed
float64 and ideal-Gaussian/PRNG model described in the paper and seals.

This distinction is part of the result, not fine print: a local clock proposes
an event time; GREENCERT decides whether that proposal is supported by the
finite-window nonlinear dynamics.

## Repository integrity

`PUBLIC_MANIFEST_SHA256.json` records every published file. Historical method,
candidate, and certificate seals retain their historical source hashes. Two
Transformer method seals replace a machine-local path with `<ROOT>` in the
public tree; `MANIFEST_SHA256.json` records and validates both the historical
source hash and the packaged hash.

## Citation and license

Citation metadata are in [`CITATION.cff`](CITATION.cff). Code is released under
the [`MIT License`](LICENSE). The paper text and original figures are released
under [`CC BY 4.0`](LICENSE-PAPER). Dataset-specific notices appear in
[`DATA.md`](DATA.md).
