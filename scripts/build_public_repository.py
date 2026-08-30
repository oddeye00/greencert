#!/usr/bin/env python3
"""Assemble the complete, path-scrubbed public GREENCERT Git repository."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "output" / "certified_local_training_events_supplement.zip"
DESTINATION = ROOT / "output" / "github" / "greencert"

PUBLIC_FILES = (
    ".gitattributes",
    ".gitignore",
    "CITATION.cff",
    "DATA.md",
    "Dockerfile",
    "FIGURES.md",
    "LICENSE",
    "LICENSE-PAPER",
    "LITERATURE_AUDIT.md",
    "Makefile",
    "README.md",
    "REPRODUCIBILITY.md",
    "environment.yml",
    "reproduce.py",
    "requirements.in",
    "requirements.txt",
    "results/figure_reproducibility_audit.json",
    "scripts/audit_public_release.py",
    "scripts/build_arxiv_release.py",
    "scripts/build_public_repository.py",
    "scripts/check_reproduction_environment.py",
    "scripts/paper_plot_style.py",
    "scripts/reproduce_figures.py",
    "scripts/update_public_manifest.py",
    "scripts/make_transformer_v3_anytime_figure.py",
    "scripts/paper_figure_new_evidence.py",
    "scripts/paper_figure_prefix_scaling.py",
    "scripts/paper_figure_transformer_green_confirmation.py",
    "scripts/paper_figures_prospective.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def copy_relative(relative: str) -> None:
    source = ROOT / relative
    destination = DESTINATION / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def clear_readonly_and_retry(function, path: str, _error) -> None:
    """Permit replacement of a previously generated read-only staging tree."""
    os.chmod(path, stat.S_IWRITE)
    function(path)


def main() -> None:
    if not ARCHIVE.is_file():
        raise FileNotFoundError("build and verify the supplement before the public repository")

    parent = (ROOT / "output" / "github").resolve()
    destination = DESTINATION.resolve()
    if destination.parent != parent or destination.name != "greencert":
        raise RuntimeError(f"refusing unsafe public-repository target: {destination}")
    parent.mkdir(parents=True, exist_ok=True)
    if DESTINATION.exists():
        shutil.rmtree(DESTINATION, onexc=clear_readonly_and_retry)
    DESTINATION.mkdir(parents=True)

    with zipfile.ZipFile(ARCHIVE) as archive:
        archive.extractall(DESTINATION)

    # Restore the authored paper source while retaining path-scrubbed historical seals.
    for relative in (
        "paper/certified_local_training_events_neurips2026.tex",
        "paper/certified_local_training_events_arxiv.tex",
        "paper/transformer_jet_appendix.tex",
        "paper/checklist.tex",
        "paper/references.bib",
        "paper/neurips_2026.sty",
    ):
        copy_relative(relative)
    for relative in PUBLIC_FILES:
        copy_relative(relative)

    workflow_source = ROOT / ".github" / "workflows" / "reproducibility.yml"
    workflow_destination = DESTINATION / ".github" / "workflows" / "reproducibility.yml"
    workflow_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(workflow_source, workflow_destination)

    paper_pdf = ROOT / "output" / "pdf" / "greencert_arxiv.pdf"
    if not paper_pdf.is_file():
        raise FileNotFoundError("build the final arXiv PDF before the public repository")
    shutil.copyfile(paper_pdf, DESTINATION / "paper" / "greencert_arxiv.pdf")
    release = ROOT / "output" / "arxiv" / "greencert_arxiv_release.json"
    shutil.copyfile(release, DESTINATION / "paper" / "greencert_arxiv_release.json")

    files = sorted(
        path for path in DESTINATION.rglob("*") if path.is_file() and ".git" not in path.parts
    )
    manifest = {
        "format": 1,
        "repository": "https://github.com/oddeye00/greencert",
        "files": {
            path.relative_to(DESTINATION).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in files
            if path.name != "PUBLIC_MANIFEST_SHA256.json"
        },
    }
    (DESTINATION / "PUBLIC_MANIFEST_SHA256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(
        json.dumps(
            {
                "destination": str(DESTINATION),
                "files": len(manifest["files"]) + 1,
                "bytes": sum(path.stat().st_size for path in DESTINATION.rglob("*") if path.is_file()),
                "manifest_sha256": digest(DESTINATION / "PUBLIC_MANIFEST_SHA256.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
