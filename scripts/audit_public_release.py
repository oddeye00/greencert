#!/usr/bin/env python3
"""Fail closed on path, credential, size, provenance, and release-manifest leaks."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from pypdf import PdfReader


TEXT_SUFFIXES = {
    ".bib", ".cff", ".csv", ".in", ".json", ".md", ".py", ".sty",
    ".tex", ".txt", ".yml", ".yaml",
}
REQUIRED = {
    "README.md",
    "REPRODUCIBILITY.md",
    "DATA.md",
    "FIGURES.md",
    "LICENSE",
    "LICENSE-PAPER",
    "LITERATURE_AUDIT.md",
    "CITATION.cff",
    "requirements.in",
    "requirements.txt",
    "PUBLIC_MANIFEST_SHA256.json",
    "paper/greencert_arxiv.pdf",
    "scripts/reproduce_figures.py",
    "scripts/paper_plot_style.py",
    "scripts/update_public_manifest.py",
}
FIGURES = (
    "paper_transformer_v3_anytime",
    "paper_real_data_confirmation",
    "paper_mechanism_scaling",
    "paper_relinearized_prefix_panel",
    "paper_transformer_green_confirmation",
    "paper_prospective_horizons",
    "paper_prospective_brackets",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()

    missing = sorted(name for name in REQUIRED if not (root / name).is_file())
    if missing:
        raise AssertionError(f"public release is missing required files: {missing}")

    bad_paths: list[str] = []
    bad_secrets: list[str] = []
    oversized: list[dict[str, int | str]] = []
    forbidden_extensions: list[str] = []
    local_patterns = (
        re.compile(r"[A-Za-z]:\\Users\\[^<\\/\s]+", re.IGNORECASE),
        re.compile("/" + r"home/[^/<\s]+/"),
        re.compile("/" + r"Users/[^/<\s]+/"),
    )
    secret_patterns = (
        re.compile(r"ghp_[A-Za-z0-9]{20,}"),
        re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile("-----" + "BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    )
    blocked = {".ckpt", ".pem", ".pt", ".pth"}

    ignored_roots = {".git", ".venv", "output", "tmp"}
    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).parts[0] not in ignored_roots
        and "__pycache__" not in path.parts
    ]
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        if size >= 95 * 1024 * 1024:
            oversized.append({"path": relative, "bytes": size})
        if path.suffix.lower() in blocked:
            forbidden_extensions.append(relative)
        if path.suffix.lower() in TEXT_SUFFIXES or path.name.startswith("LICENSE") or path.name in {"Makefile", "Dockerfile"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            if any(pattern.search(text) for pattern in local_patterns):
                bad_paths.append(relative)
            if any(pattern.search(text) for pattern in secret_patterns):
                bad_secrets.append(relative)

    if bad_paths or bad_secrets or oversized or forbidden_extensions:
        raise AssertionError(
            json.dumps(
                {
                    "local_path_leaks": bad_paths,
                    "credential_shaped_strings": bad_secrets,
                    "oversized": oversized,
                    "blocked_extensions": forbidden_extensions,
                },
                indent=2,
            )
        )

    figure_rows = {}
    for stem in FIGURES:
        pdf = root / "figures" / f"{stem}.pdf"
        png = root / "figures" / f"{stem}.png"
        if not pdf.is_file() or not png.is_file():
            raise FileNotFoundError(stem)
        metadata = PdfReader(pdf).metadata or {}
        creator = str(metadata.get("/Creator", ""))
        if not creator.startswith("Matplotlib "):
            raise AssertionError(f"figure lacks Matplotlib provenance: {stem}")
        figure_rows[stem] = {
            "creator": creator,
            "pdf_sha256": digest(pdf),
            "png_sha256": digest(png),
        }

    manifest = json.loads((root / "PUBLIC_MANIFEST_SHA256.json").read_text(encoding="utf-8"))
    manifest_files = set(manifest["files"])
    actual_files = {
        path.relative_to(root).as_posix()
        for path in files
        if path.name != "PUBLIC_MANIFEST_SHA256.json"
    }
    if manifest_files != actual_files:
        raise AssertionError("public manifest and repository payload differ")
    for relative, row in manifest["files"].items():
        path = root / relative
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise AssertionError(f"public manifest mismatch: {relative}")

    report = {
        "status": "public release audit passed",
        "root": str(root),
        "files": len(files),
        "largest_file_bytes": max(path.stat().st_size for path in files),
        "local_path_leaks": 0,
        "credential_shaped_strings": 0,
        "blocked_checkpoint_or_key_files": 0,
        "matplotlib_figures": figure_rows,
        "manifest_verified": True,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
