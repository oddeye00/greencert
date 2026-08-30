#!/usr/bin/env python3
"""Regenerate every GREENCERT paper figure and optionally prove byte stability."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
GENERATORS = (
    "scripts/make_transformer_v3_anytime_figure.py",
    "scripts/paper_figure_new_evidence.py",
    "scripts/paper_figure_prefix_scaling.py",
    "scripts/paper_figure_transformer_green_confirmation.py",
    "scripts/paper_figures_prospective.py",
)
STEMS = (
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


def generate() -> None:
    for relative in GENERATORS:
        subprocess.run([sys.executable, relative], cwd=ROOT, check=True)


def snapshot() -> dict[str, dict[str, str | int]]:
    rows: dict[str, dict[str, str | int]] = {}
    for stem in STEMS:
        for suffix in ("pdf", "png"):
            path = ROOT / "figures" / f"{stem}.{suffix}"
            if not path.is_file():
                raise FileNotFoundError(path)
            rows[path.relative_to(ROOT).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
        reader = PdfReader(ROOT / "figures" / f"{stem}.pdf")
        if len(reader.pages) != 1:
            raise AssertionError(f"{stem}.pdf is not a one-page vector figure")
        metadata = reader.metadata or {}
        if metadata.get("/Author") != "Ian Rhee":
            raise AssertionError(f"missing figure author metadata: {stem}")
        if not str(metadata.get("/Creator", "")).startswith("Matplotlib "):
            raise AssertionError(f"non-Matplotlib figure provenance: {stem}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-determinism", action="store_true")
    args = parser.parse_args()

    generate()
    first = snapshot()
    if args.check_determinism:
        generate()
        second = snapshot()
        if first != second:
            changed = sorted(name for name in first if first[name] != second[name])
            raise AssertionError(f"non-deterministic figure outputs: {changed}")

    result = {
        "status": "paper figure reproduction passed",
        "backend": "Matplotlib",
        "generators": list(GENERATORS),
        "generated_twice": bool(args.check_determinism),
        "byte_identical": True if args.check_determinism else None,
        "outputs": first,
    }
    output = ROOT / "results" / "figure_reproducibility_audit.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

