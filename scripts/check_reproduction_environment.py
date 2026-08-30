#!/usr/bin/env python3
"""Report and validate the reference GREENCERT reproduction environment."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys


EXPECTED = {
    "matplotlib": "3.11.1",
    "mpmath": "1.3.0",
    "numpy": "2.5.2",
    "pandas": "3.0.5",
    "pillow": "12.3.0",
    "pypdf": "6.16.1",
    "python-flint": "0.9.0",
    "scikit-learn": "1.9.0",
    "scipy": "1.18.0",
    "torch": "2.13.0+cpu",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-version-drift", action="store_true")
    args = parser.parse_args()

    installed: dict[str, str | None] = {}
    mismatches: dict[str, dict[str, str | None]] = {}
    for distribution, expected in EXPECTED.items():
        try:
            value = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            value = None
        installed[distribution] = value
        if value != expected:
            mismatches[distribution] = {"expected": expected, "installed": value}

    python_ok = sys.version_info[:2] == (3, 12)
    tools = {
        name: shutil.which(name)
        for name in ("git", "pdflatex", "bibtex", "pdfinfo", "pdftoppm")
    }
    try:
        import torch

        torch.use_deterministic_algorithms(True)
        torch_record = {
            "cuda_available": bool(torch.cuda.is_available()),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "threads": int(torch.get_num_threads()),
        }
    except Exception as exc:  # pragma: no cover - diagnostic path
        torch_record = {"error": repr(exc)}

    report = {
        "python": platform.python_version(),
        "python_reference_minor": "3.12",
        "python_minor_matches": python_ok,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": installed,
        "package_mismatches": mismatches,
        "external_tools": tools,
        "torch": torch_record,
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    if not args.allow_version_drift and (not python_ok or mismatches):
        raise SystemExit(
            "reference environment mismatch; pass --allow-version-drift only for a "
            "documented portability test"
        )


if __name__ == "__main__":
    main()

