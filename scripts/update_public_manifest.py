#!/usr/bin/env python3
"""Rewrite the exact-byte manifest after an intentional public-repo edit."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
IGNORED_ROOTS = {".git", ".venv", "output", "tmp"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def tracked_files() -> list[Path]:
    """Return only files already tracked by Git.

    Release manifests must not depend on a maintainer's untracked caches,
    scratch outputs, or local editor files.  ``git ls-files -z`` gives the
    repository's versioned publication boundary without making assumptions
    about filename whitespace or quoting.
    """

    completed = subprocess.run(
        ["git", "-c", "core.quotepath=false", "ls-files", "-z", "--cached"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    root_resolved = ROOT.resolve()
    files: list[Path] = []
    for raw_relative in completed.stdout.split(b"\0"):
        if not raw_relative:
            continue
        relative = Path(os.fsdecode(raw_relative))
        if relative.parts[0] in IGNORED_ROOTS:
            continue
        if "__pycache__" in relative.parts:
            continue
        if relative.name == "PUBLIC_MANIFEST_SHA256.json":
            continue
        path = (ROOT / relative).resolve()
        try:
            path.relative_to(root_resolved)
        except ValueError as exc:
            raise RuntimeError(f"tracked path escapes repository: {relative}") from exc
        if not path.is_file():
            raise RuntimeError(f"tracked release file is missing: {relative}")
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def main() -> None:
    files = tracked_files()
    manifest = {
        "format": 1,
        "repository": "https://github.com/oddeye00/greencert",
        "files": {
            path.relative_to(ROOT).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in files
        },
    }
    output = ROOT / "PUBLIC_MANIFEST_SHA256.json"
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "files": len(files) + 1,
                "manifest_sha256": digest(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
