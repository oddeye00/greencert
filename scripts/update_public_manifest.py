#!/usr/bin/env python3
"""Rewrite the exact-byte manifest after an intentional public-repo edit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IGNORED_ROOTS = {".git", ".venv", "output", "tmp"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.relative_to(ROOT).parts[0] not in IGNORED_ROOTS
        and "__pycache__" not in path.parts
        and path.name != "PUBLIC_MANIFEST_SHA256.json"
    )
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
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
