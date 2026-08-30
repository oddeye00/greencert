#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "audit_digits_signed_confirmation.py")],
        cwd=ROOT,
        check=True,
    )
    audit = json.loads(
        (ROOT / "results" / "digits_signed_confirmation_independent_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["status"] == "PASS"
    assert audit["signed_issued"] == audit["signed_covered"] == 7
    assert audit["unsigned_issued"] == audit["unsigned_covered"] == 6
    assert audit["signed_only"] == 1
    assert audit["signed_only_case"]["bracket"] == [147, 147]
    print("PASS: independent digits audit artifact is internally consistent.")


if __name__ == "__main__":
    main()
