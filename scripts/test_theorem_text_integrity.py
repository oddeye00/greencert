#!/usr/bin/env python3
"""Reject control-byte and escape corruption in authored theorem notes."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "STRUCTURED_DIRECTIONAL_TWO_RESPONSE_THEOREM.md": (
        r"\bar K_H",
        r"\mathcal P",
        r"\widetilde q",
        r"\sqrt2\eta",
    ),
    "CAUSAL_STRUCTURED_RESOLVENT_THEOREM.md": (
        r"\mathcal L_J",
        r"\mathcal A^H=0",
        r"\varepsilon_s",
        r"N(I-M)^{-1}",
    ),
}


def main() -> None:
    for relative, markers in REQUIRED.items():
        path = ROOT / relative
        raw = path.read_bytes()
        bad = [
            (index, value)
            for index, value in enumerate(raw)
            if value < 32 and value not in (9, 10, 13)
        ]
        if bad:
            raise AssertionError(f"{relative} contains control bytes: {bad[:8]}")
        text = raw.decode("utf-8")
        if "\ufffd" in text:
            raise AssertionError(f"{relative} contains replacement characters")
        missing = [marker for marker in markers if marker not in text]
        if missing:
            raise AssertionError(f"{relative} is missing LaTeX markers: {missing}")
    print("PASS: theorem UTF-8/control-byte and LaTeX-marker integrity")


if __name__ == "__main__":
    main()
