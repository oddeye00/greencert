#!/usr/bin/env python3
"""Reject a NeurIPS build if main-text prose spills beyond page nine."""
from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
PDFS = (
    PAPER / "certified_local_training_events_neurips2026_blind.pdf",
    PAPER / "certified_local_training_events_neurips2026.pdf",
)


def main() -> None:
    for path in PDFS:
        reader = PdfReader(path)
        if len(reader.pages) < 10:
            raise AssertionError(f"unexpected short PDF: {path.name}")
        page_nine = (reader.pages[8].extract_text() or "").strip()
        page_ten = (reader.pages[9].extract_text() or "").strip()
        if "Conclusion" not in page_nine:
            raise AssertionError(f"conclusion missing from page nine: {path.name}")
        if not page_ten.startswith("References"):
            raise AssertionError(
                f"main text spills onto page ten before References: {path.name}"
            )
        earlier = "\n".join(
            (page.extract_text() or "") for page in reader.pages[:9]
        )
        if "References" in earlier:
            raise AssertionError(f"references begin before page ten: {path.name}")
        print(f"PASS: {path.name}: conclusion ends on page 9; references start page 10")


if __name__ == "__main__":
    main()
