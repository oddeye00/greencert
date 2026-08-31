#!/usr/bin/env python3
"""Build and verify the authored GREENCERT arXiv PDF and minimal source bundle."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT_DIR = ROOT / "output" / "arxiv"
OUT_PDF = ROOT / "output" / "pdf" / "greencert_arxiv.pdf"
OUT_ZIP = OUT_DIR / "greencert_arxiv_source.zip"
OUT_MANIFEST = OUT_DIR / "greencert_arxiv_release.json"
JOB = "certified_local_training_events_arxiv"

SOURCE_MAP = {
    PAPER / f"{JOB}.tex": Path(f"{JOB}.tex"),
    PAPER / "certified_local_training_events_neurips2026.tex": Path(
        "certified_local_training_events_neurips2026.tex"
    ),
    PAPER / "transformer_jet_appendix.tex": Path("transformer_jet_appendix.tex"),
    PAPER / "checklist.tex": Path("checklist.tex"),
    PAPER / "references.bib": Path("references.bib"),
    PAPER / "neurips_2026.sty": Path("neurips_2026.sty"),
    ROOT / "figures" / "paper_transformer_v3_anytime.pdf": Path(
        "figures/paper_transformer_v3_anytime.pdf"
    ),
    ROOT / "figures" / "paper_real_data_confirmation.pdf": Path(
        "figures/paper_real_data_confirmation.pdf"
    ),
    ROOT / "figures" / "paper_transformer_green_confirmation.pdf": Path(
        "figures/paper_transformer_green_confirmation.pdf"
    ),
    ROOT / "figures" / "paper_relinearized_prefix_panel.pdf": Path(
        "figures/paper_relinearized_prefix_panel.pdf"
    ),
    ROOT / "figures" / "paper_mechanism_scaling.pdf": Path(
        "figures/paper_mechanism_scaling.pdf"
    ),
    ROOT / "figures" / "paper_prospective_horizons.pdf": Path(
        "figures/paper_prospective_horizons.pdf"
    ),
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def run(command: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed.stdout + completed.stderr


def deterministic_zip(path: Path, payloads: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(payloads):
            info = zipfile.ZipInfo(name, date_time=(2026, 8, 29, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payloads[name])


def main() -> None:
    missing = [str(source) for source in SOURCE_MAP if not source.is_file()]
    if missing:
        raise FileNotFoundError("missing arXiv source dependencies: " + ", ".join(missing))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "tmp").mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="greencert_arxiv_", dir=ROOT / "tmp") as tmp:
        stage = Path(tmp)
        for source, relative in SOURCE_MAP.items():
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.name == "certified_local_training_events_neurips2026.tex":
                text = source.read_text(encoding="utf-8").replace(
                    "../figures/", "figures/"
                )
                destination.write_text(text, encoding="utf-8", newline="\n")
            else:
                shutil.copyfile(source, destination)

        latex = [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            f"{JOB}.tex",
        ]
        run(latex, stage)
        run(["bibtex", JOB], stage)
        run(latex, stage)
        run(latex, stage)

        log = (stage / f"{JOB}.log").read_text(encoding="utf-8", errors="replace")
        forbidden_log_fragments = (
            "LaTeX Warning: There were undefined references",
            "LaTeX Warning: Citation",
            "multiply defined",
            "Overfull \\hbox",
            "Overfull \\vbox",
        )
        found = [fragment for fragment in forbidden_log_fragments if fragment in log]
        if found:
            raise AssertionError(f"arXiv build log contains release-blocking warnings: {found}")

        pdf = stage / f"{JOB}.pdf"
        info = run(["pdfinfo", str(pdf)], stage)
        metadata = {}
        for line in info.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()
        if metadata.get("Author") != "Ian Rhee":
            raise AssertionError(f"unexpected PDF author: {metadata.get('Author')!r}")
        if metadata.get("Pages") != "39":
            raise AssertionError(f"unexpected arXiv page count: {metadata.get('Pages')!r}")

        bbl = stage / f"{JOB}.bbl"
        readme = (
            "GREENCERT arXiv source release\n"
            "================================\n\n"
            f"Compile with: pdflatex {JOB}.tex; bibtex {JOB}; "
            f"pdflatex {JOB}.tex; pdflatex {JOB}.tex\n\n"
            "The included .bbl permits direct LaTeX compilation as well. "
            "All figures are vector PDFs.\n"
        ).encode("utf-8")

        payloads: dict[str, bytes] = {
            relative.as_posix(): (stage / relative).read_bytes()
            for relative in SOURCE_MAP.values()
        }
        payloads[f"{JOB}.bbl"] = bbl.read_bytes()
        payloads["00README.txt"] = readme
        source_manifest = {
            name: {"bytes": len(payload), "sha256": sha256(payload)}
            for name, payload in sorted(payloads.items())
        }
        payloads["SOURCE_MANIFEST_SHA256.json"] = (
            json.dumps(source_manifest, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")

        deterministic_zip(OUT_ZIP, payloads)
        shutil.copyfile(pdf, OUT_PDF)

    release = {
        "title": "GreenCert: Signed Green Operators for Certified Neural Training Transitions",
        "author": "Ian Rhee",
        "pages": 39,
        "pdf": {
            "path": OUT_PDF.relative_to(ROOT).as_posix(),
            "bytes": OUT_PDF.stat().st_size,
            "sha256": sha256(OUT_PDF.read_bytes()),
        },
        "source_bundle": {
            "path": OUT_ZIP.relative_to(ROOT).as_posix(),
            "bytes": OUT_ZIP.stat().st_size,
            "sha256": sha256(OUT_ZIP.read_bytes()),
            "files": len(payloads),
        },
        "supplement": {
            "path": "output/certified_local_training_events_supplement.zip",
            "bytes": (ROOT / "output" / "certified_local_training_events_supplement.zip").stat().st_size,
            "sha256": sha256(
                (ROOT / "output" / "certified_local_training_events_supplement.zip").read_bytes()
            ),
        },
        "build_checks": {
            "author_metadata": True,
            "page_count": True,
            "undefined_references": False,
            "overfull_boxes": False,
            "source_bundle_recompiled": True,
        },
    }
    OUT_MANIFEST.write_text(
        json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(release, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
