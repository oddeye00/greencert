"""Create a new public snapshot from tracked files and an explicit repair list.

No Git index, historical record, or existing destination is changed. The
snapshot is for audit; constructing it does not authorize publication.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
EXTRA = (
    "PUBLIC_NUMERICAL_REPLAY.md",
    "NUMERICAL_REPAIR_RESOLUTION.md",
    "artifacts/greencert_repaired_continuation_20260907.zip",
    "audit_history/numerical_release_hold_20260907.json",
    "audit_history/numerical_repair_resolution_20260907.json",
    "scripts/read_public_repair_archive.py",
    "scripts/summarize_public_repair_validation.py",
    "scripts/test_public_repair_archive.py",
    "scripts/test_public_repair_roundoff.py",
    "scripts/audit_public_repair_statistics.py",
    "scripts/stage_repaired_public_release.py",
    "scripts/scan_public_snapshot.py",
    "scripts/test_public_snapshot_scan.py",
    "scripts/test_reproducibility_fail_fast.py",
    "EXACT_DYADIC_NORM_AUDIT.md",
    "RECORDED_WINDOW_REPLAY.md",
    "scripts/extract_recorded_replay_sources.py",
    "artifacts/greencert_recorded_replay_sources_20260907.zip",
    "artifacts/greencert_recorded_replay_sources_20260907_v2.zip",
    "scripts/check_recorded_reader_short_paths.py",
    "results/recorded_window_component_tests_windows_20260907.json",
    "results/recorded_window_component_tests_arm_20260907.json",
    "scripts/exact_dyadic_norm.py",
    "scripts/test_exact_dyadic_norm.py",
    "scripts/extract_saved_norm_benchmark.py",
    "artifacts/greencert_exact_norm_benchmark_20260907.zip",
    "results/exact_dyadic_norm_benchmark_windows_20260907.json",
    "results/exact_dyadic_norm_portable_windows_20260907.json",
    "results/exact_dyadic_norm_benchmark_arm_20260907.json",
    "scripts/replay_corrected_continuation.py",
    "results/public_repair_validation_20260907.json",
    "results/public_repair_statistics_audit_20260907.json",
    "results/public_repair_descriptors_20260907.json",
    "results/repaired_numerics_manuscript_summary_20260907.json",
    "results/repair_claim_reconciliation_20260907T143307137809Z.json",
    "results/legacy_subnormal_matmul_audit_20260907T015437.656557+0000.json",
    "results/binary64_interval_products_20260907T015903.440412+0000.json",
    "results/public_repair_ledger_windows_20260907.replay.json",
    "results/public_repair_ledger_windows_20260907.isolation.json",
    "results/public_repair_outputs_windows_20260907.replay.json",
    "results/public_repair_outputs_windows_20260907.isolation.json",
    "results/public_repair_dgx_20260907/ledger.json",
    "results/public_repair_dgx_20260907/ledger.isolation.json",
    "results/public_repair_dgx_20260907/outputs.json",
    "results/public_repair_dgx_20260907/outputs.isolation.json",
    "results/public_repair_dgx_20260907/neural.json",
    "results/public_repair_dgx_20260907/neural.isolation.json",
)
OVERRIDE = {
    "paper/greencert_arxiv.pdf": "output/pdf/greencert_arxiv.pdf",
    "paper/greencert_arxiv_source.zip": "output/arxiv/greencert_arxiv_source.zip",
    "paper/greencert_arxiv_release.json": "output/arxiv/greencert_arxiv_release.json",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest().upper()


def stage(target):
    target = target.resolve()
    target.relative_to((ROOT/"output").resolve())
    if target == (ROOT/"output").resolve() or target.exists():
        raise ValueError("require a new, non-root output subdirectory")
    raw = subprocess.check_output(["git", "ls-files", "-z", "--cached"], cwd=ROOT)
    names = set(os.fsdecode(v).replace("\\", "/") for v in raw.split(b"\0") if v)
    names.update(EXTRA)
    names.discard("PUBLIC_MANIFEST_SHA256.json")
    names = sorted(names)
    sources = {}
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] in {".git", "output", "tmp", ".venv"}:
            raise ValueError("unsafe public relative path")
        source = (ROOT/OVERRIDE.get(name, name)).resolve(strict=True)
        source.relative_to(ROOT)
        if not source.is_file():
            raise ValueError("public source is not a file")
        sources[name] = source
    target.mkdir(parents=True, exist_ok=False)
    rows = {}
    for name, source in sources.items():
        dest = target/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        if digest(source) != digest(dest):
            raise ValueError("snapshot source changed during copy")
        rows[name] = {"bytes": dest.stat().st_size, "sha256": digest(dest)}
    manifest = {"format": 1, "repository": "https://github.com/oddeye00/greencert", "files": rows}
    with (target/"PUBLIC_MANIFEST_SHA256.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return {"status": "snapshot_created_not_published", "files": len(rows)+1,
            "manifest_sha256": digest(target/"PUBLIC_MANIFEST_SHA256.json"),
            "destination": target.relative_to(ROOT).as_posix(),
            "git_index_changed": False, "historical_records_changed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.destination), indent=2))
