"""Authenticate the public fixed-vector benchmark before extracting code."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT/"artifacts/greencert_exact_norm_benchmark_20260907.zip"
ARCHIVE_SHA = "1943fbfbda015e9b6907a55703a9d286e5ede568512f78714e5856a6235b076d"
MANIFEST_SHA = "5817bf337c156f7055e23108dc0d15e31935b0bff09a93e5d3899ac23baf2ea1"


def run(target=None):
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != ARCHIVE_SHA:
        raise ValueError("fixed-vector benchmark archive changed")
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = archive.namelist()
        if len(names) != 20 or len(set(names)) != len(names):
            raise ValueError("benchmark archive population changed")
        payloads = {}
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or ":" in name or "\\" in name or path.as_posix() != name:
                raise ValueError("unsafe archive path")
            payloads[name] = archive.read(name)
    if hashlib.sha256(payloads["manifest.json"]).hexdigest() != MANIFEST_SHA:
        raise ValueError("benchmark manifest changed")
    manifest = json.loads(payloads["manifest.json"])
    if set(payloads) != {"manifest.json", *manifest["files"]}:
        raise ValueError("unrecorded archive member")
    for name, expected in manifest["files"].items():
        if hashlib.sha256(payloads[name]).hexdigest() != expected:
            raise ValueError("benchmark payload changed")
    if target is not None:
        target = target.resolve()
        target.relative_to((ROOT/"output").resolve())
        if target.exists():
            raise ValueError("new output subdirectory required")
        target.mkdir(parents=True, exist_ok=False)
        for name, value in payloads.items():
            destination = target/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(value)
    return {"status": "PASS", "archive_sha256": ARCHIVE_SHA, "manifest_sha256": MANIFEST_SHA,
            "members": len(payloads), "extracted": target is not None,
            "neural_kernels_replayed": False, "event_certificate_issued": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.extract), indent=2))
