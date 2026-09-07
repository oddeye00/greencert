"""Authenticate the source-only, synthetic read-only replay test package.

The archive does not contain the complete 451k neural evidence graph.
Extraction performs no neural computation and observes no future outcome.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT/"artifacts/greencert_recorded_replay_sources_20260907_v2.zip"
ARCHIVE_SHA = "8680f837f29f516516a889fa3b598cd03de4df4c5510fb56f0e8aed29b886fa5"
MANIFEST_SHA = "4b2c7c26b9d28ebcce72c66ad402ecae8e343ae028c96b7813c378a32234d8b7"


def payload(archive=ARCHIVE):
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != ARCHIVE_SHA:
            raise ValueError("source transport identity differs")
    with zipfile.ZipFile(archive) as saved:
        raw = saved.read("manifest.json")
        if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA:
            raise ValueError("source manifest identity differs")
        manifest = json.loads(raw)
        if manifest["schema"] != "recorded_replay_source_transport_v1" or any(
            manifest[key] is not False for key in ("full_neural_evidence_included", "future_outcome_access", "event_certificate_issued")):
            raise ValueError("source transport scope differs")
        expected = {**manifest["files"], "manifest.json":MANIFEST_SHA}
        names = saved.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError("source member population differs")
        result = {}
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or name != path.as_posix() or ":" in name or "\\" in name or ".." in path.parts:
                raise ValueError("unsafe source member")
            raw = saved.read(name)
            if hashlib.sha256(raw).hexdigest() != expected[name]:
                raise ValueError("source member identity differs")
            result[name] = raw
    return manifest, result


def extract(target):
    manifest, files = payload()
    target = target.resolve()
    if target.exists():
        raise ValueError("new extraction directory required")
    target.mkdir(parents=True, exist_ok=False)
    for name, raw in files.items():
        path = target/name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
    return {"status":"authenticated_source_extraction", "files":len(files),
        "manifest_sha256":MANIFEST_SHA, "entries":manifest["entries"],
        "tests_executed":False, "full_neural_evidence_included":False, "future_outcome_access":False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(extract(args.extract), indent=2))
