"""Authenticated access to the public repair artifact; no private ledger needed."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT/"artifacts/greencert_repaired_continuation_20260907.zip"
ARCHIVE_SHA = "77f3c2b7cfe694588de33e01f4bd949729aa2676696300846cca85eef3105cbc"
MANIFEST_SHA = "2ea5e687f173a7f7f0f6c31bdc9b435c628b8f74ca2172d771ef33a77c3109cd"


def payload(archive=ARCHIVE):
    with archive.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != ARCHIVE_SHA:
            raise ValueError("public repair archive identity differs")
    with zipfile.ZipFile(archive) as saved:
        names = saved.namelist()
        manifest_bytes = saved.read("manifest.json")
        if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA:
            raise ValueError("public repair manifest identity differs")
        manifest = json.loads(manifest_bytes)
        files = {**manifest["files"], "manifest.json": MANIFEST_SHA}
        if len(names) != len(set(names)) or set(names) != set(files):
            raise ValueError("public member population differs")
        result = {}
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or name != path.as_posix() or ":" in name or "\\" in name or ".." in path.parts:
                raise ValueError("unsafe archive member")
            value = saved.read(name)
            if hashlib.sha256(value).hexdigest() != files[name]:
                raise ValueError("public member identity differs")
            result[name] = value
    return manifest, result


def recorded_brackets(archive=ARCHIVE):
    """Replay saved output counts, not the neural premises that produced them."""
    manifest, files = payload(archive)
    brackets = {}
    if manifest["unique_jobs"] != 63 or len(manifest["jobs"]) != 63 or manifest["claims"] != 79:
        raise ValueError("fixed public repair population differs")
    for job in manifest["jobs"]:
        counts = json.loads(files[job["evidence"]])["independent_counts"]
        persistence = job["persistence"]
        if type(persistence) is not int or persistence < 1 or len(counts) != job["horizon"]+1:
            raise ValueError("invalid event window")
        for index, count in enumerate(counts):
            if count["step"] != index or any(type(count[k]) is not int for k in ("guaranteed", "possible")) or not (
                    0 <= count["guaranteed"] <= count["possible"] <= job["evaluation_examples"]):
                raise ValueError("invalid recorded count enclosure")
        for event in job["events"]:
            required = event["required_correct"]
            if type(required) is not int or not 0 < required <= job["evaluation_examples"]:
                raise ValueError("invalid required count")
            def first(field):
                return next((j for j in range(len(counts)-persistence+1)
                             if all(c[field] >= required for c in counts[j:j+persistence])), None)
            left, right = first("possible"), first("guaranteed")
            original = event["original_bracket"]
            if left is None or right is None or not original[0] <= left <= right <= original[1] or event["key"] in brackets:
                raise ValueError("unretained or duplicate public event")
            brackets[event["key"]] = [left, right]
    if len(brackets) != 79:
        raise ValueError("missing public events")
    return brackets


def extract(target):
    _, files = payload()
    if target.exists():
        raise ValueError("refusing to overwrite an existing extraction")
    target.mkdir(parents=True)
    for name, value in files.items():
        path = target/name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(value)
    print(json.dumps({"status": "authenticated_and_extracted_not_numerically_replayed",
                      "files": len(files), "manifest_sha256": MANIFEST_SHA}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", type=Path)
    args = parser.parse_args()
    if args.extract is not None:
        extract(args.extract.resolve())
    else:
        print(json.dumps({"retained_recorded_brackets": len(recorded_brackets()),
                          "neural_premises_recomputed": False, "manifest_sha256": MANIFEST_SHA}, indent=2))
