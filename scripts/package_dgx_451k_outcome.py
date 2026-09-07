"""Package or extract the completed 451k observation; never run training.

All recorded payloads are copied verbatim. Extraction authenticates every
member before writing into a new destination and never imports archive code.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

SOURCE = "results/larger_transformer_third_pass"
AMENDMENT = "d8f56172db878c1a0986c6cfcaf384166b0a33a37284497e53514ec7a9d3e97a"
REVEAL = "25f204627419b5bbdfdbb902e1d0925f08f5a796aea51e8edbe48d2cb01040c3"
DELEGATION = "04db72ce5c70a44f5f307aa764761da95c2cf254f4476cc4630500675c2497bd"
AMENDMENT_ARCHIVE = "acbaba1786066a1c15acd1ad660f0da92ff2ba8bd529d7c471c0910605e98ac6"
CHECKPOINT = "5f1bd4479f4b9e802115a6109356493d74902d7bdec80e6a7db8002cd5401999"
SEAL_COMMIT = "62ba2e3fac7bdda3784f87e5cb1772f5c5fc1238"


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(name):
    p = PurePosixPath(name)
    require(bool(name) and not p.is_absolute() and all(v not in ("", ".", "..") for v in name.split("/"))
            and "\\" not in name and ":" not in name and p.as_posix() == name, "unsafe member")
    return name


def entries(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        infos = z.infolist()
        require(len(infos) <= 1000 and sum(v.file_size for v in infos) <= 64*1024*1024, "archive too large")
        require(len({v.filename.casefold() for v in infos}) == len(infos), "duplicate archive member")
        return {canonical(v.filename): z.read(v) for v in infos if not v.is_dir()}


def package(root, amendment_archive, delegation, destination):
    raw_amendment = amendment_archive.read_bytes()
    require(digest(raw_amendment) == AMENDMENT_ARCHIVE, "pre-observation source archive changed")
    records = {"amendment/"+name:raw for name,raw in entries(raw_amendment).items()}
    run = root/SOURCE/"registered_outcome_run"
    names = {"started.json", "engine_ready.json"} | {f"row_{j:03d}.json" for j in range(65)} | {
        f"intent_{j:03d}.json" for j in range(1,65)}
    require({p.name for p in run.iterdir()} == names, "unexpected or incomplete remote ledger")
    for name in sorted(names):
        records[f"root/{SOURCE}/registered_outcome_run/{name}"] = (run/name).read_bytes()
    records[f"root/{SOURCE}/reveal.json"] = (root/SOURCE/"reveal.json").read_bytes()
    records[f"root/{SOURCE}/anchor.npz"] = (root/SOURCE/"anchor.npz").read_bytes()
    records[f"root/{SOURCE}/registered_outcome_run/dgx_delegation.json"] = delegation.read_bytes()
    for name, expected in (("amendment/amendment.json", AMENDMENT),
        (f"root/{SOURCE}/reveal.json", REVEAL), (f"root/{SOURCE}/anchor.npz", CHECKPOINT),
        (f"root/{SOURCE}/registered_outcome_run/dgx_delegation.json", DELEGATION)):
        require(digest(records[name]) == expected, "pinned completed record changed")
    manifest = {"schema":"completed_451k_outcome_transport_v1", "pre_observation_seal_commit":SEAL_COMMIT,
        "amendment_sha256":AMENDMENT, "reveal_sha256":REVEAL,
        "original_payloads_changed":False, "optimizer_executed_by_packager":False,
        "scope":"Completed CPU-float64 observation ledger; certificate inputs are in the separate full recorded graph.",
        "files":{name:{"sha256":digest(raw),"bytes":len(raw)} for name,raw in sorted(records.items())}}
    records["manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True)+"\n").encode()
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name,raw in sorted(records.items()):
            info = zipfile.ZipInfo(name, (2026,9,7,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, raw)
    return {"status":"packaged_completed_observation", "sha256":digest(destination.read_bytes()),
        "bytes":destination.stat().st_size, "members":len(records), "neural_optimizer_reexecuted":False}


def extract(archive, expected, destination):
    require(not destination.exists(), "fresh extraction destination required")
    raw = archive.read_bytes()
    require(digest(raw) == expected, "outcome transport checksum mismatch")
    records = entries(raw)
    manifest = json.loads(records.pop("manifest.json"))
    require(manifest["schema"] == "completed_451k_outcome_transport_v1" and
            set(records) == set(manifest["files"]), "outcome population differs")
    for name, value in records.items():
        row = manifest["files"][name]
        require(len(value) == row["bytes"] and digest(value) == row["sha256"], "outcome member checksum mismatch")
    destination.mkdir(parents=True, exist_ok=False)
    for name,value in records.items():
        path = destination/name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(value)
    with (destination/"manifest.json").open("xb") as stream:
        stream.write(entries(raw)["manifest.json"])
    return {"status":"extracted_authenticated_completed_observation", "members":len(records)+1,
            "archive_sha256":expected, "neural_optimizer_reexecuted":False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("package","extract"), required=True)
    p.add_argument("--root", type=Path)
    p.add_argument("--amendment-archive", type=Path)
    p.add_argument("--delegation", type=Path)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--archive-sha256")
    p.add_argument("--destination", type=Path)
    a = p.parse_args()
    if a.mode == "package":
        require(a.root and a.amendment_archive and a.delegation, "package inputs required")
        result = package(a.root,a.amendment_archive,a.delegation,a.archive)
    else:
        require(a.archive_sha256 and a.destination, "extract inputs required")
        result = extract(a.archive,a.archive_sha256,a.destination)
    print(json.dumps(result, indent=2))
