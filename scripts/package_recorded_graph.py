"""Package a pinned recorded graph into content-addressed, bounded ZIP assets.

No recorded payload is rewritten. Equal digests share one transport object;
the original manifest retains every logical path. This is transport, not
new issuance, a neural recomputation, or an outcome observation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from replay_recorded_window import load_context

SOURCE_SHA = "c799935f80b06e6bfd86e71031c0b4b8bdf74264d570aea83319ef1e39c0a2de"
SOURCE_MANIFEST_SHA = "c840c041d217dd0d4a76dd67184dac760c2de2477198da6da6bfaefe21113307"
LIMIT = 1 << 30


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build(root, original, original_sha, source_archive, target):
    target = target.resolve()
    target.relative_to((root/"output").resolve())
    if target.exists() or target == (root/"output").resolve():
        raise ValueError("fresh non-root package destination required")
    reader, request, terminal, audit, policy, sources, forbidden, barrier = load_context(root, original, original_sha)
    header = json.loads(original.read_bytes())
    if digest(source_archive) != SOURCE_SHA:
        raise ValueError("tested replay source archive differs")
    with zipfile.ZipFile(source_archive) as archive:
        source_manifest = archive.read("manifest.json")
        if hashlib.sha256(source_manifest).hexdigest() != SOURCE_MANIFEST_SHA:
            raise ValueError("tested replay source manifest differs")
        for name, sha in json.loads(source_manifest)["files"].items():
            if name in reader.recorded and reader.recorded[name] != sha:
                raise ValueError("source overlay would replace frozen evidence")
    unique = {}
    for index, (name, sha) in enumerate(reader.recorded.items(), 1):
        reader.check_blob(name, sha)
        size = reader.resolve(name).stat().st_size
        if size != header["files"][name]["bytes"] or size > LIMIT:
            raise ValueError("recorded size differs or object exceeds segment limit")
        if sha in unique and unique[sha]["bytes"] != size:
            raise ValueError("digest aliases disagree on length")
        unique.setdefault(sha, {"bytes":size, "representative":name})
        if index % 1000 == 0:
            print(json.dumps({"stage":"authenticate_original_graph", "files":index, "total":len(reader.recorded)}), flush=True)
    barrier()
    target.mkdir(parents=True, exist_ok=False)
    files, objects = {}, {}
    for source, name, expected in ((original,"recorded_manifest.json",original_sha),
                                  (source_archive,"replay_sources.zip",SOURCE_SHA)):
        shutil.copyfile(source, target/name)
        if digest(target/name) != expected:
            raise ValueError("metadata changed in transport")
        files[name] = {"sha256":expected, "bytes":(target/name).stat().st_size}
    groups, group, size = [], [], 0
    for sha, row in sorted(unique.items()):
        if group and size+row["bytes"] > LIMIT:
            groups.append(group)
            group, size = [], 0
        group.append((sha,row))
        size += row["bytes"]
    if group:
        groups.append(group)
    for index, group in enumerate(groups):
        filename = f"recorded_objects_{index:03d}.zip"
        path = target/filename
        with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_STORED) as archive:
            for sha, row in group:
                member = "objects/"+sha
                info = zipfile.ZipInfo(member, date_time=(2026,9,7,0,0,0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o100644 << 16
                check, transferred = hashlib.sha256(), 0
                with reader.resolve(row["representative"]).open("rb") as source, archive.open(info,"w",force_zip64=True) as dest:
                    for block in iter(lambda: source.read(1 << 20), b""):
                        check.update(block)
                        transferred += len(block)
                        dest.write(block)
                if check.hexdigest() != sha or transferred != row["bytes"]:
                    raise ValueError("recorded object changed during archive construction")
                objects[sha] = {"archive":filename, "member":member, "bytes":row["bytes"]}
        if path.stat().st_size >= 2*(1 << 30):
            raise ValueError("segment exceeds release-asset limit")
        files[filename] = {"sha256":digest(path), "bytes":path.stat().st_size}
        print(json.dumps({"stage":"sealed_segment", "segment":index+1, "total":len(groups),
                          "bytes":path.stat().st_size}), flush=True)
    barrier()
    descriptor = {"schema":"recorded_graph_transport_v1", "recorded_manifest_sha256":original_sha,
        "replay_sources_sha256":SOURCE_SHA, "replay_source_manifest_sha256":SOURCE_MANIFEST_SHA,
        "files":files, "objects":objects, "logical_files":len(reader.recorded),
        "logical_bytes":header["logical_bytes"], "unique_payloads":len(objects),
        "object_bytes":sum(row["bytes"] for row in objects.values()),
        "archive_format":"ZIP_STORED; content-addressed exact-byte objects; deterministic sorted-digest segmentation",
        "numerical_replay_performed":False, "event_certificate_issued":False, "future_outcome_access":False,
        "original_records_changed":False, "public_upload_authorized_by_this_file":False}
    with (target/"transport.json").open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(descriptor,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write("\n")
    return {"status":"recorded_graph_packaged_not_published", "transport_sha256":digest(target/"transport.json"),
        "assets":len(files)+1, "logical_files":len(reader.recorded), "unique_payloads":len(objects),
        "transport_bytes":sum(row["bytes"] for row in files.values())+(target/"transport.json").stat().st_size,
        "future_outcome_access":False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--manifest-sha256",required=True)
    parser.add_argument("--sources",type=Path,required=True)
    parser.add_argument("--destination",type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.root.resolve(),args.manifest,args.manifest_sha256,args.sources,args.destination),indent=2))
