"""Authenticate and materialize an exact-byte recorded graph; stdlib only.

No archive code is imported or executed. Every segment, object, alias and
source is checked before this tool reports completion. Existing destinations
are refused. A failure leaves its new partial destination for inspection.
"""
import argparse
import errno
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import time
import zipfile


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    require(isinstance(value,str), "relative path required")
    value = value.replace("\\", "/")
    parts = value.split("/")
    reserved = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)",re.I)
    require(bool(value) and not value.startswith("/") and ":" not in value and
        all(p not in ("", ".", "..") and not p.endswith((" ", ".")) and not reserved.match(p)
            and all(ord(c)>=32 for c in p) for p in parts), "nonportable path")
    return PurePosixPath(*parts).as_posix()


def sha(value):
    require(isinstance(value,str) and re.fullmatch("[0-9a-f]{64}",value), "SHA256 required")
    return value


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("nonfinite JSON constant")


def floating(value):
    result = float(value)
    require(math.isfinite(result), "nonfinite JSON number")
    return result


def parse(raw):
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=reject_constant,parse_float=floating)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def checked(root,name,expected,size=None):
    path = (root/canonical(name)).resolve(strict=True)
    path.relative_to(root.resolve())
    require(path.is_file() and (size is None or path.stat().st_size==size) and digest(path)==sha(expected),
            "transport file changed")
    return path


def prepare(folder,transport_sha):
    folder = folder.resolve(strict=True)
    descriptor = parse(checked(folder,"transport.json",transport_sha).read_bytes())
    require(descriptor["schema"]=="recorded_graph_transport_v1" and all(descriptor[key] is False for key in (
        "numerical_replay_performed","event_certificate_issued","future_outcome_access","original_records_changed")),
        "transport scope differs")
    files, objects = descriptor["files"], descriptor["objects"]
    require(isinstance(files,dict) and isinstance(objects,dict) and files and objects, "empty transport")
    require(all(canonical(name)==name and "/" not in name for name in files), "noncanonical asset name")
    for name,row in files.items():
        require(set(row)=={"sha256","bytes"} and type(row["bytes"]) is int and row["bytes"]>0,
                "invalid asset descriptor")
        checked(folder,name,row["sha256"],row["bytes"])
    original = parse(checked(folder,"recorded_manifest.json",descriptor["recorded_manifest_sha256"]).read_bytes())
    require(original["schema"]=="recorded_large_evidence_manifest_v1" and original["future_outcome_access"] is False,
            "foreign original evidence index")
    logical, aliases, folded = {}, {}, set()
    for name,row in original["files"].items():
        canonical_name = canonical(name)
        require(canonical_name==name and name.casefold() not in folded, "ambiguous original path")
        folded.add(name.casefold())
        value = sha(row["sha256"])
        require(type(row["bytes"]) is int and row["bytes"]>=0, "invalid original byte length")
        logical[name] = row
        aliases.setdefault(value,[]).append(name)
    require(set(aliases)==set(objects) and descriptor["logical_files"]==len(logical) and
        descriptor["unique_payloads"]==len(objects) and descriptor["logical_bytes"]==sum(r["bytes"] for r in logical.values()),
        "incomplete object graph")
    segments = {}
    for value,row in objects.items():
        require(set(row)=={"archive","member","bytes"} and row["member"]=="objects/"+sha(value) and
            row["archive"] in files and row["archive"].startswith("recorded_objects_") and
            type(row["bytes"]) is int and 0<=row["bytes"]<=1<<30 and
            all(logical[name]["bytes"]==row["bytes"] for name in aliases[value]), "invalid object descriptor")
        segments.setdefault(row["archive"],{})[row["member"]] = (value,row["bytes"])
    require(set(files)==set(segments)|{"recorded_manifest.json","replay_sources.zip"}, "extra or missing transport asset")
    require(descriptor["object_bytes"]==sum(row["bytes"] for row in objects.values()), "object size total differs")
    overlay = {}
    with zipfile.ZipFile(checked(folder,"replay_sources.zip",descriptor["replay_sources_sha256"])) as archive:
        raw = archive.read("manifest.json")
        require(hashlib.sha256(raw).hexdigest()==sha(descriptor["replay_source_manifest_sha256"]), "source manifest differs")
        source_manifest = parse(raw)
        source_files = source_manifest["files"]
        names = archive.namelist()
        require(len(names)==len(set(names)) and set(names)==set(source_files)|{"manifest.json"}, "source member population differs")
        for name,value in source_files.items():
            require(canonical(name)==name and name.startswith("scripts/") and name.endswith(".py"), "unexpected source path")
            raw = archive.read(name)
            require(hashlib.sha256(raw).hexdigest()==sha(value), "source payload differs")
            require(name not in logical or logical[name]["sha256"]==value, "source overlay changes frozen evidence")
            overlay[name] = raw
    for filename,members in segments.items():
        with zipfile.ZipFile(folder/filename) as archive:
            names = archive.namelist()
            require(len(names)==len(set(names)) and set(names)==set(members), "segment member population differs")
            for name in names:
                info = archive.getinfo(name)
                require(info.file_size==members[name][1] and not info.is_dir() and info.compress_type==zipfile.ZIP_STORED,
                        "segment member descriptor differs")
    return descriptor,logical,aliases,segments,overlay


def materialize(folder,transport_sha,target):
    started = time.perf_counter()
    target = target.resolve()
    require(not target.exists(), "fresh materialization destination required")
    descriptor,logical,aliases,segments,overlay = prepare(folder,transport_sha)
    target.mkdir(parents=True,exist_ok=False)
    restored, linked, copied = 0,0,0
    for index,(filename,members) in enumerate(sorted(segments.items()),1):
        with zipfile.ZipFile(folder/filename) as archive:
            for member,(value,size) in members.items():
                names = aliases[value]
                first = target/names[0]
                first.parent.mkdir(parents=True,exist_ok=True)
                actual, length = hashlib.sha256(),0
                with archive.open(member) as source,first.open("xb") as output:
                    for block in iter(lambda:source.read(1<<20),b""):
                        actual.update(block)
                        length += len(block)
                        output.write(block)
                require(length==size and actual.hexdigest()==value, "object payload differs")
                for name in names[1:]:
                    dest = target/name
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    require(not dest.exists(), "alias destination already exists")
                    try:
                        os.link(first,dest)
                        linked += 1
                    except OSError as error:
                        if error.errno not in (errno.EXDEV,errno.EPERM,errno.EACCES,errno.ENOTSUP):
                            raise
                        shutil.copyfile(first,dest)
                        copied += 1
                restored += len(names)
        print(json.dumps({"stage":"materialized_segment","current":index,"total":len(segments),
                          "logical_files":restored}),flush=True)
    for name,raw in overlay.items():
        path = target/name
        if path.exists():
            require(digest(path)==hashlib.sha256(raw).hexdigest(), "frozen source changed")
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open("xb") as stream:
                stream.write(raw)
    for name,row in logical.items():
        checked(target,name,row["sha256"],row["bytes"])
    manifest_dest = target/"recorded_manifest.json"
    require(not manifest_dest.exists(), "original graph collides with transport index")
    shutil.copyfile(folder/"recorded_manifest.json",manifest_dest)
    require(digest(manifest_dest)==descriptor["recorded_manifest_sha256"], "materialized index changed")
    return {"status":"complete_recorded_graph_materialized", "transport_sha256":transport_sha,
        "recorded_manifest_sha256":descriptor["recorded_manifest_sha256"], "logical_files":restored,
        "unique_objects":len(aliases), "hardlinked_aliases":linked, "copied_aliases":copied,
        "source_overlay_files":len(overlay), "all_logical_payloads_reauthenticated":True,
        "elapsed_seconds":time.perf_counter()-started, "original_records_changed":False,
        "numerical_replay_performed":False,"event_certificate_issued":False,"future_outcome_access":False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package",type=Path,required=True)
    parser.add_argument("--transport-sha256",required=True)
    parser.add_argument("--destination",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    args = parser.parse_args()
    require(not args.report.exists(), "fresh report required")
    result = materialize(args.package.resolve(),args.transport_sha256,args.destination)
    with args.report.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps(result,indent=2))
