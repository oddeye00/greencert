"""Stdlib-only synthetic transport tests; no neural evidence or outcomes."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

from materialize_recorded_graph import digest, materialize, prepare


def write(path,value):
    path.write_text(json.dumps(value,sort_keys=True,indent=2)+"\n",encoding="utf-8",newline="\n")
    return digest(path)


def fixture(folder):
    folder.mkdir()
    payloads = {"inputs/a.bin":b"first", "inputs/alias.bin":b"first", "inputs/b.bin":b"second",
                "scripts/demo.py":b"# source must never execute\n"}
    objects = {hashlib.sha256(raw).hexdigest():raw for raw in payloads.values()}
    original = {"schema":"recorded_large_evidence_manifest_v1", "future_outcome_access":False,
                "files":{name:{"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()} for name,raw in payloads.items()}}
    original_sha = write(folder/"recorded_manifest.json",original)
    sources = {"scripts/demo.py":hashlib.sha256(payloads["scripts/demo.py"]).hexdigest()}
    source_manifest = json.dumps({"files":sources},sort_keys=True).encode()
    with zipfile.ZipFile(folder/"replay_sources.zip","w") as archive:
        archive.writestr("manifest.json",source_manifest)
        archive.writestr("scripts/demo.py",payloads["scripts/demo.py"])
    with zipfile.ZipFile(folder/"recorded_objects_000.zip","w",compression=zipfile.ZIP_STORED) as archive:
        for sha,raw in objects.items():
            archive.writestr("objects/"+sha,raw)
    descriptor = {"schema":"recorded_graph_transport_v1", "recorded_manifest_sha256":original_sha,
        "replay_sources_sha256":digest(folder/"replay_sources.zip"),
        "replay_source_manifest_sha256":hashlib.sha256(source_manifest).hexdigest(),
        "files":{name:{"bytes":(folder/name).stat().st_size,"sha256":digest(folder/name)} for name in (
            "recorded_manifest.json","replay_sources.zip","recorded_objects_000.zip")},
        "objects":{sha:{"archive":"recorded_objects_000.zip","member":"objects/"+sha,"bytes":len(raw)}
                   for sha,raw in objects.items()},
        "logical_files":len(payloads),"logical_bytes":sum(map(len,payloads.values())),
        "unique_payloads":len(objects),"object_bytes":sum(map(len,objects.values())),
        "numerical_replay_performed":False,"event_certificate_issued":False,
        "future_outcome_access":False,"original_records_changed":False}
    write(folder/"transport.json",descriptor)
    return descriptor,original,payloads


def main():
    refused = 0
    with tempfile.TemporaryDirectory(prefix="greencert_transport_tests_") as temporary:
        top = Path(temporary).resolve()
        assert top.name.startswith("greencert_transport_tests_")
        good,_,payloads = fixture(top/"good")
        sha = digest(top/"good/transport.json")
        result = materialize(top/"good",sha,top/"restored")
        assert result["logical_files"]==4 and result["unique_objects"]==3
        for name,raw in payloads.items():
            assert (top/"restored"/name).read_bytes()==raw
        cases = ("wrong_transport_hash","scope_promotion","missing_object","wrong_total","traversal",
                 "case_alias","negative_object_size","extra_member","source_conflict","coherent_payload_corruption",
                 "existing_destination","stale_segment_hash")
        for index,label in enumerate(cases):
            folder = top/f"bad{index}"
            descriptor,original,payloads = fixture(folder)
            key = next(iter(descriptor["objects"]))
            if label=="scope_promotion": descriptor["event_certificate_issued"]=True
            elif label=="missing_object": descriptor["objects"].pop(key)
            elif label=="wrong_total": descriptor["object_bytes"]+=1
            elif label in ("traversal","case_alias"):
                original["files"]["../escape" if label=="traversal" else "INPUTS/A.BIN"] = original["files"]["inputs/a.bin"]
                descriptor["recorded_manifest_sha256"] = write(folder/"recorded_manifest.json",original)
                descriptor["files"]["recorded_manifest.json"] = {
                    "bytes":(folder/"recorded_manifest.json").stat().st_size,"sha256":descriptor["recorded_manifest_sha256"]}
            elif label=="negative_object_size": descriptor["objects"][key]["bytes"]=-1
            elif label in ("extra_member","coherent_payload_corruption","stale_segment_hash"):
                original_objects = {hashlib.sha256(raw).hexdigest():raw for raw in payloads.values()}
                with zipfile.ZipFile(folder/"recorded_objects_000.zip","w",compression=zipfile.ZIP_STORED) as archive:
                    for value,raw in original_objects.items():
                        altered = bytes([raw[0]^1])+raw[1:] if label!="extra_member" and value==key else raw
                        archive.writestr("objects/"+value,altered)
                    if label=="extra_member": archive.writestr("unlisted",b"unexpected")
                if label!="stale_segment_hash":
                    descriptor["files"]["recorded_objects_000.zip"] = {
                        "bytes":(folder/"recorded_objects_000.zip").stat().st_size,
                        "sha256":digest(folder/"recorded_objects_000.zip")}
            elif label=="source_conflict":
                raw = b"# replacement source\n"
                source_manifest = json.dumps({"files":{"scripts/demo.py":hashlib.sha256(raw).hexdigest()}}).encode()
                with zipfile.ZipFile(folder/"replay_sources.zip","w") as archive:
                    archive.writestr("manifest.json",source_manifest)
                    archive.writestr("scripts/demo.py",raw)
                descriptor["replay_sources_sha256"]=digest(folder/"replay_sources.zip")
                descriptor["replay_source_manifest_sha256"]=hashlib.sha256(source_manifest).hexdigest()
                descriptor["files"]["replay_sources.zip"]={"sha256":descriptor["replay_sources_sha256"],
                                                         "bytes":(folder/"replay_sources.zip").stat().st_size}
            transport_sha = write(folder/"transport.json",descriptor)
            target = top/f"failed_restore{index}"
            if label=="existing_destination": target.mkdir()
            try:
                materialize(folder,"0"*64 if label=="wrong_transport_hash" else transport_sha,target)
            except ValueError:
                refused += 1
            else:
                raise AssertionError("incompatible transport accepted: "+label)
    return {"status":"PASS","exact_byte_restoration_cases":1,"coherent_and_integrity_refusals":refused,
            "archive_code_executed":False,"neural_evidence":False,"future_outcome_access":False}


if __name__=="__main__":
    print(json.dumps(main(),indent=2))
