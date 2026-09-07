"""Freeze the approved hardware-only amendment; never load a neural model."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from dgx_outcome_bridge import digest, validate_runtime_only, bundle_records
from registered_outcome_continuation import publish_json, utc, unopened
from verified_artifact_io import EvidenceReader
from test_dgx_outcome_amendment import audit as test_amendment

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("results/larger_transformer_third_pass")
ORIGINAL = SOURCE/"construction_03925/runtime_power2_outcome_protocol.json"
ORIGINAL_SHA = "1638644a1d5c026419e3a1d04d435cb4c6dcf6118fb264d59e22decffaf29eb4"
MANIFEST_SHA = "bc6e54b5421ffcf71cd9e905142cadcfbd3dc0856f478fd001c221cd5072b425"
TERMINAL_SHA = "8820651f723c3a60313a21355b31a96bd99dafa8ecb3a19622225ed658782ec9"
REPLAY_SHA = "42727e4d4c1de036c16d4c67c42d3fc29a73420d991d6c5bce2c98a2efb4fa89"


def freeze(destination):
    reader = EvidenceReader(ROOT)
    unopened(reader, SOURCE/"registered_outcome_run", SOURCE/"reveal.json")
    destination = destination.resolve()
    destination.relative_to((ROOT/"output").resolve())
    if destination.exists() or destination == (ROOT/"output").resolve():
        raise ValueError("new output subdirectory required")
    original, _ = reader.read_json(ORIGINAL, ORIGINAL_SHA)
    runtime = json.loads((ROOT/"results/dgx_outcome_runtime_20260907_v1.json").read_text())
    effective = {**copy.deepcopy(original), "runtime": runtime["runtime"]}
    replay = ROOT/"results/recorded_window_exact_replay_arm_20260907_v3.json"
    if digest(replay) != REPLAY_SHA:
        raise ValueError("completed ARM replay changed")
    destination.mkdir(parents=True)
    (destination/"frozen_sources").mkdir()
    for name, expected in original["sources"].items():
        source = ROOT/"scripts"/name
        if digest(source) != expected:
            raise ValueError("registered observer source changed: "+name)
        shutil.copyfile(source, destination/"frozen_sources"/name)
    shutil.copyfile(ROOT/ORIGINAL, destination/"original_protocol.json")
    shutil.copyfile(replay, destination/"completed_replay.json")
    shutil.copyfile(ROOT/"DGX_OUTCOME_AMENDMENT.md", destination/"DGX_OUTCOME_AMENDMENT.md")
    effective_sha = publish_json(destination/"effective_protocol.json", effective)
    tests_sha = publish_json(destination/"amendment_regression.json", test_amendment())
    bridge_names = ("dgx_outcome_bridge.py", "test_dgx_outcome_amendment.py", "prepare_dgx_outcome_amendment.py")
    for name in bridge_names:
        shutil.copyfile(ROOT/"scripts"/name, destination/name)
    amendment = {
        "schema": "dgx_hardware_only_outcome_amendment_v1", "created_utc": utc(),
        "user_authorized_hardware_amendment": True,
        "authorization_record": "User approved completing the frozen 451k observation on DGX before any million-scale experiment.",
        "original_protocol_sha256": ORIGINAL_SHA, "effective_protocol_sha256": effective_sha,
        "recorded_manifest_sha256": MANIFEST_SHA, "original_terminal_sha256": TERMINAL_SHA,
        "completed_replay_sha256": REPLAY_SHA, "issued_bracket": [44,44],
        "allowed_changed_protocol_fields": ["runtime"], "runtime_capture": runtime,
        "original_observer_sources": original["sources"],
        "bridge_sources": {name:digest(ROOT/"scripts"/name) for name in bridge_names},
        "document_sha256": digest(ROOT/"DGX_OUTCOME_AMENDMENT.md"),
        "regression_sha256": tests_sha, "future_outcome_access": False,
        "automatic_retry_or_resume": False, "new_event_certificate_issued": False,
        "original_records_modified": False,
    }
    validate_runtime_only(original, effective, amendment, runtime)
    amendment_sha = publish_json(destination/"amendment.json", amendment)
    bundle_records(destination, amendment_sha)
    archive = destination.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(destination).as_posix())
    unopened(reader, SOURCE/"registered_outcome_run", SOURCE/"reveal.json")
    return {"status":"amendment_frozen_before_observation", "amendment_sha256":amendment_sha,
            "effective_protocol_sha256":effective_sha, "archive_sha256":digest(archive),
            "archive_bytes":archive.stat().st_size, "future_outcome_access":False}


def delegate(bundle, amendment_sha):
    amendment, _, _ = bundle_records(bundle, amendment_sha)
    reader = EvidenceReader(ROOT)
    reader.check_blob(ORIGINAL, amendment["original_protocol_sha256"])
    run, reveal = SOURCE/"registered_outcome_run", SOURCE/"reveal.json"
    unopened(reader, run, reveal)
    reader.resolve(run).mkdir(exist_ok=False)
    record = {"schema":"dgx_outcome_source_host_delegation_v1", "created_utc":utc(),
        "amendment_sha256":amendment_sha, "original_protocol_sha256":ORIGINAL_SHA,
        "source_host_original_run_reserved":True, "source_host_neural_observations":0,
        "future_access_must_be_assumed_possible_after_delegation":True,
        "automatic_retry_or_resume":False}
    path = reader.resolve(run/"dgx_delegation.json")
    sha = publish_json(path, record)
    return {"status":"source_host_exclusively_delegated", "delegation_sha256":sha,
            "relative_path":(run/"dgx_delegation.json").as_posix()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("freeze", "delegate"), required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--amendment-sha256")
    args = parser.parse_args()
    result = freeze(args.bundle) if args.mode == "freeze" else delegate(args.bundle, args.amendment_sha256)
    print(json.dumps(result, indent=2))
