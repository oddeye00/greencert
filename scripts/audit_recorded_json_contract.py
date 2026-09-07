"""Read-only round-trip numeric audit of the authenticated recorded graph."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from replay_recorded_window import load_context
from roundtrip_evidence_reader import RoundtripEvidenceReader, NUMERIC_CONTRACT


def count_numbers(value, counts):
    if type(value) is int:
        counts["integer_visits"] += 1
    elif type(value) is float:
        counts["binary64_visits"] += 1
    elif isinstance(value, dict):
        for item in value.values():
            count_numbers(item, counts)
    elif isinstance(value, list):
        for item in value:
            count_numbers(item, counts)


def audit(args):
    started = time.perf_counter()
    original, _, _, _, _, _, _, barrier = load_context(
        args.root, args.manifest, args.manifest_sha256)
    reader = RoundtripEvidenceReader(args.root, original.recorded, numeric_contract=NUMERIC_CONTRACT)
    counts = Counter()
    for name, digest in reader.recorded.items():
        if Path(name).suffix.lower() == ".json":
            record, _ = reader.read_json(name, digest)
            count_numbers(record, counts)
            counts["json_files"] += 1
    barrier()
    if not set(reader.observed).issubset(original.recorded):
        raise ValueError("numeric audit accessed an unrecorded dependency")
    source = Path(__file__).resolve().parent
    return dict(schema="recorded_json_numeric_contract_audit_v1", status="PASS",
        numeric_contract=NUMERIC_CONTRACT, manifest_sha256=args.manifest_sha256,
        **counts, non_json_files_not_parsed=len(reader.recorded)-counts["json_files"],
        rejection_count=0, lexical_number_conversions_checked=True,
        non_json_neural_arrays_replayed=False, event_certificate_issued=False,
        future_outcome_access=False, archived_records_rewritten=False,
        sources={name:hashlib.sha256((source/name).read_bytes()).hexdigest()
                 for name in ("audit_recorded_json_contract.py", "roundtrip_evidence_reader.py")},
        elapsed_seconds=time.perf_counter()-started,
        scope="Every recorded .json dependency checked under the caller-selected round-trip binary64 contract; not a neural proof or future observation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("fresh report required")
    result = audit(args)
    with args.report.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))
