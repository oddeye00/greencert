"""Numeric-lexeme and portable-reader regressions; no neural claims."""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path
import random
import struct
import tempfile
from unittest.mock import patch

import roundtrip_evidence_reader as strict
import test_recorded_evidence_replay as legacy_tests
from verified_artifact_io import EvidenceReader, IntegrityError


def audit():
    refusals = []
    for token in ("1e-400", "-1e-400", "1e-1000000000", "-1e-1000000000",
                  "1.00000000000000001", "-1.00000000000000001",
                  "9007199254740993.0", "0.100000000000000001",
                  "2.4703282292062327e-324", "1e309", "-1e309"):
        try:
            strict.loads_roundtrip(('{"value":'+token+'}').encode())
        except IntegrityError:
            refusals.append(token)
        else:
            raise AssertionError("ambiguous numeric token accepted: "+token)
    # Format changes with equal decimal values are harmless under the
    # declared contract. The returned meaning is binary64, not rational .1.
    for token, expected in (("1e0", 1.), ("1.000", 1.), ("0.1000", .1),
                            ("5e-324", math.ulp(0.)), ("-0.0", -0.),
                            ("0e-1000000000", 0.)):
        value = strict.loads_roundtrip('{"value":'+token+'}')["value"]
        assert type(value) is float and value.hex() == expected.hex()
    exact_int = strict.loads_roundtrip('{"value":9007199254740993}')["value"]
    assert type(exact_int) is int and exact_int == 2**53+1
    rng = random.Random(20260910)
    finite = 0
    for _ in range(10000):
        value = struct.unpack(">d", rng.getrandbits(64).to_bytes(8, "big"))[0]
        if not math.isfinite(value):
            continue
        decoded = strict.loads_roundtrip(json.dumps({"value": value}, allow_nan=False))["value"]
        assert type(decoded) is float and decoded.hex() == value.hex()
        finite += 1
    factory = lambda root, recorded: strict.RoundtripEvidenceReader(
        root, recorded, numeric_contract=strict.NUMERIC_CONTRACT)
    with patch.object(legacy_tests, "RecordedEvidenceReader", factory), redirect_stdout(io.StringIO()) as log:
        legacy_tests.main()
    inherited = json.loads(log.getvalue())
    with tempfile.TemporaryDirectory(prefix="greencert-numeric-contract-") as temp:
        root = Path(temp).resolve()
        unsafe = b'{"injection":1e-400}'
        path = root/"bound.json"
        path.write_bytes(unsafe)
        digest = hashlib.sha256(unsafe).hexdigest()
        assert EvidenceReader(root).read_json("bound.json", digest)[0]["injection"] == 0.
        reader = factory(root, {"bound.json": digest})
        try:
            reader.read_json("bound.json", allow_in_progress=True)
        except IntegrityError:
            pass
        else:
            raise AssertionError("completed source-underflow record accepted")
        # A wrong hash is rejected before invoking the numeric parser.
        reader = factory(root, {"bound.json": "0"*64})
        with patch.object(strict, "loads_roundtrip", side_effect=AssertionError("parser ran before authentication")):
            try:
                reader.read_json("bound.json")
            except IntegrityError:
                pass
            else:
                raise AssertionError("wrong hash accepted")
        try:
            strict.RoundtripEvidenceReader(root, {"bound.json": digest}, numeric_contract="exact-decimal")
        except IntegrityError:
            pass
        else:
            raise AssertionError("unsupported numeric semantics accepted")
    return dict(schema="roundtrip_evidence_reader_tests_v1", status="PASS",
        numeric_contract=strict.NUMERIC_CONTRACT, lexeme_refusals=refusals,
        finite_binary64_roundtrips=finite, integer_precision_preserved=True,
        inherited_reader_refusals=inherited["refusals"],
        legacy_reader_underflow_counterexample_reproduced=True,
        authentication_precedes_numeric_parse=True, caller_selected_semantics=True,
        archived_sources_modified=False, future_outcome_access=False,
        event_certificate_issued=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = audit()
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n"
    if args.report:
        with args.report.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    print(payload, end="")
