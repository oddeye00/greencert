"""Caller-selected round-trip binary64 semantics for recorded JSON evidence.

JSON integer tokens remain exact integers. Fractional/exponent tokens must
equal a decimal spelling of Python's shortest round-trip representation of
the decoded finite binary64 value. This is NOT exact-decimal semantics.
Selecting this reader does not prove a producer's bounds or issue events.
Frozen readers and records are unchanged.
"""
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math

from recorded_evidence_replay import RecordedEvidenceReader
from verified_artifact_io import (
    EvidencePending, IntegrityError, unique_object, invalid_constant)

NUMERIC_CONTRACT = "shortest_roundtrip_binary64_json_v1"


def roundtrip_float(token):
    value = float(token)
    if not math.isfinite(value):
        raise IntegrityError("nonfinite JSON numeric value")
    try:
        source = Decimal(token)
        canonical = Decimal(repr(value))
    except InvalidOperation:
        raise IntegrityError("invalid JSON numeric exponent") from None
    if source != canonical:
        raise IntegrityError("non-roundtrip JSON number; numeric meaning would change")
    return value


def loads_roundtrip(payload):
    try:
        record = json.loads(payload, object_pairs_hook=unique_object,
                            parse_constant=invalid_constant, parse_float=roundtrip_float)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise IntegrityError("malformed completed JSON artifact") from None
    if not isinstance(record, dict):
        raise IntegrityError("artifact JSON must have an object root")
    return record


class RoundtripEvidenceReader(RecordedEvidenceReader):
    """Authenticate bytes before parsing under an explicit caller contract.

    Its numeric contract is provided by the caller, not selected by a JSON
    field in an untrusted artifact. The caller still has to establish that
    the producer's intended numbers use these round-trip binary64 semantics.
    """
    def __init__(self, root, recorded, *, numeric_contract):
        if numeric_contract != NUMERIC_CONTRACT:
            raise IntegrityError("unsupported or missing recorded numeric contract")
        super().__init__(root, recorded)
        self.numeric_contract = numeric_contract

    def read_json(self, relative, expected=None, *, allow_in_progress=False):
        # Membership and supplied-hash agreement are checked before file I/O.
        pinned = self.expected(relative, expected)
        path = self.resolve(relative)
        try:
            payload = path.read_bytes()
        except FileNotFoundError:
            raise EvidencePending(f"missing artifact: {relative}") from None
        except IsADirectoryError:
            raise IntegrityError("artifact reference is a directory") from None
        actual = hashlib.sha256(payload).hexdigest()
        # Parse the SAME authenticated byte string; no second read can race
        # the digest check. Completed corrupt records never become pending.
        self._remember(path, actual, pinned)
        return loads_roundtrip(payload), actual
