"""New POSIX one-shot observation ledger; never edits historical ledgers.

All records, including runtime and update intents, enter one hash chain.
The production writer fsyncs both files and containing directories. It does
not provide external timestamp attestation or guarantee physical disk behavior.
There is no resume/retry API. The caller must authenticate the scientific
disposition before creating a ledger and must retain any interrupted run.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


SCHEMA = "prospective_observation_ledger_v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def is_hash(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def sync_directory(path):
    if os.name != "posix":
        raise OSError("production ledger requires POSIX directory fsync")
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_new(path, value, *, sync=sync_directory):
    """Install complete bytes without replacement; fail closed on sync errors."""
    path = Path(path)
    require(path.parent.is_dir() and not path.parent.is_symlink(), "invalid publication directory")
    raw = encode(value)
    descriptor, filename = tempfile.mkstemp(prefix=".ledger-", suffix=".tmp", dir=path.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        sync(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
        sync(path.parent)
    require(path.read_bytes() == raw, "published bytes differ")
    return hashlib.sha256(raw).hexdigest()


class Phase:
    def __init__(self, horizon):
        require(type(horizon) is int and horizon > 0, "invalid horizon")
        self.horizon, self.step, self.expected = horizon, 0, "engine_ready"

    def advance(self, kind, payload):
        require(self.expected != "terminal", "ledger is terminal")
        require(type(payload) is dict, "payload must be an object")
        if kind == "interrupted":
            self.expected = "terminal"
            return
        require(kind == self.expected, "unexpected ledger phase")
        if kind in ("observation", "update_intent"):
            require(type(payload.get("step")) is int and payload["step"] == self.step,
                    "out-of-order observation or intent")
        if kind == "engine_ready":
            self.expected = "observation"
        elif kind == "update_intent":
            self.expected = "observation"
        elif kind == "observation":
            if self.step == self.horizon:
                self.expected = "completed"
            else:
                self.step += 1
                self.expected = "update_intent"
        elif kind == "completed":
            self.expected = "terminal"


class Ledger:
    def __init__(self, folder, horizon, *, sync=sync_directory):
        self.folder, self.sync = Path(folder), sync
        self.phase = Phase(horizon)
        self.sequence, self.head = 0, None
        self.failed = False

    @classmethod
    def create(cls, folder, *, horizon, bindings, sync=sync_directory):
        require(type(bindings) is dict and set(bindings) ==
                {"protocol_sha256", "source_manifest_sha256", "disposition_sha256", "checkpoint_sha256"}
                and all(is_hash(v) for v in bindings.values()), "incomplete source/anchor/disposition binding")
        ledger = cls(folder, horizon, sync=sync)
        require(ledger.folder.parent.is_dir(), "ledger parent must already exist")
        sync(ledger.folder.parent)  # Admit supported storage before reserving.
        ledger.folder.mkdir(exist_ok=False)
        sync(ledger.folder.parent)
        ledger._publish("started", {"horizon": horizon, "bindings": bindings,
                                    "automatic_retry_or_resume": False})
        return ledger

    def _publish(self, kind, payload):
        require(not self.failed, "failed ledger cannot be reused")
        value = {"schema": SCHEMA, "sequence": self.sequence, "previous_sha256": self.head,
                 "kind": kind, "utc": datetime.now(timezone.utc).isoformat(), "payload": payload}
        try:
            head = publish_new(self.folder / f"record_{self.sequence:06d}.json", value, sync=self.sync)
        except BaseException:
            self.failed = True
            raise
        self.head = head
        self.sequence += 1
        return head

    def append(self, kind, payload):
        require(not self.failed, "failed ledger cannot be reused")
        # Validate without mutating the phase; a failed write poisons this object.
        next_phase = Phase(self.phase.horizon)
        next_phase.step, next_phase.expected = self.phase.step, self.phase.expected
        next_phase.advance(kind, payload)
        head = self._publish(kind, payload)
        self.phase = next_phase
        return head


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate record key")
        result[key] = value
    return result


def invalid_constant(value):
    raise ValueError("nonfinite record constant")


def finite_float(value):
    result = float(value)
    require(math.isfinite(result), "nonfinite record float")
    return result


def audit_complete(folder, *, expected_head_sha256):
    """Rehash a completed chain against an externally supplied terminal digest.

This authenticates ledger ordering, not neural outputs or a scientific gate.
Those are separate obligations of the experiment-specific independent audit.
"""
    require(is_hash(expected_head_sha256), "external terminal digest required")
    folder = Path(folder)
    require(folder.is_dir() and not folder.is_symlink(), "invalid ledger root")
    paths = sorted(folder.iterdir())
    require(paths and all(p.is_file() and not p.is_symlink() for p in paths), "invalid record population")
    previous, phase, rows = None, None, []
    for index, path in enumerate(paths):
        require(path.name == f"record_{index:06d}.json", "missing, extra or nonsequential record")
        raw = path.read_bytes()
        record = json.loads(raw, object_pairs_hook=unique_object, parse_constant=invalid_constant,
                            parse_float=finite_float)
        require(type(record) is dict and set(record) ==
                {"schema", "sequence", "previous_sha256", "kind", "utc", "payload"}, "record fields differ")
        require(record["schema"] == SCHEMA and type(record["sequence"]) is int
                and record["sequence"] == index and record["previous_sha256"] == previous,
                "record chain differs")
        require(type(record["utc"]) is str and
                datetime.fromisoformat(record["utc"]).utcoffset() is not None, "invalid timestamp")
        payload = record["payload"]
        if index == 0:
            require(record["kind"] == "started" and type(payload) is dict
                    and payload.get("automatic_retry_or_resume") is False, "invalid start")
            bindings = payload.get("bindings")
            require(type(bindings) is dict and set(bindings) ==
                    {"protocol_sha256", "source_manifest_sha256", "disposition_sha256", "checkpoint_sha256"}
                    and all(is_hash(v) for v in bindings.values()), "invalid start bindings")
            phase = Phase(payload["horizon"])
        else:
            phase.advance(record["kind"], payload)
        rows.append(record)
        previous = hashlib.sha256(raw).hexdigest()
    require(previous == expected_head_sha256, "terminal digest differs")
    require(rows[-1]["kind"] == "completed" and phase.expected == "terminal", "run did not complete")
    return {"status": "complete_chain_authenticated", "records": len(rows),
            "head_sha256": previous, "horizon": phase.horizon,
            "timestamps_externally_attested": False, "neural_outputs_verified": False}
