"""Create-only phase bundles and strict, bounded-memory recorded-evidence IO.

The manifest authenticates bytes and phase bindings, not their mathematical
truth. Independent semantic replay is still required before event issuance.
Writers never reopen a reserved directory after interruption or completion.
"""
import hashlib
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from prospective_ledger_v1 import encode, is_hash, publish_new, sync_directory
from verified_artifact_io import EvidenceReader


SCHEMA = "final_scale_phase_bundle_v1"
SAFE_PART = re.compile(r"[a-z0-9][a-z0-9_.-]*")
BINDINGS = {"protocol_sha256", "source_manifest_sha256", "runtime_manifest_sha256", "phase_input_sha256"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative_name(value):
    require(type(value) is str and value and "\\" not in value and
            all(part not in (".", "..") and SAFE_PART.fullmatch(part) for part in value.split("/")),
            "noncanonical phase-relative path")
    require(not Path(value).is_absolute() and not Path(value).drive, "absolute artifact path")
    return value


def checked_array(value, guard):
    require(isinstance(value, np.ndarray) and value.dtype in (np.dtype("<f8"), np.dtype("<i8")) and
            value.ndim in (1, 2) and all(v > 0 for v in value.shape) and value.flags.c_contiguous,
            "only nonempty contiguous float64/int64 vectors and matrices are supported")
    flat = value.reshape(-1)
    for start in range(0, flat.size, 2**18):
        guard()
        require(np.isfinite(flat[start:start+2**18]).all(), "nonfinite artifact array")
    return value


def file_identity(root, relative):
    path = root / relative_name(relative)
    require(path.is_file() and not path.is_symlink(), "artifact must be a regular file")
    with path.open("rb") as stream:
        result = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": relative, "bytes": path.stat().st_size, "sha256": result}


def enumerate_files(root):
    require(root.is_dir() and not root.is_symlink(), "invalid phase root")
    result = []
    for path in root.rglob("*"):
        require(not path.is_symlink() and (path.is_dir() or path.is_file()), "nonregular phase entry")
        relative = path.relative_to(root).as_posix()
        relative_name(relative)
        if path.is_file():
            result.append(relative)
    return sorted(result)


class PhaseWriter:
    def __init__(self, root, *, role, bindings, guard, sync=sync_directory):
        require(role in ("selection", "construction", "observation", "audit"), "invalid phase role")
        require(type(bindings) is dict and set(bindings) == BINDINGS and all(is_hash(v) for v in bindings.values()),
                "incomplete phase binding")
        require(callable(guard), "source/resource guard required")
        self.root, self.guard, self.sync = Path(root), guard, sync
        self.failed, self.sealed = False, False
        require(self.root.parent.is_dir() and not self.root.parent.is_symlink(), "phase parent must exist")
        self._check()
        sync(self.root.parent)
        self.root.mkdir(exist_ok=False)
        sync(self.root.parent)
        self.method = {"schema": SCHEMA, "role": role, "bindings": dict(bindings), "automatic_retry_or_resume": False}
        try:
            publish_new(self.root / "method.json", self.method, sync=sync)
        except BaseException:
            self.failed = True
            raise

    def _check(self):
        require(not self.failed and not self.sealed, "phase writer is failed or sealed")
        try:
            self.guard()
        except BaseException:
            self.failed = True
            raise

    def _destination(self, relative):
        self._check()
        relative_name(relative)
        require(relative not in ("method.json", "manifest.json"), "reserved phase record")
        path = self.root / relative
        current = self.root
        for part in Path(relative).parts[:-1]:
            child = current / part
            if not child.exists():
                child.mkdir(exist_ok=False)
                self.sync(current)
            require(child.is_dir() and not child.is_symlink(), "indirect artifact directory")
            current = child
        require(not path.exists() and not path.is_symlink(), "artifact already reserved")
        return path

    def record(self, relative, value):
        self._check()
        try:
            require(type(value) is dict, "phase JSON must have an object root")
            path = self._destination(relative)
            require(path.suffix == ".json", "record extension must be json")
            publish_new(path, value, sync=self.sync)
            self._check()
            return file_identity(self.root, relative)
        except BaseException:
            self.failed = True
            raise

    def array(self, relative, value):
        self._check()
        temporary = None
        try:
            checked_array(value, self.guard)
            path = self._destination(relative)
            require(path.suffix == ".npy", "array extension must be npy")
            descriptor, filename = tempfile.mkstemp(prefix="pending-", suffix=".tmp", dir=path.parent)
            temporary = Path(filename)
            with os.fdopen(descriptor, "wb") as stream:
                np.save(stream, value, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
            self.sync(path.parent)
            temporary.unlink()
            temporary = None
            self.sync(path.parent)
            self._check()
            return {**file_identity(self.root, relative), "shape": list(value.shape), "dtype": value.dtype.str,
                    "raw_sha256": hashlib.sha256(memoryview(value).cast("B")).hexdigest()}
        except BaseException:
            self.failed = True
            raise
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)  # Only this call's own incomplete temporary.
                self.sync(temporary.parent)

    def seal(self, payload, *, maximum_bytes):
        self._check()
        require(type(maximum_bytes) is int and maximum_bytes > 0, "positive phase byte budget required")
        require(type(payload) is dict, "phase disposition must be a record")
        try:
            encode(payload)
            files, total = [], 0
            names = enumerate_files(self.root)
            require("manifest.json" not in names and "method.json" in names, "phase has already been sealed")
            for name in names:
                self._check()
                item = file_identity(self.root, name)
                total += item["bytes"]
                require(total <= maximum_bytes, "registered completed-phase byte budget exceeded")
                files.append(item)
            # Include files from cooperating numerical producers that use their
            # own writers, and synchronize their containing directories too.
            for path in sorted({self.root / Path(name).parent for name in names}, key=lambda p: len(p.parts), reverse=True):
                self.sync(path)
            manifest = {"schema": SCHEMA, "method": self.method, "files": files,
                        "total_file_bytes": total, "maximum_file_bytes": maximum_bytes, "payload": payload}
            self._check()
            digest = publish_new(self.root / "manifest.json", manifest, sync=self.sync)
            self.sealed = True
            return {"manifest_sha256": digest, "files": len(files), "bytes": total}
        except BaseException:
            self.failed = True
            raise


class PhaseReader:
    def __init__(self, root, *, expected_manifest_sha256, expected_role, expected_bindings, guard):
        require(is_hash(expected_manifest_sha256), "external phase manifest digest required")
        self.root, self.guard = Path(root), guard
        require(self.root.is_dir() and not self.root.is_symlink(), "invalid phase root")
        self.reader = EvidenceReader(self.root)
        self.manifest, actual = self.reader.read_json("manifest.json", expected_manifest_sha256)
        require(set(self.manifest) == {"schema", "method", "files", "total_file_bytes", "maximum_file_bytes", "payload"}
                and self.manifest["schema"] == SCHEMA, "phase manifest schema differs")
        method = self.manifest["method"]
        require(method == {"schema": SCHEMA, "role": expected_role, "bindings": expected_bindings,
                           "automatic_retry_or_resume": False}, "phase source/runtime/input binding differs")
        require(type(expected_bindings) is dict and set(expected_bindings) == BINDINGS and
                all(is_hash(v) for v in expected_bindings.values()), "invalid expected binding")
        require(type(self.manifest["files"]) is list and self.manifest["files"], "empty phase manifest")
        self.files = {}
        total = 0
        for entry in self.manifest["files"]:
            guard()
            require(type(entry) is dict and set(entry) == {"path", "sha256", "bytes"}, "invalid file reference")
            name = relative_name(entry["path"])
            require(name != "manifest.json" and name not in self.files and type(entry["bytes"]) is int
                    and entry["bytes"] >= 0 and is_hash(entry["sha256"]), "duplicate or invalid file reference")
            self.reader.check_blob(name, entry["sha256"])
            require((self.root / name).stat().st_size == entry["bytes"], "artifact size differs")
            self.files[name] = entry
            total += entry["bytes"]
        require(list(self.files) == sorted(self.files) and "method.json" in self.files, "noncanonical file population")
        require(set(enumerate_files(self.root)) == set(self.files) | {"manifest.json"}, "missing or extra phase file")
        require(type(self.manifest["total_file_bytes"]) is int and type(self.manifest["maximum_file_bytes"]) is int and
                total == self.manifest["total_file_bytes"] and 0 < self.manifest["maximum_file_bytes"] and
                total <= self.manifest["maximum_file_bytes"], "phase byte accounting differs")
        require(self.record("method.json") == method, "stored phase method differs")
        self.identity = actual
        guard()

    def record(self, relative):
        self.guard()
        require(relative in self.files, "unregistered phase record")
        return self.reader.read_json(relative, self.files[relative]["sha256"])[0]

    def array(self, relative, *, shape, dtype):
        self.guard()
        require(relative in self.files and relative.endswith(".npy"), "unregistered phase array")
        self.reader.check_blob(relative, self.files[relative]["sha256"])
        value = np.load(self.root / relative, mmap_mode="r", allow_pickle=False)
        require(value.shape == tuple(shape) and value.dtype == np.dtype(dtype), "array shape/dtype differs")
        checked_array(value, self.guard)
        return value
