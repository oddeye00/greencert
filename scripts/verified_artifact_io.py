"""Strict, bounded-memory readers for generated verification artifacts.

Hashes establish byte identity and provenance, not mathematical truth. An
untrusted producer still requires neural/operator proof replay. This layer
does not evaluate a network, choose a training window, or issue certificates.
"""
import hashlib
import json
import math
from pathlib import Path
import re


class EvidencePending(RuntimeError):
    pass


class IntegrityError(ValueError):
    pass


def require_sha256(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise IntegrityError("a canonical SHA256 is required")
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise IntegrityError("duplicate JSON object key")
        result[key] = value
    return result


def invalid_constant(value):
    raise IntegrityError("nonfinite JSON constant")


def finite_json_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise IntegrityError("nonfinite JSON numeric value")
    return result


class EvidenceReader:
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise IntegrityError("artifact root must be a directory")
        self.observed = {}

    def resolve(self, relative):
        if not isinstance(relative, (str, Path)):
            raise IntegrityError("artifact reference must be a relative path")
        supplied = Path(relative)
        if supplied.is_absolute() or supplied.drive or not supplied.parts or ".." in supplied.parts:
            raise IntegrityError("absolute or parent-traversing artifact reference")
        resolved = (self.root / supplied).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise IntegrityError("artifact reference escapes its root") from None
        if resolved == self.root:
            raise IntegrityError("artifact reference must name a file")
        return resolved

    def _remember(self, path, actual, expected):
        if expected is not None and require_sha256(expected) != actual:
            raise IntegrityError("artifact checksum mismatch")
        key = str(path.relative_to(self.root))
        if key in self.observed and self.observed[key] != actual:
            raise IntegrityError("artifact changed during one verification run")
        self.observed[key] = actual
        return actual

    def check_blob(self, relative, expected):
        require_sha256(expected)
        path = self.resolve(relative)
        try:
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
        except FileNotFoundError:
            raise EvidencePending(f"missing artifact: {relative}") from None
        except IsADirectoryError:
            raise IntegrityError("artifact reference is a directory") from None
        return self._remember(path, actual, expected)

    def read_json(self, relative, expected=None, *, allow_in_progress=False):
        if expected is not None:
            require_sha256(expected)
        path = self.resolve(relative)
        try:
            payload = path.read_bytes()
        except FileNotFoundError:
            raise EvidencePending(f"missing artifact: {relative}") from None
        except IsADirectoryError:
            raise IntegrityError("artifact reference is a directory") from None
        try:
            value = json.loads(payload, object_pairs_hook=unique_object, parse_constant=invalid_constant,
                               parse_float=finite_json_float)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if allow_in_progress and expected is None:
                raise EvidencePending(f"in-progress JSON artifact: {relative}") from None
            raise IntegrityError("malformed completed JSON artifact") from None
        if not isinstance(value, dict):
            raise IntegrityError("artifact JSON must have an object root")
        actual = hashlib.sha256(payload).hexdigest()
        self._remember(path, actual, expected)
        return value, actual

    def check_sources(self, method, *, required_names):
        if not isinstance(required_names, (set, frozenset)) or not required_names:
            raise IntegrityError("nonempty caller-selected source policy required")
        sources = method.get("sources")
        if not isinstance(sources, dict) or not required_names.issubset(sources):
            raise IntegrityError("required producer sources missing")
        for name, expected in sources.items():
            if not isinstance(name, str) or Path(name).name != name or not name.endswith(".py"):
                raise IntegrityError("invalid producer source name")
            self.check_blob(Path("scripts") / name, expected)
        return {name: sources[name] for name in sorted(sources)}
