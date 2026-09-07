"""Read-only portable access to a caller-pinned evidence graph.

Only path lookup changes: saved JSON/code bytes are never rewritten. This
layer authenticates recorded premises; it neither proves their numerical
validity nor issues a certificate or authorizes a future observation.
"""
from pathlib import Path, PurePosixPath
import re

from verified_artifact_io import EvidenceReader, IntegrityError, require_sha256


def canonical_path(value):
    if not isinstance(value, (str, Path)):
        raise IntegrityError("relative evidence path required")
    text = str(value).replace("\\", "/")
    if not text or text.startswith("/") or ":" in text or "\x00" in text:
        raise IntegrityError("absolute, drive, or empty evidence path")
    parts = text.split("/")
    reserved = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", re.I)
    if any(part in ("", ".", "..") or part.endswith((" ", ".")) or
           reserved.match(part) or any(ord(c) < 32 for c in part) for part in parts):
        raise IntegrityError("nonportable evidence path")
    return PurePosixPath(*parts).as_posix()


def canonical_recorded(mapping):
    if not isinstance(mapping, dict) or not mapping:
        raise IntegrityError("nonempty caller-pinned evidence graph required")
    result, folded = {}, set()
    for name, digest in mapping.items():
        key = canonical_path(name)
        if key.casefold() in folded:
            raise IntegrityError("ambiguous evidence alias or case collision")
        folded.add(key.casefold())
        result[key] = require_sha256(digest)
    return result


class RecordedEvidenceReader(EvidenceReader):
    """Enforce the same digest whether a producer reference supplies one or not.

    Folder/existence lookup is allowed for barrier checks, but opening a
    numerical/text premise requires its membership in the pinned graph.
    This does not make an untrusted manifest an authority: callers must
    authenticate its origin and digest before constructing this reader.
    """

    def __init__(self, root, recorded):
        super().__init__(root)
        self.recorded = canonical_recorded(recorded)

    def resolve(self, relative):
        return super().resolve(canonical_path(relative))

    def expected(self, relative, supplied=None):
        key = canonical_path(relative)
        if key not in self.recorded:
            raise IntegrityError("unrecorded evidence cannot be opened")
        expected = self.recorded[key]
        if supplied is not None and require_sha256(supplied) != expected:
            raise IntegrityError("producer digest differs from recorded graph")
        return expected

    def _remember(self, path, actual, expected):
        relative = path.relative_to(self.root).as_posix()
        pinned = self.expected(relative, expected)
        return super()._remember(path, actual, pinned)

    def check_blob(self, relative, expected):
        return super().check_blob(relative, self.expected(relative, expected))

    def read_json(self, relative, expected=None, **kwargs):
        # A recorded completed file never becomes an in-progress/pending JSON.
        return super().read_json(relative, self.expected(relative, expected), **kwargs)
