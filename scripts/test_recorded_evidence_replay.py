"""Portable evidence-reader contracts; synthetic data, no event issuance."""
import hashlib
import json
from pathlib import Path
import tempfile

from recorded_evidence_replay import RecordedEvidenceReader, canonical_path, canonical_recorded
from verified_artifact_io import IntegrityError, EvidencePending


def main():
    refused = []
    def reject(label, call):
        try:
            call()
        except (IntegrityError, EvidencePending):
            refused.append(label)
        else:
            raise AssertionError("expected refusal: "+label)

    assert canonical_path("data\\case/file.json") == "data/case/file.json"
    for name in ("../escape", "data/../escape", "data\\..\\escape", "/absolute",
                 "C:\\absolute", "C:relative", "\\\\server\\share", "data//file",
                 "./file", "data/./file", "", "data/file.", "data/file ",
                 "data/AUX.json", "data/COM1.txt", "data/file:stream", "data/\x00file"):
        reject("unsafe_path_"+repr(name), lambda: canonical_path(name))
    reject("separator_alias", lambda: canonical_recorded({"a/b": "0"*64, "a\\b": "0"*64}))
    reject("case_alias", lambda: canonical_recorded({"a/b": "0"*64, "A/B": "0"*64}))
    reject("invalid_digest", lambda: canonical_recorded({"a": "not-sha256"}))
    reject("empty_graph", lambda: canonical_recorded({}))

    with tempfile.TemporaryDirectory(prefix="greencert-recorded-reader-") as folder:
        # Windows CI can return an 8.3 short-name TEMP path. The production
        # reader resolves its root; pass that same canonical path when this
        # fixture deliberately exercises the internal already-resolved hook.
        root = Path(folder).resolve()
        (root/"data").mkdir()
        original = b'{"value": 1}\n'
        digest = hashlib.sha256(original).hexdigest()
        (root/"data/file.json").write_bytes(original)
        reader = RecordedEvidenceReader(root, {"data\\file.json": digest})
        assert reader.read_json("data/file.json")[0] == {"value": 1}
        assert reader.read_json("data\\file.json")[0] == {"value": 1}
        assert reader.check_blob("data\\file.json", digest) == digest
        assert not reader.resolve("future/reveal.json").exists()
        reject("foreign_supplied_hash", lambda: reader.read_json("data/file.json", "0"*64))
        (root/"unlisted.json").write_bytes(original)
        reject("unrecorded_file", lambda: reader.read_json("unlisted.json"))
        reject("unrecorded_blob", lambda: reader.check_blob("unlisted.json", digest))
        reject("direct_array_hook", lambda: reader._remember(root/"unlisted.json", digest, digest))
        (root/"data/file.json").write_bytes(b'{"value": 2}\n')
        reject("payload_changed", lambda: reader.read_json("data/file.json"))
        reject("payload_changed_blob", lambda: reader.check_blob("data/file.json", digest))
        (root/"data/file.json").write_bytes(b'{"value":')
        reject("completed_not_pending", lambda: reader.read_json("data/file.json", allow_in_progress=True))
        for name, payload in (("duplicate.json", b'{"v":1,"v":2}'),
                              ("nan.json", b'{"v":NaN}'),
                              ("overflow.json", b'{"v":1e999}')):
            (root/name).write_bytes(payload)
            specific = RecordedEvidenceReader(root, {name: hashlib.sha256(payload).hexdigest()})
            reject(name, lambda: specific.read_json(name))
        (root/"data/file.json").write_bytes(original)
        reader.check_blob("data/file.json", digest)
        assert (root/"data/file.json").read_bytes() == original

    print(json.dumps({"status": "PASS", "refusals": len(refused), "refused": refused,
                      "windows_and_posix_references_share_exact_bytes": True,
                      "original_records_rewritten": False, "synthetic_contract_only": True,
                      "neural_kernels_replayed": False, "event_certificate_issued": False,
                      "future_outcome_access": False}, indent=2))


if __name__ == "__main__":
    main()
