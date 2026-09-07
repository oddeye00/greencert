"""Check public-reader/extraction failure modes against the fixed artifact."""
import hashlib
import json
from pathlib import Path
import tempfile

import read_public_repair_archive as public


def main():
    manifest, files = public.payload()
    brackets = public.recorded_brackets()
    assert len(manifest["jobs"]) == 63 and len(brackets) == 79
    assert len([k for k in brackets if k.startswith("binary:")]) == 63
    assert len([k for k in brackets if k.startswith("mse:")]) == 16
    refusals = []
    with tempfile.TemporaryDirectory(prefix="greencert-public-archive-test-") as temporary:
        root = Path(temporary)
        target = root/"extracted"
        public.extract(target)
        for name, value in files.items():
            assert (target/name).read_bytes() == value
        assert hashlib.sha256((target/"manifest.json").read_bytes()).hexdigest() == public.MANIFEST_SHA
        try:
            public.extract(target)
        except ValueError:
            refusals.append("existing_extraction")
        else:
            raise AssertionError("reader overwrote existing directory")
        altered = root/"modified.zip"
        with altered.open("xb") as stream:
            stream.write(public.ARCHIVE.read_bytes()+b"tampered")
        try:
            public.payload(altered)
        except ValueError:
            refusals.append("modified_archive")
        else:
            raise AssertionError("reader accepted modified archive")
        assert public.payload()[0] == manifest
    print(json.dumps({"status": "PASS", "fixed_jobs": 63, "recorded_brackets": 79,
                      "members_extracted_and_byte_compared": len(files), "refusals": refusals,
                      "neural_premises_recomputed": False, "original_archive_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
