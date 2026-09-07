"""A paper edit must invalidate its derived claim report before release staging."""
import hashlib
import json
from pathlib import Path
import tempfile

from stage_repaired_public_release import check_manuscript_binding


def main():
    with tempfile.TemporaryDirectory(prefix="manuscript_binding_") as temporary:
        root = Path(temporary).resolve()
        (root/"paper").mkdir()
        (root/"results").mkdir()
        source = root/"paper/certified_local_training_events_neurips2026.tex"
        report = root/"results/greencert_manuscript_claim_audit.json"
        source.write_bytes(b"synthetic original paper\n")
        bound = hashlib.sha256(source.read_bytes()).hexdigest().upper()
        report.write_text(json.dumps({"paper_sha256": bound}), encoding="utf-8")
        check_manuscript_binding(root)
        source.write_bytes(b"synthetic revised paper\n")
        try:
            check_manuscript_binding(root)
        except ValueError as error:
            assert "stale manuscript claim audit" in str(error)
        else:
            raise AssertionError("stale derived report accepted")
        report.write_text(json.dumps({"paper_sha256": hashlib.sha256(source.read_bytes()).hexdigest().upper()}), encoding="utf-8")
        check_manuscript_binding(root)
    print("PASS: current manuscript binding, stale-report refusal, refreshed binding")


if __name__ == "__main__":
    main()
