#!/usr/bin/env python3
"""Execute the frozen four-probe response-free amplified-secant audit."""

from __future__ import annotations

import json
from pathlib import Path

from audit_transformer_v3_response_free_probe import ROOT, run


NONCE = "5a37e5ccaf6834c438fde251d52ec1de313329314377d70cc1cb25e62fc52f2a"
DOMAIN = "greencert-response-free-secant-four-probe-v1|"


def main() -> None:
    payload = run(
        nonce=NONCE,
        domain=DOMAIN,
        probes=4,
        delta=1.0e-6,
        amplification=4096.0,
        protocol_path=ROOT / "AMPLIFIED_SECANT_FOUR_PROBE_PROTOCOL.md",
        output_path_=ROOT
        / "results"
        / "transformer_v3_four_probe_audit.json",
    )
    payload["status"] = "OUTCOME-BLIND FOUR-PROBE AUDIT COMPLETED"
    payload["evidence_boundary"] = (
        "Fresh post-development four-probe block under the ideal-PRNG model; "
        "no future outcome is read. Float64 points validate geometry but are "
        "not outward scalar intervals or an exact-real proof."
    )
    output = ROOT / "results" / "transformer_v3_four_probe_audit.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
