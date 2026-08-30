#!/usr/bin/env python3
"""Frozen four-probe execution of the corrected-path secant audit."""

from pathlib import Path

import audit_transformer_relinearized_secant as audit


ROOT = Path(__file__).resolve().parents[1]
audit.OUTPUT = ROOT / "results" / "transformer_v3_relinearized_secant_four_probe_audit.json"
audit.PROTOCOL = ROOT / "RELINEARIZED_SECANT_FOUR_PROBE_PROTOCOL.md"
audit.PROBES = 4
audit.MASTER_NONCE = "611fda4bd0aa71d5a3ea2c4158a103cb32330ed279660ffe9dc35232aea14360"
audit.IDENTITY = (
    93,
    audit.CANDIDATE.seed,
    audit.CANDIDATE.gate_index,
    audit.CANDIDATE.anchor,
    audit.HORIZON,
    audit.SWEEPS,
    audit.POWER,
)


if __name__ == "__main__":
    audit.main()

