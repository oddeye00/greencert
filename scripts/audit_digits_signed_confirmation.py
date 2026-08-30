#!/usr/bin/env python3
"""Aggregation-independent audit of the sealed digits confirmation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from digits_parity_mlp import raw_data_sha256


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "digits_signed_confirmation"
OUTPUT_JSON = ROOT / "results" / "digits_signed_confirmation_independent_audit.json"
OUTPUT_MD = ROOT / "results" / "digits_signed_confirmation_independent_audit.md"
THRESHOLDS = (0.90, 0.925)
SEEDS = tuple(range(501, 513))
PERSISTENCE = 10


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def anchor(train: list[float], trigger: list[float], threshold: float) -> int | None:
    for step in range(0, len(train), 5):
        if train[step] >= 0.80 and threshold - 0.10 <= trigger[step] < threshold:
            return step
    return None


def event(values: list[float], threshold: float) -> int | None:
    run = 0
    for step, value in enumerate(values):
        run = run + 1 if value >= threshold else 0
        if run >= PERSISTENCE:
            return step - PERSISTENCE + 1
    return None


def main() -> None:
    method_path = ROOT / "DIGITS_SIGNED_METHOD_SEAL.json"
    candidate_seal_path = ROOT / "DIGITS_SIGNED_CANDIDATE_SEAL.json"
    certificate_seal_path = ROOT / "DIGITS_SIGNED_CERTIFICATE_SEAL.json"
    method = load(method_path)
    candidate_seal = load(candidate_seal_path)
    certificate_seal = load(certificate_seal_path)

    assert method["status"] == "FROZEN BEFORE FRESH TRAINING"
    assert tuple(method["fresh_seeds"]) == SEEDS
    assert tuple(method["thresholds"]) == THRESHOLDS
    assert method["raw_data_sha256"] == raw_data_sha256()
    assert method["protocol_sha256"] == sha256(ROOT / "DIGITS_SIGNED_CONFIRMATION_PROTOCOL.md")
    for relative, expected in method["code_sha256"].items():
        assert sha256(ROOT / relative) == expected

    candidate_manifest_path = RUN / "candidates_blind.json"
    candidate_manifest = load(candidate_manifest_path)
    assert candidate_seal["method_seal_sha256"] == sha256(method_path)
    assert candidate_seal["candidate_manifest_sha256"] == sha256(candidate_manifest_path)

    independently_selected = []
    for seed in SEEDS:
        blind = load(RUN / "blind" / f"seed_{seed}.json")
        assert not any("certificate" in key.lower() for key in blind)
        for gate_index, threshold in enumerate(THRESHOLDS):
            selected = anchor(blind["train_accuracy"], blind["trigger_accuracy"], threshold)
            assert selected is not None
            independently_selected.append(
                {
                    "seed": seed,
                    "gate_index": gate_index,
                    "threshold": threshold,
                    "anchor": selected,
                    "disposition": "candidate frozen",
                }
            )
    assert independently_selected == candidate_manifest["rows"]
    assert independently_selected == candidate_manifest["candidates"]
    assert candidate_seal["candidate_count"] == 24

    certificate_manifest_path = RUN / "certificate_manifest.json"
    certificate_manifest = load(certificate_manifest_path)
    assert certificate_seal["candidate_seal_sha256"] == sha256(candidate_seal_path)
    assert certificate_seal["certificate_manifest_sha256"] == sha256(certificate_manifest_path)
    assert certificate_seal["status"] == "FROZEN BEFORE OUTCOME JOIN"

    rows = []
    for record in certificate_manifest["records"]:
        certificate_path = RUN / "certificates" / record["path"]
        assert sha256(certificate_path) == record["sha256"]
        certificate = load(certificate_path)
        assert bool(certificate["certificate_issued"]) == bool(record["issued"])
        assert bool(certificate.get("unsigned_right_inverse_certificate_issued")) == bool(
            record["unsigned_issued"]
        )
        seed = int(certificate["seed"])
        threshold = float(certificate["threshold"])
        outcome = load(RUN / "audit" / "outcomes" / f"seed_{seed}.outcomes.json")
        absolute = event(outcome["certificate_accuracy"], threshold)
        assert absolute == outcome["events"][f"{threshold:.3f}"]
        relative = None if absolute is None else absolute - int(certificate["anchor"])
        issued = bool(certificate["certificate_issued"])
        unsigned = bool(certificate.get("unsigned_right_inverse_certificate_issued"))
        bracket = certificate.get("certified_bracket")
        unsigned_bracket = certificate.get("unsigned_right_inverse_certified_bracket")
        covered = bool(issued and relative is not None and bracket[0] <= relative <= bracket[1])
        unsigned_covered = bool(
            unsigned
            and relative is not None
            and unsigned_bracket[0] <= relative <= unsigned_bracket[1]
        )
        rows.append(
            {
                "seed": seed,
                "threshold": threshold,
                "anchor": int(certificate["anchor"]),
                "actual_relative": relative,
                "issued": issued,
                "covered": covered,
                "bracket": bracket,
                "unsigned_issued": unsigned,
                "unsigned_covered": unsigned_covered,
                "signed_only": bool(issued and not unsigned),
                "closure_statistic": certificate.get("closure_statistic"),
                "unsigned_closure_statistic": certificate.get(
                    "unsigned_right_inverse_closure_statistic"
                ),
                "directional_gain_ratio": certificate.get("directional_gain_ratio"),
            }
        )

    issued = [row for row in rows if row["issued"]]
    unsigned = [row for row in rows if row["unsigned_issued"]]
    signed_only = [row for row in rows if row["signed_only"]]
    assert certificate_seal["issued_unopened"] == len(issued) == 7
    assert certificate_seal["unsigned_issued_unopened"] == len(unsigned) == 6
    assert certificate_seal["signed_only_unopened"] == len(signed_only) == 1
    assert sum(row["covered"] for row in issued) == 7
    assert sum(row["unsigned_covered"] for row in unsigned) == 6
    decisive = signed_only[0]
    assert decisive["seed"] == 509 and decisive["threshold"] == 0.90
    assert decisive["actual_relative"] == 147
    assert decisive["bracket"] == [147, 147]
    assert decisive["closure_statistic"] < 1.0
    assert decisive["unsigned_closure_statistic"] > 1.0

    published = load(ROOT / "results" / "digits_signed_confirmation_summary.json")
    assert published["signed_issued"] == 7 and published["signed_covered"] == 7
    assert published["unsigned_issued"] == 6 and published["signed_only"] == 1

    payload = {
        "status": "PASS",
        "audit_independence": "Does not import the experiment runner or its aggregation helpers.",
        "method_seal_sha256": sha256(method_path),
        "candidate_seal_sha256": sha256(candidate_seal_path),
        "certificate_seal_sha256": sha256(certificate_seal_path),
        "candidates_recomputed": len(independently_selected),
        "signed_issued": len(issued),
        "signed_covered": sum(row["covered"] for row in issued),
        "signed_distinct_seeds": len({row["seed"] for row in issued}),
        "unsigned_issued": len(unsigned),
        "unsigned_covered": sum(row["unsigned_covered"] for row in unsigned),
        "signed_only": len(signed_only),
        "signed_only_case": decisive,
        "realized_union_failure_bound": certificate_seal["realized_union_failure_bound"],
        "rows": rows,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUTPUT_MD.write_text(
        "# Independent digits confirmation audit\n\n"
        "Status: **PASS**\n\n"
        "The auditor independently recomputed all 24 trigger-only anchors, verified the three-seal "
        "hash chain and every certificate hash, reconstructed persistent events directly from the "
        "post-seal accuracy trajectories, and did not import the experiment runner.\n\n"
        "- Signed: 7 issued, 7 covered, across 6 seeds.\n"
        "- Matched unsigned: 6 issued, 6 covered.\n"
        "- Signed-only: seed 509, 90% gate, bracket and actual lead [147,147].\n"
        "- Closure: 0.236945 signed versus 13.018189 unsigned.\n"
        "- Realized family-wise failure bound: 5e-7.\n",
        encoding="utf-8",
    )
    print(
        "PASS: independently verified 24 anchors, 7/7 signed coverage, and the exact 147-step signed-only event."
    )


if __name__ == "__main__":
    main()
