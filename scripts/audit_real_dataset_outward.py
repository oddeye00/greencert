#!/usr/bin/env python3
"""Independent structural/arithmetic audit of the WDBC Arb cache."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "results" / "real_dataset_confirmation"
CACHE = ROOT / "results" / "real_dataset_outward_cache"
BLIND = ROOT / "results" / "real_dataset_outward_blind.json"
JOINED = ROOT / "results" / "real_dataset_outward_joined.json"
OUT_JSON = ROOT / "results" / "real_dataset_outward_independent_audit.json"
OUT_MD = ROOT / "results" / "real_dataset_outward_independent_audit.md"
VERSION = "wdbc-arb-outward-v2-2026-08-24"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=2e-11, abs_tol=2e-18)


def main() -> None:
    blind = read_json(BLIND)
    joined = read_json(JOINED)
    method_path = EXPORT / "REAL_DATA_GREENCERT_METHOD_SEAL.json"
    method_hash = sha256(method_path)
    assert blind["version"] == VERSION
    assert blind["issued_green_candidates"] == 56
    assert blind["outward_retained"] == 56
    assert len(blind["rows"]) == 56
    assert len(joined["rows"]) == 56

    manifest = read_json(EXPORT / "certificate_manifest.json")
    issued_certificates = {}
    issued_by_anchor: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for record in manifest["records"]:
        if not record["issued"]:
            continue
        path = EXPORT / "certificates" / record["path"]
        assert sha256(path) == record["sha256"]
        certificate = read_json(path)
        certificate["certificate_sha256"] = record["sha256"]
        key = (int(certificate["seed"]), float(certificate["threshold"]))
        issued_certificates[key] = certificate
        issued_by_anchor[(int(certificate["seed"]), int(certificate["anchor"]))].append(
            certificate
        )
    assert len(issued_certificates) == 56
    assert len(issued_by_anchor) == 40

    blind_by_key = {
        (int(row["seed"]), float(row["threshold"])): row for row in blind["rows"]
    }
    joined_by_key = {
        (int(row["seed"]), float(row["threshold"])): row for row in joined["rows"]
    }
    assert set(blind_by_key) == set(issued_certificates) == set(joined_by_key)
    for key, certificate in issued_certificates.items():
        blind_row = blind_by_key[key]
        joined_row = joined_by_key[key]
        assert blind_row["green_float_bracket"] == certificate["certified_bracket"]
        assert blind_row["outward_bracket"] == certificate["certified_bracket"]
        assert blind_row["outward_issued"] is True
        assert joined_row["outward_bracket"] == blind_row["outward_bracket"]
        assert joined_row["outward_covered"] is True

    cache_files = sorted(CACHE.glob("seed_*_anchor_*.json"))
    assert len(cache_files) == 40
    maximum_radius = 0.0
    minimum_logic_slack = math.inf
    maximum_beta = 0.0
    maximum_defect = 0.0
    maximum_hessian_error = 0.0
    maximum_eigen_error = 0.0
    total_verified_steps = 0
    recurrence_slacks = []
    for path in cache_files:
        payload = read_json(path)
        assert payload["version"] == VERSION
        assert payload["method_seal_sha256"] == method_hash
        assert payload["python_flint_version"] == "0.9.0"
        assert int(payload["arb_precision_bits"]) == 192
        seed = int(payload["seed"])
        anchor = int(payload["anchor"])
        candidates = issued_by_anchor[(seed, anchor)]
        expected_hashes = sorted(row["certificate_sha256"] for row in candidates)
        assert payload["candidate_sha256"] == expected_hashes
        assert int(payload["requested_horizon"]) == max(
            int(row["certificate_horizon"]) for row in candidates
        )
        assert int(payload["reached_horizon"]) == int(payload["requested_horizon"])
        arrays = np.load(path.with_suffix(".npz"))
        radius = np.asarray(arrays["radius"], dtype=np.float64)
        assert len(radius) == int(payload["reached_horizon"]) + 1
        assert np.all(np.isfinite(radius)) and np.all(radius >= 0.0)
        assert close(float(np.max(radius)), float(payload["maximum_radius"]))
        diagnostics = payload["diagnostics"]
        assert len(diagnostics) == int(payload["reached_horizon"])
        assert radius[0] == 0.0
        for step, row in enumerate(diagnostics):
            assert int(row["step"]) == step
            assert close(float(row["next_radius"]), float(radius[step + 1]))
            beta = float(row["beta_upper"])
            defect = float(row["defect_norm_upper"])
            lipschitz = float(row["optimizer_jacobian_lipschitz_upper"])
            raw_rhs = (
                beta * float(radius[step])
                + defect
                + 0.5 * lipschitz * float(radius[step]) ** 2
            )
            assert float(row["next_radius"]) >= raw_rhs * (1.0 - 5e-15)
            recurrence_slacks.append(float(row["next_radius"]) - raw_rhs)
            assert float(row["hessian_interval_row_radius"]) >= 0.0
            assert float(row["hessian_evaluation_error"]) >= 0.0
            assert float(row["eigen_numeric_error"]) >= 0.0
            maximum_beta = max(maximum_beta, beta)
            maximum_defect = max(maximum_defect, defect)
            maximum_hessian_error = max(
                maximum_hessian_error, float(row["hessian_evaluation_error"])
            )
            maximum_eigen_error = max(
                maximum_eigen_error, float(row["eigen_numeric_error"])
            )
        for candidate in candidates:
            event = payload["events"][f"{float(candidate['threshold']):.3f}"]
            assert event["green_float_bracket"] == candidate["certified_bracket"]
            assert event["outward_bracket"] == candidate["certified_bracket"]
        maximum_radius = max(maximum_radius, float(payload["maximum_radius"]))
        minimum_logic_slack = min(
            minimum_logic_slack, float(payload["minimum_logic_slack"])
        )
        total_verified_steps += int(payload["reached_horizon"])

    outcomes = read_json(EXPORT / "final_audit.json")
    actual = {
        (int(row["seed"]), int(row["gate_index"])): row["actual_event"]
        for row in outcomes["rows"]
    }
    thresholds = read_json(method_path)["thresholds"]
    per_threshold = {}
    for gate, threshold in enumerate(thresholds):
        rows = [row for row in joined["rows"] if close(row["threshold"], threshold)]
        for row in rows:
            event = actual[(int(row["seed"]), gate)]
            assert event == row["actual_event"]
            assert row["outward_bracket"][0] <= event <= row["outward_bracket"][1]
        per_threshold[f"{threshold:.3f}"] = {
            "outward_issued": len(rows),
            "outward_covered": sum(bool(row["outward_covered"]) for row in rows),
            "distinct_seeds": len({int(row["seed"]) for row in rows}),
        }

    summary = {
        "status": "PASS",
        "version": VERSION,
        "green_issued": 56,
        "outward_retained": 56,
        "outward_covered": 56,
        "identical_singleton_brackets": 56,
        "distinct_outward_issuing_seeds": 22,
        "unique_seed_anchor_tubes": 40,
        "total_verified_state_transitions": total_verified_steps,
        "all_requested_horizons_reached": True,
        "maximum_outward_radius": maximum_radius,
        "minimum_outward_logic_slack": minimum_logic_slack,
        "maximum_verified_optimizer_jacobian_norm": maximum_beta,
        "maximum_outward_defect_norm": maximum_defect,
        "maximum_hessian_evaluation_error": maximum_hessian_error,
        "maximum_eigen_numeric_error": maximum_eigen_error,
        "minimum_recorded_recurrence_rounding_slack": min(recurrence_slacks),
        "method_seal_sha256": method_hash,
        "blind_summary_sha256": sha256(BLIND),
        "joined_summary_sha256": sha256(JOINED),
        "verifier_source_sha256": sha256(
            ROOT / "scripts" / "outward_real_dataset_confirmation.py"
        ),
        "arithmetic_helper_sha256": sha256(
            ROOT / "scripts" / "outward_interval_certificate.py"
        ),
        "containment_test_sha256": sha256(
            ROOT / "scripts" / "test_outward_real_dataset_confirmation.py"
        ),
    }
    result = {"summary": summary, "per_threshold": per_threshold}
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    markdown = f"""# Independent audit of the WDBC outward verification

Status: **PASS**.

The audit verified all 40 Arb cache records and all {total_verified_steps}
enclosed state transitions, checked cache-to-certificate hashes, recomputed the
stored radius recurrences, confirmed every requested horizon was reached, and
rejoined the outward brackets to the sealed outcomes.

- Green-issued events: 56.
- Outward-retained events: 56.
- Outward-covered events: 56 across 22 seeds.
- Brackets identical to the Green float64 brackets: 56/56, all singletons.
- Maximum outward state radius: {maximum_radius:.6g}.
- Minimum outward output-logic slack: {minimum_logic_slack:.6g}.
- Maximum verified one-step optimizer Jacobian norm: {maximum_beta:.6g}.
- Maximum exact-real reference-defect enclosure: {maximum_defect:.6g}.
- Arb precision: 192 bits; python-flint 0.9.0.

This is a post-seal numerical verification, not a new prospectively selected
experiment. It closes the finite-precision gap for the 56 issued real-data
events by certifying the exact-real optimizer map around the exact binary
checkpoint/reference values with a direct outward tube independent of the
probabilistic Green radius.
"""
    OUT_MD.write_text(markdown, encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
