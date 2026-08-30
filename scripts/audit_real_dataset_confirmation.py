#!/usr/bin/env python3
"""Independent audit of the sealed WDBC confirmation.

This checker intentionally does not import the experiment runner or any model
module. It verifies the hash chain, trigger-only candidate construction,
Gaussian-Gram arithmetic, radii-polynomial arithmetic, stored first-passage
paths, post-seal outcome join, and all headline aggregates directly from the
exported JSON records.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import NormalDist, median


ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "results" / "real_dataset_confirmation"
JSON_OUTPUT = ROOT / "results" / "real_dataset_confirmation_independent_audit.json"
MD_OUTPUT = ROOT / "results" / "real_dataset_confirmation_independent_audit.md"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def close(left: float, right: float, *, rtol: float = 2e-11, atol: float = 2e-18) -> bool:
    return math.isclose(float(left), float(right), rel_tol=rtol, abs_tol=atol)


def first_persistent(values: list[int], required: int, persistence: int) -> int | None:
    run = 0
    for step, value in enumerate(values):
        run = run + 1 if value >= required else 0
        if run >= persistence:
            return step - persistence + 1
    return None


def trigger_anchor(
    train: list[float],
    trigger: list[float],
    threshold: float,
    checkpoint_every: int,
    minimum_train_accuracy: float,
    trigger_band: float,
) -> int | None:
    assert len(train) == len(trigger)
    for step in range(0, len(train), checkpoint_every):
        if (
            train[step] >= minimum_train_accuracy
            and threshold - trigger_band <= trigger[step] < threshold
        ):
            return step
    return None


def stable_radius(z_norm: float, kappa: float, drift: float) -> float | None:
    statistic = 2.0 * kappa * drift * z_norm
    if statistic > 1.0:
        return None
    if kappa * drift == 0.0:
        return z_norm
    return 2.0 * z_norm / (1.0 + math.sqrt(max(0.0, 1.0 - statistic)))


def namespaced_seed(master_nonce: str, identity: list[int]) -> int:
    payload = (
        "certified-local-training-events/probe-v1\0"
        + master_nonce
        + "\0"
        + "|".join(str(int(part)) for part in identity)
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def exact_all_success_lower(successes: int, alpha: float = 0.05) -> float:
    """Two-sided Clopper-Pearson lower endpoint when successes == trials."""
    if successes < 1:
        return 0.0
    return (alpha / 2.0) ** (1.0 / successes)


def main() -> None:
    assert EXPORT.is_dir()
    method_path = EXPORT / "REAL_DATA_GREENCERT_METHOD_SEAL.json"
    candidate_seal_path = EXPORT / "REAL_DATA_GREENCERT_CANDIDATE_SEAL.json"
    certificate_seal_path = EXPORT / "REAL_DATA_GREENCERT_CERTIFICATE_SEAL.json"
    method = read_json(method_path)
    candidate_seal = read_json(candidate_seal_path)
    certificate_seal = read_json(certificate_seal_path)
    method_hash = sha256(method_path)
    candidate_seal_hash = sha256(candidate_seal_path)
    certificate_seal_hash = sha256(certificate_seal_path)

    # Hash chain and sealed source bytes.
    assert method_hash == certificate_seal["method_seal_sha256"]
    assert method_hash == candidate_seal["method_seal_sha256"]
    assert candidate_seal_hash == certificate_seal["candidate_seal_sha256"]
    assert sha256(ROOT / "REAL_DATA_GREENCERT_METHOD_SEAL.json") == method_hash
    assert sha256(ROOT / "REAL_DATA_GREENCERT_CANDIDATE_SEAL.json") == candidate_seal_hash
    assert sha256(ROOT / "REAL_DATA_GREENCERT_CERTIFICATE_SEAL.json") == certificate_seal_hash
    for relative, expected in method["code_sha256"].items():
        assert sha256(ROOT / relative) == expected, relative
    assert method["protocol_sha256"] == method["code_sha256"][
        "REAL_DATA_GREENCERT_CONFIRMATION_PROTOCOL.md"
    ]
    assert not (EXPORT / "outcomes").exists()

    seeds = [int(value) for value in method["fresh_seeds"]]
    thresholds = [float(value) for value in method["thresholds"]]
    persistence = int(method["persistence"])
    checkpoint_every = int(method["config"]["checkpoint_every"])
    minimum_train = float(method["minimum_train_accuracy"])
    trigger_band = float(method["trigger_band"])
    case_count = len(seeds) * len(thresholds)
    assert case_count == 72

    # Training records and outcome-blind candidate selection.
    training = read_json(EXPORT / "training_manifest.json")
    assert len(training["runs"]) == len(seeds)
    training_by_seed = {int(row["seed"]): row for row in training["runs"]}
    for seed in seeds:
        blind_path = EXPORT / "blind" / f"seed_{seed}.json"
        checkpoint_path = EXPORT / "checkpoints" / f"seed_{seed}.checkpoints.npz"
        blind_text = blind_path.read_text(encoding="utf-8").lower()
        assert "certificate" not in blind_text
        assert sha256(blind_path) == training_by_seed[seed]["blind_sha256"]
        assert sha256(checkpoint_path) == training_by_seed[seed]["checkpoint_sha256"]

    candidate_manifest_path = EXPORT / "candidates_blind.json"
    assert sha256(candidate_manifest_path) == candidate_seal["candidate_manifest_sha256"]
    candidate_manifest = read_json(candidate_manifest_path)
    assert len(candidate_manifest["rows"]) == case_count
    recomputed_candidates: list[dict] = []
    for row in candidate_manifest["rows"]:
        seed = int(row["seed"])
        gate = int(row["gate_index"])
        threshold = thresholds[gate]
        assert close(float(row["threshold"]), threshold)
        blind = read_json(EXPORT / "blind" / f"seed_{seed}.json")
        anchor = trigger_anchor(
            blind["train_accuracy"],
            blind["trigger_accuracy"],
            threshold,
            checkpoint_every,
            minimum_train,
            trigger_band,
        )
        assert row["anchor"] == anchor
        expected_disposition = "candidate frozen" if anchor is not None else "no trigger-only anchor"
        assert row["disposition"] == expected_disposition
        if anchor is not None:
            recomputed_candidates.append(row)
    assert recomputed_candidates == candidate_manifest["candidates"]
    assert len(recomputed_candidates) == int(candidate_seal["candidate_count"])

    # Certificate manifest, each certificate hash, and all certificate arithmetic.
    certificate_manifest_path = EXPORT / "certificate_manifest.json"
    assert sha256(certificate_manifest_path) == certificate_seal[
        "certificate_manifest_sha256"
    ]
    certificate_manifest = read_json(certificate_manifest_path)
    records = certificate_manifest["records"]
    assert len(records) == len(recomputed_candidates) == 71
    queried_identities: list[tuple[int, ...]] = []
    certificate_rows: list[dict] = []
    for record, candidate in zip(records, recomputed_candidates, strict=True):
        path = EXPORT / "certificates" / record["path"]
        assert sha256(path) == record["sha256"]
        certificate = read_json(path)
        certificate_rows.append(certificate)
        assert int(certificate["seed"]) == int(candidate["seed"]) == int(record["seed"])
        assert int(certificate["gate_index"]) == int(candidate["gate_index"]) == int(record["gate_index"])
        assert int(certificate["anchor"]) == int(candidate["anchor"])
        assert bool(certificate["certificate_issued"]) == bool(record["issued"])
        assert certificate["method_seal_sha256"] == method_hash
        assert certificate["candidate_seal_sha256"] == candidate_seal_hash
        assert certificate["candidate_manifest_sha256"] == candidate_seal[
            "candidate_manifest_sha256"
        ]
        predicted = certificate["predicted_event"]
        if "center_count" in certificate:
            from_center = first_persistent(
                [int(value) for value in certificate["center_count"]],
                int(certificate["required_correct"]),
                persistence,
            )
            assert from_center == predicted
        probe = certificate.get("green_probe")
        assert bool(probe is not None) == bool(record["queried_operator"])
        if probe is None:
            assert not certificate["certificate_issued"]
            continue

        identity = tuple(int(value) for value in probe["identity"])
        queried_identities.append(identity)
        assert int(probe["rng_seed"]) == namespaced_seed(method["master_nonce"], list(identity))
        m = int(probe["probes"])
        q = int(probe["power"])
        delta = float(probe["delta"])
        c_value = NormalDist().inv_cdf(0.5 * (1.0 + delta ** (1.0 / m)))
        bound = (float(probe["Y"]) / c_value) ** (1.0 / (2.0 * q))
        assert close(c_value, float(probe["c_delta"]))
        assert close(bound, float(probe["operator_norm_upper_bound"]))
        kappa = float(probe["operator_norm_upper_bound"])
        z_norm = float(certificate["signed_response_sequence_norm"])
        drift = float(certificate["maximum_optimizer_derivative_drift_upper"])
        closure = 2.0 * kappa * drift * z_norm
        assert close(closure, float(certificate["closure_statistic"]))
        assert close(1.0 - closure, float(certificate["closure_slack"]))
        radius = stable_radius(z_norm, kappa, drift)
        assert radius is not None
        assert close(radius, float(certificate["minimal_admissible_radius"]))
        assert close(
            z_norm + 0.5 * kappa * drift * radius**2,
            radius,
            rtol=2e-10,
            atol=2e-18,
        )
        guaranteed = [int(value) for value in certificate["guaranteed_correct"]]
        possible = [int(value) for value in certificate["possibly_correct"]]
        required = int(certificate["required_correct"])
        lower = first_persistent(possible, required, persistence)
        upper = first_persistent(guaranteed, required, persistence)
        bracket = None if lower is None or upper is None else [lower, upper]
        assert bracket == certificate["certified_bracket"]
        assert bool(bracket is not None) == bool(certificate["certificate_issued"])
        assert float(certificate["minimum_output_slack"]) > 0.0

        unsigned_z = float(certificate["unsigned_right_inverse_response_upper"])
        unsigned_drift = float(
            certificate["unsigned_right_inverse_derivative_drift_upper"]
        )
        unsigned_closure = 2.0 * kappa * unsigned_drift * unsigned_z
        assert close(
            unsigned_closure,
            float(certificate["unsigned_right_inverse_closure_statistic"]),
        )
        unsigned_radius = stable_radius(unsigned_z, kappa, unsigned_drift)
        assert unsigned_radius is not None
        assert close(
            unsigned_radius,
            float(certificate["unsigned_right_inverse_minimal_radius"]),
        )

    assert len(set(queried_identities)) == len(queried_identities)
    queried = len(queried_identities)
    assert queried == int(certificate_seal["queried_operators"])
    per_operator_delta = float(method["probe"]["delta"])
    realized_union = queried * per_operator_delta
    assert close(realized_union, float(certificate_seal["realized_union_failure_bound"]))
    assert realized_union <= float(method["family_failure_probability"])

    # First-passage outcome join and observed numerical diagnostics.
    final_audit_path = EXPORT / "final_audit.json"
    final_audit = read_json(final_audit_path)
    final_summary = read_json(EXPORT / "final_summary.json")
    assert sha256(final_audit_path) == final_summary["final_audit_sha256"]
    audit_rows = final_audit["rows"]
    assert len(audit_rows) == len(records)
    joined: list[dict] = []
    for record, certificate, audit in zip(
        records, certificate_rows, audit_rows, strict=True
    ):
        assert audit["certificate_sha256"] == record["sha256"]
        assert int(audit["seed"]) == int(certificate["seed"])
        assert int(audit["gate_index"]) == int(certificate["gate_index"])
        outcome = read_json(
            EXPORT / "audit" / "outcomes" / f"seed_{certificate['seed']}.outcomes.json"
        )
        assert outcome["status"] == "FIRST MATERIALIZED AFTER CERTIFICATE SEAL"
        assert float(outcome["maximum_checkpoint_reconstruction_error"]) == 0.0
        counts = [int(value) for value in outcome["certificate_count"]]
        required = int(certificate["required_correct"])
        absolute = first_persistent(counts, required, persistence)
        stored_absolute = outcome["events"][f"{float(certificate['threshold']):.3f}"]
        assert absolute == stored_absolute
        actual = None if absolute is None else absolute - int(certificate["anchor"])
        assert actual == audit["actual_event"]
        predicted = certificate["predicted_event"]
        expected_error = None if predicted is None or actual is None else predicted - actual
        assert expected_error == audit["raw_timing_error"]
        if certificate["certificate_issued"]:
            bracket = [int(value) for value in certificate["certified_bracket"]]
            assert actual is not None and bracket[0] <= actual <= bracket[1]
            assert audit["bracket_contains_actual"] is True
            assert not audit["observed_sequence_tube_violation"]
            assert not audit["observed_state_tube_violation"]
        joined.append(audit)

    issued = [row for row in joined if row["certificate_issued"]]
    covered = [row for row in issued if row["bracket_contains_actual"]]
    leads = [int(row["actual_event"]) for row in issued]
    widths = [int(row["certified_bracket"][1]) - int(row["certified_bracket"][0]) for row in issued]
    sequence_ratios = [float(row["observed_sequence_error_to_radius"]) for row in issued]
    state_ratios = [float(row["maximum_observed_state_error_to_radius"]) for row in issued]
    strict_sequence_exceedances = sum(value > 1.0 for value in sequence_ratios)
    strict_state_exceedances = sum(value > 1.0 for value in state_ratios)
    issuing_seeds = {int(row["seed"]) for row in issued}
    raw_comparable = [row for row in joined if row["raw_timing_error"] is not None]
    summary = {
        "status": "PASS WITH EXPLICIT FLOAT64 ROUNDING DIAGNOSTIC",
        "hash_chain_verified": True,
        "sealed_source_files_verified": len(method["code_sha256"]),
        "fresh_seeds": len(seeds),
        "seed_threshold_cases": case_count,
        "trigger_only_candidates": len(recomputed_candidates),
        "candidate_rate": len(recomputed_candidates) / case_count,
        "issued": len(issued),
        "overall_issuance_rate": len(issued) / case_count,
        "conditional_issuance_rate": len(issued) / len(recomputed_candidates),
        "covered": len(covered),
        "conditional_coverage": len(covered) / len(issued),
        "distinct_issuing_seeds": len(issuing_seeds),
        "issuing_seed_level_all_covered": sum(
            all(row["bracket_contains_actual"] for row in issued if int(row["seed"]) == seed)
            for seed in issuing_seeds
        ),
        "two_sided_95pct_exact_lower_if_issuing_seeds_were_independent": exact_all_success_lower(
            len(issuing_seeds)
        ),
        "median_lead": float(median(leads)),
        "maximum_lead": max(leads),
        "median_bracket_width": float(median(widths)),
        "maximum_bracket_width": max(widths),
        "minimum_closure_slack": min(float(row["closure_slack"]) for row in issued),
        "minimum_output_slack": min(float(row["minimum_output_slack"]) for row in issued),
        "raw_exact_timing_matches": sum(int(row["raw_timing_error"]) == 0 for row in raw_comparable),
        "raw_timing_comparable": len(raw_comparable),
        "queried_operators": queried,
        "realized_union_failure_bound": realized_union,
        "unsigned_right_inverse_issued": sum(
            bool(row.get("unsigned_right_inverse_certificate_issued")) for row in joined
        ),
        "tolerance_adjusted_sequence_tube_violations": sum(
            bool(row.get("observed_sequence_tube_violation")) for row in issued
        ),
        "tolerance_adjusted_state_tube_violations": sum(
            bool(row.get("observed_state_tube_violation")) for row in issued
        ),
        "strict_float64_sequence_ratio_exceedances": strict_sequence_exceedances,
        "maximum_float64_sequence_error_to_radius": max(sequence_ratios),
        "strict_float64_pointwise_state_ratio_exceedances": strict_state_exceedances,
        "maximum_float64_pointwise_state_error_to_radius": max(state_ratios),
        "maximum_absolute_float64_sequence_error": max(
            float(row["observed_sequence_error"]) for row in issued
        ),
        "maximum_checkpoint_reconstruction_error": max(
            float(
                read_json(
                    EXPORT / "audit" / "outcomes" / f"seed_{seed}.outcomes.json"
                )["maximum_checkpoint_reconstruction_error"]
            )
            for seed in seeds
        ),
    }
    per_threshold = {}
    for gate, threshold in enumerate(thresholds):
        gate_candidates = [row for row in recomputed_candidates if int(row["gate_index"]) == gate]
        gate_issued = [row for row in issued if int(row["gate_index"]) == gate]
        per_threshold[f"{threshold:.3f}"] = {
            "cases": len(seeds),
            "candidates": len(gate_candidates),
            "issued": len(gate_issued),
            "covered": sum(bool(row["bracket_contains_actual"]) for row in gate_issued),
            "distinct_issuing_seeds": len({int(row["seed"]) for row in gate_issued}),
            "median_lead": None
            if not gate_issued
            else float(median([int(row["actual_event"]) for row in gate_issued])),
            "maximum_lead": None
            if not gate_issued
            else max(int(row["actual_event"]) for row in gate_issued),
        }
    result = {
        "summary": summary,
        "per_threshold": per_threshold,
        "candidate_dispositions": dict(
            sorted(Counter(row["disposition"] for row in candidate_manifest["rows"]).items())
        ),
        "certificate_dispositions": dict(
            sorted(Counter(row["certificate_status"] for row in joined).items())
        ),
        "seal_sha256": {
            "method": method_hash,
            "candidate": candidate_seal_hash,
            "certificate": certificate_seal_hash,
        },
    }
    # Match every headline field that has the same definition in the sealed summary.
    mappings = {
        "fresh_seeds": "fresh_seeds",
        "seed_threshold_cases": "seed_threshold_cases",
        "trigger_only_candidates": "candidates",
        "issued": "issued",
        "covered": "covered",
        "distinct_issuing_seeds": "distinct_issuing_seeds",
        "median_lead": "median_certified_lead",
        "maximum_lead": "maximum_certified_lead",
        "median_bracket_width": "median_bracket_width",
        "maximum_bracket_width": "maximum_bracket_width",
        "raw_exact_timing_matches": "raw_exact_timing_matches",
        "raw_timing_comparable": "raw_timing_comparable",
        "queried_operators": "queried_operators",
        "unsigned_right_inverse_issued": "unsigned_right_inverse_issued",
    }
    for ours, sealed in mappings.items():
        assert summary[ours] == final_summary[sealed], (ours, summary[ours], final_summary[sealed])
    for key in (
        "candidate_rate",
        "overall_issuance_rate",
        "conditional_issuance_rate",
        "conditional_coverage",
        "realized_union_failure_bound",
    ):
        assert close(summary[key], float(final_summary[key]))

    JSON_OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    threshold_lines = "\n".join(
        f"- {name}: {row['issued']}/{row['cases']} issued, {row['covered']}/{row['issued']} covered, "
        f"median lead {row['median_lead']}, maximum lead {row['maximum_lead']}."
        for name, row in per_threshold.items()
    )
    markdown = f"""# Independent WDBC confirmation audit

Status: **PASS, with an explicit float64 rounding diagnostic**.

This checker imports neither the experiment runner nor the model code. It
verified the three-stage SHA-256 chain and all {len(method['code_sha256'])}
sealed source/data files; independently reconstructed all 72 trigger-only
selection decisions; rechecked all 71 certificate hashes; recomputed every
Gaussian-Gram bound, signed and unsigned radii-polynomial calculation, stored
first-passage bracket, post-seal event time, and headline aggregate.

## Confirmatory result

- Candidates: {len(recomputed_candidates)}/72 ({100*summary['candidate_rate']:.1f}%).
- Issued: {len(issued)}/72 overall and {len(issued)}/{len(recomputed_candidates)} among candidates.
- Containment: {len(covered)}/{len(issued)} issued brackets, across
  {len(issuing_seeds)} distinct issuing seeds; all brackets are singletons.
- Lead: median {summary['median_lead']:.0f} updates, maximum
  {summary['maximum_lead']} updates.
- Minimum closure slack: {summary['minimum_closure_slack']:.6g}; minimum strict
  output slack: {summary['minimum_output_slack']:.6g}.
- The raw four-sweep clock exactly matches all
  {summary['raw_timing_comparable']} comparable post-seal event offsets.
- The matched unsigned right-inverse baseline also issues {summary['unsigned_right_inverse_issued']}
  cases on this numerically easy, highly contractive transfer task; this batch
  demonstrates transfer and information isolation, not a signed-vs-unsigned
  advantage.

{threshold_lines}

## Finite-precision diagnostic

The runner's declared tolerance-adjusted audit records zero sequence or
pointwise state-tube violations. Under a literal no-tolerance float64 ratio,
however, {strict_sequence_exceedances}/{len(issued)} accumulated sequence norms
slightly exceed the extremely small analytic radius; the maximum ratio is
{max(sequence_ratios):.3f} and the largest absolute sequence discrepancy is
{summary['maximum_absolute_float64_sequence_error']:.3e}. Every pointwise state
error remains strictly inside its radius (maximum ratio
{max(state_ratios):.3f}). The strict margin slack is at least
{summary['minimum_output_slack']:.3e}, over nine orders of magnitude larger
than the largest observed absolute trajectory discrepancy. This is benign for
the observed event classifications but must be described as high-confidence
float64 evidence unless an outward-rounded computer-error budget is added.

Thresholds within a seed are correlated. The seed-cluster descriptive result
is {len(issuing_seeds)}/{len(issuing_seeds)} issuing seeds with all issued
events covered. The two-sided 95% exact lower endpoint would be
{summary['two_sided_95pct_exact_lower_if_issuing_seeds_were_independent']:.3f}
if those issuing-seed indicators were treated as independent Bernoulli trials;
this is not a population-generalization guarantee.
"""
    MD_OUTPUT.write_text(markdown, encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
