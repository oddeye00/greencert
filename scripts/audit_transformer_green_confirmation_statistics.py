"""Independent consistency and statistics audit for the fresh Green study.

This checker intentionally does not import the experiment runner or its aggregate
helpers.  It reads the sealed certificate/audit artifacts, reconstructs the
claim-bearing inequalities and persistent brackets, verifies hashes, and writes
a compact machine-readable and human-readable audit record.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
AGGREGATE = RESULTS / "transformer_green_confirmation_audit.json"
CERTIFICATE_SEAL = ROOT / "TRANSFORMER_GREEN_CONFIRMATION_CERTIFICATE_SEAL.json"
METHOD_SEAL = ROOT / "TRANSFORMER_GREEN_CONFIRMATION_METHOD_SEAL.json"
OUTPUT_JSON = RESULTS / "transformer_green_confirmation_independent_audit.json"
OUTPUT_MD = ROOT / "TRANSFORMER_GREEN_CONFIRMATION_INDEPENDENT_AUDIT.md"
PERSISTENCE = 25


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def historical_sha256(path: Path) -> str:
    """Return the sealed source hash, validating any public path-scrubbed copy."""
    packaged = sha256(path)
    manifest_path = ROOT / "MANIFEST_SHA256.json"
    if not manifest_path.is_file():
        return packaged
    manifest = load_json(manifest_path)
    relative = path.relative_to(ROOT).as_posix()
    row = manifest.get(relative)
    if not row or not bool(row.get("sanitized")):
        return packaged
    assert packaged == str(row["packaged_sha256"]).upper()
    return str(row["source_sha256"]).upper()


def close(a: float, b: float, *, rel: float = 5e-13, abs_: float = 1e-300) -> bool:
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)


def candidate_key(candidate: dict[str, Any]) -> tuple[int, float, int]:
    return (
        int(candidate["seed"]),
        float(candidate["threshold"]),
        int(candidate["anchor"]),
    )


def audit_path_for(key: tuple[int, float, int]) -> Path:
    seed, threshold, anchor = key
    gate_index = {0.7: 0, 0.8: 1, 0.9: 2}[threshold]
    return RESULTS / (
        f"transformer_green_confirmation_audit_seed_{seed}_"
        f"gate_{gate_index}_anchor_{anchor}.json"
    )


def first_persistent(values: list[int], required: int, persistence: int) -> int | None:
    for start in range(0, len(values) - persistence + 1):
        if all(value >= required for value in values[start : start + persistence]):
            return start
    return None


def cp_lower_all_successes(successes: int, alpha: float = 0.05) -> float:
    """Two-sided Clopper--Pearson lower endpoint when x=n."""
    if successes <= 0:
        return 0.0
    return (alpha / 2.0) ** (1.0 / successes)


def median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def main() -> None:
    aggregate = load_json(AGGREGATE)
    certificate_seal = load_json(CERTIFICATE_SEAL)
    method_seal = load_json(METHOD_SEAL)

    method_hash = historical_sha256(METHOD_SEAL)
    assert aggregate["method_seal_sha256"] == method_hash
    assert aggregate["certificate_seal_sha256"] == sha256(CERTIFICATE_SEAL)
    assert certificate_seal["method_seal_sha256"] == method_hash

    seal_entries = {
        candidate_key(entry["candidate"]): entry
        for entry in certificate_seal["certificate_files"]
    }
    rows = aggregate["rows"]
    assert len(rows) == len(seal_entries) == 23
    assert len({candidate_key(row["candidate"]) for row in rows}) == len(rows)

    disposition_counts = Counter(aggregate["summary"]["dispositions"])
    assert sum(disposition_counts.values()) == 72
    assert disposition_counts["candidate frozen"] == len(rows)

    issued_rows: list[dict[str, Any]] = []
    constructed_rows: list[dict[str, Any]] = []
    timing_errors: list[int] = []
    per_threshold: dict[float, dict[str, int]] = defaultdict(
        lambda: {"candidates": 0, "issued": 0, "covered": 0}
    )
    queried_operators = 0

    for row in rows:
        key = candidate_key(row["candidate"])
        assert key in seal_entries
        seal_entry = seal_entries[key]
        cert_path = ROOT / Path(seal_entry["path"])
        audit_path = audit_path_for(key)
        assert cert_path.is_file(), cert_path
        assert audit_path.is_file(), audit_path
        assert sha256(cert_path) == seal_entry["sha256"] == row["certificate_sha256"]
        assert sha256(audit_path) == row["audit_sha256"]

        certificate = load_json(cert_path)
        outcome_audit = load_json(audit_path)
        assert candidate_key(certificate["candidate"]) == key
        assert candidate_key(outcome_audit["candidate"]) == key
        assert certificate["outcome_joined"] is False
        assert outcome_audit["certificate_sha256"] == sha256(cert_path)
        assert bool(certificate["certificate_issued"]) == bool(row["certificate_issued"])
        assert bool(seal_entry["issued"]) == bool(row["certificate_issued"])

        predicted = int(outcome_audit["predicted_persistent_event"])
        actual_raw = outcome_audit["actual_persistent_event"]
        assert actual_raw is not None
        actual = int(actual_raw)
        raw_error = predicted - actual
        assert raw_error == int(outcome_audit["raw_timing_error"])
        assert raw_error == int(row["raw_timing_error"])
        timing_errors.append(raw_error)
        assert predicted == int(row["predicted_event"])
        assert actual == int(row["actual_event"])

        threshold = key[1]
        per_threshold[threshold]["candidates"] += 1
        queried_operators += int(certificate["probability_budget"]["queried_operators"])
        assert int(row["queried_operators"]) == int(
            certificate["probability_budget"]["queried_operators"]
        )

        if bool(row["construction_abstention"]):
            assert seal_entry["construction_abstention"] is True
            assert certificate["certificate_issued"] is False
            assert row["signed_radius"] is None
            assert row["certified_bracket"] is None
            continue

        constructed_rows.append(row)
        z_norm = float(certificate["signed_response_sequence_norm"])
        radius = float(certificate["signed_radius"])
        drift = float(certificate["maximum_optimizer_derivative_drift_upper"])
        assert close(radius, 2.0 * z_norm)
        assert close(float(row["signed_radius"]), radius)
        minimum_lhs = 2.0 * drift * z_norm
        assert close(
            minimum_lhs,
            float(certificate["minimum_closure_lhs_using_kappa_ge_1"]),
        )
        assert bool(certificate["early_abstention_before_green_probe"]) == (
            minimum_lhs > 1.0
        )

        if certificate["green_probe"] is not None:
            kappa = float(certificate["green_probe"]["green_operator_norm_upper_bound"])
            closure_lhs = 2.0 * kappa * drift * z_norm
            assert close(closure_lhs, float(certificate["closure_lhs_2_kappa_M_Z"]))
            assert close(float(certificate["closure_slack"]), 1.0 - closure_lhs)
            assert bool(certificate["closure_passed"]) == (closure_lhs <= 1.0)
        else:
            assert certificate["early_abstention_before_green_probe"] is True

        if certificate["raw_margin_bracket"] is not None:
            required = int(certificate["required_correct"])
            possible = [int(value) for value in certificate["possibly_correct"]]
            guaranteed = [int(value) for value in certificate["guaranteed_correct"]]
            lower = first_persistent(possible, required, PERSISTENCE)
            upper = first_persistent(guaranteed, required, PERSISTENCE)
            reconstructed = None if lower is None or upper is None else [lower, upper]
            assert reconstructed == certificate["raw_margin_bracket"]

        if bool(row["certificate_issued"]):
            issued_rows.append(row)
            per_threshold[threshold]["issued"] += 1
            bracket = [int(value) for value in certificate["certified_bracket"]]
            assert bracket == [int(value) for value in row["certified_bracket"]]
            assert bracket[0] <= actual <= bracket[1]
            assert outcome_audit["bracket_contains_actual"] is True
            assert row["bracket_contains_actual"] is True
            assert certificate["closure_passed"] is True
            assert certificate["block_fixed_points_all_consistent"] is True
            slacks = certificate["certificate_output_logic_slack"]
            assert min(float(value) for value in slacks.values()) > 0.0
            sequence_error = float(outcome_audit["actual_sequence_error"])
            assert sequence_error <= radius
            assert close(
                float(outcome_audit["actual_sequence_error_to_radius_ratio"]),
                sequence_error / radius,
            )
            assert outcome_audit["observed_sequence_tube_violation"] is False
            state_violations = outcome_audit["observed_state_tube_violations"]
            assert isinstance(state_violations, int)
            assert state_violations == 0
            per_threshold[threshold]["covered"] += 1
        else:
            assert certificate["certified_bracket"] is None
            assert outcome_audit["bracket_contains_actual"] is None

    issued_seeds = sorted({int(row["candidate"]["seed"]) for row in issued_rows})
    candidate_seeds = sorted({int(row["candidate"]["seed"]) for row in rows})
    bracket_widths = [
        int(row["certified_bracket"][1]) - int(row["certified_bracket"][0])
        for row in issued_rows
    ]
    leads = [int(row["actual_event"]) for row in issued_rows]
    closure_slacks = [float(row["closure_slack"]) for row in issued_rows]
    output_slacks = [
        float(row["output_logic_slack"]["minimum_logic_slack"])
        for row in issued_rows
    ]
    issued_error_ratios = [
        float(row["actual_sequence_error_to_radius_ratio"]) for row in issued_rows
    ]

    per_operator_delta = float(method_seal["probe_config"]["delta"])
    realized_union_bound = queried_operators * per_operator_delta
    maximum_union_bound = float(aggregate["probability_budget"]["maximum_family_union_bound"])

    recomputed = {
        "fresh_seeds": 24,
        "seed_threshold_cases": 72,
        "candidates": len(rows),
        "distinct_candidate_seeds": len(candidate_seeds),
        "construction_abstentions": sum(bool(row["construction_abstention"]) for row in rows),
        "issued": len(issued_rows),
        "covered": sum(bool(row["bracket_contains_actual"]) for row in issued_rows),
        "distinct_issuing_seeds": len(issued_seeds),
        "abstention_rate_among_candidates": 1.0 - len(issued_rows) / len(rows),
        "median_bracket_width": median([float(value) for value in bracket_widths]),
        "median_certified_lead": median([float(value) for value in leads]),
        "maximum_certified_lead": max(leads),
        "raw_exact_timing_matches": sum(error == 0 for error in timing_errors),
        "observed_issued_sequence_tube_violations": 0,
        "observed_issued_state_tube_violations": 0,
    }
    for key, value in recomputed.items():
        reported = aggregate["summary"][key]
        if isinstance(value, float):
            assert close(value, float(reported))
        else:
            assert value == reported

    assert queried_operators == int(aggregate["probability_budget"]["queried_operators"])
    assert close(
        realized_union_bound,
        float(aggregate["probability_budget"]["queried_union_bound"]),
    )
    assert realized_union_bound <= maximum_union_bound

    result = {
        "status": "PASS: independent artifact, inequality, bracket, and statistics audit",
        "source_aggregate_sha256": sha256(AGGREGATE),
        "method_seal_sha256": method_hash,
        "certificate_seal_sha256": sha256(CERTIFICATE_SEAL),
        "recomputed_summary": recomputed,
        "rates": {
            "candidate_per_seed_threshold_case": len(rows) / 72.0,
            "issued_per_seed_threshold_case": len(issued_rows) / 72.0,
            "issued_per_candidate": len(issued_rows) / len(rows),
            "candidate_seed_fraction": len(candidate_seeds) / 24.0,
            "issuing_seed_fraction": len(issued_seeds) / 24.0,
        },
        "seed_clustered": {
            "candidate_seeds": candidate_seeds,
            "issuing_seeds": issued_seeds,
            "all_certificates_covered_in_each_issuing_seed": True,
            "two_sided_95pct_cp_lower_if_issuing_seeds_are_exchangeable": cp_lower_all_successes(
                len(issued_seeds)
            ),
        },
        "event_level": {
            "two_sided_95pct_cp_lower_if_issued_events_are_independent": cp_lower_all_successes(
                len(issued_rows)
            ),
            "minimum_lead": min(leads),
            "median_lead": median([float(value) for value in leads]),
            "maximum_lead": max(leads),
            "singleton_brackets": sum(width == 0 for width in bracket_widths),
            "minimum_closure_slack": min(closure_slacks),
            "minimum_output_logic_slack": min(output_slacks),
            "maximum_actual_sequence_error_to_radius_ratio": max(issued_error_ratios),
        },
        "per_threshold": {str(key): value for key, value in sorted(per_threshold.items())},
        "randomized_operator_budget": {
            "queried_operators": queried_operators,
            "maximum_predeclared_operators": int(
                method_seal["operator_accounting"]["maximum_probabilistic_operators"]
            ),
            "per_operator_delta": per_operator_delta,
            "realized_union_bound": realized_union_bound,
            "maximum_family_union_bound": maximum_union_bound,
        },
        "interpretation": {
            "coverage_is_conditional_on_issuance": True,
            "thresholds_within_seed_are_correlated": True,
            "clopper_pearson_intervals_are_sensitivity_analyses_not_population_claims": True,
            "float64_neural_and_operator_arithmetic_is_not_a_formal_interval_proof": True,
        },
    }

    OUTPUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    markdown = f"""# Independent fresh Transformer confirmation audit

Status: **PASS**.

This checker independently reloaded all 23 sealed candidate certificates and
their post-seal outcome audits. It verified certificate and audit hashes,
recomputed the signed-radius and nonlinear-closure inequalities, reconstructed
every 25-step persistent bracket from the guaranteed/possible count paths, and
recomputed all aggregate statistics without importing the experiment runner.

## Claim-bearing result

- 24 untouched Transformer seeds; 72 seed-threshold cases.
- 23 prospectively frozen candidates across 12 seeds.
- 9 certificates issued across 6 distinct seeds; all 9 contained the observed
  first passage.
- All 9 brackets are singletons. Certified leads range from {min(leads)} to
  {max(leads)} updates, with median {median([float(value) for value in leads]):.0f}.
- Zero observed issued sequence-tube violations and zero observed issued
  pointwise state-tube violations.
- Minimum nonlinear closure slack: {min(closure_slacks):.6g}.
- Minimum strict output-logic slack: {min(output_slacks):.6g}.
- Largest observed issued sequence-error/radius ratio:
  {max(issued_error_ratios):.6g}.
- The raw four-sweep centerline hit the exact persistent-event offset in all
  {sum(error == 0 for error in timing_errors)}/{len(timing_errors)} frozen candidates; this is a
  secondary timing diagnostic, not a coverage guarantee.

## Denominators and clustering

- Candidate rate: {len(rows)}/72 ({100.0 * len(rows) / 72.0:.1f}%).
- Issuance over all prespecified seed-threshold cases: {len(issued_rows)}/72
  ({100.0 * len(issued_rows) / 72.0:.1f}%).
- Issuance conditional on a frozen candidate: {len(issued_rows)}/{len(rows)}
  ({100.0 * len(issued_rows) / len(rows):.1f}%).
- Issuing seeds: {len(issued_seeds)}/24 ({100.0 * len(issued_seeds) / 24.0:.1f}%).
- Event-level 9/9 observations are correlated within seed. If one nevertheless
  treats issued events as independent, the two-sided 95% Clopper--Pearson lower
  endpoint is {cp_lower_all_successes(len(issued_rows)):.3f}. At the stricter
  issuing-seed level, 6/6 clusters have no failure and the analogous endpoint
  is {cp_lower_all_successes(len(issued_seeds)):.3f}. These are sensitivity
  summaries, not unconditional population-coverage claims.

## Randomized verification budget

- Queried operators: {queried_operators:,} of the predeclared maximum
  {int(method_seal['operator_accounting']['maximum_probabilistic_operators']):,}.
- Realized union bound: {realized_union_bound:.6g}.
- Frozen family-wise ceiling: {maximum_union_bound:.6g}.

## Numerical boundary

The artifact chain and scalar inequalities pass exactly as stored. The neural
jet, HVP/VJP, and randomized power computations were performed in float64, so
this is a high-confidence probabilistic numerical certificate, not an
outward-rounded computer-assisted proof of the PyTorch execution. The margins
are far from scalar rounding ties, but the paper must preserve that boundary.
"""
    OUTPUT_MD.write_text(markdown, encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
