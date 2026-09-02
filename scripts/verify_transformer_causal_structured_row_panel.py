#!/usr/bin/env python3
"""Independent scalar verification of the frozen structured row panel."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
import subprocess

from transformer_certificate_protocol import Candidate
from transformer_v3_certificate import output_path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
AUDIT = RESULTS / "transformer_causal_structured_row_panel_audit.json"
BASELINE = RESULTS / "transformer_v3_relinearized_prefix_panel_audit.json"
CACHE = RESULTS / "transformer_causal_structured_row_panel_cache"
OUTPUT = RESULTS / "transformer_causal_structured_row_panel_verification.json"
EXPECTED_CASES = 15
PERSISTENCE = 25


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(left: float, right: float, *, absolute: float = 1.0e-24) -> bool:
    return math.isclose(float(left), float(right), rel_tol=2.0e-12, abs_tol=absolute)


def close_vector(left: list[float], right: list[float], name: str) -> None:
    require(len(left) == len(right), f"{name} length changed")
    require(
        all(close(a, b) for a, b in zip(left, right)),
        f"{name} values changed",
    )


def first_persistent(values: list[bool]) -> int | None:
    for start in range(max(0, len(values) - PERSISTENCE + 1)):
        if all(values[start : start + PERSISTENCE]):
            return start
    return None


def persistent_bracket(
    guarantee: list[float], exclusion: list[float]
) -> tuple[list[int] | None, float | None]:
    lower = first_persistent([value <= 0.0 for value in exclusion])
    upper = first_persistent([value > 0.0 for value in guarantee])
    if lower is None or upper is None or lower > upper:
        return None, None
    prior = [max(exclusion[start : start + PERSISTENCE]) for start in range(lower)]
    lower_slack = math.inf if not prior else min(prior)
    upper_slack = min(guarantee[upper : upper + PERSISTENCE])
    return [lower, upper], min(lower_slack, upper_slack)


def candidate_key(candidate: Candidate) -> str:
    return f"seed_{candidate.seed}_gate_{candidate.gate_index}_anchor_{candidate.anchor}"


def block_path(candidate: Candidate, offset: int) -> Path:
    return CACHE / f"{candidate_key(candidate)}_offset_{offset}.json"


def verify_frozen_sources(audit: dict) -> None:
    commit = audit["frozen_git_commit"]
    subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    for name, expected in audit["source_hashes"].items():
        blob = subprocess.run(
            ["git", "show", f"{commit}:{name}"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        require(sha256_bytes(blob) == expected, f"frozen source hash changed: {name}")
    audit_name = "scripts/audit_transformer_causal_structured_row_panel.py"
    blob = subprocess.run(
        ["git", "show", f"{commit}:{audit_name}"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    require(
        sha256_bytes(blob) == audit["audit_script_sha256"],
        "frozen audit script hash changed",
    )


def verify_cache_identity(block: dict, audit: dict) -> None:
    candidate = Candidate(**block["candidate"])
    identity = {
        "candidate": block["candidate"],
        "offset": int(block["probe_offset"]),
        "block_probes": 4,
        "maximum_probes": 8,
        "stage_delta": float(audit["stage_delta"]),
        "certificate_sha256": sha256(output_path(candidate)),
        "source_hashes": audit["source_hashes"],
        "audit_script_sha256": audit["audit_script_sha256"],
    }
    expected = sha256_bytes(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    )
    require(block["cache_identity"] == expected, "cache identity changed")
    require(int(block["outcome_files_read"]) == 0, "cache block read an outcome")
    require(block["closure_channel"] == "structured_parameter", "wrong channel")
    require(int(block["sweeps"]) == 4, "wrong variational sweep count")


def recompute(blocks: list[dict], stage_delta: float) -> dict:
    reference = blocks[0]
    horizon = int(reference["horizon"])
    total_probes = sum(int(block["probes"]) for block in blocks)
    offsets: list[int] = []
    for block in blocks:
        for name in (
            "candidate",
            "horizon",
            "probe_seed",
            "probe_stream_size",
            "centerline_sha256",
            "corrected_path_sha256",
            "domain_radius",
            "sealed_four_sweep_bracket",
            "raw_event_slacks",
            "output_first_derivative_bounds",
            "active_curvature_bounds",
            "active_forcing_error_bounds",
        ):
            require(block[name] == reference[name], f"block identity changed: {name}")
        offset = int(block["probe_offset"])
        offsets.extend(range(offset, offset + int(block["probes"])))
    require(sorted(offsets) == list(range(total_probes)), "probe blocks overlap or gap")
    require(
        total_probes <= int(reference["probe_stream_size"]),
        "probe prefix exceeds the stream",
    )

    row_budget = float(stage_delta) / horizon
    calibration = NormalDist().inv_cdf(
        0.5 * (1.0 + row_budget ** (1.0 / total_probes))
    )
    maxima = [
        max(float(block["row_image_maxima"][step]) for block in blocks)
        for step in range(horizon)
    ]
    gains = [value / calibration for value in maxima]
    signed = [
        max(float(block["signed_response_row_norms"][step]) for block in blocks)
        for step in range(horizon)
    ]
    errors = [float(value) for value in reference["active_forcing_error_bounds"]]
    curvature = [float(value) for value in reference["active_curvature_bounds"]]

    affine = []
    error_sum = 0.0
    for step in range(horizon):
        error_sum += errors[step] ** 2
        affine.append(signed[step] + gains[step] * math.sqrt(error_sum))
    radii = []
    nonlinear_sum = 0.0
    for step in range(horizon):
        radius = affine[step] + gains[step] * math.sqrt(nonlinear_sum)
        radii.append(radius)
        if step + 1 < horizon:
            forcing = 0.5 * curvature[step + 1] * radius**2
            nonlinear_sum += forcing**2

    domain = float(reference["domain_radius"])
    domain_passed = all(math.isfinite(value) and value <= domain for value in radii)
    margins = [0.0]
    bracket = None
    logic_slack = None
    if domain_passed:
        first = reference["output_first_derivative_bounds"]
        margins.extend(
            math.sqrt(2.0) * float(first[step]) * radii[step - 1]
            for step in range(1, horizon + 1)
        )
        raw = reference["raw_event_slacks"]
        guarantee = [float(pair[0]) - margin for pair, margin in zip(raw, margins)]
        exclusion = [float(pair[1]) - margin for pair, margin in zip(raw, margins)]
        bracket, logic_slack = persistent_bracket(guarantee, exclusion)
    issued = bracket is not None and logic_slack is not None and logic_slack > 0.0
    return {
        "calibration": calibration,
        "maxima": maxima,
        "gains": gains,
        "affine": affine,
        "radii": radii,
        "domain_passed": domain_passed,
        "issued": issued,
        "bracket": bracket,
        "logic_slack": logic_slack,
        "maximum_margin_radius": max(margins),
    }


def verify_row(row: dict, audit: dict) -> None:
    candidate = Candidate(**row["candidate"])
    prefixes = int(row["prefixes_computed"])
    offsets = [0] if prefixes == 4 else [0, 4]
    require(prefixes in (4, 8), "undeclared probe prefix")
    blocks = [load(block_path(candidate, offset)) for offset in offsets]
    for block in blocks:
        verify_cache_identity(block, audit)
    rebuilt = recompute(blocks, float(audit["stage_delta"]))
    require(bool(row["issued"]) == rebuilt["issued"], "issuance changed")
    require(row["bracket"] == rebuilt["bracket"], "bracket changed")
    require(bool(row["row_domain_passed"]) == rebuilt["domain_passed"], "domain changed")
    require(row["bracket"] == row["sealed_four_sweep_bracket"], "sealed bracket changed")
    require(bool(row["released_corrected_path_match"]), "released path mismatch")
    require(int(row["outcome_files_read"]) == 0, "aggregate row read an outcome")
    close_vector(row["row_image_maxima"], rebuilt["maxima"], "row maxima")
    close_vector(row["row_gain_bounds"], rebuilt["gains"], "row gains")
    close_vector(row["row_affine_bounds"], rebuilt["affine"], "affine bounds")
    close_vector(row["row_radii"], rebuilt["radii"], "row radii")
    require(close(row["logic_slack"], rebuilt["logic_slack"]), "logic slack changed")
    require(
        close(row["maximum_margin_radius"], rebuilt["maximum_margin_radius"]),
        "maximum margin changed",
    )


def main() -> None:
    audit = load(AUDIT)
    baseline = load(BASELINE)
    verify_frozen_sources(audit)
    require(int(audit["cases"]) == EXPECTED_CASES, "case denominator changed")
    require(all(int(row["exit_code"]) == 0 for row in audit["preaudit_tests"]), "test failed")

    baseline_candidates = {
        tuple(row["candidate"].values()) for row in baseline["rows"]
    }
    audit_candidates = {tuple(row["candidate"].values()) for row in audit["rows"]}
    require(audit_candidates == baseline_candidates, "candidate panel changed")
    for row in audit["rows"]:
        verify_row(row, audit)

    prefixes = [int(row["prefixes_computed"]) for row in audit["rows"]]
    row_random = sum(prefixes)
    row_signed = EXPECTED_CASES
    common_first_response = EXPECTED_CASES
    row_closure = row_random + row_signed
    row_full = common_first_response + row_closure
    baseline_gram = int(baseline["new_total_green_gram_applications"])
    baseline_random = 2 * baseline_gram
    baseline_full = int(baseline["new_total_theoretical_linearized_sweeps"])
    baseline_closure = baseline_full - common_first_response
    require(baseline_full == 144, "released baseline accounting changed")
    require(baseline_full - baseline_random == 16, "baseline residual accounting changed")
    require(int(baseline["direct_forcing_response_cases"]) == 1, "baseline response count changed")

    minimum_slack = min(float(row["logic_slack"]) for row in audit["rows"])
    maximum_domain_fraction = max(
        float(row["maximum_row_radius"]) / float(row["domain_radius"])
        for row in audit["rows"]
    )
    payload = {
        "status": "structured causal-row panel independently verified",
        "audit_sha256": sha256(AUDIT),
        "frozen_git_commit": audit["frozen_git_commit"],
        "cases_recomputed": len(audit["rows"]),
        "issued_recomputed": sum(bool(row["issued"]) for row in audit["rows"]),
        "brackets_retained_recomputed": sum(
            row["bracket"] == row["sealed_four_sweep_bracket"] for row in audit["rows"]
        ),
        "four_probe_cases": sum(value == 4 for value in prefixes),
        "eight_probe_cases": sum(value == 8 for value in prefixes),
        "minimum_logic_slack": minimum_slack,
        "maximum_radius_to_domain_ratio": maximum_domain_fraction,
        "outcome_files_read": 0,
        "matched_cost_accounting": {
            "baseline_random_forward_plus_transpose_sweeps": baseline_random,
            "row_random_forward_sweeps": row_random,
            "random_operator_sweep_reduction": baseline_random / row_random,
            "baseline_closure_sweeps": baseline_closure,
            "row_closure_sweeps": row_closure,
            "closure_sweep_reduction": baseline_closure / row_closure,
            "baseline_full_post_reference_sweeps": baseline_full,
            "row_full_post_reference_sweeps": row_full,
            "full_post_reference_sweep_reduction": baseline_full / row_full,
            "baseline_transpose_sweeps": baseline_gram,
            "row_transpose_sweeps": 0,
        },
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
