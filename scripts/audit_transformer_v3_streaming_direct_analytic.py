#!/usr/bin/env python3
"""Independent arithmetic and provenance audit of the optimized certificate."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

from analytic_jet_release import analytic_jet_release, logit_margin_radius
from transformer_certificate_protocol import Candidate
from transformer_v3_certificate import (
    _persistent_bracket,
    output_path,
    safe_json,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "transformer_v3_streaming_direct_analytic_audit.json"
SCRIPT = ROOT / "scripts" / "benchmark_transformer_v3_streaming_direct_analytic.py"
IDENTITY = RESULTS / "transformer_seed_366_streaming_prefix_identity.json"
CONTINUATION = RESULTS / "transformer_seed_366_matched_continuation.json"
DIRECT_PANEL = RESULTS / "transformer_direct_image_green_panel_audit.json"
PREFIX_PANEL = RESULTS / "transformer_v3_relinearized_prefix_panel_audit.json"
CLAIM_AUDIT = RESULTS / "greencert_manuscript_claim_audit.json"
CANDIDATE = Candidate(366, 0.8, 1120)
REPLICATES = tuple(
    RESULTS
    / (
        "transformer_v3_streaming_direct_analytic_seed_366_gate_1_"
        f"anchor_1120_replicate-{index}.json"
    )
    for index in range(1, 4)
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def close(left: float, right: float, tolerance: float = 3.0e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=1.0e-300)


def index_row(payload: dict, candidate: Candidate) -> dict:
    for row in payload["rows"]:
        if row["candidate"] == candidate.__dict__:
            return row
    raise RuntimeError(f"candidate absent from source panel: {candidate}")


def main() -> None:
    identity = safe_json(IDENTITY)
    continuation = safe_json(CONTINUATION)
    direct = index_row(safe_json(DIRECT_PANEL), CANDIDATE)
    prefix = index_row(safe_json(PREFIX_PANEL), CANDIDATE)
    claim_audit = safe_json(CLAIM_AUDIT)
    certificate_path = output_path(CANDIDATE)
    certificate = safe_json(certificate_path)
    trigger = safe_json(RESULTS / "transformer_hvp_prospective_seed_366.json")

    selected = direct["stage_rows"][0]["direct"]
    kappa = float(selected["operator_norm_upper_bound"])
    forcing_response = kappa * float(prefix["total_corrected_injection_upper"])
    rows = certificate["output_rows"]
    release = analytic_jet_release(
        kappa=kappa,
        corrected_defect_response_bound=forcing_response,
        correction_max_state_norm=float(prefix["correction_max_state_norm"]),
        domain_radius=float(prefix["domain_radius"]),
        learning_rate=float(trigger["config"]["learning_rate"]),
        transition_jets=[
            (
                float(row["block_first"]),
                float(row["block_second"]),
                float(row["block_third"]),
            )
            for row in rows[:-1]
        ],
        output_first_bounds=[float(row["block_first"]) for row in rows],
    )
    if not release.closure.closure_passed:
        raise RuntimeError("independent analytic closure abstained")
    state_radius = float(release.state_radius_about_original_reference)
    # The first persistent start is 2 and step 1 is a certified failure, so
    # step 0 cannot participate in an earlier K=25 run.  This reconstructs the
    # bracket from the committed checkpoint rows without loading the omitted
    # 31.9 MB model archive.
    raw = [(-math.inf, math.inf)] + [
        (float(row["raw_guarantee_slack"]), float(row["raw_exclusion_slack"]))
        for row in rows
    ]
    margins = [0.0] + [
        logit_margin_radius(
            first=float(row["block_first"]), state_radius=state_radius
        )
        for row in rows
    ]
    guarantee = [float(pair[0]) - margin for pair, margin in zip(raw, margins)]
    exclusion = [float(pair[1]) - margin for pair, margin in zip(raw, margins)]
    if exclusion[1] <= 0.0:
        raise RuntimeError("step 1 no longer excludes an earlier persistent event")
    bracket = _persistent_bracket(guarantee, exclusion)
    if bracket != certificate["certified_bracket"] or bracket != direct["bracket"]:
        raise RuntimeError("independent event bracket differs")

    script_hash = sha256(SCRIPT)
    identity_hash = sha256(IDENTITY)
    continuation_hash = sha256(CONTINUATION)
    direct_hash = sha256(DIRECT_PANEL)
    prefix_hash = sha256(PREFIX_PANEL)
    certificate_hash = sha256(certificate_path)
    observed = []
    for path in REPLICATES:
        result = safe_json(path)
        if result["candidate"] != CANDIDATE.__dict__:
            raise RuntimeError(f"candidate mismatch in {path.name}")
        expected_hashes = {
            "script_sha256": script_hash,
            "identity_record_sha256": identity_hash,
            "certificate_sha256": certificate_hash,
            "direct_panel_sha256": direct_hash,
            "corrected_prefix_panel_sha256": prefix_hash,
        }
        for key, expected in expected_hashes.items():
            if result[key] != expected:
                raise RuntimeError(f"{key} mismatch in {path.name}")
        if result["matched_continuation"]["source_sha256"] != continuation_hash:
            raise RuntimeError(f"continuation hash mismatch in {path.name}")
        if int(result["outcome_files_read"]) != 0:
            raise RuntimeError(f"outcome file read in {path.name}")
        if result["certified_bracket"] != bracket or not bool(result["same_bracket"]):
            raise RuntimeError(f"bracket mismatch in {path.name}")
        if int(result["green_forward_probes"]) != 4:
            raise RuntimeError(f"forward-probe count changed in {path.name}")
        if int(result["green_transpose_probes"]) != 0:
            raise RuntimeError(f"transpose query occurred in {path.name}")
        if int(result["randomized_output_operators"]) != 0:
            raise RuntimeError(f"randomized output query occurred in {path.name}")
        if not close(result["green_operator_norm_upper_bound"], kappa):
            raise RuntimeError(f"Green bound mismatch in {path.name}")
        if not close(result["state_radius_about_original_reference"], state_radius):
            raise RuntimeError(f"state radius mismatch in {path.name}")
        if result["analytic_release"] != release.as_dict():
            raise RuntimeError(f"analytic release arithmetic mismatch in {path.name}")
        elapsed = float(result["timings_seconds"]["end_to_end"])
        component_sum = sum(
            float(value)
            for key, value in result["timings_seconds"].items()
            if key != "end_to_end"
        )
        if elapsed < component_sum:
            raise RuntimeError(f"end-to-end time is below component sum in {path.name}")
        if not close(
            result["matched_continuation"]["certificate_to_26_step_ratio"],
            elapsed / float(continuation["median_short_seconds"]),
        ):
            raise RuntimeError(f"26-step timing ratio mismatch in {path.name}")
        if not close(
            result["matched_continuation"]["certificate_to_300_step_ratio"],
            elapsed / float(continuation["median_full_seconds"]),
        ):
            raise RuntimeError(f"300-step timing ratio mismatch in {path.name}")
        observed.append(
            {
                "source": str(path.relative_to(ROOT)).replace("\\", "/"),
                "source_sha256": sha256(path),
                "end_to_end_seconds": elapsed,
            }
        )

    timings = [row["end_to_end_seconds"] for row in observed]
    median_seconds = statistics.median(timings)
    historical_minutes = float(
        claim_audit["implementation"]["historical_candidate_minutes"]
    )
    payload = {
        "status": "independent streamed direct-image analytic-jet audit passed",
        "candidate": CANDIDATE.__dict__,
        "replicates": len(observed),
        "all_same_bracket": True,
        "certified_bracket": bracket,
        "median_end_to_end_seconds": median_seconds,
        "minimum_end_to_end_seconds": min(timings),
        "maximum_end_to_end_seconds": max(timings),
        "matched_26_step_continuation_seconds": float(
            continuation["median_short_seconds"]
        ),
        "matched_300_step_continuation_seconds": float(
            continuation["median_full_seconds"]
        ),
        "median_certificate_to_26_step_ratio": median_seconds
        / float(continuation["median_short_seconds"]),
        "median_certificate_to_300_step_ratio": median_seconds
        / float(continuation["median_full_seconds"]),
        "historical_fixed_q8_cross_batch_median_seconds": historical_minutes * 60.0,
        "historical_to_optimized_cross_batch_ratio": historical_minutes
        * 60.0
        / median_seconds,
        "green_forward_probes": 4,
        "green_transpose_probes": 0,
        "randomized_output_operators": 0,
        "combined_family_failure_upper": 1.0e-6,
        "identity_record_sha256": identity_hash,
        "continuation_record_sha256": continuation_hash,
        "certificate_sha256": certificate_hash,
        "direct_panel_sha256": direct_hash,
        "corrected_prefix_panel_sha256": prefix_hash,
        "benchmark_script_sha256": script_hash,
        "audit_script_sha256": sha256(Path(__file__).resolve()),
        "replicate_rows": observed,
        "outcome_files_read": 0,
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
