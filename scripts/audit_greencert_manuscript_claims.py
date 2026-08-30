#!/usr/bin/env python3
"""Cross-check headline manuscript claims against immutable result records.

This audit is intentionally independent of the experiment aggregators.  It
loads their final JSON products, recomputes the cross-study totals and several
easy-to-misstate edge cases, then checks that the manuscript contains the
corresponding scoped language and no known stale wording.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "certified_local_training_events_neurips2026.tex"
OUTPUT = ROOT / "results" / "greencert_manuscript_claim_audit.json"


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(left: float, right: float, tolerance: float = 5.0e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=0.0)


def main() -> None:
    paper = PAPER.read_text(encoding="utf-8")
    wdbc = load("results/real_dataset_confirmation/final_summary.json")
    wdbc_audit = load("results/real_dataset_confirmation/final_audit.json")
    digits = load("results/digits_signed_confirmation_summary.json")
    selectivity = load("results/forecast_selectivity_audit.json")
    transformer = load("results/transformer_green_confirmation_audit.json")
    v3 = load("results/transformer_v3_confirmation_audit.json")
    unsigned = load("results/transformer_unsigned_right_inverse_audit.json")
    wdbc_outward = load("results/real_dataset_outward_independent_audit.json")
    digits_outward = load("results/digits_outward_joined.json")
    online = load("results/transformer_v3_online_policy_matched_audit.json")
    panel = load("results/transformer_v3_role_sparse_panel_audit.json")
    combined = load(
        "results/transformer_v3_combined_online_role_seed_366_gate_1_anchor_1120_"
        "matched-combined-v5_independent_audit.json"
    )
    inexact = load("results/transformer_v3_inexact_operator_tolerance_postseal_audit.json")
    outward_inexact = load("results/transformer_v3_outward_inexact_root_postseal_audit.json")
    chi = load("results/transformer_v3_chi_block_postseal_audit.json")
    mixed = load("results/transformer_v3_mixed_precision_residual_independent_audit.json")
    mixed_timing = load("results/transformer_v3_mixed_precision_timing_aggregate_audit.json")
    weighted = load("results/transformer_v3_weighted_green_postseal_audit.json")
    scale = load("results/transformer_batched_scaling_benchmark.json")
    legacy_scale = load("results/transformer_scaling_benchmark.json")
    output_recenter = load("results/transformer_v3_output_recentering_independent_audit.json")
    two_response = load("results/transformer_v3_two_response_independent_audit.json")
    two_response_postseal = load(
        "results/transformer_v3_two_response_postseal_audit.json"
    )
    two_response_timing = load(
        "results/transformer_v3_two_response_paired_benchmark_independent_audit.json"
    )
    amplified = load(
        "results/transformer_v3_amplified_secant_independent_audit.json"
    )
    response_free = load(
        "results/transformer_v3_response_free_probe_independent_audit.json"
    )
    four_probe = load(
        "results/transformer_v3_four_probe_independent_audit.json"
    )
    arb_secant = load(
        "results/transformer_v3_arb_secant_full_v2_independent_audit.json"
    )
    relinearized_literal = load(
        "results/transformer_v3_relinearized_green_audit.json"
    )
    relinearized_sixteen = load(
        "results/transformer_v3_relinearized_secant_audit.json"
    )
    relinearized_four_path = (
        ROOT / "results/transformer_v3_relinearized_secant_four_probe_audit.json"
    )
    relinearized_four = json.loads(relinearized_four_path.read_text(encoding="utf-8"))
    relinearized_four_independent = load(
        "results/transformer_v3_relinearized_secant_four_probe_independent_audit.json"
    )
    relinearized_timing = load(
        "results/transformer_v3_relinearized_probe_block_benchmark.json"
    )
    prefix_panel_path = (
        ROOT / "results/transformer_v3_relinearized_prefix_panel_audit.json"
    )
    prefix_panel = json.loads(prefix_panel_path.read_text(encoding="utf-8"))
    prefix_panel_independent = load(
        "results/transformer_v3_relinearized_prefix_panel_independent_audit.json"
    )
    direct_panel_path = (
        ROOT / "results/transformer_direct_image_green_panel_audit.json"
    )
    direct_panel = json.loads(direct_panel_path.read_text(encoding="utf-8"))
    direct_panel_independent = load(
        "results/transformer_direct_image_green_panel_independent_audit.json"
    )
    streaming = load("results/transformer_streaming_centerline_benchmark.json")
    analytic_jet_path = (
        ROOT / "results/transformer_analytic_jet_release_postseal_audit.json"
    )
    analytic_jet = json.loads(analytic_jet_path.read_text(encoding="utf-8"))
    analytic_jet_independent = load(
        "results/transformer_analytic_jet_release_independent_audit.json"
    )

    ts = transformer["summary"]
    studies = {
        "wdbc": {
            "cases": int(wdbc["seed_threshold_cases"]),
            "candidates": int(wdbc["candidates"]),
            "issued": int(wdbc["issued"]),
            "covered": int(wdbc["covered"]),
            "seeds": int(wdbc["distinct_issuing_seeds"]),
        },
        "digits": {
            "cases": int(digits["seed_threshold_cases"]),
            "candidates": int(digits["candidates"]),
            "issued": int(digits["signed_issued"]),
            "covered": int(digits["signed_covered"]),
            "seeds": int(digits["signed_distinct_seeds"]),
        },
        "transformer_fixed": {
            "cases": int(ts["seed_threshold_cases"]),
            "candidates": int(ts["candidates"]),
            "issued": int(ts["issued"]),
            "covered": int(ts["covered"]),
            "seeds": int(ts["distinct_issuing_seeds"]),
        },
        "transformer_response": {
            "cases": int(v3["seed_threshold_cases"]),
            "candidates": int(v3["candidates"]),
            "issued": int(v3["v3_issued"]),
            "covered": int(v3["v3_covered"]),
            "seeds": int(v3["v3_distinct_issuing_seeds"]),
        },
    }
    expected = {
        "wdbc": (72, 71, 56, 56, 22),
        "digits": (24, 24, 7, 7, 6),
        "transformer_fixed": (72, 23, 9, 9, 6),
        "transformer_response": (72, 19, 11, 11, 7),
    }
    for name, values in expected.items():
        observed = studies[name]
        require(
            tuple(observed[key] for key in ("cases", "candidates", "issued", "covered", "seeds"))
            == values,
            f"headline count mismatch for {name}: {observed}",
        )
    total_issued = sum(row["issued"] for row in studies.values())
    total_covered = sum(row["covered"] for row in studies.values())
    require((total_issued, total_covered) == (83, 83), "83/83 total mismatch")
    require(
        int(wdbc_outward["summary"]["outward_retained"])
        + int(digits_outward["summary"]["outward_issued"])
        == 63,
        "63 direct outward brackets mismatch",
    )

    # The WDBC count that is easiest to describe incorrectly: 59 comparable
    # clocks consist of 56 issued future events plus three gates already present
    # at the anchor.  All 15 non-issued candidates lack a strictly future clock.
    wdbc_rows = wdbc_audit["rows"]
    comparable = [row for row in wdbc_rows if row["raw_timing_error"] is not None]
    step_zero = [
        row
        for row in comparable
        if int(row["predicted_event"]) == 0 and int(row["actual_event"]) == 0
    ]
    nonissued = [row for row in wdbc_rows if not bool(row["certificate_issued"])]
    require(len(comparable) == 59, "WDBC comparable timing count changed")
    require(all(int(row["raw_timing_error"]) == 0 for row in comparable), "WDBC clock mismatch")
    require(len(step_zero) == 3, "WDBC step-zero gate count changed")
    require(len(nonissued) == 15, "WDBC non-issued count changed")
    require(
        all(row["certificate_status"] == "centerline has no future certification-set event" for row in nonissued),
        "WDBC non-issued disposition changed",
    )

    digit_selectivity = selectivity["digits"]
    require(
        (
            int(digit_selectivity["finite_prediction_outcome_pairs"]),
            int(digit_selectivity["exact_finite_predictions"]),
            int(digit_selectivity["inaccurate_finite_predictions"]),
            int(digit_selectivity["inaccurate_abstained"]),
        )
        == (16, 14, 2, 2),
        "digits selectivity counts changed",
    )
    require(
        all(not bool(row["randomized_operator_queried"]) for row in digit_selectivity["inaccurate_records"]),
        "an inaccurate digits clock now consumed a random query",
    )
    require(int(digits["signed_only"]) == 1, "signed-only digits count changed")
    signed_only = [row for row in digits["rows"] if bool(row["signed_only"])]
    require(len(signed_only) == 1 and signed_only[0]["bracket"] == [147, 147], "147-step event changed")

    require(int(ts["raw_exact_timing_matches"]) == 23, "fixed Transformer clock count changed")
    require(int(v3["exact_finite_predictions"]) == 19, "v3 exact clock count changed")
    require(v3["earliest_power_distribution"] == {"1": 1, "2": 7, "3": 2, "4": 1}, "v3 power distribution changed")
    us = unsigned["summary"]
    require(
        (int(us["signed_issued_in_matched_cases"]), int(us["strong_unsigned_issued_in_matched_cases"])) == (9, 1),
        "Transformer signed/unsigned issuance changed",
    )
    require(close(us["median_unsigned_to_signed_response_ratio"], 319.70120352667), "320x directional median changed")

    require(close(online["measured_operator_time_speedup"], 8.03182674660282), "online operator speedup changed")
    require(close(online["measured_end_to_end_speedup"], 2.19253009513), "online end-to-end speedup changed")
    ps = panel["summary"]
    require(close(ps["aggregate_output_wall_speedup"], 1.34811401978939), "panel output speedup changed")
    require(close(ps["aggregate_centerline_plus_output_speedup"], 1.11035612154906), "panel total speedup changed")
    require(close(combined["matched_end_to_end_replay_speedup"], 1.0183768029508034), "monolithic speedup changed")
    require(close(inexact["tolerances"]["common"]["lower_passing_relative_gram_residual"], 0.5999285448884594), "inexact tolerance changed")
    require(close(outward_inexact["strict_interior_safety_factor"], 0.99), "outward inexact safety factor changed")
    require(outward_inexact["same_bracket"] and outward_inexact["bracket"] == [2, 2], "outward inexact bracket changed")
    require(close(chi["replays"]["bonferroni_hybrid"]["closure"]["total_pointwise_radius"], 2.0704641923066263e-15), "chi hybrid radius changed")
    require(mixed["same_bracket"], "mixed-precision bracket changed")
    require(close(mixed["maximum_measured_relative_residual"], 3.3593789499600366e-7), "mixed residual changed")
    require(close(mixed["admissible_to_measured_ratio"], 1785831.7082554684), "mixed residual headroom changed")
    require(int(mixed_timing["invocations"]) == 4, "mixed invocation count changed")
    require(int(mixed_timing["paired_trials"]) == 20, "mixed paired-trial count changed")
    require(bool(mixed_timing["all_paired_speedups_above_one"]), "a mixed timing pair no longer improves")
    require(close(mixed_timing["pooled_median_paired_speedup"], 1.8272291746452054), "mixed pooled speedup changed")
    require(close(mixed_timing["pooled_minimum_paired_speedup"], 1.0368268314666373), "mixed minimum speedup changed")
    require(close(mixed_timing["pooled_maximum_paired_speedup"], 2.3793902960657647), "mixed maximum speedup changed")
    require(int(weighted["earliest_issuing_power"]) == 2, "weighted Green depth changed")
    million = next(row for row in scale["profiles"] if row["profile"] == "1m")
    require(int(million["parameter_count"]) == 1_008_864, "million-parameter profile changed")
    require(close(million["projection_h300"]["projected_batched_core_seconds"] / 3600.0, 11.3729967397635), "11.37-hour projection changed")
    legacy_cost = legacy_scale["existing_paper_scale_cost"]
    require(close(legacy_cost["median_candidate_minutes"], 58.0174975633335), "58.02-minute historical cost changed")
    require(close(legacy_cost["median_measured_300_step_continuation_seconds"], 5.308161827501317), "5.31-second continuation changed")
    require(close(legacy_cost["certificate_to_300_step_continuation_ratio"], 655.7919609317236), "655.8x cost ratio changed")
    require(int(output_recenter["old_issued_retained"]) == 11, "output recentering retention changed")
    require(close(output_recenter["median_maximum_margin_radius_ratio"], 0.36505254056767006), "output recentering median changed")
    require(close(output_recenter["minimum_maximum_margin_radius_ratio"], 0.09077829636531191), "output recentering minimum changed")
    require(close(output_recenter["maximum_maximum_margin_radius_ratio"], 0.6020999753221202), "output recentering maximum changed")
    require(int(two_response["green_evaluable_records"]) == 15, "two-response evaluable count changed")
    require(
        (int(two_response_postseal["certificate_records"]), int(two_response_postseal["evaluable_records"]))
        == (19, 15),
        "two-response operator-cohort denominator changed",
    )
    unavailable_green = [row for row in two_response_postseal["rows"] if not bool(row["evaluable"])]
    require(
        len(unavailable_green) == 4
        and all(row["reason"] == "no sealed Green trace" for row in unavailable_green)
        and all(not bool(row["old_certificate_issued"]) for row in unavailable_green),
        "the four non-evaluable candidates are no longer exactly the pre-query abstentions",
    )
    require(int(two_response["surrogate_issued"]) == 15, "two-response closure count changed")
    require(int(two_response["old_issued"]) == 11, "two-response baseline count changed")
    require(int(two_response["converted_old_abstentions"]) == 4, "two-response conversion count changed")
    require(int(two_response["earlier_power_cases"]) == 5, "two-response earlier-power count changed")
    require(int(two_response["adaptive_response_invocations"]) == 9, "two-response invocation count changed")
    require(int(two_response["unchanged_cases_without_added_response"]) == 6, "two-response no-work count changed")
    require(int(two_response["progressive_power_levels_saved"]) == 22, "two-response saved-depth count changed")
    require(int(two_response["net_hvp_calls_saved_excluding_third_products"]) == 167250, "two-response HVP accounting changed")
    require(int(two_response["directional_third_products"]) == 1797, "two-response third-product count changed")
    require(int(two_response["local_fourth_order_passes"]) == 9, "two-response fourth-order pass count changed")
    require(close(two_response["minimum_local_fourth_headroom_ratio"], 55.30410964848143), "two-response fourth-order headroom changed")
    require(int(two_response_timing["repeats"]) == 4, "two-response timing repeat count changed")
    require(close(two_response_timing["median_directional_seconds"], 78.58018739998806), "two-response directional median changed")
    require(close(two_response_timing["median_additional_gram_power_seconds"], 233.44951584999217), "two-response Gram-power median changed")
    require(close(two_response_timing["median_paired_speedup"], 2.9129207123058674), "two-response timing median speedup changed")
    require(close(two_response_timing["minimum_paired_speedup"], 2.773219603089279), "two-response timing minimum changed")
    require(close(two_response_timing["maximum_paired_speedup"], 3.2810227163066106), "two-response timing maximum changed")
    require(amplified["candidate"] == {"seed": 366, "threshold": 0.7, "anchor": 1040}, "amplified-secant candidate changed")
    require(int(amplified["horizon"]) == 52, "amplified-secant horizon changed")
    require(close(amplified["amplification"], 4096.0), "amplified-secant lambda changed")
    require(amplified["bracket"] == [28, 28], "amplified-secant bracket changed")
    require(close(amplified["analytic_headroom_ratio"], 24.413592002410677), "amplified-secant headroom changed")
    require(close(amplified["remaining_arithmetic_and_recurrence_headroom"], 6.014744428490773e-20), "amplified-secant arithmetic headroom changed")
    require(close(amplified["median_secant_seconds"], 6.558657099987613), "amplified-secant median changed")
    require(close(amplified["median_third_product_seconds"], 8.100442700000713), "third-product matched median changed")
    require(close(amplified["median_additional_gram_power_seconds"], 26.007475799997337), "amplified matched Gram median changed")
    require(close(amplified["median_paired_third_over_secant_speedup"], 1.1923708890424844), "amplified-versus-third speedup changed")
    require(close(amplified["median_paired_power_over_secant_speedup"], 4.0413260665889394), "amplified-versus-power speedup changed")
    require(close(amplified["minimum_power_over_secant_speedup"], 3.8198985845418694), "amplified minimum paired speedup changed")
    require(close(amplified["maximum_power_over_secant_speedup"], 4.370522476848509), "amplified maximum paired speedup changed")
    require(response_free["status"] == "INDEPENDENT RESPONSE-FREE PROBE AUDIT PASSED", "response-free independent audit status changed")
    require(response_free["candidate"] == {"seed": 366, "threshold": 0.7, "anchor": 1040}, "response-free candidate changed")
    require(int(response_free["probes"]) == 16, "response-free probe count changed")
    require(close(response_free["bound_to_observed_ratio"], 4.6202928718764555), "response-free bound inflation changed")
    require(close(response_free["forcing_headroom_ratio"], 22.76982331522108), "response-free headroom changed")
    require(response_free["bracket"] == [28, 28], "response-free bracket changed")
    require(int(response_free["outcome_files_read"]) == 0, "response-free evidence boundary changed")
    require(four_probe["status"] == "INDEPENDENT FOUR-PROBE AUDIT PASSED", "four-probe audit status changed")
    require(int(four_probe["probes"]) == 4, "four-probe count changed")
    require(close(four_probe["bound_to_observed_ratio"], 32.13468336403352), "four-probe bound inflation changed")
    require(close(four_probe["forcing_headroom_ratio"], 16.25303212587122), "four-probe headroom changed")
    require(four_probe["bracket"] == [28, 28], "four-probe bracket changed")
    require(int(four_probe["outcome_files_read"]) == 0, "four-probe evidence boundary changed")
    require(arb_secant["status"] == "INDEPENDENT FULL-SEQUENCE ARB SECANT AUDIT V2 PASSED", "outward secant audit status changed")
    require(int(arb_secant["intervals_recomputed"]) == 204, "outward secant interval count changed")
    require(close(arb_secant["secant_forcing_norm_upper"], 1.3054822832359867e-29), "outward secant forcing bound changed")
    require(close(arb_secant["forcing_headroom_ratio"], 24.413591878344434), "outward secant headroom changed")
    require(close(arb_secant["total_wall_seconds"] / 60.0, 9.799390046666667), "outward secant wall time changed")
    require(arb_secant["bracket"] == [28, 28], "outward secant bracket changed")
    require(int(arb_secant["outcome_files_read"]) == 0, "outward secant outcome boundary changed")
    require(not bool(relinearized_literal["closure"]["closure_passed"]), "literal corrected-defect failure disappeared")
    require(close(relinearized_literal["corrected_defect_sequence_norm"], 3.101357457947468e-15), "literal corrected-defect norm changed")
    require(int(relinearized_literal["outcome_files_read"]) == 0, "literal corrected-path evidence boundary changed")
    require(close(relinearized_sixteen["old_mixed_coefficient"], 0.6273518016998689), "old mixed coefficient changed")
    require(close(relinearized_sixteen["new_mixed_coefficient"], 0.0), "relinearized mixed coefficient changed")
    require(close(relinearized_sixteen["forcing_headroom_ratio"], 173.7330443370273), "16-probe relinearized headroom changed")
    require(relinearized_sixteen["bracket"] == [28, 28], "16-probe relinearized bracket changed")
    require(int(relinearized_sixteen["extra_causal_response_sweeps"]) == 0, "relinearized response count changed")
    require(int(relinearized_sixteen["outcome_files_read"]) == 0, "16-probe relinearized evidence boundary changed")
    require(
        relinearized_four_independent["status"]
        == "INDEPENDENT FOUR-PROBE RELINEARIZED SECANT AUDIT PASSED",
        "independent four-probe relinearized audit status changed",
    )
    require(
        relinearized_four_independent["result_sha256"] == sha256(relinearized_four_path),
        "independent relinearized audit points to a different claim result",
    )
    require(int(relinearized_four["probe"]["probes"]) == 4, "relinearized probe count changed")
    require(int(relinearized_four["probe"]["gram_applications"]) == 4, "relinearized Gram count changed")
    require(close(relinearized_four["forcing_headroom_ratio"], 2.002784762434598), "four-probe relinearized headroom changed")
    require(relinearized_four["bracket"] == [28, 28], "four-probe relinearized bracket changed")
    require(close(relinearized_four["closure"]["total_radius_about_original_reference"], 2.8580475301165783e-15), "four-probe relinearized radius changed")
    require(int(relinearized_four["outcome_files_read"]) == 0, "four-probe relinearized evidence boundary changed")
    require(close(relinearized_timing["median_four_probe_seconds"], 6.512193500006106), "four-probe matched timing changed")
    require(close(relinearized_timing["median_sixteen_probe_seconds"], 23.639498800010188), "16-probe matched timing changed")
    require(close(relinearized_timing["median_paired_speedup"], 3.6300363003630074), "relinearized matched speedup changed")
    require(close(relinearized_timing["minimum_paired_speedup"], 3.4758805809955424), "relinearized minimum speedup changed")
    require(close(relinearized_timing["maximum_paired_speedup"], 4.095553522693955), "relinearized maximum speedup changed")
    require(close(relinearized_timing["logical_gram_application_reduction"], 4.0), "relinearized logical reduction changed")
    require(int(relinearized_timing["outcome_files_read"]) == 0, "relinearized benchmark evidence boundary changed")
    require(prefix_panel["status"] == "OUTCOME-BLIND RELINEARIZED PREFIX PANEL COMPLETED", "prefix-panel status changed")
    require((int(prefix_panel["cases"]), int(prefix_panel["issued"]), int(prefix_panel["same_as_directional_bracket"])) == (15, 15, 15), "prefix-panel issuance changed")
    require(prefix_panel["prefix_distribution"] == {"4": 14, "8": 1, "16": 0}, "prefix-panel stopping distribution changed")
    require((int(prefix_panel["old_total_green_gram_applications"]), int(prefix_panel["new_total_green_gram_applications"])) == (560, 64), "prefix-panel Green accounting changed")
    require(close(prefix_panel["aggregate_green_gram_reduction"], 8.75), "prefix-panel Green reduction changed")
    require((int(prefix_panel["old_total_theoretical_linearized_sweeps"]), int(prefix_panel["new_total_theoretical_linearized_sweeps"])) == (1150, 144), "prefix-panel sweep accounting changed")
    require(close(prefix_panel["aggregate_theoretical_linearized_sweep_reduction"], 7.986111111111111), "prefix-panel sweep reduction changed")
    require(close(prefix_panel["minimum_issued_forcing_headroom"], 2.293290248676895), "prefix-panel forcing headroom changed")
    require(close(prefix_panel["combined_family_failure_upper"], 2.0e-6), "prefix-panel probability accounting changed")
    require(int(prefix_panel["outcome_files_read"]) == 0, "prefix-panel outcome boundary changed")
    require(prefix_panel_independent["status"] == "INDEPENDENT RELINEARIZED PREFIX-PANEL AUDIT PASSED", "independent prefix-panel status changed")
    require(prefix_panel_independent["result_sha256"] == sha256(prefix_panel_path), "independent prefix-panel hash changed")
    require(int(prefix_panel_independent["unique_probe_hashes"]) == 64, "prefix-panel probe uniqueness changed")
    require(close(prefix_panel_independent["minimum_logic_slack"], 1.446608571861377e-6), "prefix-panel logic slack changed")
    require(direct_panel["status"] == "STAGED DIRECT-IMAGE/GRAM PANEL COMPLETED", "direct-image panel status changed")
    require((int(direct_panel["cases"]), int(direct_panel["issued"])) == (15, 15), "direct-image panel issuance changed")
    require(direct_panel["route_distribution"] == {"direct_image": 4, "gram_fallback": 11}, "direct-image route distribution changed")
    require((int(direct_panel["panel_green_probe_sweeps"]), int(direct_panel["staged_green_probe_sweeps"])) == (128, 112), "direct-image sweep accounting changed")
    require(close(direct_panel["aggregate_probe_sweep_reduction"], 1.1428571428571428), "direct-image sweep reduction changed")
    require(int(direct_panel["transpose_sweeps_avoided"]) == 16, "direct-image transpose savings changed")
    require(int(direct_panel["outcome_files_read"]) == 0, "direct-image outcome boundary changed")
    require(direct_panel_independent["status"] == "INDEPENDENT DIRECT-IMAGE/GREEN PANEL AUDIT PASSED", "independent direct-image status changed")
    require(direct_panel_independent["result_sha256"] == sha256(direct_panel_path), "independent direct-image hash changed")
    require(int(direct_panel_independent["unique_probe_hashes"]) == 64, "direct-image probe uniqueness changed")
    require(streaming["status"] == "STREAMING PREFIX-LOCAL CENTERLINE BENCHMARK COMPLETED", "streaming benchmark status changed")
    require(bool(streaming["all_bitwise_equal"]), "streaming centerline is no longer bitwise identical")
    require([int(row["horizon"]) for row in streaming["rows"]] == [26, 131, 299], "streaming horizons changed")
    require(all(int(row["outcome_files_read"]) == 0 for row in streaming["rows"]), "streaming outcome boundary changed")
    for row, speed, memory in zip(
        streaming["rows"],
        (9.939919583703775, 2.0904569958018846, 1.0587661893705587),
        (39.60526315789474, 10.524475524475525, 4.839228295819936),
    ):
        require(close(row["speedup"], speed), "streaming speedup changed")
        require(close(row["estimated_centerline_memory_reduction"], memory), "streaming memory reduction changed")
    require(
        analytic_jet["status"]
        == "POST-SEAL DETERMINISTIC ANALYTIC-JET RELEASE AUDIT PASSED",
        "analytic-jet release status changed",
    )
    require(
        (
            int(analytic_jet["cases"]),
            int(analytic_jet["analytic_jet_issued"]),
            int(analytic_jet["probe_fallback_required"]),
            int(analytic_jet["staged_total_issued"]),
        )
        == (15, 8, 7, 15),
        "analytic-jet staged disposition changed",
    )
    require(bool(analytic_jet["same_brackets"]), "analytic-jet bracket changed")
    require(
        int(analytic_jet["randomized_output_operators_eliminated"]) == 1432,
        "analytic-jet operator reduction changed",
    )
    require(
        int(analytic_jet["randomized_output_gram_applications_eliminated"])
        == 22912,
        "analytic-jet Gram reduction changed",
    )
    require(
        close(analytic_jet["minimum_analytic_logic_slack"], 1.804045650110337e-4),
        "analytic-jet logic slack changed",
    )
    require(
        close(analytic_jet["maximum_analytic_margin_radius"], 4.554086605439273e-10),
        "analytic-jet margin radius changed",
    )
    require(
        int(analytic_jet["future_outcome_files_read"]) == 0,
        "analytic-jet evidence boundary changed",
    )
    require(
        analytic_jet_independent["status"]
        == "INDEPENDENT ANALYTIC-JET RELEASE AUDIT PASSED",
        "independent analytic-jet status changed",
    )
    require(
        analytic_jet_independent["source_sha256"] == sha256(analytic_jet_path),
        "independent analytic-jet source hash changed",
    )
    require(
        int(analytic_jet_independent["analytic_jet_issued"]) == 8
        and bool(analytic_jet_independent["same_brackets"]),
        "independent analytic-jet result changed",
    )

    required_phrases = (
        "56/72 WDBC, 7/24 digits, and 9/72 plus 11/72",
        "all 83 subsequently revealed crossings lie in their",
        "three of these gates are already present at the anchor",
        "no strictly future centerline",
        "Coverage is conditional on",
        "not a population-generalization bound",
        "high-confidence numerical certificates",
        "1.018\\times",
        "3.36\\times10^{-7}",
        "1.83\\times",
        "all 20 paired q=1 kernel speedups exceed",
        "Precision-adaptive Gram queries",
        "Certificate-aware operator-cap budget",
        "Residualized two-response interface",
        "Cancellation-safe directional second response",
        "Amplified-secant second response",
        "Fresh-probe forcing and Green-image residual",
        "167,250 net objective HVPs",
        "Directional third products added & 1,797",
        "55.3\\times",
        "four-pair range",
        "$2.77$--$3.28\\times$",
        "$24.4\\times$ analytic headroom",
        "Three rotating-order triples",
        "6.56\\,s versus 8.10\\,s",
        "$4.04\\times$ paired speedup",
        "204 outward 192-bit scalar jets",
        "$1.31\\times10^{-29}$",
        "$24.41\\times$ headroom",
        "9.80 minutes wall time",
        "Corrected-path relinearization",
        "$0.627\\to0$",
        "Four-probe bracket / forcing headroom & $[28,28]$ / $2.00\\times$",
        "Three alternating matched pairs give medians 6.51 versus",
        "23.64 seconds, or $3.63\\times$",
        "$3.48$--$4.10\\times$",
        "$560\\to64$ ($8.75\\times$)",
        "$1150\\to144$ ($7.99\\times$)",
        "all 64 unique Gaussian vectors",
        "complete pre-existing operator cohort, not a pass-selected subset",
        "failed the original\ndeterministic pre-query screen",
        "image screening under the same Gaussian event issues four cases",
        "Prefix streaming reduces centerline",
        "Deterministic neural-jet release",
        "8/15 cases",
        "1,432",
        "22,912",
        "$1.80\\times10^{-4}$",
        "combined upper bound $2\\times10^{-6}$",
        "counts above unchanged.",
        "recentered output radii",
        "median 0.365",
        "58.02 minutes",
        "5.31 seconds",
        "655.8\\times",
        "outcome-independent proof object",
        "naughton2026certified",
        "surrogate verification",
        "admissible-residual sensitivities, not estimates of floating-point error",
    )
    for phrase in required_phrases:
        require(phrase in paper, f"manuscript lost required scoped phrase: {phrase}")
    forbidden_phrases = (
        "all 59 cases having a comparable future event",
        "Untouched Transformer confirmation",
        "untouched dense triggers",
        "Related work and novelty boundary",
        "remains far from practical",
        "8.03\\times$ less operator",
        "15/15 prospective",
        "15 fresh two-response certificates",
        "2.91\\times prospective",
        "4.04\\times prospective",
        "fresh amplified-secant certificate",
        "outward response-free certificate",
        "response-free prospective certificate",
        "complete computer-assisted Transformer certificate",
        "fully outward Transformer certificate",
        "four-probe prospective certificate",
        "fully outward corrected-path certificate",
        "computer-assisted corrected-path Transformer certificate",
    )
    for phrase in forbidden_phrases:
        require(phrase not in paper, f"stale/attackable manuscript phrase remains: {phrase}")

    payload = {
        "status": "GREENCERT manuscript claim audit passed",
        "paper": str(PAPER.relative_to(ROOT)),
        "paper_sha256": sha256(PAPER),
        "studies": studies,
        "totals": {"issued": total_issued, "covered": total_covered, "direct_outward": 63},
        "wdbc_timing_edge_case": {
            "comparable": len(comparable),
            "exact": sum(int(row["raw_timing_error"]) == 0 for row in comparable),
            "step_zero": len(step_zero),
            "nonissued_no_strictly_future_event": len(nonissued),
        },
        "digits_selectivity": {
            "finite": 16,
            "exact": 14,
            "inaccurate": 2,
            "inaccurate_abstained_before_random_query": 2,
        },
        "implementation": {
            "online_operator_speedup": online["measured_operator_time_speedup"],
            "online_end_to_end_speedup": online["measured_end_to_end_speedup"],
            "role_panel_output_speedup": ps["aggregate_output_wall_speedup"],
            "monolithic_end_to_end_speedup": combined["matched_end_to_end_replay_speedup"],
            "mixed_precision_q1_kernel_speedup": mixed_timing["pooled_median_paired_speedup"],
            "mixed_precision_timing_invocations": mixed_timing["invocations"],
            "mixed_precision_timing_pairs": mixed_timing["paired_trials"],
            "mixed_precision_residual_headroom": mixed["admissible_to_measured_ratio"],
            "million_parameter_core_hours": million["projection_h300"]["projected_batched_core_seconds"] / 3600.0,
            "historical_candidate_minutes": legacy_cost["median_candidate_minutes"],
            "direct_300_step_seconds": legacy_cost["median_measured_300_step_continuation_seconds"],
            "historical_certificate_to_continuation_ratio": legacy_cost["certificate_to_300_step_continuation_ratio"],
            "output_recenter_median_radius_ratio": output_recenter["median_maximum_margin_radius_ratio"],
            "two_response_green_evaluable": two_response["green_evaluable_records"],
            "two_response_closures": two_response["surrogate_issued"],
            "two_response_converted_abstentions": two_response["converted_old_abstentions"],
            "two_response_saved_power_levels": two_response["progressive_power_levels_saved"],
            "two_response_net_hvp_savings": two_response["net_hvp_calls_saved_excluding_third_products"],
            "two_response_minimum_fourth_headroom": two_response["minimum_local_fourth_headroom_ratio"],
            "two_response_median_paired_speedup": two_response_timing["median_paired_speedup"],
            "two_response_minimum_paired_speedup": two_response_timing["minimum_paired_speedup"],
            "two_response_maximum_paired_speedup": two_response_timing["maximum_paired_speedup"],
            "amplified_secant_lambda": amplified["amplification"],
            "amplified_secant_bracket": amplified["bracket"],
            "amplified_secant_analytic_headroom": amplified["analytic_headroom_ratio"],
            "amplified_secant_arithmetic_headroom": amplified["remaining_arithmetic_and_recurrence_headroom"],
            "amplified_secant_median_seconds": amplified["median_secant_seconds"],
            "amplified_secant_vs_third_speedup": amplified["median_paired_third_over_secant_speedup"],
            "amplified_secant_vs_power_speedup": amplified["median_paired_power_over_secant_speedup"],
            "response_free_probe_count": response_free["probes"],
            "response_free_bound_inflation": response_free["bound_to_observed_ratio"],
            "response_free_forcing_headroom": response_free["forcing_headroom_ratio"],
            "response_free_bracket": response_free["bracket"],
            "fresh_four_probe_count": four_probe["probes"],
            "fresh_four_probe_bound_inflation": four_probe["bound_to_observed_ratio"],
            "fresh_four_probe_forcing_headroom": four_probe["forcing_headroom_ratio"],
            "outward_secant_intervals": arb_secant["intervals_recomputed"],
            "outward_secant_forcing_norm_upper": arb_secant["secant_forcing_norm_upper"],
            "outward_secant_forcing_headroom": arb_secant["forcing_headroom_ratio"],
            "outward_secant_wall_seconds": arb_secant["total_wall_seconds"],
            "relinearized_old_mixed_coefficient": relinearized_sixteen["old_mixed_coefficient"],
            "relinearized_new_mixed_coefficient": relinearized_sixteen["new_mixed_coefficient"],
            "relinearized_sixteen_probe_headroom": relinearized_sixteen["forcing_headroom_ratio"],
            "relinearized_four_probe_headroom": relinearized_four["forcing_headroom_ratio"],
            "relinearized_gram_application_reduction": relinearized_timing["logical_gram_application_reduction"],
            "relinearized_median_paired_speedup": relinearized_timing["median_paired_speedup"],
            "prefix_panel_issued": prefix_panel["issued"],
            "prefix_panel_green_applications": [
                prefix_panel["old_total_green_gram_applications"],
                prefix_panel["new_total_green_gram_applications"],
            ],
            "prefix_panel_green_reduction": prefix_panel["aggregate_green_gram_reduction"],
            "prefix_panel_theoretical_sweeps": [
                prefix_panel["old_total_theoretical_linearized_sweeps"],
                prefix_panel["new_total_theoretical_linearized_sweeps"],
            ],
            "prefix_panel_combined_failure_upper": prefix_panel["combined_family_failure_upper"],
            "direct_image_routes": direct_panel["route_distribution"],
            "direct_image_staged_sweeps": direct_panel["staged_green_probe_sweeps"],
            "analytic_jet_issued": analytic_jet["analytic_jet_issued"],
            "analytic_jet_fallback": analytic_jet["probe_fallback_required"],
            "analytic_jet_output_operators_eliminated": analytic_jet[
                "randomized_output_operators_eliminated"
            ],
            "analytic_jet_output_gram_applications_eliminated": analytic_jet[
                "randomized_output_gram_applications_eliminated"
            ],
            "streaming_speedups": [row["speedup"] for row in streaming["rows"]],
            "streaming_memory_reductions": [
                row["estimated_centerline_memory_reduction"]
                for row in streaming["rows"]
            ],
        },
        "checked_required_phrases": list(required_phrases),
        "checked_forbidden_phrases": list(forbidden_phrases),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**payload, "output_sha256": sha256(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
