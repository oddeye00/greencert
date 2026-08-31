#!/usr/bin/env python3
"""Build a compact, path-sanitized anonymous supplementary archive."""
from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "certified_local_training_events_supplement.zip"

FILES = [
    "SUPPLEMENT_README.md",
    "LITERATURE_AUDIT.md",
    "GREEN_OPERATOR_SHADOWING_THEOREM.md",
    "VARIATIONAL_SHADOWING_THEOREM.md",
    "PROJECTED_HVP_SHADOWING_THEOREM.md",
    "RESPONSE_CENTERED_EVENT_TRANSPORT_THEOREM.md",
    "DIRECTIONAL_TWO_RESPONSE_THEOREM.md",
    "AMPLIFIED_SECANT_RESPONSE_THEOREM.md",
    "RANDOMIZED_RESIDUAL_PROBE_THEOREM.md",
    "RELINEARIZED_GREEN_THEOREM.md",
    "NESTED_PREFIX_GRAM_THEOREM.md",
    "COST_AWARE_FORCING_THEOREM.md",
    "CAUSAL_PREFIX_RECENTERING_THEOREM.md",
    "DIRECT_IMAGE_GREEN_THEOREM.md",
    "ANALYTIC_JET_RELEASE_THEOREM.md",
    "STRUCTURED_PARAMETER_GREEN_THEOREM.md",
    "STRUCTURED_PARAMETER_GREEN_THEOREM_V1_INDEXING_NOTE.md",
    "STRUCTURED_PARAMETER_GREEN_THEOREM_V2.md",
    "STRUCTURED_PARAMETER_GREEN_SOURCE_SUPERSESSION.md",
    "AMPLIFIED_SECANT_PROBE_PROTOCOL.md",
    "AMPLIFIED_SECANT_FOUR_PROBE_PROTOCOL.md",
    "AMPLIFIED_SECANT_OUTWARD_EXECUTION_PROTOCOL.md",
    "AMPLIFIED_SECANT_OUTWARD_EXECUTION_PROTOCOL_V2.md",
    "RELINEARIZED_GREEN_AUDIT_PROTOCOL.md",
    "RELINEARIZED_SECANT_AUDIT_PROTOCOL.md",
    "RELINEARIZED_SECANT_FOUR_PROBE_PROTOCOL.md",
    "RELINEARIZED_PREFIX_PANEL_PROTOCOL.md",
    "RELINEARIZED_PREFIX_PANEL_RESULT.md",
    "RELINEARIZED_PREFIX_PANEL_EXECUTION_DEVIATIONS.md",
    "DIRECT_IMAGE_GREEN_PANEL_PROTOCOL.md",
    "STRUCTURED_PARAMETER_GREEN_AUDIT_PROTOCOL.md",
    "STRUCTURED_PARAMETER_GREEN_AUDIT_ABORTED_V1.md",
    "STRUCTURED_PARAMETER_GREEN_AUDIT_PROTOCOL_V2.md",
    "ANCHOR_FIXED_STRUCTURED_PARAMETER_GREEN_AUDIT_PROTOCOL.md",
    "TRANSFORMER_GREEN_CONFIRMATION_PROTOCOL.md",
    "TRANSFORMER_GREEN_CONFIRMATION_METHOD_SEAL.json",
    "TRANSFORMER_GREEN_CONFIRMATION_CANDIDATE_SEAL.json",
    "TRANSFORMER_GREEN_CONFIRMATION_EXECUTION_AMENDMENT_V1_1.md",
    "TRANSFORMER_GREEN_CONFIRMATION_EXECUTION_AMENDMENT_SEAL_V1_1.json",
    "TRANSFORMER_GREEN_CONFIRMATION_CERTIFICATE_SEAL.json",
    "TRANSFORMER_FOUR_SWEEP_INDEPENDENT_AUDIT.md",
    "TRANSFORMER_CERTIFICATE_TRACEABILITY.md",
    "TRANSFORMER_GREEN_CONFIRMATION_INDEPENDENT_AUDIT.md",
    "TRANSFORMER_GREEN_PROOF_AND_NUMERICAL_AUDIT.md",
    "REAL_DATA_GREENCERT_CONFIRMATION_PROTOCOL.md",
    "REAL_DATA_GREENCERT_METHOD_SEAL.json",
    "REAL_DATA_GREENCERT_CANDIDATE_SEAL.json",
    "REAL_DATA_GREENCERT_CERTIFICATE_SEAL.json",
    "DIGITS_SIGNED_CONFIRMATION_PROTOCOL.md",
    "DIGITS_SIGNED_METHOD_SEAL.json",
    "DIGITS_SIGNED_CANDIDATE_SEAL.json",
    "DIGITS_SIGNED_CERTIFICATE_SEAL.json",
    "PROSPECTIVE_V2_PRIMARY_PROTOCOL.md",
    "PROSPECTIVE_V2_CODE_MANIFEST.json",
    "PROSPECTIVE_V2_PREFREEZE_AUDIT.json",
    "DISJOINT_HVP_PROSPECTIVE_PROTOCOL.md",
    "DISJOINT_HVP_PROSPECTIVE_SEAL.json",
    "TRANSFORMER_HVP_PROSPECTIVE_PROTOCOL_V2.md",
    "TRANSFORMER_HVP_PROSPECTIVE_SEAL_V2.json",
    "TRANSFORMER_HVP_CANDIDATE_SEAL_V2.json",
    "paper/certified_local_training_events_neurips2026.tex",
    "paper/certified_local_training_events_neurips2026_blind.tex",
    "paper/certified_local_training_events_arxiv.tex",
    "paper/transformer_jet_appendix.tex",
    "paper/references.bib",
    "paper/checklist.tex",
    "paper/neurips_2026.sty",
    "figures/paper_prospective_horizons.pdf",
    "figures/paper_prospective_horizons.png",
    "figures/paper_prospective_brackets.pdf",
    "figures/paper_prospective_brackets.png",
    "figures/paper_transformer_green_confirmation.pdf",
    "figures/paper_transformer_green_confirmation.png",
    "figures/paper_disjoint_hvp.pdf",
    "figures/paper_real_data_confirmation.pdf",
    "figures/paper_real_data_confirmation.png",
    "figures/paper_mechanism_scaling.pdf",
    "figures/paper_mechanism_scaling.png",
    "figures/paper_relinearized_prefix_panel.pdf",
    "figures/paper_relinearized_prefix_panel.png",
    "figures/paper_composed_runtime.pdf",
    "figures/paper_composed_runtime.png",
    "data/wdbc_breast_cancer.csv",
    "scripts/prospective_v2_primary.py",
    "scripts/outward_interval_certificate.py",
    "scripts/generate_prospective_v2_seed.py",
    "scripts/generate_smooth_mlp_seed.py",
    "scripts/smooth_mlp_modular_grokking.py",
    "scripts/smooth_mlp_certificate.py",
    "scripts/variational_mlp_certificate.py",
    "scripts/variational_shadowing.py",
    "scripts/modular_accuracy_certificate.py",
    "scripts/replay_smooth_mlp_thresholds.py",
    "scripts/test_prospective_v2_primary.py",
    "scripts/test_outward_interval_certificate.py",
    "scripts/disjoint_large_mlp.py",
    "scripts/matrix_free_mlp.py",
    "scripts/projected_variational_shadowing.py",
    "scripts/hvp_projected_mlp_certificate.py",
    "scripts/run_disjoint_hvp_certificate.py",
    "scripts/run_disjoint_hvp_prospective_audit.py",
    "scripts/test_matrix_free_mlp.py",
    "scripts/run_transformer_green_confirmation.py",
    "scripts/transformer_green_confirmation_protocol.py",
    "scripts/transformer_green_confirmation_certificate.py",
    "scripts/transformer_green_protocol.py",
    "scripts/transformer_green_operator.py",
    "scripts/transformer_green_development_audit.py",
    "scripts/transformer_four_sweep_development_audit.py",
    "scripts/transformer_certificate_protocol.py",
    "scripts/transformer_optimizer_probe.py",
    "scripts/transformer_block_envelope.py",
    "scripts/block_jet_bound.py",
    "scripts/probe_jacobian_bound.py",
    "scripts/transformer_hvp_grokking.py",
    "scripts/transformer_modal_forecast.py",
    "scripts/verify_transformer_green_result.py",
    "scripts/verify_transformer_green_confirmation.py",
    "scripts/transformer_green_confirmation_execution_amendment_v1_1.py",
    "scripts/audit_transformer_green_confirmation_statistics.py",
    "scripts/paper_figure_transformer_green_confirmation.py",
    "scripts/test_transformer_green_confirmation_protocol.py",
    "scripts/test_transformer_green_confirmation_execution_amendment_v1_1.py",
    "scripts/test_transformer_green_operator.py",
    "scripts/test_transformer_certificate_protocol.py",
    "scripts/test_probe_jacobian_bound.py",
    "scripts/test_block_jet_bound.py",
    "scripts/test_transformer_hvp_grokking.py",
    "scripts/test_transformer_modal_forecast.py",
    "scripts/real_dataset_mlp.py",
    "scripts/real_dataset_jet_bound.py",
    "scripts/real_dataset_greencert.py",
    "scripts/run_real_dataset_confirmation.py",
    "scripts/audit_real_dataset_confirmation.py",
    "scripts/outward_real_dataset_confirmation.py",
    "scripts/audit_real_dataset_outward.py",
    "scripts/test_real_dataset_mlp.py",
    "scripts/test_real_dataset_jet_bound.py",
    "scripts/test_real_dataset_greencert_replay.py",
    "scripts/test_real_dataset_confirmation_protocol.py",
    "scripts/test_real_dataset_confirmation_e2e.py",
    "scripts/test_outward_real_dataset_confirmation.py",
    "scripts/audit_wdbc_direct_validated_baseline.py",
    "scripts/test_wdbc_direct_validated_baseline.py",
    "scripts/digits_parity_mlp.py",
    "scripts/run_digits_signed_confirmation.py",
    "scripts/test_digits_parity_mlp.py",
    "scripts/test_digits_signed_confirmation_protocol.py",
    "scripts/audit_digits_signed_confirmation.py",
    "scripts/test_digits_signed_confirmation_audit.py",
    "scripts/outward_digits_confirmation.py",
    "scripts/batched_green_operator.py",
    "scripts/test_batched_green_operator.py",
    "scripts/benchmark_batched_certificate_primitives.py",
    "scripts/audit_transformer_batched_replay.py",
    "scripts/adamw_optimizer_probe.py",
    "scripts/test_adamw_optimizer_probe.py",
    "scripts/audit_modern_transformer_primitives.py",
    "scripts/audit_postseal_hardening.py",
    "scripts/test_postseal_hardening_audit.py",
    "scripts/develop_digits_signed_screen.py",
    "scripts/develop_digits_certificate_screen.py",
    "scripts/audit_transformer_unsigned_right_inverse.py",
    "scripts/test_transformer_unsigned_right_inverse_audit.py",
    "scripts/audit_transformer_sweep_ablation.py",
    "scripts/test_transformer_sweep_ablation.py",
    "scripts/benchmark_transformer_scaling.py",
    "scripts/test_transformer_scaling_benchmark.py",
    "scripts/test_transformer_jet_analytic_constants.py",
    "scripts/test_transformer_readout_relaxation.py",
    "scripts/paper_figure_new_evidence.py",
    "scripts/paper_plot_style.py",
    "scripts/reproduce_figures.py",
    "results/figure_reproducibility_audit.json",
    "results/transformer_green_confirmation_no_artifact_audit.json",
    "results/transformer_green_confirmation_candidates_blind.json",
    "results/transformer_green_confirmation_audit.json",
    "results/transformer_green_confirmation_independent_audit.json",
    "results/transformer_four_sweep_development_audit.json",
    "results/transformer_green_development_seed_321_gate_0_anchor_1440.json",
    "results/transformer_green_development_seed_322_gate_0_anchor_2400.json",
    "results/transformer_green_development_seed_322_gate_1_anchor_2640.json",
    "results/prospective_v2_primary_triggers_blind.json",
    "results/prospective_v2_primary_outcomes.json",
    "results/prospective_v2_primary.json",
    "results/prospective_v2_interval_blind.json",
    "results/prospective_v2_interval.json",
    "results/prospective_v2_integrity_audit.json",
    "results/disjoint_hvp_prospective_audit.json",
    "results/posthoc_projected_theorem_audit.json",
    "results/transformer_hvp_prospective_candidates_blind.json",
    "results/transformer_hvp_prospective_audit.json",
    "results/real_dataset_confirmation_independent_audit.json",
    "results/real_dataset_confirmation_independent_audit.md",
    "results/real_dataset_outward_independent_audit.json",
    "results/real_dataset_outward_independent_audit.md",
    "results/real_dataset_outward_blind.json",
    "results/real_dataset_outward_joined.json",
    "results/wdbc_direct_validated_baseline_audit.json",
    "results/wdbc_direct_validated_baseline_audit.md",
    "results/digits_signed_confirmation_summary.json",
    "results/forecast_selectivity_audit.json",
    "results/digits_signed_confirmation_independent_audit.json",
    "results/digits_signed_confirmation_independent_audit.md",
    "results/digits_outward_blind.json",
    "results/digits_outward_joined.json",
    "results/transformer_unsigned_right_inverse_audit.json",
    "results/transformer_unsigned_right_inverse_audit.md",
    "results/transformer_sweep_ablation.json",
    "results/transformer_sweep_ablation.md",
    "results/transformer_scaling_benchmark.json",
    "results/transformer_scaling_benchmark.md",
    "results/transformer_scaling_benchmark_paper.json",
    "results/transformer_scaling_benchmark_100k.json",
    "results/transformer_scaling_benchmark_1m.json",
    "results/transformer_batched_scaling_benchmark.json",
    "results/transformer_batched_scaling_benchmark.md",
    "results/transformer_batched_scaling_benchmark_paper.json",
    "results/transformer_batched_scaling_benchmark_100k.json",
    "results/transformer_batched_scaling_benchmark_1m.json",
    "results/transformer_batched_replay_seed_333_gate_0_anchor_3000.json",
    "results/transformer_batched_replay_seed_345_gate_1_anchor_1320.json",
    "results/modern_transformer_primitive_audit.json",
    "results/modern_transformer_primitive_audit.md",
    "results/postseal_hardening_independent_audit.json",
    "results/postseal_hardening_independent_audit.md",
]

# Response-centered v3, anytime/sparse practical refinements, and their audits.
FILES.extend([
    "ONE_SHOT_RECENTER_THEOREM.md",
    "HETEROGENEOUS_RECENTER_THEOREM.md",
    "ASYNC_ANYTIME_GREENCERT_THEOREM.md",
    "INEXACT_OPERATOR_GREENCERT_THEOREM.md",
    "WEIGHTED_GREENCERT_THEOREM.md",
    "TRANSFORMER_V3_CONFIRMATION_PROTOCOL.md",
    "TRANSFORMER_V3_METHOD_SEAL.json",
    "TRANSFORMER_V3_CANDIDATE_SEAL.json",
    "TRANSFORMER_V3_CERTIFICATE_SEAL.json",
    "TRANSFORMER_V3_EXECUTION_AMENDMENT.md",
    "TRANSFORMER_V3_EXECUTION_AMENDMENT_SEAL.json",
    "TRANSFORMER_V3_EXECUTION_AMENDMENT_JOIN_SEAL.json",
    "figures/paper_transformer_v3_anytime.pdf",
    "figures/paper_transformer_v3_anytime.png",
    "scripts/transformer_v3_protocol.py",
    "scripts/transformer_v3_certificate.py",
    "scripts/transformer_v3_execution_amendment.py",
    "scripts/run_transformer_v3_confirmation.py",
    "scripts/one_shot_recenter_closure.py",
    "scripts/audit_one_shot_signed_recenter.py",
    "scripts/heterogeneous_recenter_closure.py",
    "scripts/online_progressive_gram.py",
    "scripts/predictable_failure_budget.py",
    "scripts/strict_transformer_block_envelope.py",
    "scripts/audit_transformer_v3_power_grid.py",
    "scripts/audit_transformer_v3_heterogeneous_recenter.py",
    "scripts/audit_transformer_v3_witness_sparse.py",
    "scripts/adaptive_witness_policy.py",
    "scripts/audit_transformer_v3_adaptive_witness.py",
    "scripts/benchmark_transformer_v3_role_sparse.py",
    "scripts/audit_transformer_v3_role_sparse_benchmark.py",
    "scripts/audit_transformer_v3_role_sparse_panel.py",
    "scripts/audit_transformer_v3_outward_calibration.py",
    "scripts/audit_transformer_v3_outward_calibration_result.py",
    "scripts/benchmark_transformer_v3_combined_online_role.py",
    "scripts/audit_transformer_v3_combined_online_role.py",
    "scripts/inexact_anytime_gram.py",
    "scripts/test_inexact_anytime_gram.py",
    "scripts/precision_budget_controller.py",
    "scripts/test_precision_budget_controller.py",
    "scripts/response_centered_event_transport.py",
    "scripts/test_response_centered_event_transport.py",
    "scripts/test_persistent_first_passage_exhaustive.py",
    "scripts/audit_transformer_v3_output_recentering.py",
    "scripts/audit_transformer_v3_output_recentering_result.py",
    "scripts/outward_inexact_anytime_gram.py",
    "scripts/test_outward_inexact_anytime_gram.py",
    "scripts/causal_response_residual.py",
    "scripts/test_causal_response_residual.py",
    "scripts/inexact_variational_recenter.py",
    "scripts/test_inexact_variational_recenter.py",
    "scripts/chi_block_gram.py",
    "scripts/test_chi_block_gram.py",
    "scripts/weighted_recenter_closure.py",
    "scripts/test_weighted_recenter_closure.py",
    "scripts/weighted_green_operator.py",
    "scripts/test_weighted_green_operator.py",
    "scripts/audit_transformer_v3_inexact_operator_tolerance.py",
    "scripts/audit_transformer_v3_outward_inexact_root.py",
    "scripts/audit_transformer_v3_chi_block_bound.py",
    "scripts/audit_transformer_v3_weighted_green.py",
    "scripts/audit_transformer_v3_mixed_precision_residual.py",
    "scripts/audit_transformer_v3_mixed_precision_residual_result.py",
    "scripts/audit_transformer_v3_mixed_precision_timing_aggregate.py",
    "scripts/verify_immutable_mixed_precision_chain.py",
    "scripts/audit_greencert_manuscript_claims.py",
    "scripts/transformer_two_response.py",
    "scripts/directional_two_response.py",
    "scripts/transformer_fourth_jet_bound.py",
    "scripts/test_transformer_two_response.py",
    "scripts/test_directional_two_response.py",
    "scripts/test_directional_two_response_theorem.py",
    "scripts/test_transformer_fourth_jet_bound.py",
    "scripts/audit_transformer_v3_two_response.py",
    "scripts/audit_transformer_v3_two_response_policy.py",
    "scripts/audit_transformer_v3_two_response_local_fourth.py",
    "scripts/audit_transformer_v3_two_response_result.py",
    "scripts/benchmark_transformer_v3_two_response.py",
    "scripts/audit_transformer_v3_two_response_benchmark.py",
    "scripts/amplified_secant_response.py",
    "scripts/test_amplified_secant_response.py",
    "scripts/test_amplified_secant_theorem.py",
    "scripts/audit_transformer_v3_amplified_secant.py",
    "scripts/audit_transformer_v3_amplified_secant_full.py",
    "scripts/benchmark_transformer_v3_amplified_secant.py",
    "scripts/audit_transformer_v3_amplified_secant_result.py",
    "scripts/randomized_residual_certificate.py",
    "scripts/test_randomized_residual_certificate.py",
    "scripts/audit_transformer_v3_response_free_probe.py",
    "scripts/audit_transformer_v3_response_free_probe_result.py",
    "scripts/audit_transformer_v3_four_probe.py",
    "scripts/audit_transformer_v3_four_probe_result.py",
    "scripts/arb_transformer_objective.py",
    "scripts/arb_transformer_multijet.py",
    "scripts/test_arb_transformer_objective.py",
    "scripts/test_arb_transformer_multijet.py",
    "scripts/test_arb_transformer_multijet_randomized.py",
    "scripts/audit_arb_multijet_randomized_test.py",
    "scripts/audit_arb_transformer_secant_checkpoint.py",
    "scripts/audit_arb_transformer_secant_full.py",
    "scripts/audit_arb_transformer_secant_full_result.py",
    "scripts/relinearized_green_closure.py",
    "scripts/test_relinearized_green_closure.py",
    "scripts/audit_transformer_relinearized_green.py",
    "scripts/audit_transformer_relinearized_secant.py",
    "scripts/audit_transformer_relinearized_secant_four_probe.py",
    "scripts/audit_transformer_relinearized_secant_four_probe_result.py",
    "scripts/benchmark_relinearized_green_probe_block.py",
    "scripts/prefix_gram_enclosure.py",
    "scripts/test_prefix_gram_enclosure.py",
    "scripts/cost_aware_forcing.py",
    "scripts/test_cost_aware_forcing.py",
    "scripts/streaming_variational_centerline.py",
    "scripts/test_streaming_variational_centerline.py",
    "scripts/benchmark_streaming_transformer_centerline.py",
    "scripts/direct_image_green_bound.py",
    "scripts/test_direct_image_green_bound.py",
    "scripts/audit_transformer_relinearized_prefix_panel.py",
    "scripts/audit_transformer_relinearized_prefix_panel_result.py",
    "scripts/audit_transformer_direct_image_green_panel.py",
    "scripts/audit_transformer_direct_image_green_panel_result.py",
    "scripts/analytic_jet_release.py",
    "scripts/test_analytic_jet_release.py",
    "scripts/audit_transformer_analytic_jet_release.py",
    "scripts/audit_transformer_analytic_jet_release_compact.py",
    "scripts/audit_transformer_analytic_jet_release_result.py",
    "scripts/structured_parameter_green.py",
    "scripts/test_structured_parameter_green.py",
    "scripts/audit_structured_parameter_green_transformer.py",
    "scripts/verify_structured_parameter_green_audit.py",
    "scripts/structured_parameter_green_v2.py",
    "scripts/test_structured_parameter_green_v2.py",
    "scripts/structured_parameter_green_sealed_v1.py",
    "scripts/test_structured_parameter_green_sealed_v1.py",
    "scripts/structured_parameter_green_source_bridge.py",
    "scripts/audit_anchor_fixed_structured_parameter_green_transformer.py",
    "scripts/verify_anchor_fixed_structured_parameter_green_audit.py",
    "scripts/paper_figure_prefix_scaling.py",
    "scripts/paper_figure_composed_runtime.py",
    "scripts/benchmark_transformer_matched_continuation.py",
    "scripts/benchmark_transformer_v3_streaming_direct_analytic.py",
    "scripts/audit_transformer_v3_streaming_direct_analytic.py",
    "scripts/seal_transformer_streaming_prefix_identity.py",
    "scripts/verify_neurips_page_boundary.py",
    "scripts/audit_transformer_block_postfixed.py",
    "scripts/benchmark_transformer_v3_online_policy.py",
    "scripts/audit_matched_online_benchmark.py",
    "scripts/test_transformer_v3_protocol.py",
    "scripts/test_transformer_v3_preseal.py",
    "scripts/test_one_shot_signed_recenter.py",
    "scripts/test_heterogeneous_recenter_closure.py",
    "scripts/test_online_progressive_gram.py",
    "scripts/test_predictable_failure_budget.py",
    "scripts/test_transformer_v3_witness_sparse.py",
    "scripts/test_adaptive_witness_policy.py",
    "scripts/test_strict_transformer_block_envelope.py",
    "results/transformer_v3_candidates_blind.json",
    "results/transformer_v3_confirmation_audit.json",
    "results/transformer_v3_no_artifact_audit.json",
    "results/transformer_v3_power_grid_postseal_audit.json",
    "results/transformer_v3_heterogeneous_recenter_seed_372_gate_0_anchor_3440.json",
    "results/transformer_v3_heterogeneous_recenter_seed_372_gate_0_anchor_3440_role_budget.json",
    "results/transformer_v3_online_policy_matched_audit.json",
    "results/transformer_v3_online_policy_seed_366_gate_1_anchor_1120_matched-online.json",
    "results/transformer_v3_online_policy_seed_366_gate_1_anchor_1120_matched-full-q8.json",
    "results/transformer_v3_witness_sparse_postseal_audit.json",
    "results/transformer_v3_adaptive_witness_postseal_audit.json",
    "results/transformer_v3_role_sparse_seed_366_gate_1_anchor_1120_audit.json",
    "results/transformer_v3_role_sparse_seed_366_gate_1_anchor_1120_cache.json",
    "results/transformer_v3_role_sparse_seed_366_gate_1_anchor_1120_independent_audit.json",
    "results/transformer_v3_role_sparse_seed_366_gate_0_anchor_1040_audit.json",
    "results/transformer_v3_role_sparse_seed_366_gate_0_anchor_1040_cache.json",
    "results/transformer_v3_role_sparse_seed_366_gate_0_anchor_1040_independent_audit.json",
    "results/transformer_v3_role_sparse_seed_366_gate_2_anchor_1360_audit.json",
    "results/transformer_v3_role_sparse_seed_366_gate_2_anchor_1360_cache.json",
    "results/transformer_v3_role_sparse_seed_366_gate_2_anchor_1360_independent_audit.json",
    "results/transformer_v3_role_sparse_seed_369_gate_1_anchor_4480_audit.json",
    "results/transformer_v3_role_sparse_seed_369_gate_1_anchor_4480_cache.json",
    "results/transformer_v3_role_sparse_seed_369_gate_1_anchor_4480_independent_audit.json",
    "results/transformer_v3_role_sparse_panel_audit.json",
    "results/transformer_v3_outward_calibration_postseal_audit.json",
    "results/transformer_v3_outward_calibration_independent_audit.json",
    "results/transformer_v3_block_postfixed_shortest_postseal_audit.json",
    "results/transformer_v3_combined_online_role_seed_366_gate_1_anchor_1120_matched-combined-v5.json",
    "results/transformer_v3_combined_online_role_seed_366_gate_1_anchor_1120_matched-combined-v5_independent_audit.json",
    "results/transformer_v3_inexact_operator_tolerance_postseal_audit.json",
    "results/transformer_v3_inexact_operator_tolerance_pre_timing_reframe.json",
    "results/transformer_v3_inexact_operator_tolerance_pre_core_hash.json",
    "results/transformer_v3_outward_inexact_root_postseal_audit.json",
    "results/transformer_v3_outward_inexact_root_pre_core_hash.json",
    "results/transformer_v3_chi_block_postseal_audit.json",
    "results/transformer_v3_weighted_green_postseal_audit.json",
    "results/transformer_v3_output_recentering_postseal_audit.json",
    "results/transformer_v3_output_recentering_independent_audit.json",
    "results/transformer_v3_mixed_precision_residual_postseal_audit.json",
    "results/transformer_v3_mixed_precision_residual_independent_audit.json",
    "results/transformer_v3_mixed_precision_residual_replication1_postseal_audit.json",
    "results/transformer_v3_mixed_precision_residual_replication1_independent_audit.json",
    "results/transformer_v3_mixed_precision_residual_replication2_postseal_audit.json",
    "results/transformer_v3_mixed_precision_residual_replication2_independent_audit.json",
    "results/transformer_v3_mixed_precision_residual_replication3_postseal_audit.json",
    "results/transformer_v3_mixed_precision_residual_replication3_independent_audit.json",
    "results/transformer_v3_mixed_precision_timing_three_invocation_audit.json",
    "results/transformer_v3_mixed_precision_timing_aggregate_audit.json",
    "results/greencert_manuscript_claim_audit.json",
    "results/transformer_v3_two_response_postseal_audit.json",
    "results/transformer_v3_two_response_policy_audit.json",
    "results/transformer_v3_two_response_local_fourth_audit.json",
    "results/transformer_v3_two_response_independent_audit.json",
    "results/transformer_v3_two_response_paired_benchmark.json",
    "results/transformer_v3_two_response_paired_benchmark_independent_audit.json",
    "results/transformer_v3_amplified_secant_one_step_audit.json",
    "results/transformer_v3_amplified_secant_full_audit.json",
    "results/transformer_v3_amplified_secant_paired_benchmark.json",
    "results/transformer_v3_amplified_secant_independent_audit.json",
    "results/transformer_v3_response_free_probe_audit.json",
    "results/transformer_v3_response_free_probe_independent_audit.json",
    "results/transformer_v3_four_probe_audit.json",
    "results/transformer_v3_four_probe_independent_audit.json",
    "results/transformer_v3_arb_secant_checkpoint_audit.json",
    "results/transformer_v3_arb_secant_full_audit.json",
    "results/transformer_v3_arb_secant_full_v2_audit.json",
    "results/transformer_v3_arb_secant_full_v2_independent_audit.json",
    "results/transformer_arb_multijet_randomized_test_audit.json",
    "results/transformer_v3_relinearized_green_audit.json",
    "results/transformer_v3_relinearized_secant_audit.json",
    "results/transformer_v3_relinearized_secant_four_probe_audit.json",
    "results/transformer_v3_relinearized_secant_four_probe_independent_audit.json",
    "results/transformer_v3_relinearized_probe_block_benchmark.json",
    "results/transformer_v3_relinearized_prefix_panel_audit.json",
    "results/transformer_v3_relinearized_prefix_panel_independent_audit.json",
    "results/transformer_streaming_centerline_benchmark.json",
    "results/transformer_direct_image_green_panel_audit.json",
    "results/transformer_direct_image_green_panel_independent_audit.json",
    "results/transformer_analytic_jet_release_postseal_audit.json",
    "results/transformer_analytic_jet_release_independent_audit.json",
    "results/structured_parameter_green_transformer_audit.json",
    "results/structured_parameter_green_independent_audit.json",
    "results/anchor_fixed_structured_parameter_green_transformer_audit.json",
    "results/anchor_fixed_structured_parameter_green_independent_audit.json",
    "results/transformer_seed_366_matched_continuation.json",
    "results/transformer_seed_366_streaming_prefix_identity.json",
    "results/transformer_v3_streaming_direct_analytic_seed_366_gate_1_anchor_1120_replicate-1.json",
    "results/transformer_v3_streaming_direct_analytic_seed_366_gate_1_anchor_1120_replicate-2.json",
    "results/transformer_v3_streaming_direct_analytic_seed_366_gate_1_anchor_1120_replicate-3.json",
    "results/transformer_v3_streaming_direct_analytic_audit.json",
])

# The 24 compact training summaries and their separately joined outcomes are the
# primary fresh confirmation batch.  Large checkpoint tensors are deliberately
# excluded; all claim-bearing candidate, certificate, and audit JSONs are kept.
for seed in range(331, 355):
    FILES.extend([
        f"results/transformer_hvp_prospective_seed_{seed}.json",
        f"results/transformer_hvp_prospective_seed_{seed}.outcomes.json",
    ])

# The independent response-centered cohort uses the same compact training
# summaries, with its own prospectively sealed candidate/certificate chain.
for seed in range(355, 379):
    FILES.extend([
        f"results/transformer_hvp_prospective_seed_{seed}.json",
        f"results/transformer_hvp_prospective_seed_{seed}.outcomes.json",
    ])

v3_candidate_manifest = json.loads(
    (ROOT / "results" / "transformer_v3_candidates_blind.json").read_text(
        encoding="utf-8"
    )
)
for record in v3_candidate_manifest["records"]:
    if record.get("forecast_file"):
        FILES.append(record["forecast_file"].replace("\\", "/"))

v3_certificate_seal = json.loads(
    (ROOT / "TRANSFORMER_V3_CERTIFICATE_SEAL.json").read_text(encoding="utf-8")
)
for entry in v3_certificate_seal["certificate_files"]:
    FILES.append(entry["path"].replace("\\", "/"))
for path in (ROOT / "results").glob("transformer_v3_audit_seed_*_gate_*_anchor_*.json"):
    FILES.append(path.relative_to(ROOT).as_posix())

# The WDBC records are compact enough to retain in full, including the complete
# post-seal audit chain and every outward-rounded cache record.
for directory in (
    ROOT / "results" / "real_dataset_confirmation",
    ROOT / "results" / "real_dataset_outward_cache",
    ROOT / "results" / "digits_signed_confirmation",
    ROOT / "results" / "digits_outward_cache",
):
    for path in directory.rglob("*"):
        if path.is_file():
            FILES.append(path.relative_to(ROOT).as_posix())

for path in (ROOT / "results").glob("digits_signed_development*.json"):
    FILES.append(path.relative_to(ROOT).as_posix())

# Corrected-prefix and staged direct/Gram cache rows are the independently
# replayable claim records for the 15-case post-seal scalability panels.
for directory in (
    ROOT / "results" / "transformer_v3_relinearized_prefix_panel_cache",
    ROOT / "results" / "transformer_direct_image_green_panel_cache",
    ROOT / "results" / "structured_parameter_green_transformer_cache",
    ROOT / "results" / "anchor_fixed_structured_parameter_green_transformer_cache",
):
    for path in directory.glob("*.json"):
        FILES.append(path.relative_to(ROOT).as_posix())

candidate_manifest = json.loads(
    (ROOT / "results" / "transformer_green_confirmation_candidates_blind.json").read_text(
        encoding="utf-8"
    )
)
for record in candidate_manifest["records"]:
    if record.get("forecast_file"):
        FILES.append(record["forecast_file"].replace("\\", "/"))

certificate_seal = json.loads(
    (ROOT / "TRANSFORMER_GREEN_CONFIRMATION_CERTIFICATE_SEAL.json").read_text(
        encoding="utf-8"
    )
)
gate_index = {0.7: 0, 0.8: 1, 0.9: 2}
for entry in certificate_seal["certificate_files"]:
    candidate = entry["candidate"]
    FILES.append(entry["path"].replace("\\", "/"))
    FILES.append(
        "results/transformer_green_confirmation_audit_"
        f"seed_{candidate['seed']}_gate_{gate_index[candidate['threshold']]}_"
        f"anchor_{candidate['anchor']}.json"
    )

FILES = list(dict.fromkeys(FILES))

TEXT_SUFFIXES = {".md", ".json", ".py", ".tex", ".bib", ".sty", ".csv"}


def sanitize(text: str) -> str:
    root = str(ROOT)
    replacements = (
        (root.replace("\\", "\\\\"), "<ROOT>"),
        (root, "<ROOT>"),
        ("C:\\\\Users\\\\oddey", "<USER_HOME>"),
        ("C:\\Users\\oddey", "<USER_HOME>"),
        ("Ian Rhee", "Anonymous Author"),
        ("oddey", "anonymous"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def main() -> None:
    missing = [rel for rel in FILES if not (ROOT / rel).is_file()]
    if missing:
        raise FileNotFoundError("missing supplement files: " + ", ".join(missing))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, str | bool]] = {}
    forbidden = (
        b"Ian Rhee",
        b"oddey",
        b"C:\\Users\\",
        b"C:\\\\Users\\\\",
        b"C:/Users/",
    )
    (ROOT / "tmp").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cert_event_supp_", dir=ROOT / "tmp") as tmp:
        stage = Path(tmp)
        for rel in FILES:
            src = ROOT / rel
            dst = stage / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            source_payload = src.read_bytes()
            if src.suffix.lower() in TEXT_SUFFIXES:
                payload = sanitize(source_payload.decode("utf-8")).encode("utf-8")
            else:
                payload = source_payload
            if src.suffix.lower() in TEXT_SUFFIXES:
                leaked = [token.decode("utf-8", errors="replace") for token in forbidden if token in payload]
                if leaked:
                    raise RuntimeError(f"anonymous payload still contains identity/path tokens: {rel}: {leaked}")
            dst.write_bytes(payload)
            manifest[rel] = {
                "source_sha256": hashlib.sha256(source_payload).hexdigest().upper(),
                "packaged_sha256": hashlib.sha256(payload).hexdigest().upper(),
                "sanitized": payload != source_payload,
            }

        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        (stage / "MANIFEST_SHA256.json").write_bytes(manifest_bytes)
        readme = (
            "# Anonymous supplement integrity\n\n"
            "Text files are path- and author-sanitized for double-blind review. "
            "`MANIFEST_SHA256.json` records both the original source hash (the "
            "hash cited by the paper and seal chain) and the packaged sanitized "
            "hash. A `sanitized` flag identifies every changed payload. Binary "
            "artifacts are copied byte-for-byte. Sanitization changes only local "
            "paths and author strings; it does not rewrite numerical records, "
            "code logic, or source-manifest hashes embedded in the records.\n"
        ).encode("utf-8")
        (stage / "ANONYMIZATION_README.md").write_bytes(readme)

        with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(stage).as_posix())

    print(json.dumps({
        "output": str(OUTPUT),
        "files": len(FILES) + 2,
        "bytes": OUTPUT.stat().st_size,
        "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    }, indent=2))


if __name__ == "__main__":
    main()
