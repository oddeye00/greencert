#!/usr/bin/env python3
"""Verify hashes, required v3 evidence, and anonymity of the supplement ZIP."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "output" / "certified_local_training_events_supplement.zip"
TEXT_SUFFIXES = {".md", ".json", ".py", ".tex", ".bib", ".sty", ".csv"}
FORBIDDEN = (b"Ian Rhee", b"oddey", b"C:\\Users\\", b"C:\\\\Users\\\\", b"C:/Users/")
REQUIRED = {
    "TRANSFORMER_V3_METHOD_SEAL.json",
    "TRANSFORMER_V3_CANDIDATE_SEAL.json",
    "TRANSFORMER_V3_CERTIFICATE_SEAL.json",
    "results/transformer_v3_confirmation_audit.json",
    "results/transformer_v3_witness_sparse_postseal_audit.json",
    "results/transformer_v3_block_postfixed_shortest_postseal_audit.json",
    "DIRECTIONAL_TWO_RESPONSE_THEOREM.md",
    "AMPLIFIED_SECANT_RESPONSE_THEOREM.md",
    "RANDOMIZED_RESIDUAL_PROBE_THEOREM.md",
    "RELINEARIZED_GREEN_THEOREM.md",
    "ANALYTIC_JET_RELEASE_THEOREM.md",
    "STRUCTURED_PARAMETER_GREEN_THEOREM.md",
    "STRUCTURED_PARAMETER_GREEN_THEOREM_V1_INDEXING_NOTE.md",
    "STRUCTURED_PARAMETER_GREEN_THEOREM_V2.md",
    "CAUSAL_ROW_GREEN_THEOREM.md",
    "CAUSAL_STRUCTURED_ROW_PANEL_PROTOCOL.md",
    "CAUSAL_STRUCTURED_ROW_PANEL_RESULT.md",
    "DIRECTIONAL_ENVELOPE_TRANSPORT_THEOREM.md",
    "DIRECTIONAL_ENVELOPE_TRANSPORT_AUDIT_PROTOCOL.md",
    "DIRECTIONAL_ENVELOPE_TRANSPORT_SOURCE_ISOLATION_AMENDMENT.md",
    "STRUCTURED_PARAMETER_GREEN_AUDIT_PROTOCOL_V2.md",
    "ANCHOR_FIXED_STRUCTURED_PARAMETER_GREEN_AUDIT_PROTOCOL.md",
    "AMPLIFIED_SECANT_PROBE_PROTOCOL.md",
    "AMPLIFIED_SECANT_FOUR_PROBE_PROTOCOL.md",
    "AMPLIFIED_SECANT_OUTWARD_EXECUTION_PROTOCOL_V2.md",
    "RELINEARIZED_GREEN_AUDIT_PROTOCOL.md",
    "RELINEARIZED_SECANT_AUDIT_PROTOCOL.md",
    "RELINEARIZED_SECANT_FOUR_PROBE_PROTOCOL.md",
    "results/transformer_v3_two_response_independent_audit.json",
    "results/transformer_v3_two_response_paired_benchmark_independent_audit.json",
    "results/transformer_v3_amplified_secant_independent_audit.json",
    "results/transformer_v3_response_free_probe_independent_audit.json",
    "results/transformer_v3_four_probe_independent_audit.json",
    "results/transformer_v3_arb_secant_full_v2_independent_audit.json",
    "results/transformer_v3_relinearized_green_audit.json",
    "results/transformer_v3_relinearized_secant_audit.json",
    "results/transformer_v3_relinearized_secant_four_probe_independent_audit.json",
    "results/transformer_v3_relinearized_probe_block_benchmark.json",
    "scripts/analytic_jet_release.py",
    "scripts/test_analytic_jet_release.py",
    "scripts/audit_transformer_analytic_jet_release.py",
    "scripts/audit_transformer_analytic_jet_release_compact.py",
    "scripts/audit_transformer_analytic_jet_release_result.py",
    "results/transformer_analytic_jet_release_postseal_audit.json",
    "results/transformer_analytic_jet_release_independent_audit.json",
    "scripts/structured_parameter_green.py",
    "scripts/test_structured_parameter_green.py",
    "scripts/verify_structured_parameter_green_audit.py",
    "scripts/structured_parameter_green_v2.py",
    "scripts/test_structured_parameter_green_v2.py",
    "scripts/verify_anchor_fixed_structured_parameter_green_audit.py",
    "results/structured_parameter_green_transformer_audit.json",
    "results/structured_parameter_green_independent_audit.json",
    "results/anchor_fixed_structured_parameter_green_transformer_audit.json",
    "results/anchor_fixed_structured_parameter_green_independent_audit.json",
    "scripts/causal_row_green.py",
    "scripts/diagnose_transformer_causal_row_green.py",
    "scripts/combine_causal_row_probe_blocks.py",
    "scripts/audit_transformer_causal_structured_row_panel.py",
    "scripts/verify_transformer_causal_structured_row_panel.py",
    "scripts/test_causal_row_green.py",
    "scripts/test_causal_structured_row_green.py",
    "scripts/test_combine_causal_row_probe_blocks.py",
    "scripts/test_causal_row_green_transformer_batch.py",
    "results/transformer_causal_structured_row_panel_audit.json",
    "results/transformer_causal_structured_row_panel_verification.json",
    "scripts/test_transformer_directional_envelope_transport.py",
    "scripts/test_transformer_envelope_geometry_cache.py",
    "scripts/audit_transformer_directional_envelope_transport.py",
    "scripts/transformer_block_envelope_v15.py",
    "scripts/transformer_hvp_grokking_v15.py",
    "scripts/transformer_modal_forecast_v15.py",
    "scripts/transformer_optimizer_probe_v15.py",
    "scripts/transformer_mixed_directional_jet_v15.py",
    "scripts/streaming_variational_centerline_v15.py",
    "results/transformer_directional_envelope_transport_audit.json",
    "results/transformer_directional_envelope_transport_audit_preisolation_v1.json",
    "results/transformer_arb_multijet_randomized_test_audit.json",
    "figures/paper_transformer_v3_anytime.pdf",
    "scripts/paper_plot_style.py",
    "scripts/reproduce_figures.py",
    "results/figure_reproducibility_audit.json",
    "paper/certified_local_training_events_neurips2026_blind.tex",
    "paper/certified_local_training_events_arxiv.tex",
    "MANIFEST_SHA256.json",
    "ANONYMIZATION_README.md",
}


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def main() -> None:
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = set(archive.namelist())
        missing = sorted(REQUIRED - names)
        if missing:
            raise AssertionError(f"archive lacks required v3 evidence: {missing}")
        manifest = json.loads(archive.read("MANIFEST_SHA256.json"))
        if set(manifest) != names - {"MANIFEST_SHA256.json", "ANONYMIZATION_README.md"}:
            raise AssertionError("manifest and archive payload sets differ")

        for cache_prefix, expected_rows in (
            ("results/structured_parameter_green_transformer_cache/", 15),
            ("results/anchor_fixed_structured_parameter_green_transformer_cache/", 15),
            ("results/transformer_causal_structured_row_panel_cache/", 17),
        ):
            cache_rows = [name for name in names if name.startswith(cache_prefix)]
            if len(cache_rows) != expected_rows:
                raise AssertionError(
                    f"expected {expected_rows} independently replayable cache rows under {cache_prefix}"
                )

        sanitized = 0
        for name, row in manifest.items():
            payload = archive.read(name)
            if digest(payload) != row["packaged_sha256"]:
                raise AssertionError(f"packaged hash mismatch: {name}")
            source = ROOT / name
            if source.is_file() and digest(source.read_bytes()) != row["source_sha256"]:
                raise AssertionError(f"source hash mismatch: {name}")
            if bool(row["sanitized"]):
                sanitized += 1
            if Path(name).suffix.lower() in TEXT_SUFFIXES:
                leaked = [token for token in FORBIDDEN if token in payload]
                if leaked:
                    raise AssertionError(f"identity/path leak in {name}: {leaked}")

        method_name = "TRANSFORMER_V3_METHOD_SEAL.json"
        method_source = manifest[method_name]["source_sha256"]
        method_seal = json.loads(archive.read(method_name))
        if method_seal["status"] != "FROZEN BEFORE V3 FRESH TRAINING":
            raise AssertionError("v3 method-seal status changed")
        protocol_name = method_seal["protocol"].replace("\\", "/")
        if method_seal["protocol_sha256"] != manifest[protocol_name]["source_sha256"]:
            raise AssertionError("v3 protocol hash no longer matches its method seal")
        for relative, expected in method_seal["code_manifest"].items():
            relative = relative.replace("\\", "/")
            if relative not in manifest or manifest[relative]["source_sha256"] != expected:
                raise AssertionError(f"v3 method-seal dependency changed: {relative}")

        causal_audit_name = "results/transformer_causal_structured_row_panel_audit.json"
        causal_verification_name = (
            "results/transformer_causal_structured_row_panel_verification.json"
        )
        causal_audit = json.loads(archive.read(causal_audit_name))
        causal_verification = json.loads(archive.read(causal_verification_name))
        if (
            int(causal_audit["cases"]) != 15
            or int(causal_audit["holdout_issued"]) != 14
            or int(causal_audit["brackets_retained"]) != 15
            or not causal_audit["promotion_passed"]
        ):
            raise AssertionError("causal-row panel invariants changed")
        if (
            causal_verification["status"]
            != "structured causal-row panel independently verified"
            or causal_verification["audit_sha256"]
            != manifest[causal_audit_name]["source_sha256"]
            or int(causal_verification["issued_recomputed"]) != 15
        ):
            raise AssertionError("causal-row independent verification changed")

        transported = json.loads(
            archive.read("results/transformer_directional_envelope_transport_audit.json")
        )
        transported_rows = transported["rows"]
        if (
            transported["status"] != "DIRECTIONAL ENVELOPE TRANSPORT AUDIT PASSED"
            or not transported["protocol_gates_passed"]
            or int(transported["outcome_files_read"]) != 0
            or len(transported_rows) != 4
            or sum(int(row["issued"]) for row in transported_rows) != 4
            or sum(int(row["transport_checks"]) for row in transported_rows) != 9420
        ):
            raise AssertionError("directional-envelope transport invariants changed")

    print(
        json.dumps(
            {
                "archive": str(ARCHIVE),
                "files": len(names),
                "sanitized_text_payloads": sanitized,
                "archive_sha256": digest(ARCHIVE.read_bytes()),
                "v3_method_source_sha256": method_source,
                "anonymity_scan_passed": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
