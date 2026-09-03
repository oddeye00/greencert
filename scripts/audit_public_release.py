#!/usr/bin/env python3
"""Fail closed on path, credential, size, provenance, and release-manifest leaks."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from pypdf import PdfReader

from audit_directional_replay_dependency_closure import build_record as build_directional_dependency_record
from transformer_directional_anchor_bundle import verify as verify_directional_anchor_bundle


TEXT_SUFFIXES = {
    ".bib", ".cff", ".csv", ".in", ".json", ".md", ".py", ".sty",
    ".tex", ".txt", ".yml", ".yaml",
}
REQUIRED = {
    "README.md",
    "REPRODUCIBILITY.md",
    "DATA.md",
    "FIGURES.md",
    "GREENCERT_ADVERSARIAL_AUDIT.md",
    "LICENSE",
    "LICENSE-PAPER",
    "LITERATURE_AUDIT.md",
    "CITATION.cff",
    "requirements.in",
    "requirements.txt",
    "PUBLIC_MANIFEST_SHA256.json",
    "paper/greencert_arxiv.pdf",
    "paper/greencert_arxiv_release.json",
    "paper/greencert_arxiv_source.zip",
    "paper/greencert_supplement.zip",
    "STRUCTURED_PARAMETER_GREEN_THEOREM_V2.md",
    "STRUCTURED_PARAMETER_GREEN_SOURCE_SUPERSESSION.md",
    "STRUCTURED_PARAMETER_GREEN_AUDIT_PROTOCOL_V2.md",
    "ANCHOR_FIXED_STRUCTURED_PARAMETER_GREEN_AUDIT_PROTOCOL.md",
    "CAUSAL_ROW_GREEN_THEOREM.md",
    "CAUSAL_STRUCTURED_ROW_PANEL_PROTOCOL.md",
    "CAUSAL_STRUCTURED_ROW_PANEL_RESULT.md",
    "DIRECTIONAL_BLOCK_REMAINDER_THEOREM.md",
    "DIRECTIONAL_BLOCK_REMAINDER_THEOREM_V2.md",
    "DIRECTIONAL_BLOCK_REMAINDER_SOURCE_SUPERSESSION.md",
    "DIRECTIONAL_BLOCK_REMAINDER_PROTOCOL.md",
    "DIRECTIONAL_THREE_SWEEP_EVENT_PROTOCOL.md",
    "MIXED_DIRECTIONAL_JET_AUDIT_PROTOCOL.md",
    "DIRECTIONAL_ENVELOPE_TRANSPORT_THEOREM.md",
    "DIRECTIONAL_ENVELOPE_TRANSPORT_AUDIT_PROTOCOL.md",
    "DIRECTIONAL_ENVELOPE_TRANSPORT_SOURCE_ISOLATION_AMENDMENT.md",
    "ADAPTIVE_SWEEP_COHORT_PROTOCOL.md",
    "TRANSFORMER_V3_METHOD_SEAL.json",
    "scripts/test_structured_parameter_green.py",
    "scripts/test_progressive_gram_bound.py",
    "scripts/test_structured_parameter_green_v2.py",
    "scripts/structured_parameter_green_sealed_v1.py",
    "scripts/test_structured_parameter_green_sealed_v1.py",
    "scripts/structured_parameter_green_source_bridge.py",
    "scripts/verify_structured_parameter_green_audit.py",
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
    "scripts/transformer_directional_fourth_bound.py",
    "scripts/test_transformer_directional_fourth_bound.py",
    "scripts/test_directional_block_symmetrization.py",
    "scripts/verify_directional_block_theorem_supersession.py",
    "scripts/diagnose_transformer_directional_block_remainder.py",
    "scripts/audit_transformer_directional_three_sweep_events.py",
    "scripts/transformer_mixed_directional_jet.py",
    "scripts/test_transformer_mixed_directional_jet.py",
    "scripts/audit_transformer_mixed_directional_cohort.py",
    "scripts/test_transformer_directional_envelope_transport.py",
    "scripts/test_transformer_envelope_geometry_cache.py",
    "scripts/audit_transformer_directional_envelope_transport.py",
    "scripts/transformer_block_envelope_v15.py",
    "scripts/transformer_hvp_grokking_v15.py",
    "scripts/transformer_modal_forecast_v15.py",
    "scripts/transformer_optimizer_probe_v15.py",
    "scripts/transformer_mixed_directional_jet_v15.py",
    "scripts/streaming_variational_centerline_v15.py",
    "scripts/audit_transformer_adaptive_sweep_cohort.py",
    "scripts/corrected_path_closure.py",
    "scripts/audit_directional_replay_dependency_closure.py",
    "scripts/transformer_directional_anchor_bundle.py",
    "scripts/test_transformer_directional_sparse_checkpoint_loader.py",
    "scripts/paper_figure_directional_block_remainder.py",
    "results/transformer_directional_block_remainder_diagnostic.json",
    "results/transformer_directional_three_sweep_event_audit.json",
    "results/transformer_mixed_directional_cohort_audit.json",
    "results/transformer_directional_envelope_transport_audit.json",
    "results/transformer_directional_envelope_transport_audit_preisolation_v1.json",
    "results/transformer_fully_recentered_three_sweep_audit.json",
    "results/directional_replay_dependency_closure.json",
    "results/transformer_directional_anchor_states.npz",
    "results/transformer_directional_anchor_states_manifest.json",
    "scripts/reproduce_figures.py",
    "scripts/audit_transformer_v3_streaming_direct_analytic.py",
    "scripts/benchmark_transformer_v3_streaming_direct_analytic.py",
    "scripts/paper_figure_composed_runtime.py",
    "scripts/paper_plot_style.py",
    "scripts/update_public_manifest.py",
    "results/transformer_seed_366_matched_continuation.json",
    "results/transformer_seed_366_streaming_prefix_identity.json",
    "results/transformer_v3_streaming_direct_analytic_audit.json",
    "results/transformer_v3_streaming_direct_analytic_seed_366_gate_1_anchor_1120_replicate-1.json",
    "results/transformer_v3_streaming_direct_analytic_seed_366_gate_1_anchor_1120_replicate-2.json",
    "results/transformer_v3_streaming_direct_analytic_seed_366_gate_1_anchor_1120_replicate-3.json",
} | {
    f"results/transformer_hvp_prospective_seed_{seed}.checkpoints.npz"
    for seed in (360, 361, 366, 369, 370, 372, 373, 375, 378)
}
FIGURES = (
    "paper_transformer_v3_anytime",
    "paper_real_data_confirmation",
    "paper_mechanism_scaling",
    "paper_composed_runtime",
    "paper_relinearized_prefix_panel",
    "paper_transformer_green_confirmation",
    "paper_prospective_horizons",
    "paper_prospective_brackets",
    "paper_directional_block_remainder",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()

    missing = sorted(name for name in REQUIRED if not (root / name).is_file())
    if missing:
        raise AssertionError(f"public release is missing required files: {missing}")

    bad_paths: list[str] = []
    bad_secrets: list[str] = []
    oversized: list[dict[str, int | str]] = []
    forbidden_extensions: list[str] = []
    local_patterns = (
        re.compile(r"[A-Za-z]:\\Users\\[^<\\/\s]+", re.IGNORECASE),
        re.compile("/" + r"home/[^/<\s]+/"),
        re.compile("/" + r"Users/[^/<\s]+/"),
    )
    secret_patterns = (
        re.compile(r"ghp_[A-Za-z0-9]{20,}"),
        re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile("-----" + "BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    )
    blocked = {".ckpt", ".pem", ".pt", ".pth"}

    ignored_roots = {".git", ".venv", "output", "tmp"}
    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).parts[0] not in ignored_roots
        and "__pycache__" not in path.parts
    ]
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        if size >= 95 * 1024 * 1024:
            oversized.append({"path": relative, "bytes": size})
        if path.suffix.lower() in blocked:
            forbidden_extensions.append(relative)
        if path.suffix.lower() in TEXT_SUFFIXES or path.name.startswith("LICENSE") or path.name in {"Makefile", "Dockerfile"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            if any(pattern.search(text) for pattern in local_patterns):
                bad_paths.append(relative)
            if any(pattern.search(text) for pattern in secret_patterns):
                bad_secrets.append(relative)

    if bad_paths or bad_secrets or oversized or forbidden_extensions:
        raise AssertionError(
            json.dumps(
                {
                    "local_path_leaks": bad_paths,
                    "credential_shaped_strings": bad_secrets,
                    "oversized": oversized,
                    "blocked_extensions": forbidden_extensions,
                },
                indent=2,
            )
        )

    figure_rows = {}
    for stem in FIGURES:
        pdf = root / "figures" / f"{stem}.pdf"
        png = root / "figures" / f"{stem}.png"
        if not pdf.is_file() or not png.is_file():
            raise FileNotFoundError(stem)
        metadata = PdfReader(pdf).metadata or {}
        creator = str(metadata.get("/Creator", ""))
        if not creator.startswith("Matplotlib "):
            raise AssertionError(f"figure lacks Matplotlib provenance: {stem}")
        figure_rows[stem] = {
            "creator": creator,
            "pdf_sha256": digest(pdf),
            "png_sha256": digest(png),
        }

    paper_pdf = root / "paper" / "greencert_arxiv.pdf"
    paper_reader = PdfReader(paper_pdf)
    paper_metadata = paper_reader.metadata or {}
    if len(paper_reader.pages) != 45:
        raise AssertionError(f"unexpected public preprint length: {len(paper_reader.pages)}")
    if str(paper_metadata.get("/Author", "")) != "Ian Rhee":
        raise AssertionError("public preprint author metadata changed")
    release = json.loads(
        (root / "paper" / "greencert_arxiv_release.json").read_text(encoding="utf-8")
    )
    if int(release["pages"]) != 45 or release["pdf"]["sha256"] != digest(paper_pdf):
        raise AssertionError("public preprint and arXiv release manifest differ")
    release_payloads = {
        "source_bundle": root / "paper" / "greencert_arxiv_source.zip",
        "supplement": root / "paper" / "greencert_supplement.zip",
    }
    for key, path in release_payloads.items():
        if (
            release[key]["sha256"] != digest(path)
            or int(release[key]["bytes"]) != path.stat().st_size
        ):
            raise AssertionError(f"public {key} and arXiv release manifest differ")

    v3_method_seal = json.loads(
        (root / "TRANSFORMER_V3_METHOD_SEAL.json").read_text(encoding="utf-8")
    )
    for relative, expected in v3_method_seal["code_manifest"].items():
        path = root / relative
        if not path.is_file() or digest(path) != expected:
            raise AssertionError(f"v3 method-seal source mismatch: {relative}")

    expected_audits = {
        "results/structured_parameter_green_independent_audit.json":
            "INDEPENDENT STRUCTURED PARAMETER GREEN AUDIT PASSED",
        "results/anchor_fixed_structured_parameter_green_independent_audit.json":
            "INDEPENDENT ANCHOR-FIXED STRUCTURED AUDIT PASSED",
        "results/transformer_directional_block_remainder_diagnostic.json":
            "directional block remainder cohort diagnostic complete",
        "results/transformer_directional_three_sweep_event_audit.json":
            "directional three-sweep event audit complete",
        "results/transformer_mixed_directional_cohort_audit.json":
            "mixed directional cohort audit complete",
        "results/transformer_directional_envelope_transport_audit.json":
            "DIRECTIONAL ENVELOPE TRANSPORT AUDIT PASSED",
        "results/transformer_causal_structured_row_panel_audit.json":
            "frozen structured causal-row Transformer panel audit complete",
        "results/transformer_causal_structured_row_panel_verification.json":
            "structured causal-row panel independently verified",
    }
    for relative, status in expected_audits.items():
        record = json.loads((root / relative).read_text(encoding="utf-8"))
        if record.get("status") != status:
            raise AssertionError(f"public independent audit status changed: {relative}")

    directional = json.loads(
        (root / "results/transformer_directional_block_remainder_diagnostic.json")
        .read_text(encoding="utf-8")
    )
    holdout = directional["nondevelopment_cases"]
    if (
        not holdout["every_step_directional_no_larger"]
        or int(holdout["newly_closed"]) != 3
    ):
        raise AssertionError("directional block promotion invariants changed")
    event = json.loads(
        (root / "results/transformer_directional_three_sweep_event_audit.json")
        .read_text(encoding="utf-8")
    )
    if int(event["issued"]) != 4 or int(event["retained_sealed_bracket"]) != 4:
        raise AssertionError("directional event-retention invariants changed")
    mixed = json.loads(
        (root / "results/transformer_mixed_directional_cohort_audit.json")
        .read_text(encoding="utf-8")
    )
    if (
        not mixed["all_local_and_closure_results_reproduced"]
        or float(mixed["maximum_local_relative_error"]) > 3.0e-12
    ):
        raise AssertionError("mixed directional equivalence invariants changed")
    if mixed["prespecified_audit_passed"]:
        raise AssertionError("failed mixed-jet speed gate was silently reclassified")

    transported = json.loads(
        (root / "results/transformer_directional_envelope_transport_audit.json")
        .read_text(encoding="utf-8")
    )
    transported_rows = transported["rows"]
    if (
        not transported["protocol_gates_passed"]
        or not transported["source_isolated_replay"]
        or int(transported["outcome_files_read"]) != 0
        or (len(transported_rows), sum(int(row["issued"]) for row in transported_rows))
        != (4, 4)
        or sum(int(row["transport_checks"]) for row in transported_rows) != 9420
        or not all(bool(row["same_centerline"]) for row in transported_rows)
        or not all(bool(row["same_corrected_path"]) for row in transported_rows)
    ):
        raise AssertionError("directional-envelope transport invariants changed")
    transported_dependencies = {
        "closure_parent": "results/transformer_directional_block_remainder_diagnostic.json",
        "full_parent": "results/transformer_fully_recentered_three_sweep_audit.json",
        "event_parent": "results/transformer_directional_three_sweep_event_audit.json",
        "protocol": "DIRECTIONAL_ENVELOPE_TRANSPORT_AUDIT_PROTOCOL.md",
        "theorem": "DIRECTIONAL_ENVELOPE_TRANSPORT_THEOREM.md",
        "source_isolation_amendment": "DIRECTIONAL_ENVELOPE_TRANSPORT_SOURCE_ISOLATION_AMENDMENT.md",
        "block_envelope_v15": "scripts/transformer_block_envelope_v15.py",
        "hvp_v15": "scripts/transformer_hvp_grokking_v15.py",
        "modal_v15": "scripts/transformer_modal_forecast_v15.py",
        "mixed_jet_v15": "scripts/transformer_mixed_directional_jet_v15.py",
        "streaming_centerline_v15": "scripts/streaming_variational_centerline_v15.py",
        "historical_mixed_jet": "scripts/transformer_mixed_directional_jet.py",
        "historical_streaming_centerline": "scripts/streaming_variational_centerline.py",
        "optimizer_probe_v15": "scripts/transformer_optimizer_probe_v15.py",
        "v3_method_seal": "TRANSFORMER_V3_METHOD_SEAL.json",
        "script": "scripts/audit_transformer_directional_envelope_transport.py",
    }
    for key, relative in transported_dependencies.items():
        if transported["source_hashes"][key] != digest(root / relative):
            raise AssertionError(
                f"directional-envelope source lock changed: {relative}"
            )

    causal_row = json.loads(
        (root / "results/transformer_causal_structured_row_panel_audit.json")
        .read_text(encoding="utf-8")
    )
    causal_verification = json.loads(
        (root / "results/transformer_causal_structured_row_panel_verification.json")
        .read_text(encoding="utf-8")
    )
    if (
        (int(causal_row["cases"]), int(causal_row["holdout_issued"]),
         int(causal_row["brackets_retained"])) != (15, 14, 15)
        or not causal_row["promotion_passed"]
        or int(causal_row["outcome_files_read"]) != 0
    ):
        raise AssertionError("causal-row public panel invariants changed")
    if (
        causal_verification["audit_sha256"]
        != digest(root / "results/transformer_causal_structured_row_panel_audit.json")
        or int(causal_verification["issued_recomputed"]) != 15
        or int(causal_verification["outcome_files_read"]) != 0
    ):
        raise AssertionError("causal-row public verification invariants changed")

    directional_dependency_path = (
        root / "results/directional_replay_dependency_closure.json"
    )
    committed_directional_dependency = json.loads(
        directional_dependency_path.read_text(encoding="utf-8")
    )
    rebuilt_directional_dependency = build_directional_dependency_record()
    if committed_directional_dependency != rebuilt_directional_dependency:
        raise AssertionError("directional replay dependency lock is stale")
    directional_anchor_audit = verify_directional_anchor_bundle(root)
    if int(directional_anchor_audit["anchors"]) != 15:
        raise AssertionError("directional anchor bundle cardinality changed")

    manifest = json.loads((root / "PUBLIC_MANIFEST_SHA256.json").read_text(encoding="utf-8"))
    manifest_files = set(manifest["files"])
    actual_files = {
        path.relative_to(root).as_posix()
        for path in files
        if path.name != "PUBLIC_MANIFEST_SHA256.json"
    }
    if manifest_files != actual_files:
        raise AssertionError("public manifest and repository payload differ")
    for relative, row in manifest["files"].items():
        path = root / relative
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise AssertionError(f"public manifest mismatch: {relative}")

    report = {
        "status": "public release audit passed",
        "root": str(root),
        "files": len(files),
        "largest_file_bytes": max(path.stat().st_size for path in files),
        "local_path_leaks": 0,
        "credential_shaped_strings": 0,
        "blocked_checkpoint_or_key_files": 0,
        "matplotlib_figures": figure_rows,
        "manifest_verified": True,
        "preprint_pages": len(paper_reader.pages),
        "claim_audits_verified": len(expected_audits) + 1,
        "directional_anchors_verified": int(directional_anchor_audit["anchors"]),
        "v3_method_seal_files_verified": len(v3_method_seal["code_manifest"]),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
