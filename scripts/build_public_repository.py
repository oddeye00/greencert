#!/usr/bin/env python3
"""Assemble the complete, path-scrubbed public GREENCERT Git repository."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "output" / "certified_local_training_events_supplement.zip"
DESTINATION = ROOT / "output" / "github" / "greencert"

PUBLIC_FILES = (
    ".github/workflows/causal-resolvent-diagnostic.yml",
    ".gitattributes",
    ".gitignore",
    "CITATION.cff",
    "CAUSAL_STRUCTURED_RESOLVENT_THEOREM.md",
    "CAUSAL_ROW_GREEN_THEOREM.md",
    "CAUSAL_STRUCTURED_ROW_PANEL_PROTOCOL.md",
    "CAUSAL_STRUCTURED_ROW_PANEL_RESULT.md",
    "ADAPTIVE_SWEEP_COHORT_PROTOCOL.md",
    "DATA.md",
    "Dockerfile",
    "FIGURES.md",
    "GREENCERT_ADVERSARIAL_AUDIT.md",
    "LICENSE",
    "LICENSE-PAPER",
    "LITERATURE_AUDIT.md",
    "Makefile",
    "README.md",
    "REPRODUCIBILITY.md",
    "SUPPLEMENT_README.md",
    "STRUCTURED_DIRECTIONAL_TWO_RESPONSE_THEOREM.md",
    "STRUCTURED_PARAMETER_GREEN_SOURCE_SUPERSESSION.md",
    "environment.yml",
    "reproduce.py",
    "requirements.in",
    "requirements.txt",
    "data/transformer_seed_366_anchor_1120.json",
    "data/transformer_seed_366_anchor_1120_corrected_parameter.npy",
    "data/transformer_seed_366_anchor_1120_corrected_path.json",
    "data/transformer_seed_366_anchor_1120_correction_parameter.npy",
    "data/transformer_seed_366_anchor_1120_parameter.npy",
    "data/transformer_seed_366_anchor_1120_velocity.npy",
    "results/causal_forward_radius_synthetic_audit.json",
    "results/figure_reproducibility_audit.json",
    "results/greencert_manuscript_claim_audit.json",
    "results/directional_replay_dependency_closure.json",
    "results/transformer_directional_anchor_states.npz",
    "results/transformer_directional_anchor_states_manifest.json",
    "results/transformer_hvp_prospective_seed_360.checkpoints.npz",
    "results/transformer_hvp_prospective_seed_361.checkpoints.npz",
    "results/transformer_hvp_prospective_seed_366.checkpoints.npz",
    "results/transformer_hvp_prospective_seed_369.checkpoints.npz",
    "results/transformer_hvp_prospective_seed_370.checkpoints.npz",
    "results/transformer_hvp_prospective_seed_372.checkpoints.npz",
    "results/transformer_hvp_prospective_seed_373.checkpoints.npz",
    "results/transformer_hvp_prospective_seed_375.checkpoints.npz",
    "results/transformer_hvp_prospective_seed_378.checkpoints.npz",
    "results/transformer_direct_image_green_panel_audit.json",
    "results/transformer_causal_structured_row_panel_audit.json",
    "results/transformer_causal_structured_row_panel_verification.json",
    "results/transformer_hvp_prospective_seed_366.json",
    "results/transformer_low_rank_resolvent_diagnostic.json",
    "results/transformer_low_rank_resolvent_independent_audit.json",
    "results/transformer_seed_366_anchor_1120_regeneration_bridge.json",
    "results/transformer_seed_366_matched_continuation.json",
    "results/transformer_seed_366_streaming_prefix_identity.json",
    "results/transformer_v3_relinearized_prefix_panel_audit.json",
    "results/transformer_v3_streaming_direct_analytic_audit.json",
    "results/transformer_v3_streaming_direct_analytic_seed_366_gate_1_anchor_1120_replicate-1.json",
    "results/transformer_v3_streaming_direct_analytic_seed_366_gate_1_anchor_1120_replicate-2.json",
    "results/transformer_v3_streaming_direct_analytic_seed_366_gate_1_anchor_1120_replicate-3.json",
    "results/transformer_segmented_resolvent_diagnostic.json",
    "results/transformer_segmented_resolvent_independent_audit.json",
    "figures/paper_composed_runtime.pdf",
    "figures/paper_composed_runtime.png",
    "scripts/audit_greencert_manuscript_claims.py",
    "scripts/audit_directional_replay_dependency_closure.py",
    "scripts/audit_transformer_adaptive_sweep_cohort.py",
    "scripts/audit_causal_forward_radius.py",
    "scripts/audit_public_release.py",
    "scripts/audit_transformer_v3_streaming_direct_analytic.py",
    "scripts/benchmark_transformer_matched_continuation.py",
    "scripts/benchmark_transformer_v3_streaming_direct_analytic.py",
    "scripts/build_anonymous_supplement.py",
    "scripts/build_arxiv_release.py",
    "scripts/build_public_repository.py",
    "scripts/check_reproduction_environment.py",
    "scripts/causal_structured_resolvent.py",
    "scripts/causal_row_green.py",
    "scripts/combine_causal_row_probe_blocks.py",
    "scripts/corrected_path_closure.py",
    "scripts/diagnose_transformer_causal_row_green.py",
    "scripts/audit_transformer_causal_structured_row_panel.py",
    "scripts/diagnose_transformer_low_rank_resolvent.py",
    "scripts/diagnose_transformer_segmented_resolvent.py",
    "scripts/export_transformer_anchor_state.py",
    "scripts/export_transformer_corrected_parameter_path.py",
    "scripts/materialize_transformer_anchor_checkpoint.py",
    "scripts/paper_plot_style.py",
    "scripts/reproduce_figures.py",
    "scripts/regenerate_transformer_checkpoint.py",
    "scripts/update_public_manifest.py",
    "scripts/verify_anonymous_supplement.py",
    "scripts/make_transformer_v3_anytime_figure.py",
    "scripts/paper_figure_new_evidence.py",
    "scripts/paper_figure_composed_runtime.py",
    "scripts/paper_figure_prefix_scaling.py",
    "scripts/paper_figure_transformer_green_confirmation.py",
    "scripts/paper_figures_prospective.py",
    "scripts/seal_transformer_streaming_prefix_identity.py",
    "scripts/structured_directional_two_response.py",
    "scripts/structured_parameter_green_sealed_v1.py",
    "scripts/structured_parameter_green_source_bridge.py",
    "scripts/test_causal_structured_resolvent.py",
    "scripts/test_causal_row_green.py",
    "scripts/test_causal_structured_row_green.py",
    "scripts/test_combine_causal_row_probe_blocks.py",
    "scripts/test_causal_row_green_transformer_batch.py",
    "scripts/test_transformer_directional_sparse_checkpoint_loader.py",
    "scripts/test_structured_directional_two_response.py",
    "scripts/test_structured_parameter_green_sealed_v1.py",
    "scripts/test_theorem_text_integrity.py",
    "scripts/verify_transformer_low_rank_resolvent_diagnostic.py",
    "scripts/verify_transformer_causal_structured_row_panel.py",
    "scripts/verify_transformer_segmented_resolvent_diagnostic.py",
    "scripts/transformer_directional_anchor_bundle.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def copy_relative(relative: str) -> None:
    source = ROOT / relative
    destination = DESTINATION / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def clear_readonly_and_retry(function, path: str, _error) -> None:
    """Permit replacement of a previously generated read-only staging tree."""
    os.chmod(path, stat.S_IWRITE)
    function(path)


def main() -> None:
    if not ARCHIVE.is_file():
        raise FileNotFoundError("build and verify the supplement before the public repository")

    parent = (ROOT / "output" / "github").resolve()
    destination = DESTINATION.resolve()
    if destination.parent != parent or destination.name != "greencert":
        raise RuntimeError(f"refusing unsafe public-repository target: {destination}")
    parent.mkdir(parents=True, exist_ok=True)
    preserved_git = parent / ".greencert-git-preserved"
    git_metadata = DESTINATION / ".git"
    if preserved_git.exists():
        raise RuntimeError(f"refusing to overwrite preserved Git metadata: {preserved_git}")
    if git_metadata.exists():
        git_metadata.rename(preserved_git)
    try:
        if DESTINATION.exists():
            shutil.rmtree(DESTINATION, onexc=clear_readonly_and_retry)
        DESTINATION.mkdir(parents=True)
    finally:
        if preserved_git.exists():
            DESTINATION.mkdir(parents=True, exist_ok=True)
            preserved_git.rename(DESTINATION / ".git")

    with zipfile.ZipFile(ARCHIVE) as archive:
        archive.extractall(DESTINATION)

    # Restore the authored paper source while retaining path-scrubbed historical seals.
    for relative in (
        "paper/certified_local_training_events_neurips2026.tex",
        "paper/certified_local_training_events_arxiv.tex",
        "paper/transformer_jet_appendix.tex",
        "paper/checklist.tex",
        "paper/references.bib",
        "paper/neurips_2026.sty",
    ):
        copy_relative(relative)
    for relative in PUBLIC_FILES:
        copy_relative(relative)

    workflow_source = ROOT / ".github" / "workflows" / "reproducibility.yml"
    workflow_destination = DESTINATION / ".github" / "workflows" / "reproducibility.yml"
    workflow_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(workflow_source, workflow_destination)

    paper_pdf = ROOT / "output" / "pdf" / "greencert_arxiv.pdf"
    if not paper_pdf.is_file():
        raise FileNotFoundError("build the final arXiv PDF before the public repository")
    shutil.copyfile(paper_pdf, DESTINATION / "paper" / "greencert_arxiv.pdf")
    release = ROOT / "output" / "arxiv" / "greencert_arxiv_release.json"
    shutil.copyfile(release, DESTINATION / "paper" / "greencert_arxiv_release.json")
    source_bundle = ROOT / "output" / "arxiv" / "greencert_arxiv_source.zip"
    shutil.copyfile(source_bundle, DESTINATION / "paper" / "greencert_arxiv_source.zip")
    shutil.copyfile(ARCHIVE, DESTINATION / "paper" / "greencert_supplement.zip")

    files = sorted(
        path for path in DESTINATION.rglob("*") if path.is_file() and ".git" not in path.parts
    )
    manifest = {
        "format": 1,
        "repository": "https://github.com/oddeye00/greencert",
        "files": {
            path.relative_to(DESTINATION).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in files
            if path.name != "PUBLIC_MANIFEST_SHA256.json"
        },
    }
    (DESTINATION / "PUBLIC_MANIFEST_SHA256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(
        json.dumps(
            {
                "destination": str(DESTINATION),
                "files": len(manifest["files"]) + 1,
                "bytes": sum(path.stat().st_size for path in DESTINATION.rglob("*") if path.is_file()),
                "manifest_sha256": digest(DESTINATION / "PUBLIC_MANIFEST_SHA256.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
