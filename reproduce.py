#!/usr/bin/env python3
"""Human-readable entry point for GREENCERT reproduction tiers."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent

TIERS = {
    "smoke": (
        "scripts/test_batched_green_operator.py",
        "scripts/test_structured_parameter_green.py",
        "scripts/test_structured_parameter_green_v2.py",
        "scripts/structured_parameter_green_source_bridge.py",
        "scripts/test_one_shot_signed_recenter.py",
        "scripts/test_online_progressive_gram.py",
        "scripts/test_progressive_gram_bound.py",
        "scripts/test_predictable_failure_budget.py",
        "scripts/test_inexact_anytime_gram.py",
        "scripts/test_response_centered_event_transport.py",
        "scripts/test_persistent_first_passage_exhaustive.py",
        "scripts/test_prefix_gram_enclosure.py",
        "scripts/test_direct_image_green_bound.py",
        "scripts/test_analytic_jet_release.py",
        "scripts/verify_directional_block_theorem_supersession.py",
        "scripts/test_directional_block_symmetrization.py",
        "scripts/test_transformer_directional_fourth_bound.py",
        "scripts/test_transformer_mixed_directional_jet.py",
        "scripts/transformer_directional_anchor_bundle.py",
        "scripts/test_transformer_directional_sparse_checkpoint_loader.py",
        "scripts/audit_directional_replay_dependency_closure.py",
        "scripts/audit_greencert_manuscript_claims.py",
    ),
    "artifact-audit": (
        "scripts/audit_real_dataset_confirmation.py",
        "scripts/audit_digits_signed_confirmation.py",
        "scripts/audit_transformer_green_confirmation_statistics.py",
        "scripts/audit_transformer_relinearized_prefix_panel_result.py",
        "scripts/audit_transformer_direct_image_green_panel_result.py",
        "scripts/audit_transformer_analytic_jet_release_compact.py",
        "scripts/audit_transformer_v3_streaming_direct_analytic.py",
        "scripts/verify_structured_parameter_green_audit.py",
        "scripts/verify_anchor_fixed_structured_parameter_green_audit.py",
        "scripts/audit_greencert_manuscript_claims.py",
    ),
    "outward": (
        "scripts/test_outward_real_dataset_confirmation.py",
        "scripts/audit_real_dataset_outward.py",
        "scripts/test_digits_signed_confirmation_audit.py",
    ),
    "figures": ("scripts/reproduce_figures.py",),
    "paper": (
        "scripts/audit_greencert_manuscript_claims.py",
        "scripts/build_arxiv_release.py",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tier", nargs="?", choices=sorted(TIERS))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    if args.list:
        print(json.dumps(TIERS, indent=2))
        return
    if args.tier is None:
        parser.error("choose a tier or pass --list")

    report = {"tier": args.tier, "commands": [], "passed": True}
    for relative in TIERS[args.tier]:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        command = [sys.executable, relative]
        if relative.endswith("reproduce_figures.py"):
            command.append("--check-determinism")
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=ROOT, check=False)
        elapsed = time.perf_counter() - started
        row = {
            "command": command,
            "returncode": completed.returncode,
            "seconds": elapsed,
        }
        report["commands"].append(row)
        if completed.returncode:
            report["passed"] = False
            if not args.keep_going:
                break

    output = ROOT / "output" / "reproduction_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
