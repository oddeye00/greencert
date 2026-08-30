#!/usr/bin/env python3
"""Burned-seed end-to-end dry run of every sealed confirmation phase."""
from __future__ import annotations

import concurrent.futures
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import run_real_dataset_confirmation as runner
from probe_jacobian_bound import ProbeConfig
from real_dataset_mlp import RealMLPConfig


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        run_root = base / "run"
        export_root = base / "export"
        method_seal = base / "method_seal.json"
        candidate_seal = base / "candidate_seal.json"
        certificate_seal = base / "certificate_seal.json"
        config = RealMLPConfig(
            **{
                **asdict(runner.BASE_CONFIG),
                "steps": 400,
                "threads": 1,
            }
        )
        with (
            mock.patch.object(runner, "FRESH_SEEDS", (0,)),
            mock.patch.object(runner, "THRESHOLDS", (0.925,)),
            mock.patch.object(runner, "MAXIMUM_OPERATORS", 1),
            mock.patch.object(
                runner,
                "PROBE",
                ProbeConfig(probes=8, power=4, delta=1e-6),
            ),
            mock.patch.object(runner, "BASE_CONFIG", config),
            mock.patch.object(runner, "METHOD_SEAL_PATH", method_seal),
            mock.patch.object(runner, "CANDIDATE_SEAL_PATH", candidate_seal),
            mock.patch.object(runner, "CERTIFICATE_SEAL_PATH", certificate_seal),
            mock.patch.object(runner, "EXPORT_ROOT", export_root),
            mock.patch.object(
                runner.concurrent.futures,
                "ProcessPoolExecutor",
                concurrent.futures.ThreadPoolExecutor,
            ),
        ):
            runner.phase_seal(run_root)
            runner.phase_train(run_root, workers=1)
            assert not (run_root / "outcomes").exists()
            runner.phase_select(run_root)
            assert not (run_root / "outcomes").exists()
            runner.phase_certify(run_root, workers=1)
            assert not (run_root / "audit").exists()
            runner.phase_join(run_root)
            runner.phase_export(run_root)
            summary = runner.read_json(run_root / "final_summary.json")
            assert summary["seed_threshold_cases"] == 1
            assert summary["candidates"] == 1
            assert summary["issued"] == 1
            assert summary["covered"] == 1
            assert summary["maximum_checkpoint_reconstruction_error"] == 0.0
            assert (export_root / "final_audit.json").exists()
    print("PASS burned-seed end-to-end sealed confirmation dry run")


if __name__ == "__main__":
    main()
