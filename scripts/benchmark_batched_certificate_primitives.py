#!/usr/bin/env python3
"""Benchmark block-batched GreenCert probe primitives on CPU.

The frozen verifier uses 16 independent Gaussian directions.  This post-seal
benchmark compares evaluating those directions serially with evaluating the
identical block through ``is_grads_batched=True``.  The probability theorem,
probe count, power count, and model are unchanged.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from batched_green_operator import (
    make_batched_output_gram_operator,
    objective_hvp_batch,
    relative_error,
)
from benchmark_transformer_scaling import peak_rss_bytes
from probe_jacobian_bound import make_gram_operator
from transformer_hvp_grokking import (
    TransformerConfig,
    flat_spec,
    flatten_parameters,
    make_disjoint_split,
    make_template,
    objective_hvp,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "transformer_batched_scaling_benchmark.json"
OUTPUT_MD = RESULTS / "transformer_batched_scaling_benchmark.md"
HORIZON = 300
PROBES = 16
POWER = 8
PROFILES = {
    "paper": {"model_dim": 32, "hidden_dim": 128, "heads": 4, "repeats": 2},
    "100k": {"model_dim": 96, "hidden_dim": 384, "heads": 4, "repeats": 1},
    "1m": {"model_dim": 288, "hidden_dim": 1152, "heads": 4, "repeats": 1},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def profile_path(name: str) -> Path:
    return RESULTS / f"transformer_batched_scaling_benchmark_{name}.json"


def timed(callable_, repeats: int) -> tuple[float, list[float], torch.Tensor]:
    rows = []
    result = None
    for _ in range(repeats):
        gc.collect()
        started = time.perf_counter()
        result = callable_()
        rows.append(time.perf_counter() - started)
    assert result is not None
    return statistics.median(rows), rows, result


def run_profile(name: str) -> dict:
    profile = PROFILES[name]
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20260824)
    config = TransformerConfig(
        modulus=17,
        model_dim=int(profile["model_dim"]),
        hidden_dim=int(profile["hidden_dim"]),
        heads=int(profile["heads"]),
        depth=1,
        train_fraction=0.60,
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=0.01,
        steps=1,
        seed=20260824,
        threads=4,
        dtype="float64",
        loss="cross_entropy",
        normalization="none",
    )
    template = make_template(config)
    spec = flat_spec(template)
    parameter = flatten_parameters(template)
    train_pairs, train_labels, _, _, _, _ = make_disjoint_split(config)
    all_pairs = torch.cartesian_prod(
        torch.arange(config.modulus), torch.arange(config.modulus)
    ).long()
    generator = torch.Generator().manual_seed(20260825)
    vectors = torch.randn(
        PROBES,
        parameter.numel(),
        generator=generator,
        dtype=parameter.dtype,
    )
    scalar_output, _ = make_gram_operator(parameter, all_pairs, template, spec)
    batched_output = make_batched_output_gram_operator(
        parameter, all_pairs, template, spec
    )

    def scalar_hvp_block() -> torch.Tensor:
        return torch.stack(
            [
                objective_hvp(
                    parameter,
                    vector,
                    train_pairs,
                    train_labels,
                    template,
                    spec,
                    config,
                )
                for vector in vectors
            ]
        )

    def batched_hvp_block() -> torch.Tensor:
        return objective_hvp_batch(
            parameter,
            vectors,
            train_pairs,
            train_labels,
            template,
            spec,
            config,
        )

    def scalar_output_block() -> torch.Tensor:
        return torch.stack([scalar_output(vector) for vector in vectors])

    def batched_output_block() -> torch.Tensor:
        return batched_output(vectors)

    repeats = int(profile["repeats"])
    scalar_hvp_time, scalar_hvp_rows, scalar_hvp = timed(scalar_hvp_block, repeats)
    batched_hvp_time, batched_hvp_rows, batched_hvp = timed(batched_hvp_block, repeats)
    scalar_output_time, scalar_output_rows, scalar_out = timed(scalar_output_block, repeats)
    batched_output_time, batched_output_rows, batched_out = timed(
        batched_output_block, repeats
    )
    hvp_error = relative_error(batched_hvp, scalar_hvp)
    output_error = relative_error(batched_out, scalar_out)
    if hvp_error >= 5e-11 or output_error >= 5e-11:
        raise RuntimeError(
            f"batched products failed equivalence: HVP={hvp_error}, output={output_error}"
        )
    result = {
        "status": "complete post-seal batched-probe primitive benchmark",
        "profile": name,
        "config": config.__dict__,
        "parameter_count": int(parameter.numel()),
        "probe_block_size": PROBES,
        "repeats": repeats,
        "timings_seconds": {
            "serial_16_hvp_median": scalar_hvp_time,
            "serial_16_hvp_samples": scalar_hvp_rows,
            "batched_16_hvp_median": batched_hvp_time,
            "batched_16_hvp_samples": batched_hvp_rows,
            "serial_16_output_gram_median": scalar_output_time,
            "serial_16_output_gram_samples": scalar_output_rows,
            "batched_16_output_gram_median": batched_output_time,
            "batched_16_output_gram_samples": batched_output_rows,
        },
        "speedups": {
            "hvp_probe_block": scalar_hvp_time / batched_hvp_time,
            "output_gram_probe_block": scalar_output_time / batched_output_time,
        },
        "equivalence": {
            "hvp_relative_error": hvp_error,
            "output_gram_relative_error": output_error,
        },
        "observed_process_peak_rss_bytes": peak_rss_bytes(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "logical_cpu_count": __import__("os").cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "dtype": "float64",
            "device": "cpu",
        },
    }
    profile_path(name).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(profile_path(name)),
        "parameters": result["parameter_count"],
        "hvp_speedup": result["speedups"]["hvp_probe_block"],
        "output_speedup": result["speedups"]["output_gram_probe_block"],
        "peak_gib": result["observed_process_peak_rss_bytes"] / 2**30,
    }, indent=2), flush=True)
    return result


def projection(profile: dict, serial_profile: dict) -> dict:
    timing = profile["timings_seconds"]
    serial_timing = serial_profile["timings_seconds"]
    # One batched block replaces 16 serial directions.  K^T K requires two
    # optimizer HVP blocks per power and time step; output J^T J requires one.
    centerline = 5 * HORIZON * float(serial_timing["objective_hvp_median"])
    green = 2 * POWER * HORIZON * float(timing["batched_16_hvp_median"])
    output = POWER * (HORIZON + 1) * float(
        timing["batched_16_output_gram_median"]
    )
    matched_serial_green = 2 * POWER * HORIZON * float(
        timing["serial_16_hvp_median"]
    )
    matched_serial_output = POWER * (HORIZON + 1) * float(
        timing["serial_16_output_gram_median"]
    )
    matched_serial_core = centerline + matched_serial_green + matched_serial_output
    legacy_serial_core = float(
        serial_profile["projection_h300"]["projected_core_certificate_seconds"]
    )
    batched_core = centerline + green + output
    return {
        "horizon": HORIZON,
        "probe_count": PROBES,
        "power_iterations": POWER,
        "projected_centerline_seconds": centerline,
        "projected_batched_green_seconds": green,
        "projected_batched_output_seconds": output,
        "projected_batched_core_seconds": batched_core,
        "matched_serial_green_seconds": matched_serial_green,
        "matched_serial_output_seconds": matched_serial_output,
        "matched_serial_core_seconds": matched_serial_core,
        "matched_projected_core_speedup": matched_serial_core / batched_core,
        "legacy_isolated_primitive_projection_seconds": legacy_serial_core,
        "legacy_projection_to_batched_ratio": legacy_serial_core / batched_core,
        "excluded": "analytic envelopes, Python orchestration, checkpoint I/O, and cache effects",
    }


def aggregate() -> dict:
    old = json.loads(
        (RESULTS / "transformer_scaling_benchmark.json").read_text(encoding="utf-8")
    )
    old_profiles = {row["profile"]: row for row in old["profiles"]}
    profiles = []
    for name in PROFILES:
        row = json.loads(profile_path(name).read_text(encoding="utf-8"))
        row["projection_h300"] = projection(row, old_profiles[name])
        profiles.append(row)
    output = {
        "status": "complete post-seal block-batched scaling audit",
        "interpretation": (
            "All 16 committed Gaussian directions are unchanged; vectorizing their "
            "reverse products reduces wall time without weakening the probability "
            "budget or altering the certificate theorem."
        ),
        "benchmark_source_sha256": sha256(Path(__file__)),
        "implementation_source_sha256": sha256(
            ROOT / "scripts" / "batched_green_operator.py"
        ),
        "profiles": profiles,
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Block-batched GreenCert probe benchmark",
        "",
        "The same 16 Gaussian directions and the same m=16, q=8 theorem are",
        "evaluated as one reverse-mode block rather than 16 serial calls.",
        "",
        "| profile | parameters | HVP block speedup | output-Gram block speedup | matched serial H=300 | batched H=300 | matched speedup | peak RSS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in profiles:
        proj = row["projection_h300"]
        lines.append(
            f"| {row['profile']} | {row['parameter_count']:,} | "
            f"{row['speedups']['hvp_probe_block']:.2f}x | "
            f"{row['speedups']['output_gram_probe_block']:.2f}x | "
            f"{proj['matched_serial_core_seconds']/3600:.2f} h | "
            f"{proj['projected_batched_core_seconds']/3600:.2f} h | "
            f"{proj['matched_projected_core_speedup']:.2f}x | "
            f"{row['observed_process_peak_rss_bytes']/2**30:.2f} GiB |"
        )
    lines.extend([
        "",
        "The projection retains all 16 probes and eight powers. It is an operation-",
        "matched serial-versus-batched wall-clock estimate, not a measured end-to-end",
        "certificate. The older isolated-primitive projection is retained in JSON",
        "for traceability but is not used as the acceleration denominator. The table",
        "excludes analytic envelopes, orchestration, checkpoint I/O, and cache effects.",
        "",
    ])
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT)}, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(PROFILES))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if sum((args.profile is not None, args.all, args.aggregate)) != 1:
        parser.error("choose exactly one of --profile, --all, or --aggregate")
    if args.profile:
        run_profile(args.profile)
    elif args.all:
        for name in PROFILES:
            subprocess.run(
                [sys.executable, str(Path(__file__)), "--profile", name],
                cwd=ROOT,
                check=True,
            )
        aggregate()
    else:
        aggregate()


if __name__ == "__main__":
    main()
