#!/usr/bin/env python3
"""CPU scaling and certificate-cost benchmark for the matrix-free Transformer.

The benchmark widens the exact smooth one-block architecture used by the paper
to approximately 100k and 1M parameters.  It measures the primitives used by
the certificate (gradient, objective HVP, and output-Jacobian Gram product) and
then reports a transparent operation-count projection for H=300.  It is a
scaling study, not a claim that a million-parameter certificate was issued.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from probe_jacobian_bound import make_gram_operator
from transformer_hvp_grokking import (
    TransformerConfig,
    flatten_parameters,
    flat_spec,
    gradient,
    make_disjoint_split,
    make_template,
    objective,
    objective_hvp,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "transformer_scaling_benchmark.json"
OUTPUT_MD = RESULTS / "transformer_scaling_benchmark.md"
HORIZON = 300
PROBES = 16
POWER = 8

PROFILES = {
    "paper": {"model_dim": 32, "hidden_dim": 128, "heads": 4, "repeats": 3},
    "100k": {"model_dim": 96, "hidden_dim": 384, "heads": 4, "repeats": 2},
    "1m": {"model_dim": 288, "hidden_dim": 1152, "heads": 4, "repeats": 1},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def profile_path(name: str) -> Path:
    return RESULTS / f"transformer_scaling_benchmark_{name}.json"


def peak_rss_bytes() -> int:
    if os.name == "nt":
        class ProcessMemoryCountersEx(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCountersEx),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        process = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        )
        if not ok:
            raise ctypes.WinError()
        return int(counters.PeakWorkingSetSize)
    # Linux fallback for portability; ru_maxrss is KiB on Linux.
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def timed(callable_, repeats: int, *, warmup: bool) -> tuple[float, list[float], object]:
    if warmup:
        value = callable_()
        del value
        gc.collect()
    rows = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = callable_()
        rows.append(time.perf_counter() - started)
    return statistics.median(rows), rows, result


def run_profile(name: str) -> dict:
    if name not in PROFILES:
        raise ValueError(name)
    profile = PROFILES[name]
    repeats = int(profile["repeats"])
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
    started = time.perf_counter()
    template = make_template(config)
    spec = flat_spec(template)
    parameter = flatten_parameters(template)
    train_pairs, train_labels, _, _, _, _ = make_disjoint_split(config)
    all_pairs = torch.cartesian_prod(
        torch.arange(config.modulus), torch.arange(config.modulus)
    ).long()
    vector = torch.randn_like(parameter)
    vector /= torch.linalg.vector_norm(vector)

    forward_median, forward_rows, forward_value = timed(
        lambda: objective(
            parameter, train_pairs, train_labels, template, spec, config
        ).detach(),
        repeats,
        warmup=True,
    )
    gradient_median, gradient_rows, gradient_value = timed(
        lambda: gradient(
            parameter, train_pairs, train_labels, template, spec, config
        ),
        repeats,
        warmup=True,
    )
    hvp_median, hvp_rows, hvp_value = timed(
        lambda: objective_hvp(
            parameter,
            vector,
            train_pairs,
            train_labels,
            template,
            spec,
            config,
        ),
        repeats,
        warmup=(name != "1m"),
    )
    output_gram, output_meta = make_gram_operator(parameter, all_pairs, template, spec)
    output_median, output_rows, output_value = timed(
        lambda: output_gram(vector),
        repeats,
        warmup=(name == "paper"),
    )

    for value in (forward_value, gradient_value, hvp_value, output_value):
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"non-finite benchmark output in profile {name}")
    parameter_count = int(parameter.numel())
    state_count = 2 * parameter_count
    peak = peak_rss_bytes()
    result = {
        "status": "complete matrix-free primitive benchmark",
        "profile": name,
        "config": config.__dict__,
        "parameter_count": parameter_count,
        "optimizer_state_dimension": state_count,
        "train_examples": int(len(train_pairs)),
        "output_examples": int(len(all_pairs)),
        "output_coordinates": int(output_meta["output_dimension"]),
        "repeats": repeats,
        "timings_seconds": {
            "objective_forward_median": forward_median,
            "objective_forward_samples": forward_rows,
            "gradient_median": gradient_median,
            "gradient_samples": gradient_rows,
            "objective_hvp_median": hvp_median,
            "objective_hvp_samples": hvp_rows,
            "output_jacobian_gram_median": output_median,
            "output_jacobian_gram_samples": output_rows,
        },
        "observed_process_peak_rss_bytes": peak,
        "observed_process_peak_rss_gib": peak / 2**30,
        "total_elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "dtype": "float64",
            "device": "cpu",
        },
    }
    path = profile_path(name)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(path),
                "parameters": parameter_count,
                "hvp_seconds": hvp_median,
                "output_gram_seconds": output_median,
                "peak_gib": result["observed_process_peak_rss_gib"],
            },
            indent=2,
        ),
        flush=True,
    )
    return result


def existing_costs() -> dict:
    certificates = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(
            RESULTS.glob("transformer_green_confirmation_certificate_seed_*.json")
        )
    ]
    elapsed = [
        float(row["elapsed_seconds"])
        for row in certificates
        if row.get("elapsed_seconds") is not None
    ]
    issued_elapsed = [
        float(row["elapsed_seconds"])
        for row in certificates
        if row.get("elapsed_seconds") is not None and row["certificate_issued"]
    ]
    method_seal = json.loads(
        (ROOT / "TRANSFORMER_GREEN_CONFIRMATION_METHOD_SEAL.json").read_text(
            encoding="utf-8"
        )
    )
    fresh_seeds = [int(seed) for seed in method_seal["fresh_seeds"]]
    training_paths = [
        RESULTS / f"transformer_hvp_prospective_seed_{seed}.json"
        for seed in fresh_seeds
    ]
    training = [
        json.loads(path.read_text(encoding="utf-8")) for path in training_paths
    ]
    training_seconds_per_step = [
        float(row["summary"]["elapsed_seconds"]) / int(row["config"]["steps"])
        for row in training
    ]
    continuation = [HORIZON * value for value in training_seconds_per_step]
    return {
        "candidate_certificates_with_runtime": len(elapsed),
        "aggregate_candidate_hours": sum(elapsed) / 3600.0,
        "median_candidate_minutes": statistics.median(elapsed) / 60.0,
        "median_issued_candidate_minutes": statistics.median(issued_elapsed) / 60.0,
        "training_runs": len(training),
        "median_measured_300_step_continuation_seconds": statistics.median(continuation),
        "certificate_to_300_step_continuation_ratio": statistics.median(elapsed)
        / statistics.median(continuation),
    }


def projected_cost(profile: dict) -> dict:
    timings = profile["timings_seconds"]
    hvp = float(timings["objective_hvp_median"])
    output = float(timings["output_jacobian_gram_median"])
    gradient_time = float(timings["gradient_median"])
    center_hvp_calls = 5 * HORIZON
    green_hvp_calls = 2 * PROBES * POWER * HORIZON
    output_gram_calls = PROBES * POWER * (HORIZON + 1)
    return {
        "horizon": HORIZON,
        "centerline_objective_hvp_calls": center_hvp_calls,
        "green_probe_objective_hvp_calls": green_hvp_calls,
        "output_probe_gram_calls": output_gram_calls,
        "projected_centerline_seconds": center_hvp_calls * hvp,
        "projected_green_probe_seconds": green_hvp_calls * hvp,
        "projected_output_probe_seconds": output_gram_calls * output,
        "projected_core_certificate_seconds": (
            (center_hvp_calls + green_hvp_calls) * hvp
            + output_gram_calls * output
        ),
        "projected_300_step_training_seconds": HORIZON * gradient_time,
        "projected_core_certificate_to_training_ratio": (
            (center_hvp_calls + green_hvp_calls) * hvp
            + output_gram_calls * output
        )
        / (HORIZON * gradient_time),
        "excluded_from_projection": (
            "analytic derivative envelopes, Python orchestration, checkpoint I/O, "
            "and cache effects; operation-count projection only"
        ),
    }


def aggregate() -> dict:
    profiles = []
    for name in PROFILES:
        path = profile_path(name)
        if not path.exists():
            raise FileNotFoundError(path)
        profile = json.loads(path.read_text(encoding="utf-8"))
        profile["projection_h300"] = projected_cost(profile)
        profiles.append(profile)

    paper = profiles[0]
    largest = profiles[-1]
    parameter_ratio = largest["parameter_count"] / paper["parameter_count"]
    hvp_ratio = (
        largest["timings_seconds"]["objective_hvp_median"]
        / paper["timings_seconds"]["objective_hvp_median"]
    )
    output_ratio = (
        largest["timings_seconds"]["output_jacobian_gram_median"]
        / paper["timings_seconds"]["output_jacobian_gram_median"]
    )
    scaling = {
        "paper_to_largest_parameter_ratio": parameter_ratio,
        "paper_to_largest_hvp_runtime_ratio": hvp_ratio,
        "paper_to_largest_output_gram_runtime_ratio": output_ratio,
        "two_point_hvp_parameter_exponent": math.log(hvp_ratio) / math.log(parameter_ratio),
        "two_point_output_gram_parameter_exponent": math.log(output_ratio)
        / math.log(parameter_ratio),
        "largest_profile_completes_matrix_free_hvp": True,
        "largest_profile_completes_output_gram_product": True,
    }
    output = {
        "status": "complete Transformer matrix-free scaling and cost audit",
        "interpretation": (
            "The matrix-free primitives execute at approximately one million parameters, "
            "but the frozen m=16, q=8, H=300 verification workload is orders of magnitude "
            "more expensive than continuing training. This supports scalability of memory "
            "representation, not current end-to-end computational practicality."
        ),
        "benchmark_source_sha256": sha256(Path(__file__)),
        "profile_source_hashes": {
            name: sha256(profile_path(name)) for name in PROFILES
        },
        "existing_paper_scale_cost": existing_costs(),
        "scaling": scaling,
        "profiles": profiles,
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Transformer matrix-free scaling and certificate cost",
        "",
        "All profiles use the paper's smooth one-block, no-normalization Transformer",
        "with float64 CPU arithmetic; only width changes.",
        "",
        "| profile | parameters | HVP (s) | output Gram (s) | observed peak RSS (GiB) | projected H=300 core cert. | projected 300-step train |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in profiles:
        timing = profile["timings_seconds"]
        projection = profile["projection_h300"]
        lines.append(
            f"| {profile['profile']} | {profile['parameter_count']:,} | "
            f"{timing['objective_hvp_median']:.4f} | "
            f"{timing['output_jacobian_gram_median']:.4f} | "
            f"{profile['observed_process_peak_rss_gib']:.2f} | "
            f"{projection['projected_core_certificate_seconds']/3600:.2f} h | "
            f"{projection['projected_300_step_training_seconds']:.2f} s |"
        )
    costs = output["existing_paper_scale_cost"]
    lines.extend(
        [
            "",
            "Measured on the frozen paper batch, candidate construction consumed",
            f"**{costs['aggregate_candidate_hours']:.2f} aggregate hours**, with a median",
            f"of **{costs['median_candidate_minutes']:.2f} minutes per constructed candidate**.",
            f"A measured 300-step continuation took a median **{costs['median_measured_300_step_continuation_seconds']:.2f} seconds**.",
            "",
            "The benchmark establishes that the method's matrix-free HVP and output-Gram",
            "primitives do not require dense Hessians at roughly one million parameters.",
            "It simultaneously makes the current limitation explicit: the number of",
            "probabilistic operator applications, not dense storage, dominates runtime.",
            "The projection omits analytic-envelope and orchestration overhead and should",
            "therefore be read as an operation-count estimate, not a measured full certificate.",
            "",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "scaling": scaling}, indent=2))
    return output


def run_all() -> None:
    for name in PROFILES:
        command = [sys.executable, str(Path(__file__)), "--profile", name]
        subprocess.run(command, cwd=ROOT, check=True)
    aggregate()


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
        run_all()
    else:
        aggregate()


if __name__ == "__main__":
    main()
