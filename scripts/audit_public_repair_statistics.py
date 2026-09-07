"""Reconstruct the arithmetic appendix statistics using only public inputs."""
import argparse
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from read_public_repair_archive import ROOT, MANIFEST_SHA, payload

DESCRIPTORS = "results/public_repair_descriptors_20260907.json"
DESCRIPTORS_SHA = "5215b2046123e348b81b4a7d6a34561ca81963a165fb7f9639497417dab6cbb1"
ORIGINAL = "results/repaired_numerics_manuscript_summary_20260907.json"
ORIGINAL_SHA = "697f36ad8503486d8f349c98c7d3d7b51e7e129029b74c5e4ab4bbe07f4a5395"


def read_checked(name, expected):
    raw = (ROOT/name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("descriptive evidence changed")
    return json.loads(raw)


def audit():
    manifest, files = payload()
    descriptors = read_checked(DESCRIPTORS, DESCRIPTORS_SHA)
    original = read_checked(ORIGINAL, ORIGINAL_SHA)
    if descriptors["manifest_sha256"] != MANIFEST_SHA or len(descriptors["rows"]) != 63:
        raise ValueError("descriptor manifest/population differs")
    groups = {}
    for job, descriptor in zip(manifest["jobs"], descriptors["rows"]):
        if (job["ordinal"], job["id"], job["study"], job["original_summary_sha256"]) != (
                descriptor["ordinal"], descriptor["job_id"], descriptor["study"], descriptor["original_summary_sha256"]):
            raise ValueError("descriptor/job correspondence differs")
        if not (math.isfinite(descriptor["elapsed_seconds"]) and descriptor["elapsed_seconds"] >= 0):
            raise ValueError("invalid descriptive time")
        evidence = json.loads(files[job["evidence"]])
        with np.load(io.BytesIO(files[job["enclosures"]]), allow_pickle=False) as saved:
            radius = saved["radius"].copy()
        if radius.dtype != np.float64 or radius.shape != (job["horizon"]+1,) or not np.isfinite(radius).all() or radius[0] != 0:
            raise ValueError("invalid public radius path")
        maximum_radius = float(np.max(radius))
        if maximum_radius != descriptor["maximum_radius"]:
            raise ValueError("supplied maximum radius not reproduced")
        diagnostics = evidence["diagnostics"]
        if len(diagnostics) != job["horizon"]:
            raise ValueError("diagnostic horizon differs")
        recurrence_slacks = []
        key = "lipschitz_upper" if job["backend"] == "mse" else "optimizer_jacobian_lipschitz_upper"
        for index, row in enumerate(diagnostics):
            r, following = Fraction(float(radius[index])), Fraction(float(radius[index+1]))
            if row["step"] != index or following != Fraction(row["next_radius"]):
                raise ValueError("scalar row identity differs")
            need = Fraction(row["beta_upper"])*r + Fraction(row["defect_norm_upper"]) + Fraction(row[key])*r*r/2
            if following < need:
                raise ValueError("inward scalar recurrence")
            recurrence_slacks.append(following-need)
        groups.setdefault(job["study"], []).append((job, descriptor, diagnostics,
            maximum_radius, min(recurrence_slacks), sum(row["margins"] for row in evidence["independent_counts"])))
    result = {}
    for study, rows in sorted(groups.items()):
        diagnostics = [d for _, _, ds, _, _, _ in rows for d in ds]
        platforms = {}
        for _, descriptor, _, _, _, _ in rows:
            group = platforms.setdefault(descriptor["platform"], {"jobs": 0, "summed_seconds": 0.})
            group["jobs"] += 1
            group["summed_seconds"] += descriptor["elapsed_seconds"]
        slack = min(value for _, _, _, _, value, _ in rows)
        result[study] = {
            "unique_jobs": len(rows),
            "historical_events_retained": sum(len(job["events"]) for job, *_ in rows),
            "unique_transitions": sum(job["horizon"] for job, *_ in rows),
            "independent_output_margins": sum(margins for *_, margins in rows),
            "maximum_radius": max(value for _, _, _, value, _, _ in rows),
            "minimum_logic_slack": None if study == "modular_mse" else min(d["minimum_logic_slack"] for _, d, *_ in rows),
            "maximum_optimizer_norm_upper": max(d["optimizer_norm_upper"] for d in diagnostics),
            "maximum_basis_orthogonality_upper": max(d["basis_orthogonality_frobenius_upper"] for d in diagnostics),
            "maximum_hessian_basis_residual_upper": max(d["hessian_basis_residual_frobenius_upper"] for d in diagnostics),
            "maximum_optimizer_norm_inflation_descriptive_float64": max(d["optimizer_norm_upper"]-d["diagonal_optimizer_norm_upper"] for d in diagnostics),
            "minimum_exact_recurrence_slack": str(slack),
            "minimum_recurrence_slack_descriptive_float64": float(slack),
            "recorded_job_cost_by_platform_not_a_controlled_comparison": platforms,
        }
    if result != original["studies"]:
        raise ValueError("public reconstruction differs from the original descriptive summary")
    return {"status": "PASS", "all_study_statistics_equal": True,
            "studies": result, "manifest_sha256": MANIFEST_SHA,
            "descriptor_sha256": DESCRIPTORS_SHA, "original_summary_sha256": ORIGINAL_SHA,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "new_prospective_events": 0, "neural_kernels_recomputed": False,
            "scope": "Public-array/scalar reconstruction of manuscript statistics; original per-job timings and strict-output-slack descriptors are authenticated inputs, not new timing or output experiments."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = audit()
    if args.report:
        with args.report.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    print(json.dumps(result, indent=2))
