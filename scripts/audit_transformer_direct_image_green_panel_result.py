#!/usr/bin/env python3
"""Independent scalar/cache audit of the staged direct-image panel."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist, median


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "transformer_direct_image_green_panel_audit.json"
PANEL = ROOT / "results" / "transformer_v3_relinearized_prefix_panel_audit.json"
OUTPUT = ROOT / "results" / "transformer_direct_image_green_panel_independent_audit.json"
CACHE = ROOT / "results" / "transformer_direct_image_green_panel_cache"
EXPECTED_RESULT = "931CBF5750510C49DEB92F16F77E8CCA355C7969A18BCF4EFA1A0701335ED705"
EXPECTED_PANEL = "08E501B51FEAC3D96FFE02BE0B5D84E0E682C2E73CB906C083ED0FEF7E75E12B"
EXPECTED_PROTOCOL = "11813379A4B388EB71C409BE4709EB366E6E5BE8BF53C27AA5334802226EC428"
STAGE_DELTA = 1.0e-6 / 45.0
NORMAL = NormalDist()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def close(left: float, right: float, tolerance: float = 3.0e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=1e-300)


def calibration(prefix: int) -> float:
    return NORMAL.inv_cdf(0.5 * (1.0 + STAGE_DELTA ** (1.0 / prefix)))


def stable_closure(kappa: float, drift: float, forcing: float) -> tuple[float, float | None]:
    coefficient = kappa * drift
    discriminant = 1.0 - 2.0 * coefficient * forcing
    if discriminant < 0.0:
        return discriminant, None
    if forcing == 0.0:
        return discriminant, 0.0
    if coefficient == 0.0:
        return discriminant, forcing
    return discriminant, 2.0 * forcing / (1.0 + math.sqrt(discriminant))


def cache_path(candidate: dict) -> Path:
    gate = (0.7, 0.8, 0.9).index(round(float(candidate["threshold"]), 2))
    return CACHE / (
        f"seed_{candidate['seed']}_gate_{gate}_anchor_{candidate['anchor']}_v1.json"
    )


def main() -> None:
    require(sha256(RESULT) == EXPECTED_RESULT, "direct-image result changed")
    require(sha256(PANEL) == EXPECTED_PANEL, "prefix panel changed")
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    panel = json.loads(PANEL.read_text(encoding="utf-8"))
    require(result["protocol_sha256"] == EXPECTED_PROTOCOL, "protocol hash changed")
    for relative, digest in result["dependency_sha256"].items():
        require(sha256(ROOT / relative) == digest, f"dependency changed: {relative}")
    panel_index = {
        (
            row["candidate"]["seed"],
            row["candidate"]["threshold"],
            row["candidate"]["anchor"],
        ): row
        for row in panel["rows"]
    }

    routes = {"direct_image": 0, "gram_fallback": 0}
    prefixes = {4: 0, 8: 0, 16: 0}
    old_sweeps = 0
    new_sweeps = 0
    transposes_avoided = 0
    reductions = []
    all_hashes = set()
    minimum_logic = math.inf
    for row in result["rows"]:
        candidate = row["candidate"]
        key = (candidate["seed"], candidate["threshold"], candidate["anchor"])
        require(key in panel_index, f"candidate outside panel: {key}")
        parent = panel_index[key]
        cached = json.loads(cache_path(candidate).read_text(encoding="utf-8"))
        require(cached == row, f"cache/result mismatch: {key}")
        require(row["panel_result_sha256"] == EXPECTED_PANEL, f"parent hash: {key}")
        require(row["protocol_sha256"] == EXPECTED_PROTOCOL, f"protocol: {key}")
        require(row["bracket"] == parent["bracket"], f"bracket: {key}")
        require(int(row["outcome_files_read"]) == 0, f"outcome read: {key}")
        require(
            row["probe_hashes"] == parent["probe_hashes"][: row["prefix"]],
            f"probe stream: {key}",
        )
        for digest in row["probe_hashes"]:
            require(digest not in all_hashes, f"probe hash collision: {key}")
            all_hashes.add(digest)

        for stage_index, stage in enumerate(row["stage_rows"]):
            prefix = int(stage["prefix"])
            require(prefix == (4, 8, 16)[stage_index], f"prefix order: {key}")
            cd = calibration(prefix)
            direct = stage["direct"]
            y_direct = max(row["direct_image_norms"][:prefix])
            direct_kappa = y_direct / cd
            require(close(direct["Y_direct"], y_direct), f"direct Y: {key}")
            require(close(direct["c_delta"], cd), f"direct c: {key}")
            require(
                close(direct["operator_norm_upper_bound"], direct_kappa),
                f"direct kappa: {key}",
            )
            direct_forcing = direct_kappa * float(parent["total_corrected_injection_upper"])
            require(
                close(direct["forcing_response_upper"], direct_forcing),
                f"direct forcing: {key}",
            )
            discriminant, radius = stable_closure(
                direct_kappa,
                float(parent["derivative_drift_upper"]),
                direct_forcing,
            )
            require(
                close(direct["closure"]["discriminant"], discriminant),
                f"direct discriminant: {key}",
            )
            if radius is None:
                require(
                    direct["closure"]["remainder_radius"] is None,
                    f"direct failed radius: {key}",
                )
            else:
                require(
                    close(direct["closure"]["remainder_radius"], radius),
                    f"direct radius: {key}",
                )

            gram = stage["gram"]
            if direct["issued"]:
                require(gram is None, f"transpose computed after direct issue: {key}")
                require(stage_index + 1 == len(row["stage_rows"]), f"continued after direct: {key}")
            else:
                require(gram is not None, f"missing Gram fallback: {key}")
                y_gram = max(row["gram_norms_computed"][:prefix])
                gram_kappa = math.sqrt(y_gram / cd)
                require(close(gram["Y"], y_gram), f"Gram Y: {key}")
                require(
                    close(gram["operator_norm_upper_bound"], gram_kappa),
                    f"Gram kappa: {key}",
                )
                require(
                    row["gram_norms_computed"][:prefix]
                    == parent["final_probe_norms"][:prefix],
                    f"stored Gram norm replay: {key}",
                )
                gram_forcing = gram_kappa * float(parent["total_corrected_injection_upper"])
                require(
                    close(gram["forcing_response_upper"], gram_forcing),
                    f"Gram forcing: {key}",
                )
                gram_discriminant, gram_radius = stable_closure(
                    gram_kappa,
                    float(parent["derivative_drift_upper"]),
                    gram_forcing,
                )
                require(
                    close(gram["closure"]["discriminant"], gram_discriminant),
                    f"Gram discriminant: {key}",
                )
                if gram_radius is None:
                    require(gram["closure"]["remainder_radius"] is None, f"Gram failed radius: {key}")
                else:
                    require(
                        close(gram["closure"]["remainder_radius"], gram_radius),
                        f"Gram radius: {key}",
                    )
                if stage_index + 1 < len(row["stage_rows"]):
                    require(not gram["issued"], f"continued after Gram issue: {key}")

        final = row["stage_rows"][-1]
        selected = final["direct"] if row["route"] == "direct_image" else final["gram"]
        require(selected["issued"], f"selected route did not issue: {key}")
        require(selected["bracket"] == row["bracket"], f"selected bracket: {key}")
        require(int(row["prefix"]) == int(final["prefix"]), f"stopping prefix: {key}")
        require(
            int(row["logical_forward_green_applications"]) == int(row["prefix"]),
            f"forward count: {key}",
        )
        expected_transpose = 0 if row["route"] == "direct_image" else int(row["prefix"])
        require(
            int(row["logical_transpose_green_applications"]) == expected_transpose,
            f"transpose count: {key}",
        )
        routes[row["route"]] += 1
        prefixes[int(row["prefix"])] += 1
        old_sweeps += int(row["panel_logical_green_sweeps"])
        new_sweeps += int(row["logical_total_green_sweeps"])
        transposes_avoided += int(row["transpose_sweeps_avoided"])
        reductions.append(float(row["probe_sweep_reduction"]))
        minimum_logic = min(minimum_logic, float(selected["logic_slack"]))

    require(len(result["rows"]) == len(panel_index) == 15, "candidate count changed")
    require(routes == {"direct_image": 4, "gram_fallback": 11}, "route count changed")
    require(prefixes == {4: 14, 8: 1, 16: 0}, "prefix count changed")
    require(len(all_hashes) == 64, "probe hash family changed")
    require(old_sweeps == 128 and new_sweeps == 112, "sweep totals changed")
    require(transposes_avoided == 16, "transpose saving changed")
    require(result["outcome_files_read"] == 0, "aggregate outcome read")

    audit = {
        "status": "INDEPENDENT DIRECT-IMAGE/GREEN PANEL AUDIT PASSED",
        "result_sha256": EXPECTED_RESULT,
        "cases": 15,
        "issued": 15,
        "route_distribution": routes,
        "prefix_distribution": {str(key): value for key, value in prefixes.items()},
        "unique_probe_hashes": len(all_hashes),
        "panel_green_probe_sweeps": old_sweeps,
        "staged_green_probe_sweeps": new_sweeps,
        "aggregate_probe_sweep_reduction": old_sweeps / new_sweeps,
        "median_pairwise_probe_sweep_reduction": median(reductions),
        "transpose_sweeps_avoided": transposes_avoided,
        "minimum_logic_slack": minimum_logic,
        "combined_family_failure_upper": 2.0e-6,
        "outcome_files_read": 0,
    }
    OUTPUT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
