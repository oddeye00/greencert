#!/usr/bin/env python3
"""Independent scalar/RNG/cache audit of the frozen prefix-panel result."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist, median

import torch


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "transformer_v3_relinearized_prefix_panel_audit.json"
OUTPUT = (
    ROOT
    / "results"
    / "transformer_v3_relinearized_prefix_panel_independent_audit.json"
)
PROTOCOL = ROOT / "RELINEARIZED_PREFIX_PANEL_PROTOCOL.md"
TWO_RESPONSE = ROOT / "results" / "transformer_v3_two_response_postseal_audit.json"
CACHE = ROOT / "results" / "transformer_v3_relinearized_prefix_panel_cache"

EXPECTED_PROTOCOL = "6740E5D32B5A5841E81AE0B25F17FDE316322CA4B279F9A2A1EB8F0C55BE1358"
EXPECTED_SOURCE = "BE35D0771CF49B53B2D0721AA4BF3035EE9A9BF2F2DFA1BABB2B9B37A47A2B58"
EXPECTED_CASE_SET = "A34AEBB6651B05C4FE18A5379D1778838B276C4709D530936285A600FE2030FB"
EXPECTED_TWO_RESPONSE = "2DAF416457E016F9D9A77F2E49B4B8B4FBAC8C69C98C9A3051D9105F11A3C287"
MASTER_NONCE = "b3fe3a46aafe29d1ea08c5d1e24a547932e0dff73c67be8e4899011f328fdd05"
PARAMETER_DIMENSION = 13_792
PREFIXES = (4, 8, 16)
STAGE_DELTA = 1.0e-6 / (15 * 3)
NORMAL = NormalDist()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def close(left: float, right: float, tolerance: float = 3.0e-13) -> bool:
    return math.isclose(
        float(left), float(right), rel_tol=tolerance, abs_tol=1.0e-300
    )


def probe_seed(identity: list[int]) -> int:
    payload = (
        "certified-local-training-events/probe-v1\0"
        + MASTER_NONCE
        + "\0"
        + "|".join(str(int(part)) for part in identity)
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (
        2**63 - 1
    )


def calibration(delta: float, probes: int) -> float:
    return NORMAL.inv_cdf(0.5 * (1.0 + float(delta) ** (1.0 / probes)))


def closure(
    *, kappa: float, drift: float, forcing: float, correction: float, domain: float
) -> dict:
    coefficient = kappa * drift
    discriminant = 1.0 - 2.0 * coefficient * forcing
    radius = None
    if discriminant >= 0.0:
        if forcing == 0.0:
            radius = 0.0
        elif coefficient == 0.0:
            radius = forcing
        else:
            radius = 2.0 * forcing / (1.0 + math.sqrt(discriminant))
    return {
        "discriminant": discriminant,
        "radius": radius,
        "passed": radius is not None and correction + radius <= domain,
    }


def capacity(*, kappa: float, drift: float, correction: float, domain: float) -> dict:
    available = max(0.0, domain - correction)
    coefficient = kappa * drift
    if kappa <= 0.0:
        return {"radius": available, "response": math.inf, "injection": math.inf}
    if coefficient <= 0.0:
        return {
            "radius": available,
            "response": available,
            "injection": available / kappa,
        }
    radius = min(available, 1.0 / coefficient)
    response = max(0.0, radius - 0.5 * coefficient * radius * radius)
    return {"radius": radius, "response": response, "injection": response / kappa}


def cache_path(candidate: dict) -> Path:
    gate = (0.7, 0.8, 0.9).index(round(float(candidate["threshold"]), 2))
    return CACHE / (
        f"seed_{candidate['seed']}_gate_{gate}_"
        f"anchor_{candidate['anchor']}_v2.json"
    )


def main() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    require(sha256(PROTOCOL) == EXPECTED_PROTOCOL, "protocol hash changed")
    require(payload["protocol_sha256"] == EXPECTED_PROTOCOL, "result protocol mismatch")
    require(payload["source_sha256"] == EXPECTED_SOURCE, "result source mismatch")
    require(payload["case_set_sha256"] == EXPECTED_CASE_SET, "case set changed")
    require(sha256(TWO_RESPONSE) == EXPECTED_TWO_RESPONSE, "two-response record changed")
    require(
        payload["two_response_source_sha256"] == EXPECTED_TWO_RESPONSE,
        "two-response source mismatch",
    )
    for relative, digest in payload["dependency_sha256"].items():
        require(sha256(ROOT / relative) == digest, f"dependency changed: {relative}")

    directional_payload = json.loads(TWO_RESPONSE.read_text(encoding="utf-8"))
    directional = {
        (
            int(row["candidate"]["seed"]),
            float(row["candidate"]["threshold"]),
            int(row["candidate"]["anchor"]),
        ): row
        for row in directional_payload["rows"]
        if row.get("evaluable")
    }

    keys = []
    all_probe_hashes = set()
    old_total = 0
    new_total = 0
    old_sweeps = 0
    new_sweeps = 0
    reductions = []
    prefix_counts = {4: 0, 8: 0, 16: 0}
    route_counts = {"norm_only": 0, "direct_response": 0}
    direct_computed = 0
    conservative_short_circuits = []
    minimum_logic = math.inf
    minimum_headroom = math.inf
    regenerated_vectors = 0

    torch.set_num_threads(4)
    for row in payload["rows"]:
        candidate = row["candidate"]
        key = (
            int(candidate["seed"]),
            float(candidate["threshold"]),
            int(candidate["anchor"]),
        )
        keys.append(key)
        require(int(row["version"]) == 2, f"wrong row version: {key}")
        require(int(row["outcome_files_read"]) == 0, f"outcome read: {key}")
        require(row["case_set_sha256"] == EXPECTED_CASE_SET, f"row case set: {key}")
        require(row["protocol_sha256"] == EXPECTED_PROTOCOL, f"row protocol: {key}")
        require(row["source_sha256"] == EXPECTED_SOURCE, f"row source: {key}")
        require(
            row["combined_family_failure_upper"] == 2.0e-6,
            f"combined budget: {key}",
        )
        require(close(row["stage_delta"], STAGE_DELTA), f"stage delta: {key}")
        require(
            sha256(ROOT / row["certificate_path"]) == row["certificate_sha256"],
            f"certificate hash: {key}",
        )
        cached = json.loads(cache_path(candidate).read_text(encoding="utf-8"))
        require(cached == row, f"aggregate/cache row mismatch: {key}")

        identity = [int(value) for value in row["operator_identity"]]
        seed = probe_seed(identity)
        require(seed == int(row["probe_seed"]), f"probe seed: {key}")
        dimension = int(row["horizon"]) * 2 * PARAMETER_DIMENSION
        generator = torch.Generator(device="cpu").manual_seed(seed)
        require(
            len(row["probe_hashes"])
            == len(row["initial_probe_norms"])
            == len(row["final_probe_norms"])
            == int(row["prefixes_computed"]),
            f"paired probe lengths: {key}",
        )
        for index, (expected_hash, expected_norm) in enumerate(
            zip(row["probe_hashes"], row["initial_probe_norms"])
        ):
            vector = torch.randn(dimension, generator=generator, dtype=torch.float64)
            digest = hashlib.sha256(vector.numpy().tobytes(order="C")).hexdigest().upper()
            require(digest == expected_hash, f"probe hash {index}: {key}")
            require(
                close(float(torch.linalg.vector_norm(vector)), expected_norm, 2.0e-15),
                f"initial probe norm {index}: {key}",
            )
            require(digest not in all_probe_hashes, f"probe collision: {key}")
            all_probe_hashes.add(digest)
            regenerated_vectors += 1

        require(row["issued"], f"nonissued row in completed panel: {key}")
        require(
            row["bracket"] == directional[key]["surrogate_bracket"],
            f"directional bracket mismatch: {key}",
        )
        require(row["same_as_directional_bracket"], f"agreement flag: {key}")

        for stage_index, stage in enumerate(row["stage_rows"]):
            prefix = int(stage["probes"])
            require(prefix == PREFIXES[stage_index], f"prefix order: {key}")
            y = max(float(value) for value in row["final_probe_norms"][:prefix])
            cd = calibration(STAGE_DELTA, prefix)
            kappa = math.sqrt(y / cd)
            require(close(stage["Y"], y), f"Y: {key} prefix {prefix}")
            require(close(stage["c_delta"], cd), f"calibration: {key} prefix {prefix}")
            require(
                close(stage["operator_norm_upper_bound"], kappa),
                f"kappa: {key} prefix {prefix}",
            )
            error = (
                float(row["directional_quadratic_taylor_error_upper"])
                + float(row["measured_response_recurrence_residual_norm"])
            )
            norm_forcing = kappa * (
                float(row["quadratic_surrogate_injection_norm"]) + error
            )
            require(
                close(
                    stage["forcing_release"]["norm_only_response_upper"],
                    norm_forcing,
                ),
                f"norm forcing: {key} prefix {prefix}",
            )
            selected = norm_forcing
            method = "norm_only"
            if row["direct_forcing_response_used"]:
                response_forcing = float(row["direct_forcing_response_norm"]) + kappa * (
                    error
                    + float(
                        row[
                            "direct_forcing_response_recurrence_residual_norm"
                        ]
                    )
                )
                stored_response = stage["forcing_release"]["response_aware_upper"]
                if stored_response is None:
                    require(
                        stage["norm_only_attempt"]["issued"] and stage_index > 0,
                        f"unexplained omitted response release: {key} prefix {prefix}",
                    )
                    conservative_short_circuits.append(
                        {
                            "candidate": list(key),
                            "prefix": prefix,
                            "stored_norm_only_response_upper": norm_forcing,
                            "latent_response_aware_upper": response_forcing,
                            "latent_to_stored_ratio": response_forcing / norm_forcing,
                        }
                    )
                else:
                    if response_forcing < selected:
                        selected = response_forcing
                        method = "direct_response"
                    require(
                        close(stored_response, response_forcing),
                        f"response forcing: {key} prefix {prefix}",
                    )
            require(stage["selected_method"] == method, f"selected route: {key}")
            require(
                close(stage["corrected_defect_response_bound"], selected),
                f"selected forcing: {key} prefix {prefix}",
            )
            independently_closed = closure(
                kappa=kappa,
                drift=float(row["derivative_drift_upper"]),
                forcing=selected,
                correction=float(row["correction_max_state_norm"]),
                domain=float(row["domain_radius"]),
            )
            stored_closure = stage["closure"]
            require(
                close(stored_closure["discriminant"], independently_closed["discriminant"]),
                f"discriminant: {key} prefix {prefix}",
            )
            if independently_closed["radius"] is None:
                require(
                    stored_closure["remainder_radius"] is None,
                    f"failed-closure radius: {key} prefix {prefix}",
                )
            else:
                require(
                    close(
                        stored_closure["remainder_radius"],
                        independently_closed["radius"],
                    ),
                    f"radius: {key} prefix {prefix}",
                )
            require(
                bool(stored_closure["closure_passed"]) == independently_closed["passed"],
                f"closure flag: {key} prefix {prefix}",
            )
            cap = capacity(
                kappa=kappa,
                drift=float(row["derivative_drift_upper"]),
                correction=float(row["correction_max_state_norm"]),
                domain=float(row["domain_radius"]),
            )
            headroom = cap["response"] / selected
            require(
                close(stage["selected_forcing_headroom_ratio"], headroom),
                f"headroom: {key} prefix {prefix}",
            )
            if stage_index + 1 < len(row["stage_rows"]):
                require(not stage["issued"], f"continued after issuance: {key}")
        final = row["stage_rows"][-1]
        require(final["issued"], f"final stage did not issue: {key}")
        require(row["bracket"] == final["bracket"], f"final bracket: {key}")
        require(
            int(row["prefixes_computed"]) == int(final["probes"]),
            f"stopping prefix: {key}",
        )
        require(
            int(row["batched_gram_calls"]) == len(row["stage_rows"]),
            f"batched call count: {key}",
        )
        require(
            row["direct_forcing_response_used"]
            == (not row["stage_rows"][0]["norm_only_attempt"]["issued"]),
            f"cost-aware fallback rule: {key}",
        )

        prefix_counts[int(final["probes"])] += 1
        route_counts[final["selected_method"]] += 1
        direct_computed += int(row["direct_forcing_response_used"])
        minimum_logic = min(minimum_logic, float(final["logic_slack"]))
        minimum_headroom = min(
            minimum_headroom, float(final["selected_forcing_headroom_ratio"])
        )
        old_total += int(row["directional_baseline_green_gram_applications"])
        new_total += int(row["relinearized_green_gram_applications"])
        old_sweeps += int(row["directional_baseline_theoretical_linearized_sweeps"])
        new_sweeps += int(row["relinearized_theoretical_linearized_sweeps"])
        reductions.append(float(row["green_gram_application_reduction"]))

    require(len(keys) == len(set(keys)) == 15, "candidate family is not 15 unique rows")
    require(set(keys) == set(directional), "candidate family differs from two-response rows")
    require(len(all_probe_hashes) == regenerated_vectors == 64, "probe family size changed")
    require(prefix_counts == {4: 14, 8: 1, 16: 0}, "prefix distribution changed")
    require(route_counts == {"norm_only": 15, "direct_response": 0}, "route counts changed")
    require(direct_computed == 1, "direct fallback computation count changed")
    require(
        len(conservative_short_circuits) == 1,
        "conservative response-release short-circuit count changed",
    )
    require(old_total == 560 and new_total == 64, "aggregate Gram counts changed")
    require(old_sweeps == 1150 and new_sweeps == 144, "sweep counts changed")
    require(payload["outcome_files_read"] == 0, "aggregate outcome read count changed")
    require(payload["issued"] == 15, "issuance count changed")

    audit = {
        "status": "INDEPENDENT RELINEARIZED PREFIX-PANEL AUDIT PASSED",
        "result_sha256": sha256(RESULT),
        "cases": len(keys),
        "issued": 15,
        "same_as_directional_bracket": 15,
        "regenerated_probe_vectors": regenerated_vectors,
        "unique_probe_hashes": len(all_probe_hashes),
        "prefix_distribution": {str(k): v for k, v in prefix_counts.items()},
        "direct_response_computed": direct_computed,
        "conservative_response_release_short_circuits": (
            conservative_short_circuits
        ),
        "selected_method_distribution": route_counts,
        "old_green_gram_applications": old_total,
        "new_green_gram_applications": new_total,
        "aggregate_green_reduction": old_total / new_total,
        "median_pairwise_green_reduction": median(reductions),
        "old_theoretical_linearized_sweeps": old_sweeps,
        "new_theoretical_linearized_sweeps": new_sweeps,
        "aggregate_theoretical_sweep_reduction": old_sweeps / new_sweeps,
        "minimum_logic_slack": minimum_logic,
        "minimum_forcing_headroom": minimum_headroom,
        "new_green_family_failure_upper": 1.0e-6,
        "combined_output_green_failure_upper": 2.0e-6,
        "outcome_files_read": 0,
    }
    OUTPUT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
