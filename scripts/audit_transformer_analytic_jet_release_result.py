#!/usr/bin/env python3
"""Independent replay of the deterministic analytic-jet release result."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from transformer_certificate_protocol import Candidate, PERSISTENCE
from transformer_hvp_grokking import logits
from transformer_v3_certificate import _gate_raw_slacks, load_candidate, output_path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOURCE = RESULTS / "transformer_analytic_jet_release_postseal_audit.json"
DIRECT = RESULTS / "transformer_direct_image_green_panel_audit.json"
PREFIX = RESULTS / "transformer_v3_relinearized_prefix_panel_audit.json"
OUTPUT = RESULTS / "transformer_analytic_jet_release_independent_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def key(candidate: dict) -> tuple[int, float, int]:
    return int(candidate["seed"]), float(candidate["threshold"]), int(candidate["anchor"])


def first_persistent(values: list[bool]) -> int | None:
    for start in range(max(0, len(values) - PERSISTENCE + 1)):
        if all(values[start : start + PERSISTENCE]):
            return start
    return None


def bracket(guarantee: list[float], exclusion: list[float]) -> list[int] | None:
    lower = first_persistent([value <= 0.0 for value in exclusion])
    upper = first_persistent([value > 0.0 for value in guarantee])
    if lower is None or upper is None or lower > upper:
        return None
    return [lower, upper]


def logic_slack(
    interval: list[int], guarantee: list[float], exclusion: list[float]
) -> float:
    lower, upper = interval
    upper_slack = min(guarantee[upper : upper + PERSISTENCE])
    prior = [max(exclusion[start : start + PERSISTENCE]) for start in range(lower)]
    return min(upper_slack, math.inf if not prior else min(prior))


def close(left: float, right: float, rel: float = 2.0e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=rel, abs_tol=1.0e-300)


def main() -> None:
    source = read(SOURCE)
    direct = read(DIRECT)
    prefix = read(PREFIX)
    if source["direct_panel_sha256"] != sha256(DIRECT):
        raise AssertionError("direct-panel hash mismatch")
    if source["prefix_panel_sha256"] != sha256(PREFIX):
        raise AssertionError("prefix-panel hash mismatch")
    direct_index = {key(row["candidate"]): row for row in direct["rows"]}
    prefix_index = {key(row["candidate"]): row for row in prefix["rows"]}
    recomputed = []

    for claimed in source["rows"]:
        identity = key(claimed["candidate"])
        seed, threshold, anchor = identity
        candidate = Candidate(seed, threshold, anchor)
        drow = direct_index[identity]
        prow = prefix_index[identity]
        certificate = read(output_path(candidate))
        terminal = drow["stage_rows"][-1]
        selected = terminal["direct"] if drow["route"] == "direct_image" else terminal["gram"]
        rows = certificate["output_rows"]
        eta = 0.01
        drift = 0.0
        for row in rows[:-1]:
            first = float(row["block_first"])
            second = float(row["block_second"])
            third = float(row["block_third"])
            objective_third = 2.0 * first**3 + 1.5 * first * second + math.sqrt(2.0) * third
            drift = max(drift, math.sqrt(2.0) * eta * objective_third)
        kappa = float(selected["operator_norm_upper_bound"])
        forcing = float(selected["forcing_response_upper"])
        correction = float(prow["correction_max_state_norm"])
        domain = float(prow["domain_radius"])
        discriminant = 1.0 - 2.0 * kappa * drift * forcing
        radius = None
        if discriminant >= 0.0:
            radius = 2.0 * forcing / (1.0 + math.sqrt(discriminant))
            if correction + radius > domain:
                radius = None

        interval = None
        slack = None
        maximum_margin = None
        if radius is not None:
            config, template, spec, data, parameter, _ = load_candidate(candidate)
            _, _, _, _, cert_pairs, cert_labels = data
            raw_zero = _gate_raw_slacks(
                logits(parameter, cert_pairs, template, spec),
                cert_labels,
                int(certificate["required_correct"]),
            )
            guarantees = [float(raw_zero[0])]
            exclusions = [float(raw_zero[1])]
            margins = [0.0]
            total_radius = correction + radius
            for row in rows:
                margin = math.sqrt(2.0) * float(row["block_first"]) * total_radius
                margins.append(margin)
                guarantees.append(float(row["raw_guarantee_slack"]) - margin)
                exclusions.append(float(row["raw_exclusion_slack"]) - margin)
            interval = bracket(guarantees, exclusions)
            maximum_margin = max(margins)
            if interval is not None:
                slack = logic_slack(interval, guarantees, exclusions)

        issued = interval is not None
        if issued != bool(claimed["analytic_jet_issued"]):
            raise AssertionError(f"issuance mismatch for {candidate}")
        if interval != claimed["analytic_jet_bracket"]:
            raise AssertionError(f"bracket mismatch for {candidate}")
        if not close(drift, claimed["maximum_optimizer_jacobian_drift"]):
            raise AssertionError(f"drift mismatch for {candidate}")
        if issued:
            if not close(slack, claimed["analytic_logic_slack"]):
                raise AssertionError(f"logic-slack mismatch for {candidate}")
            if not close(maximum_margin, claimed["maximum_analytic_margin_radius"]):
                raise AssertionError(f"margin mismatch for {candidate}")
        recomputed.append(
            {
                "candidate": claimed["candidate"],
                "issued": issued,
                "bracket": interval,
                "drift": drift,
                "discriminant": discriminant,
                "remainder_radius": radius,
                "logic_slack": slack,
                "maximum_margin_radius": maximum_margin,
            }
        )

    issued_rows = [row for row in recomputed if row["issued"]]
    if len(issued_rows) != 8:
        raise AssertionError("independent issuance count changed")
    payload = {
        "status": "INDEPENDENT ANALYTIC-JET RELEASE AUDIT PASSED",
        "source_sha256": sha256(SOURCE),
        "source_script_sha256": source["source_sha256"],
        "cases": len(recomputed),
        "analytic_jet_issued": len(issued_rows),
        "same_brackets": all(
            row["bracket"] == direct_index[key(row["candidate"])]["bracket"]
            for row in issued_rows
        ),
        "future_outcome_files_read": 0,
        "rows": recomputed,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
