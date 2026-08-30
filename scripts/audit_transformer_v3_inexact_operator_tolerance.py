#!/usr/bin/env python3
"""Post-seal tolerance of one combined Transformer certificate to Gram defects."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from adaptive_witness_policy import WitnessQuery, acquire_witnesses
from inexact_anytime_gram import q1_relative_terminal_residual_upper
from one_shot_recenter_closure import conservative_one_shot_closure
from transformer_block_envelope import objective_hessian_lipschitz
from transformer_certificate_protocol import Candidate
from transformer_v3_certificate import load_candidate, output_path, safe_json


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOURCE = RESULTS / (
    "transformer_v3_combined_online_role_seed_366_gate_1_anchor_1120_"
    "matched-combined-v5.json"
)
OUTPUT = RESULTS / "transformer_v3_inexact_operator_tolerance_postseal_audit.json"
THEOREM = ROOT / "INEXACT_OPERATOR_GREENCERT_THEOREM.md"
THEOREM_CORE_MARKER = "\n## 5. "


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def theorem_core_sha256() -> str:
    text = THEOREM.read_text(encoding="utf-8")
    if THEOREM_CORE_MARKER not in text:
        raise AssertionError("inexact theorem core marker is missing")
    core = text.split(THEOREM_CORE_MARKER, 1)[0].rstrip() + "\n"
    return hashlib.sha256(core.encode("utf-8")).hexdigest().upper()


def main() -> None:
    source = safe_json(SOURCE)
    candidate = Candidate(**source["candidate"])
    certificate = safe_json(output_path(candidate))
    config, _, _, _, _, _ = load_candidate(candidate)
    rows = {int(row["step"]): row for row in source["output_rows"]}
    green = source["green_rows"][0]
    if source["decision"]["q_green"] != 1 or source["decision"]["q_output"] != 1:
        raise AssertionError("tolerance audit is specialized to the sealed q=1 case")
    event = int(source["frozen_predicted_persistent_event"])
    horizon = int(source["horizon"])
    persistence = int(certificate["protocol"]["persistence"])
    geometry = source["geometry"]
    raw_exclusions = {
        step: float(row["raw_exclusion_slack"]) for step, row in rows.items()
    }

    def replay(green_relative: float, output_relative: float) -> dict:
        corrected_outputs = {
            step: q1_relative_terminal_residual_upper(
                terminal_norm=float(row["Y"]),
                calibration=float(row["c_delta"]),
                relative_residual=output_relative,
            )
            for step, row in rows.items()
        }
        maximum_map_drift = 0.0
        for step in range(1, horizon):
            row = rows[step]
            first_ball = corrected_outputs[step] + float(row["block_second"]) * float(
                geometry["outer_domain_radius"]
            )
            objective_drift = objective_hessian_lipschitz(
                first_ball,
                float(row["block_second"]),
                float(row["block_third"]),
            )
            maximum_map_drift = max(
                maximum_map_drift,
                math.sqrt(2.0) * config.learning_rate * objective_drift,
            )
        corrected_green = q1_relative_terminal_residual_upper(
            terminal_norm=float(green["Y"]),
            calibration=float(green["c_delta"]),
            relative_residual=green_relative,
        )
        closure = conservative_one_shot_closure(
            kappa=corrected_green,
            derivative_drift=maximum_map_drift,
            response_sequence_norm=float(geometry["signed_response_sequence_norm"]),
            response_max_state_norm=float(geometry["signed_response_max_state_norm"]),
            domain_radius=float(geometry["outer_domain_radius"]),
        )
        if not closure.closure_passed:
            return {
                "issued": False,
                "reason": "optimizer closure failed",
                "green_upper": corrected_green,
                "maximum_map_drift": maximum_map_drift,
                "closure": closure.as_dict(),
            }
        radius = float(closure.total_pointwise_radius)
        queried_slacks: list[float] = []

        def query(step: int) -> WitnessQuery:
            row = rows[step]
            margin = math.sqrt(2.0) * (
                corrected_outputs[step] * radius
                + 0.5 * float(row["block_second"]) * radius * radius
            )
            guarantee_slack = float(row["raw_guarantee_slack"]) - margin
            exclusion_slack = float(row["raw_exclusion_slack"]) - margin
            if event <= step < event + persistence:
                queried_slacks.append(guarantee_slack)
            else:
                queried_slacks.append(exclusion_slack)
            return WitnessQuery(step, guarantee_slack > 0.0, exclusion_slack > 0.0)

        policy = acquire_witnesses(
            event=event,
            persistence=persistence,
            horizon=horizon,
            raw_exclusion_slacks=raw_exclusions,
            query=query,
            exact_failures={0},
        )
        return {
            "issued": bool(policy.issued),
            "reason": policy.reason,
            "green_upper": corrected_green,
            "maximum_output_upper": max(corrected_outputs.values()),
            "maximum_map_drift": maximum_map_drift,
            "closure": closure.as_dict(),
            "minimum_queried_logic_slack": min(queried_slacks),
            "query_order": list(policy.query_order),
        }

    baseline = replay(0.0, 0.0)
    if not baseline["issued"]:
        raise AssertionError("zero-residual replay failed")

    def critical(mode: str) -> dict:
        def evaluate(relative: float) -> dict:
            return replay(
                relative if mode in ("green", "common") else 0.0,
                relative if mode in ("output", "common") else 0.0,
            )

        lower = 0.0
        upper = 1.0
        while evaluate(upper)["issued"] and upper < 1.0e18:
            lower = upper
            upper *= 2.0
        if upper >= 1.0e18 and evaluate(upper)["issued"]:
            return {
                "mode": mode,
                "lower_passing_relative_gram_residual": upper,
                "upper_failing_relative_gram_residual": None,
                "operator_norm_inflation_at_lower": math.sqrt(1.0 + upper),
                "lower_replay": evaluate(upper),
                "upper_replay": None,
            }
        for _ in range(90):
            middle = 0.5 * (lower + upper)
            if evaluate(middle)["issued"]:
                lower = middle
            else:
                upper = middle
        return {
            "mode": mode,
            "lower_passing_relative_gram_residual": lower,
            "upper_failing_relative_gram_residual": upper,
            "operator_norm_inflation_at_lower": math.sqrt(1.0 + lower),
            "lower_replay": evaluate(lower),
            "upper_replay": evaluate(upper),
        }

    tolerances = {mode: critical(mode) for mode in ("green", "output", "common")}
    payload = {
        "status": "post-seal inexact-operator tolerance audit passed",
        "scope": (
            "Sensitivity conditional on verified q=1 Gram residual envelopes; "
            "not an estimate of actual float64 error and not a new certificate."
        ),
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "theorem": str(THEOREM.relative_to(ROOT)),
        "theorem_core_scope": "Sections 1--4, ending before the empirical audit record",
        "theorem_core_marker": THEOREM_CORE_MARKER.strip(),
        "theorem_core_sha256": theorem_core_sha256(),
        "candidate": candidate.__dict__,
        "bracket": source["combined_bracket"],
        "baseline": baseline,
        "tolerances": tolerances,
        "interpretation": (
            "For q=1, xi <= r*Y inflates the randomized operator bound by "
            "sqrt(1+r).  Each reported lower endpoint still issues; the adjacent "
            "upper endpoint abstains."
        ),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**payload, "output_sha256": sha256(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
