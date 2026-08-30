#!/usr/bin/env python3
"""Post-seal matched unsigned right-inverse baseline for the Transformer batch.

The frozen certificate uses the signed response ``Z_s = ||K_H s||``.  The
standard norm-tube replacement is ``Z_u = kappa ||s||``.  This audit gives the
unsigned baseline two advantages:

* it uses the same high-confidence finite-window Green norm bound, rather than
  a product of one-step Jacobian norms; and
* whenever closure is possible, it uses the smaller radii-polynomial root
  instead of the frozen signed protocol's convenient radius ``2 Z_s``.

For cases with ``Z_u > 2 Z_s``, the unsigned ball must contain the signed
protocol ball.  The shipped analytic jet envelope is monotone in the radius.
Consequently, if ``2 kappa M(2 Z_s) Z_u > 1``, no radius accepted by the same
unsigned radii-polynomial construction can exist.  Only cases surviving that
favourable necessary test need an expensive centreline reconstruction.

This is an explicitly post-seal baseline audit.  It does not alter candidate
selection, the frozen certificates, or any outcome-blind artifact.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from transformer_block_envelope import ball_valid_envelope, objective_hessian_lipschitz
from transformer_certificate_protocol import Candidate
from transformer_four_sweep_development_audit import count_envelope, persistent_bracket
from transformer_green_confirmation_certificate import load_candidate
from transformer_green_development_audit import (
    build_frozen_centerline,
    gate_slacks,
    persistent_certificate_slack,
)
from transformer_hvp_grokking import logits


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT_JSON = RESULTS / "transformer_unsigned_right_inverse_audit.json"
OUTPUT_MD = RESULTS / "transformer_unsigned_right_inverse_audit.md"
CERTIFICATE_GLOB = "transformer_green_confirmation_certificate_seed_*.json"
AGGREGATE_AUDIT = RESULTS / "transformer_green_confirmation_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def stable_minimal_radius(zero_order: float, kappa: float, drift: float) -> float | None:
    """Small root of Z + (kappa M / 2) R^2 <= R."""
    closure = 2.0 * kappa * drift * zero_order
    if not math.isfinite(closure) or closure > 1.0:
        return None
    if drift == 0.0:
        return zero_order
    return 2.0 * zero_order / (1.0 + math.sqrt(max(0.0, 1.0 - closure)))


def candidate_key(candidate: dict) -> tuple[int, float, int]:
    return (
        int(candidate["seed"]),
        round(float(candidate["threshold"]), 8),
        int(candidate["anchor"]),
    )


def all_certificates() -> list[tuple[Path, dict]]:
    rows = []
    for path in sorted(RESULTS.glob(CERTIFICATE_GLOB)):
        rows.append((path, json.loads(path.read_text(encoding="utf-8"))))
    if len(rows) != 23:
        raise RuntimeError(f"expected 23 frozen candidates, found {len(rows)}")
    return rows


def outcome_map() -> dict[tuple[int, float, int], dict]:
    payload = json.loads(AGGREGATE_AUDIT.read_text(encoding="utf-8"))
    return {candidate_key(row["candidate"]): row for row in payload["rows"]}


@torch.no_grad()
def evaluate_survivor(certificate: dict) -> dict:
    """Run the full valid unsigned tube and output transport for one survivor."""
    started = time.perf_counter()
    raw = certificate["candidate"]
    candidate = Candidate(int(raw["seed"]), float(raw["threshold"]), int(raw["anchor"]))
    config, template, spec, data, parameter, velocity = load_candidate(candidate)
    torch.set_num_threads(config.threads)
    train_pairs, train_labels, _, _, cert_pairs, cert_labels = data
    path = build_frozen_centerline(
        config,
        template,
        spec,
        train_pairs,
        train_labels,
        parameter,
        velocity,
    )
    if path["centerline_sha256"] != certificate["centerline_sha256"]:
        raise RuntimeError("reconstructed centreline differs from the frozen certificate")

    horizon = int(certificate["protocol"]["horizon"])
    center = path["center"][: horizon + 1]
    dimension = parameter.numel()
    kappa = float(certificate["green_probe"]["green_operator_norm_upper_bound"])
    defect = float(certificate["defect_sequence_norm"])
    unsigned_zero_order = kappa * defect
    outer_radius = 2.0 * unsigned_zero_order

    geometry_by_step = {int(row["step"]): row for row in certificate["geometry"]}
    outer_drift = 0.0
    outer_fixed_points = True
    outer_rows = []
    for step in range(1, horizon + 1):
        theta = center[step, :dimension]
        output_upper = float(
            geometry_by_step[step]["output_probe"]["jacobian_norm_upper_bound"]
        )
        block = ball_valid_envelope(
            theta,
            spec,
            config,
            epsilon=outer_radius,
            exact_values=True,
            sphere=True,
        )
        outer_fixed_points &= bool(block["fixed_point_consistent"])
        first_ball = output_upper + block["second"] * outer_radius
        objective_lipschitz = objective_hessian_lipschitz(
            first_ball, block["second"], block["third"]
        )
        map_drift = math.sqrt(2.0) * config.learning_rate * objective_lipschitz
        if step < horizon:
            outer_drift = max(outer_drift, map_drift)
        outer_rows.append(
            {
                "step": step,
                "fixed_point_consistent": bool(block["fixed_point_consistent"]),
                "fixed_point_iterations": int(block["fixed_point_iterations_used"]),
                "block_second": float(block["second"]),
                "block_third": float(block["third"]),
                "objective_hessian_lipschitz_upper": float(objective_lipschitz),
                "optimizer_derivative_drift_upper": float(map_drift),
            }
        )

    outer_closure = 2.0 * kappa * outer_drift * unsigned_zero_order
    minimal_radius = (
        stable_minimal_radius(unsigned_zero_order, kappa, outer_drift)
        if outer_fixed_points
        else None
    )
    if minimal_radius is None:
        return {
            "full_evaluation": True,
            "outer_radius": outer_radius,
            "outer_derivative_drift_upper": outer_drift,
            "outer_closure_statistic": outer_closure,
            "outer_fixed_points_all_consistent": outer_fixed_points,
            "minimal_admissible_radius": None,
            "unsigned_certificate_issued": False,
            "unsigned_certified_bracket": None,
            "elapsed_seconds": time.perf_counter() - started,
            "outer_geometry": outer_rows,
        }

    required = int(math.ceil(candidate.threshold * len(cert_pairs)))
    center_counts = np.asarray(
        [
            int(
                (
                    logits(state[:dimension], cert_pairs, template, spec).argmax(1)
                    == cert_labels
                ).sum()
            )
            for state in center
        ],
        dtype=np.int64,
    )
    guaranteed = [int(center_counts[0])]
    possible = [int(center_counts[0])]
    guarantee_slacks = [gate_slacks(logits(center[0, :dimension], cert_pairs, template, spec), cert_labels, 0.0, required)[0]]
    exclusion_slacks = [gate_slacks(logits(center[0, :dimension], cert_pairs, template, spec), cert_labels, 0.0, required)[1]]
    output_rows = []
    fixed_points = True
    for step in range(1, horizon + 1):
        theta = center[step, :dimension]
        output_upper = float(
            geometry_by_step[step]["output_probe"]["jacobian_norm_upper_bound"]
        )
        block = ball_valid_envelope(
            theta,
            spec,
            config,
            epsilon=minimal_radius,
            exact_values=True,
            sphere=True,
        )
        fixed_points &= bool(block["fixed_point_consistent"])
        margin_radius = math.sqrt(2.0) * (
            output_upper * minimal_radius
            + 0.5 * block["second"] * minimal_radius * minimal_radius
        )
        center_logits = logits(theta, cert_pairs, template, spec)
        lower, upper = count_envelope(center_logits, cert_labels, margin_radius)
        guarantee, exclusion = gate_slacks(
            center_logits, cert_labels, margin_radius, required
        )
        guaranteed.append(lower)
        possible.append(upper)
        guarantee_slacks.append(guarantee)
        exclusion_slacks.append(exclusion)
        output_rows.append(
            {
                "step": step,
                "fixed_point_consistent": bool(block["fixed_point_consistent"]),
                "margin_radius": float(margin_radius),
                "guaranteed_correct": lower,
                "possibly_correct": upper,
            }
        )

    bracket = None
    if fixed_points and outer_fixed_points and outer_closure <= 1.0:
        bracket = persistent_bracket(
            np.asarray(guaranteed, dtype=np.int64),
            np.asarray(possible, dtype=np.int64),
            required,
        )
    return {
        "full_evaluation": True,
        "outer_radius": outer_radius,
        "outer_derivative_drift_upper": outer_drift,
        "outer_closure_statistic": outer_closure,
        "outer_fixed_points_all_consistent": outer_fixed_points,
        "minimal_admissible_radius": minimal_radius,
        "minimal_radius_to_signed_radius_ratio": minimal_radius
        / float(certificate["signed_radius"]),
        "output_fixed_points_all_consistent": fixed_points,
        "unsigned_certified_bracket": bracket,
        "unsigned_certificate_issued": bracket is not None,
        "unsigned_certificate_output_logic_slack": persistent_certificate_slack(
            bracket, guarantee_slacks, exclusion_slacks
        ),
        "maximum_margin_radius": max(row["margin_radius"] for row in output_rows),
        "elapsed_seconds": time.perf_counter() - started,
        "outer_geometry": outer_rows,
        "output_geometry": output_rows,
    }


def main() -> None:
    started = time.perf_counter()
    certificates = all_certificates()
    outcomes = outcome_map()
    rows = []
    survivors: list[tuple[int, dict]] = []

    for index, (path, certificate) in enumerate(certificates):
        candidate = certificate["candidate"]
        row = {
            "candidate": candidate,
            "certificate_file": str(path.relative_to(ROOT)),
            "certificate_sha256": sha256(path),
            "signed_certificate_issued": bool(certificate["certificate_issued"]),
            "signed_certified_bracket": certificate["certified_bracket"],
            "actual_event": outcomes[candidate_key(candidate)]["actual_event"],
            "green_operator_available": certificate["green_probe"] is not None,
            "unsigned_certificate_issued": None,
            "unsigned_certified_bracket": None,
        }
        if certificate["green_probe"] is None:
            row["disposition"] = (
                "construction abstention before residual"
                if certificate.get("defect_sequence_norm") is None
                else "outside matched operator comparison: frozen Green probe was not queried"
            )
            rows.append(row)
            continue

        kappa = float(certificate["green_probe"]["green_operator_norm_upper_bound"])
        defect = float(certificate["defect_sequence_norm"])
        signed = float(certificate["signed_response_sequence_norm"])
        signed_radius = float(certificate["signed_radius"])
        signed_drift = float(certificate["maximum_optimizer_derivative_drift_upper"])
        unsigned = kappa * defect
        ratio = unsigned / signed
        favourable_closure = 2.0 * kappa * signed_drift * unsigned
        row.update(
            {
                "horizon": int(certificate["protocol"]["horizon"]),
                "green_operator_norm_upper_bound": kappa,
                "defect_sequence_norm": defect,
                "signed_response_sequence_norm": signed,
                "unsigned_response_upper": unsigned,
                "unsigned_to_signed_response_ratio": ratio,
                "signed_radius": signed_radius,
                "signed_radius_derivative_drift_upper": signed_drift,
                "favourable_unsigned_closure_using_signed_radius_drift": favourable_closure,
            }
        )
        if not ratio > 2.0:
            raise RuntimeError("monotone-envelope shortcut requires Z_u > 2 Z_s")
        if favourable_closure > 1.0:
            row.update(
                {
                    "disposition": "unsigned closure impossible under the same monotone analytic envelope",
                    "radii_polynomial_impossible": True,
                    "unsigned_certificate_issued": False,
                }
            )
        else:
            row["disposition"] = "survived favourable closure screen; full baseline evaluated"
            survivors.append((index, certificate))
        rows.append(row)

    if len(survivors) != 1:
        raise RuntimeError(f"expected one full-evaluation survivor, found {len(survivors)}")
    for row_index, certificate in survivors:
        full = evaluate_survivor(certificate)
        rows[row_index].update(full)
        bracket = full["unsigned_certified_bracket"]
        actual = rows[row_index]["actual_event"]
        rows[row_index]["unsigned_bracket_contains_actual"] = (
            None if bracket is None else bool(bracket[0] <= actual <= bracket[1])
        )

    comparable = [row for row in rows if row["green_operator_available"]]
    signed_issued = [row for row in comparable if row["signed_certificate_issued"]]
    unsigned_issued = [row for row in comparable if row["unsigned_certificate_issued"]]
    conversions = [
        row
        for row in comparable
        if row["signed_certificate_issued"] and not row["unsigned_certificate_issued"]
    ]
    ratios = [row["unsigned_to_signed_response_ratio"] for row in comparable]
    summary = {
        "frozen_candidates": len(rows),
        "matched_green_operator_cases": len(comparable),
        "signed_issued_in_matched_cases": len(signed_issued),
        "strong_unsigned_issued_in_matched_cases": len(unsigned_issued),
        "signed_certificates_lost_by_unsigned_norm_replacement": len(conversions),
        "signed_issued_retention_fraction_under_unsigned_baseline": len(unsigned_issued)
        / len(signed_issued),
        "unsigned_covered": sum(
            row.get("unsigned_bracket_contains_actual") is True for row in unsigned_issued
        ),
        "minimum_unsigned_to_signed_response_ratio": min(ratios),
        "median_unsigned_to_signed_response_ratio": statistics.median(ratios),
        "maximum_unsigned_to_signed_response_ratio": max(ratios),
        "cases_eliminated_without_reconstruction": sum(
            row.get("radii_polynomial_impossible") is True for row in comparable
        ),
        "full_reconstructions": len(survivors),
    }
    payload = {
        "status": "complete post-seal matched unsigned finite-window right-inverse audit",
        "baseline": {
            "zero_order_term": "Z_u = kappa ||s||_2",
            "operator": "same finite-window causal Green right inverse as the signed method",
            "closure": "Z_u + (kappa M / 2) R^2 <= R",
            "radius": "smallest radii-polynomial root using M valid on B(2 Z_u)",
            "note": "strictly stronger than a product-of-one-step-norm baseline",
        },
        "monotonicity_argument": (
            "The analytic block-jet envelope is coordinatewise nondecreasing in the ball radius. "
            "When Z_u > 2 Z_s, every admissible unsigned radius R >= Z_u contains the frozen "
            "signed ball B(2 Z_s). Hence favourable closure failure using M(2 Z_s) rules out "
            "the same unsigned radii-polynomial certificate at every admissible R."
        ),
        "source_hashes": {
            "aggregate_audit": sha256(AGGREGATE_AUDIT),
            "script": sha256(Path(__file__)),
        },
        "summary": summary,
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    survivor = next(row for row in comparable if row.get("full_evaluation"))
    md = f"""# Matched unsigned finite-window baseline audit

This post-seal audit replaces the signed Green response $\\lVert K_Hs\\rVert$ by the
standard norm-only upper bound $\\kappa\\lVert s\\rVert$, while retaining the **same
finite-window right inverse**. It also gives the baseline the smallest valid
radii-polynomial root, so this is stronger than the common product-of-local-
norms comparison.

## Result

- Frozen candidates with a queried Green operator: **{len(comparable)}**.
- Frozen signed certificates in that matched set: **{len(signed_issued)}**.
- Strong unsigned certificates: **{len(unsigned_issued)}**.
- Signed certificates destroyed solely by discarding the defect direction:
  **{len(conversions)}/{len(signed_issued)}**.
- Unsigned/signed zero-order inflation: median **{statistics.median(ratios):.2f}x**,
  range **{min(ratios):.2f}x--{max(ratios):.2f}x**.
- {summary['cases_eliminated_without_reconstruction']} cases fail closure even
  under the favourable smaller signed-ball derivative envelope.

The sole survivor of that favourable screen was seed
{survivor['candidate']['seed']} at threshold
{survivor['candidate']['threshold']:.2f}. Its own outer-ball closure statistic
is **{survivor['outer_closure_statistic']:.6g}** and its strongest unsigned
bracket is **{survivor['unsigned_certified_bracket']}**.

## Interpretation

The finite-window inverse itself is not enough. The gain comes from applying
that inverse to the **signed, time-ordered defect** before taking a norm. On
the main Transformer confirmation, this cancellation is the difference
between {len(signed_issued)} issued certificates and {len(unsigned_issued)}
under an otherwise matched validated-dynamics construction.

This audit is post-seal and does not modify the frozen candidate, certificate,
or outcome artifacts. Cases where the frozen protocol never queried a Green
operator are excluded from the matched-operator denominator.
"""
    OUTPUT_MD.write_text(md, encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT_JSON), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
