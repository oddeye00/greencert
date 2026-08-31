#!/usr/bin/env python3
"""Dependency-free audit of the triangular nonlinear envelope.

This script intentionally does not import the implementation under test.  It
generates scalar causal Volterra systems, solves each nonlinear recurrence
chronologically, and independently evaluates the proposed block majorant.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "causal_forward_radius_synthetic_audit.json"
SEED = 918_442_731
CASES = 10_000


def solve_case(
    affine: list[float],
    blocks: list[list[float]],
    curvature: list[float],
    nonlinear_signs: list[float],
) -> tuple[list[float], list[float]]:
    horizon = len(affine)
    exact = [0.0]
    forcing = [0.0] * horizon
    radii = [0.0]
    majorant_forcing = [0.0] * horizon
    for output_step in range(horizon):
        if output_step > 0:
            forcing[output_step] = (
                0.5
                * curvature[output_step]
                * nonlinear_signs[output_step]
                * exact[output_step] ** 2
            )
            majorant_forcing[output_step] = (
                0.5
                * curvature[output_step]
                * radii[output_step] ** 2
            )
        exact.append(
            affine[output_step]
            + sum(
                blocks[output_step][forcing_step] * forcing[forcing_step]
                for forcing_step in range(output_step + 1)
            )
        )
        radii.append(
            abs(affine[output_step])
            + sum(
                abs(blocks[output_step][forcing_step])
                * majorant_forcing[forcing_step]
                for forcing_step in range(output_step + 1)
            )
        )
    return exact, radii


def main() -> None:
    generator = random.Random(SEED)
    minimum_slack = math.inf
    maximum_ratio = 0.0
    checkpoints = 0
    for _case in range(CASES):
        horizon = generator.randint(1, 12)
        affine = [generator.uniform(-0.02, 0.02) for _ in range(horizon)]
        curvature = [generator.uniform(0.0, 4.0) for _ in range(horizon)]
        nonlinear_signs = [
            -1.0 if generator.random() < 0.5 else 1.0
            for _ in range(horizon)
        ]
        blocks = [
            [
                generator.uniform(-0.25, 0.25)
                if forcing_step <= output_step
                else 0.0
                for forcing_step in range(horizon)
            ]
            for output_step in range(horizon)
        ]
        exact, radii = solve_case(
            affine, blocks, curvature, nonlinear_signs
        )
        for observed, radius in zip(exact[1:], radii[1:]):
            slack = radius - abs(observed)
            if slack < -2.0e-15:
                raise AssertionError(
                    f"causal envelope violated by {-slack:.3e}"
                )
            minimum_slack = min(minimum_slack, slack)
            if radius > 0.0:
                maximum_ratio = max(maximum_ratio, abs(observed) / radius)
            checkpoints += 1

    witness_affine = [1.0e-2, 1.0e-2, 1.0e-8, 1.0e-2]
    witness_blocks = [
        [1.0 if row == column else 0.0 for column in range(4)]
        for row in range(4)
    ]
    witness_curvature = [0.0, 0.0, 0.0, 1.0e8]
    witness_exact, witness_radii = solve_case(
        witness_affine,
        witness_blocks,
        witness_curvature,
        [1.0, 1.0, 1.0, 1.0],
    )
    global_affine_norm = math.sqrt(
        sum(value * value for value in witness_affine)
    )
    global_discriminant = 1.0 - 2.0e8 * global_affine_norm
    if global_discriminant >= 0.0:
        raise AssertionError("the scalar-root separation witness disappeared")
    if max(witness_radii[1:]) >= 0.010001:
        raise AssertionError("the causal witness unexpectedly inflated")

    result = {
        "status": "causal forward radius synthetic audit passed",
        "algorithm": (
            "independent scalar Volterra solve versus chronological "
            "absolute block majorant"
        ),
        "seed": SEED,
        "systems": CASES,
        "checkpoints": checkpoints,
        "minimum_radius_slack": minimum_slack,
        "maximum_observed_to_radius_ratio": maximum_ratio,
        "scalar_root_separation_witness": {
            "affine_bounds": witness_affine,
            "curvature_profile": witness_curvature,
            "global_sequence_discriminant": global_discriminant,
            "causal_radii": witness_radii[1:],
            "exact_parameter_errors": witness_exact[1:],
        },
        "outcome_files_read": 0,
    }
    OUTPUT.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
