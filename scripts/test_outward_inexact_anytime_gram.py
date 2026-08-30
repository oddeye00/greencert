#!/usr/bin/env python3
"""Arb regression tests for the inexact Gram scalar supersolution."""
from __future__ import annotations

import json
import math

import numpy as np
from flint import ctx

from inexact_anytime_gram import inexact_gram_operator_upper_bound
from outward_inexact_anytime_gram import (
    folded_normal_calibration_lower,
    outward_inexact_gram_operator_upper_bound,
    outward_operator_supersolution_value,
)


def main() -> None:
    old_precision = ctx.prec
    ctx.prec = 256
    try:
        run_tests()
    finally:
        ctx.prec = old_precision


def run_tests() -> None:
    calibrations = 0
    for probes in (1, 2, 8, 16, 32):
        for delta in (1.0e-12, 1.0e-8, 1.0e-3):
            value = folded_normal_calibration_lower(delta=delta, probes=probes)
            if not math.isfinite(value) or value <= 0.0:
                raise AssertionError("invalid outward calibration")
            calibrations += 1

    generator = np.random.default_rng(270826)
    cases = 0
    maximum_inflation = 1.0
    for q in range(1, 9):
        for _ in range(125):
            terminal = float(10.0 ** generator.uniform(-20.0, 20.0))
            calibration = float(10.0 ** generator.uniform(-3.0, -0.05))
            residuals = tuple(
                float(terminal * 10.0 ** generator.uniform(-8.0, 1.0))
                for _power in range(q)
            )
            ordinary = inexact_gram_operator_upper_bound(
                terminal_norm=terminal,
                calibration=calibration,
                residual_norms=residuals,
            )
            outward = outward_inexact_gram_operator_upper_bound(
                terminal_norm=terminal,
                calibration_lower=calibration,
                residual_norms=residuals,
            )
            if outward < ordinary * (1.0 - 5.0e-14):
                raise AssertionError((q, ordinary, outward))
            polynomial = outward_operator_supersolution_value(
                outward,
                terminal_norm=terminal,
                calibration_lower=calibration,
                residual_norms=residuals,
            )
            if float(polynomial.lower()) < 0.0:
                raise AssertionError("outward root failed supersolution check")
            maximum_inflation = max(maximum_inflation, outward / ordinary)
            cases += 1

    q1 = outward_inexact_gram_operator_upper_bound(
        terminal_norm=7.0,
        calibration_lower=0.4,
        residual_norms=(1.75,),
    )
    exact = math.sqrt((7.0 + 1.75) / 0.4)
    if q1 < exact:
        raise AssertionError("q=1 outward root is below the exact value")
    degenerate = outward_inexact_gram_operator_upper_bound(
        terminal_norm=0.0,
        calibration_lower=0.5,
        residual_norms=(2.0, 0.0),
    )
    if degenerate < 2.0:
        raise AssertionError("degenerate positive root was missed")
    print(
        json.dumps(
            {
                "status": "outward inexact Gram scalar tests passed",
                "calibration_cases": calibrations,
                "root_cases": cases + 2,
                "maximum_outward_to_float_bound_ratio": maximum_inflation,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
