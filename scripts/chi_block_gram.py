#!/usr/bin/env python3
"""Chi-calibrated Gaussian block bounds for matrix-free Gram powers."""
from __future__ import annotations

import math

from scipy.stats import chi


def chi_lower_calibration(*, probes: int, delta: float) -> float:
    """Lower-tail chi quantile used by the block-Frobenius theorem."""

    if probes < 1:
        raise ValueError("probes must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0,1)")
    value = float(chi.ppf(delta, df=probes))
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError("chi lower-tail calibration was not finite and positive")
    return value


def chi_block_operator_upper_bound(
    *, terminal_frobenius_norm: float, calibration: float, power: int
) -> float:
    """Return ``(||A^q G||_F / c)^(1/(2q))``."""

    terminal = float(terminal_frobenius_norm)
    calibration = float(calibration)
    if not math.isfinite(terminal) or terminal < 0.0:
        raise ValueError("terminal_frobenius_norm must be finite and nonnegative")
    if not math.isfinite(calibration) or calibration <= 0.0:
        raise ValueError("calibration must be finite and positive")
    if power < 1:
        raise ValueError("power must be positive")
    return (
        0.0
        if terminal == 0.0
        else (terminal / calibration) ** (1.0 / (2.0 * power))
    )
