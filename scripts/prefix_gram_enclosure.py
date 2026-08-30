#!/usr/bin/env python3
"""Arithmetic for Gaussian Gram bounds at prespecified probe prefixes."""
from __future__ import annotations

import math
from collections.abc import Sequence

from probe_jacobian_bound import c_delta


def validate_prefixes(prefixes: Sequence[int], available: int | None = None) -> tuple[int, ...]:
    rows = tuple(int(value) for value in prefixes)
    if not rows or any(value < 1 for value in rows):
        raise ValueError("prefixes must be positive and nonempty")
    if any(right <= left for left, right in zip(rows, rows[1:])):
        raise ValueError("prefixes must be strictly increasing")
    if available is not None and rows[-1] > int(available):
        raise ValueError("a prefix exceeds the available probe block")
    return rows


def equal_family_stage_delta(
    *, family_failure: float, operators: int, prefixes: Sequence[int]
) -> float:
    rows = validate_prefixes(prefixes)
    if not 0.0 < float(family_failure) < 1.0:
        raise ValueError("family_failure must lie in (0,1)")
    if int(operators) < 1:
        raise ValueError("operators must be positive")
    return float(family_failure) / (int(operators) * len(rows))


def prefix_increment(*, initial_count: int, final_count: int, target: int) -> int:
    """Return the exact number of new paired probes needed at one prefix."""

    initial_count = int(initial_count)
    final_count = int(final_count)
    target = int(target)
    if initial_count < 0 or final_count < 0 or target < 1:
        raise ValueError("probe counts must be nonnegative and target positive")
    if initial_count != final_count:
        raise ValueError("initial and final probe counts have drifted")
    if final_count > target:
        raise ValueError("the requested target precedes the current prefix")
    return target - final_count


def prefix_gram_rows(
    *,
    final_norms: Sequence[float],
    initial_norms: Sequence[float],
    prefixes: Sequence[int],
    power: int,
    stage_delta: float,
) -> list[dict]:
    rows = validate_prefixes(prefixes, available=len(final_norms))
    if len(final_norms) != len(initial_norms):
        raise ValueError("initial and final norm arrays must have equal length")
    if int(power) < 1:
        raise ValueError("power must be positive")
    if not 0.0 < float(stage_delta) < 1.0:
        raise ValueError("stage_delta must lie in (0,1)")
    finals = [float(value) for value in final_norms]
    initials = [float(value) for value in initial_norms]
    if any(not math.isfinite(value) or value < 0.0 for value in finals + initials):
        raise ValueError("probe norms must be finite and nonnegative")

    output = []
    exponent = 1.0 / (2.0 * int(power))
    for prefix in rows:
        terminal = max(finals[:prefix], default=0.0)
        lower = max(
            (
                (final / initial) ** exponent
                for final, initial in zip(finals[:prefix], initials[:prefix])
                if initial > 0.0 and final > 0.0
            ),
            default=0.0,
        )
        calibration = c_delta(float(stage_delta), prefix)
        upper = 0.0 if terminal <= 0.0 else (terminal / calibration) ** exponent
        output.append(
            {
                "probes": prefix,
                "power": int(power),
                "Y": terminal,
                "c_delta": calibration,
                "delta": float(stage_delta),
                "operator_norm_upper_bound": upper,
                "operator_norm_lower_estimate": lower,
                "gram_applications": prefix * int(power),
            }
        )
    return output


def family_failure_upper(*, stage_delta: float, operators: int, prefixes: Sequence[int]) -> float:
    rows = validate_prefixes(prefixes)
    value = float(stage_delta) * int(operators) * len(rows)
    if not 0.0 <= value < 1.0:
        raise ValueError("combined family failure must lie in [0,1)")
    return value
