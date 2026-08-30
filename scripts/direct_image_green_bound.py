#!/usr/bin/env python3
"""Direct-image Gaussian operator bounds at prespecified probe prefixes."""
from __future__ import annotations

import math
from collections.abc import Sequence

from prefix_gram_enclosure import validate_prefixes
from probe_jacobian_bound import c_delta


def direct_image_rows(
    *,
    image_norms: Sequence[float],
    initial_norms: Sequence[float],
    prefixes: Sequence[int],
    stage_delta: float,
) -> list[dict]:
    """Return ``||T|| <= max_i ||Tg_i||/c_delta`` at each prefix."""

    rows = validate_prefixes(prefixes, available=len(image_norms))
    if len(image_norms) != len(initial_norms):
        raise ValueError("initial and image norm arrays must have equal length")
    if not 0.0 < float(stage_delta) < 1.0:
        raise ValueError("stage_delta must lie in (0,1)")
    images = [float(value) for value in image_norms]
    initials = [float(value) for value in initial_norms]
    if any(not math.isfinite(value) or value < 0.0 for value in images + initials):
        raise ValueError("probe norms must be finite and nonnegative")

    output = []
    for prefix in rows:
        terminal = max(images[:prefix], default=0.0)
        lower = max(
            (
                image / initial
                for image, initial in zip(images[:prefix], initials[:prefix])
                if initial > 0.0
            ),
            default=0.0,
        )
        calibration = c_delta(float(stage_delta), prefix)
        upper = 0.0 if terminal <= 0.0 else terminal / calibration
        output.append(
            {
                "probes": prefix,
                "Y_direct": terminal,
                "c_delta": calibration,
                "delta": float(stage_delta),
                "operator_norm_upper_bound": upper,
                "operator_norm_lower_estimate": lower,
                "forward_green_applications": prefix,
                "transpose_green_applications": 0,
            }
        )
    return output
