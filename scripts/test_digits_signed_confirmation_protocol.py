#!/usr/bin/env python3
from __future__ import annotations

from digits_parity_mlp import make_selection_split, parameter_spec
from run_digits_signed_confirmation import (
    BASE_CONFIG,
    FAMILY_FAILURE_PROBABILITY,
    FRESH_SEEDS,
    HORIZON,
    MAXIMUM_OPERATORS,
    PROBE,
    SWEEPS,
    THRESHOLDS,
    seed_config,
)


def main() -> None:
    assert FRESH_SEEDS == tuple(range(501, 513))
    assert THRESHOLDS == (0.90, 0.925)
    assert HORIZON == 400 and SWEEPS == 3
    assert MAXIMUM_OPERATORS == 24
    assert PROBE.probes == 8 and PROBE.power == 4
    assert PROBE.delta * MAXIMUM_OPERATORS == FAMILY_FAILURE_PROBABILITY
    assert BASE_CONFIG.width == 8 and BASE_CONFIG.learning_rate == 0.03
    config = seed_config(501)
    selection = make_selection_split(config)
    assert "certificate_x" not in selection and "certificate_y" not in selection
    assert parameter_spec(config).size == 538
    print("PASS: prospective digits protocol constants and outcome barrier are frozen.")


if __name__ == "__main__":
    main()
