from transformer_certificate_protocol import Candidate
from transformer_v3_protocol import (
    FAMILY_FAILURE_PROBABILITY,
    HORIZON,
    MAXIMUM_POWER,
    SEEDS,
    THRESHOLDS,
    candidate_universe,
    maximum_operator_count,
    probe_config,
)


def main() -> None:
    assert SEEDS == tuple(range(355, 379))
    assert THRESHOLDS == (0.70, 0.80, 0.90)
    assert HORIZON == 300
    assert MAXIMUM_POWER == 8
    accounting = maximum_operator_count()
    assert accounting["maximum_candidates"] == 72
    assert accounting["operators_per_candidate"] == 302
    assert accounting["maximum_probabilistic_operators"] == 21_744
    probe = probe_config()
    assert probe.probes == 16 and probe.power == 8
    assert (
        probe.delta * accounting["maximum_probabilistic_operators"]
        == FAMILY_FAILURE_PROBABILITY
    )
    candidates = (
        Candidate(355, 0.70, 40),
        Candidate(378, 0.90, 5_000),
    )
    horizons = {candidates[0]: 31, candidates[1]: 300}
    universe = candidate_universe(candidates, horizons)
    assert len(universe) == (31 + 1) + 1 + (300 + 1) + 1
    assert len(universe) == len(set(universe))
    print("PASS: Transformer v3 protocol constants and identities are coherent.")


if __name__ == "__main__":
    main()
