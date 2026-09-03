#!/usr/bin/env python3
"""Regression for invariant-geometry reuse in Transformer jet envelopes."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from strict_transformer_block_envelope import strict_ball_valid_envelope
from transformer_block_envelope_v15 import (
    _compose,
    anchor_majorized_parameter_geometry,
    ball_valid_envelope,
    exact_stage_values,
    parameter_geometry,
)
from transformer_certificate_protocol import Candidate
from transformer_mixed_directional_jet_v15 import (
    mixed_directional_objective_fourth_bound,
)
from transformer_v3_certificate import load_candidate


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "results" / "transformer_v3_certificate_seed_366_gate_1_anchor_1120.json"


def main() -> None:
    payload = json.loads(RECORD.read_text(encoding="utf-8"))
    raw = payload["candidate"]
    candidate = Candidate(int(raw["seed"]), float(raw["threshold"]), int(raw["anchor"]))
    config, _, spec, _, parameter, _ = load_candidate(candidate)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    epsilon = float(payload["outer_domain_radius"])

    centre = exact_stage_values(parameter, spec, config)
    geometry = parameter_geometry(parameter, spec, config)
    expected_geometry_keys = {
        "query_weight", "query_bias", "key_weight", "key_bias",
        "value_weight", "value_bias", "out_proj_weight", "out_proj_bias",
        "linear1_weight", "linear1_bias", "linear2_weight", "linear2_bias",
        "readout_weight",
    }
    assert set(geometry) == expected_geometry_keys
    anchor_majorized = anchor_majorized_parameter_geometry(
        parameter, parameter, spec, config, anchor_norms=geometry
    )
    assert anchor_majorized == geometry

    stages_uncached = {}
    stages_cached = {}
    uncached = _compose(
        parameter,
        spec,
        config,
        centre,
        {name: 0.0 for name in centre},
        epsilon,
        exact_values=True,
        sphere=True,
        stage_jets=stages_uncached,
    )
    cached = _compose(
        parameter,
        spec,
        config,
        centre,
        {name: 0.0 for name in centre},
        epsilon,
        exact_values=True,
        sphere=True,
        stage_jets=stages_cached,
        geometry=geometry,
    )
    assert uncached == cached
    assert stages_uncached == stages_cached

    default_envelope = ball_valid_envelope(
        parameter, spec, config, epsilon=epsilon, exact_values=True, sphere=True
    )
    uncached_envelope = ball_valid_envelope(
        parameter,
        spec,
        config,
        epsilon=epsilon,
        exact_values=True,
        sphere=True,
        reuse_geometry=False,
    )
    shared_envelope = ball_valid_envelope(
        parameter,
        spec,
        config,
        epsilon=epsilon,
        exact_values=True,
        sphere=True,
        centre_values=centre,
        parameter_norms=geometry,
    )
    for key in (
        "value", "first", "second", "third", "centre_values", "inflation",
        "stage_first", "fixed_point_consistent", "first_iterations",
        "fixed_point_iterations_used", "jet",
    ):
        assert default_envelope[key] == uncached_envelope[key], key
        assert default_envelope[key] == shared_envelope[key], key

    generator = torch.Generator().manual_seed(1701)
    direction = torch.randn(parameter.shape, dtype=parameter.dtype, generator=generator)
    direction *= 1.0e-12 / torch.linalg.vector_norm(direction)
    default_mixed = mixed_directional_objective_fourth_bound(
        parameter, direction, spec, config
    )
    uncached_mixed = mixed_directional_objective_fourth_bound(
        parameter, direction, spec, config, reuse_geometry=False
    )
    shared_mixed = mixed_directional_objective_fourth_bound(
        parameter,
        direction,
        spec,
        config,
        centre_values=centre,
        parameter_norms=geometry,
    )
    assert default_mixed == uncached_mixed == shared_mixed

    strict = strict_ball_valid_envelope(
        parameter, spec, config, epsilon=epsilon, exact_values=True, sphere=True
    )
    assert strict["strict_binary64_postfixed"]

    dominance_checks = 0
    for scale in (1.0e-8, 1.0e-6, 1.0e-4):
        perturbed = parameter + scale * direction / 1.0e-12
        exact_geometry = parameter_geometry(perturbed, spec, config)
        majorized_geometry = anchor_majorized_parameter_geometry(
            perturbed, parameter, spec, config, anchor_norms=geometry
        )
        for key in expected_geometry_keys:
            assert majorized_geometry[key] >= exact_geometry[key] * (1.0 - 2.0e-14)
            dominance_checks += 1
        perturbed_centre = exact_stage_values(perturbed, spec, config)
        exact_bound = ball_valid_envelope(
            perturbed,
            spec,
            config,
            epsilon=epsilon,
            centre_values=perturbed_centre,
            parameter_norms=exact_geometry,
        )
        majorized_bound = ball_valid_envelope(
            perturbed,
            spec,
            config,
            epsilon=epsilon,
            centre_values=perturbed_centre,
            parameter_norms=majorized_geometry,
        )
        assert majorized_bound["fixed_point_consistent"]
        for key in ("first", "second", "third"):
            assert majorized_bound[key] >= exact_bound[key] * (1.0 - 2.0e-14)
            dominance_checks += 1
    print(
        {
            "status": "Transformer envelope geometry-cache regression passed",
            "geometry_entries": len(geometry),
            "envelope_iterations": default_envelope["fixed_point_iterations_used"],
            "mixed_iterations": default_mixed["fixed_point_iterations_used"],
            "anchor_majorization_checks": dominance_checks,
        }
    )


if __name__ == "__main__":
    main()
