#!/usr/bin/env python3
"""Reusable, outcome-blind GreenCert construction for the real-data MLP."""
from __future__ import annotations

import hashlib
import math
import time
from typing import Iterable

import numpy as np
import torch

from matrix_free_mlp import signed_variational_recenter
from probe_jacobian_bound import ProbeConfig, ProbeRegistry, gram_norm_bound
from real_dataset_jet_bound import (
    cross_entropy_hessian_lipschitz,
    margin_jet_bound,
    output_jet_bound,
)
from real_dataset_mlp import (
    ParameterSpec,
    RealMLPConfig,
    count_correct,
    logits,
    objective_hvp,
    optimizer_map,
)
from transformer_green_operator import make_causal_green_products
from transformer_modal_forecast import affine_reference, first_persistent


def trigger_only_anchor(
    train_accuracy: Iterable[float],
    trigger_accuracy: Iterable[float],
    *,
    threshold: float,
    checkpoint_every: int,
    minimum_train_accuracy: float,
    trigger_band: float,
) -> int | None:
    train = list(train_accuracy)
    trigger = list(trigger_accuracy)
    if len(train) != len(trigger):
        raise ValueError("train and trigger histories must have equal length")
    for step in range(0, len(trigger), checkpoint_every):
        if (
            train[step] >= minimum_train_accuracy
            and threshold - trigger_band <= trigger[step] < threshold
        ):
            return step
    return None


def persistent_bracket(
    guaranteed: np.ndarray,
    possible: np.ndarray,
    required: int,
    persistence: int,
) -> list[int] | None:
    lower = first_persistent(possible, required, persistence)
    upper = first_persistent(guaranteed, required, persistence)
    if lower is None or upper is None or lower > upper:
        return None
    return [int(lower), int(upper)]


def minimal_admissible_radius(z_norm: float, kappa: float, drift: float) -> float | None:
    """Small root of Z + kappa*M*R^2/2 <= R, evaluated stably."""
    statistic = 2.0 * kappa * drift * z_norm
    if statistic > 1.0:
        return None
    if kappa * drift == 0.0:
        return z_norm
    return 2.0 * z_norm / (1.0 + math.sqrt(max(0.0, 1.0 - statistic)))


def build_centerline(
    parameter: torch.Tensor,
    data: dict,
    spec: ParameterSpec,
    config: RealMLPConfig,
    *,
    horizon: int,
    sweeps: int,
) -> tuple[list[torch.Tensor], list[dict]]:
    def map_step(point):
        return optimizer_map(point, data["train_x"], data["train_y"], spec, config)

    def jvp(center, direction):
        return direction - config.learning_rate * objective_hvp(
            center,
            direction,
            data["train_x"],
            data["train_y"],
            spec,
            config,
        )

    raw = affine_reference(
        parameter,
        map_step,
        lambda direction: jvp(parameter, direction),
        horizon=horizon,
    )
    paths = [raw]
    diagnostics = []
    for sweep in range(sweeps):
        corrected, diagnostic = signed_variational_recenter(
            paths[-1], map_step, jvp, numeric_cap=1e4
        )
        diagnostics.append({"sweep": sweep + 1, **diagnostic})
        if diagnostic["reached_horizon"] != horizon:
            raise RuntimeError(f"recentring sweep {sweep + 1} truncated")
        paths.append(corrected)
    return paths, diagnostics


@torch.no_grad()
def _count_path(path, features, labels, spec) -> np.ndarray:
    return np.asarray(
        [count_correct(point, features, labels, spec) for point in path],
        dtype=np.int64,
    )


def certify_candidate(
    parameter: torch.Tensor,
    data: dict,
    spec: ParameterSpec,
    config: RealMLPConfig,
    *,
    seed: int,
    gate_index: int,
    threshold: float,
    anchor: int,
    horizon: int,
    persistence: int,
    sweeps: int,
    probe: ProbeConfig,
    registry: ProbeRegistry,
    identity: tuple[int, ...],
) -> dict:
    """Construct one certificate without reading any future outcome artifact."""
    started = time.perf_counter()
    paths, diagnostics = build_centerline(
        parameter, data, spec, config, horizon=horizon, sweeps=sweeps
    )
    required = int(math.ceil(threshold * len(data["certificate_y"])))
    center_counts = _count_path(
        paths[-1], data["certificate_x"], data["certificate_y"], spec
    )
    predicted = first_persistent(center_counts, required, persistence)
    base = {
        "seed": seed,
        "gate_index": gate_index,
        "threshold": threshold,
        "anchor": anchor,
        "required_correct": required,
        "predicted_event": predicted,
        "centerline_sha256": hashlib.sha256(
            paths[-1].detach().cpu().numpy().tobytes(order="C")
        ).hexdigest().upper(),
        "sweep_diagnostics": diagnostics,
        "outcome_joined": False,
    }
    if predicted is None or predicted <= 0:
        return {
            **base,
            "status": "centerline has no future certification-set event",
            "certificate_issued": False,
            "elapsed_seconds": time.perf_counter() - started,
        }

    certificate_horizon = predicted + persistence - 1
    center = paths[-1][: certificate_horizon + 1]

    def map_step(point):
        return optimizer_map(point, data["train_x"], data["train_y"], spec, config)

    def jvp(point, vector):
        return vector - config.learning_rate * objective_hvp(
            point,
            vector,
            data["train_x"],
            data["train_y"],
            spec,
            config,
        )

    defects = torch.stack(
        [map_step(center[step]) - center[step + 1] for step in range(certificate_horizon)]
    )
    products = [lambda vector, point=point: jvp(point, vector) for point in center[:-1]]
    apply_green, transpose_green = make_causal_green_products(
        products, products, spec.size
    )
    signed_response = apply_green(defects.reshape(-1)).reshape(
        certificate_horizon, spec.size
    )
    defect_norm = float(torch.linalg.vector_norm(defects))
    z_norm = float(torch.linalg.vector_norm(signed_response))
    provisional_radius = 2.0 * z_norm
    drift = max(
        config.learning_rate
        * cross_entropy_hessian_lipschitz(
            output_jet_bound(point, data["train_x"], spec, provisional_radius)
        )
        for point in center[:-1]
    )
    early_statistic = 2.0 * drift * z_norm
    common = {
        **base,
        "certificate_horizon": certificate_horizon,
        "defect_sequence_norm": defect_norm,
        "signed_response_sequence_norm": z_norm,
        "maximum_optimizer_derivative_drift_upper": drift,
        "minimum_closure_statistic_using_kappa_ge_1": early_statistic,
    }
    if early_statistic > 1.0:
        return {
            **common,
            "status": "safe early closure abstention",
            "certificate_issued": False,
            "elapsed_seconds": time.perf_counter() - started,
        }

    def gram(vector):
        return transpose_green(apply_green(vector))

    green = gram_norm_bound(
        gram,
        dimension=certificate_horizon * spec.size,
        dtype=parameter.dtype,
        device=parameter.device,
        config=probe,
        identity=identity,
        registry=registry,
    )
    kappa = float(green["operator_norm_upper_bound"])
    closure = 2.0 * kappa * drift * z_norm
    radius = minimal_admissible_radius(z_norm, kappa, drift)
    unsigned_z = kappa * defect_norm
    unsigned_drift = max(
        config.learning_rate
        * cross_entropy_hessian_lipschitz(
            output_jet_bound(point, data["train_x"], spec, 2.0 * unsigned_z)
        )
        for point in center[:-1]
    )
    unsigned_closure = 2.0 * kappa * unsigned_drift * unsigned_z
    unsigned_radius = minimal_admissible_radius(unsigned_z, kappa, unsigned_drift)
    common.update(
        {
            "green_probe": green,
            "closure_statistic": closure,
            "closure_slack": 1.0 - closure,
            "unsigned_right_inverse_response_upper": unsigned_z,
            "directional_gain_ratio": unsigned_z / max(z_norm, 1e-300),
            "unsigned_right_inverse_derivative_drift_upper": unsigned_drift,
            "unsigned_right_inverse_closure_statistic": unsigned_closure,
            "unsigned_right_inverse_minimal_radius": unsigned_radius,
            "unsigned_right_inverse_certified_bracket": None,
            "unsigned_right_inverse_certificate_issued": False,
            "unsigned_right_inverse_maximum_margin_radius": None,
            "unsigned_right_inverse_minimum_output_slack": None,
        }
    )
    if radius is None:
        return {
            **common,
            "status": "Green closure abstention",
            "certificate_issued": False,
            "elapsed_seconds": time.perf_counter() - started,
        }

    def output_logic(test_radius: float) -> dict:
        guaranteed: list[int] = []
        possible: list[int] = []
        margin_radii: list[float] = []
        strict_slacks: list[float] = []
        for point in center:
            values = logits(point, data["certificate_x"], spec)
            labels = data["certificate_y"]
            other = 1 - labels
            margins = values.gather(1, labels[:, None]).squeeze(1) - values.gather(
                1, other[:, None]
            ).squeeze(1)
            jet = margin_jet_bound(
                point, data["certificate_x"], spec, 1, 0, test_radius
            )
            margin_radius = (
                jet["first"] * test_radius
                + 0.5 * jet["second"] * test_radius**2
            )
            lower = margins - margin_radius
            upper = margins + margin_radius
            guaranteed.append(int((lower > 0.0).sum()))
            possible.append(int(len(labels) - (upper < 0.0).sum()))
            margin_radii.append(margin_radius)
            strict_slacks.extend(torch.abs(torch.cat((lower, upper))).tolist())
        event_bracket = persistent_bracket(
            np.asarray(guaranteed), np.asarray(possible), required, persistence
        )
        return {
            "bracket": event_bracket,
            "guaranteed": guaranteed,
            "possible": possible,
            "maximum_margin_radius": max(margin_radii),
            "minimum_output_slack": min(strict_slacks),
        }

    signed_output = output_logic(radius)
    unsigned_output = None if unsigned_radius is None else output_logic(unsigned_radius)
    event_bracket = signed_output["bracket"]
    issued = event_bracket is not None
    return {
        **common,
        "status": "certificate issued" if issued else "output-logic abstention",
        "minimal_admissible_radius": radius,
        "fixed_two_z_radius": 2.0 * z_norm,
        "radius_reduction_fraction": 1.0 - radius / max(2.0 * z_norm, 1e-300),
        "certified_bracket": event_bracket,
        "certificate_issued": issued,
        "maximum_margin_radius": signed_output["maximum_margin_radius"],
        "minimum_output_slack": signed_output["minimum_output_slack"],
        "unsigned_right_inverse_certified_bracket": (
            None if unsigned_output is None else unsigned_output["bracket"]
        ),
        "unsigned_right_inverse_certificate_issued": bool(
            unsigned_output is not None and unsigned_output["bracket"] is not None
        ),
        "unsigned_right_inverse_maximum_margin_radius": (
            None if unsigned_output is None else unsigned_output["maximum_margin_radius"]
        ),
        "unsigned_right_inverse_minimum_output_slack": (
            None if unsigned_output is None else unsigned_output["minimum_output_slack"]
        ),
        "center_count": center_counts[: certificate_horizon + 1].tolist(),
        "guaranteed_correct": signed_output["guaranteed"],
        "possibly_correct": signed_output["possible"],
        "elapsed_seconds": time.perf_counter() - started,
    }
