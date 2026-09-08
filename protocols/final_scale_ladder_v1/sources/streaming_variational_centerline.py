#!/usr/bin/env python3
"""Prefix-local streaming construction of variationally recentered paths."""
from __future__ import annotations

import hashlib
from collections.abc import Callable

import torch
from torch import Tensor

from transformer_four_sweep_development_audit import to_scaled
from transformer_modal_forecast import optimizer_jvp, optimizer_map


@torch.no_grad()
def streaming_signed_recentered_reference(
    anchor: Tensor,
    map_step: Callable[[Tensor], Tensor],
    jacobian_vector: Callable[[Tensor, Tensor], Tensor],
    anchor_jacobian_vector: Callable[[Tensor], Tensor],
    *,
    maximum_horizon: int,
    sweeps: int,
    numeric_cap: float,
    stop_when: Callable[[int, Tensor], bool] | None = None,
) -> tuple[Tensor, list[dict]]:
    """Build all causal recentering sweeps in one time-forward pass.

    Only the final-sweep path is retained.  At step ``j``, each sweep needs the
    preceding sweep only at ``j`` and ``j+1`` plus its own current correction,
    so the sweep dimension can be pipelined without changing the recurrence.
    ``stop_when`` is evaluated on the final-sweep state after each completed
    step and may end the construction causally.
    """

    if maximum_horizon < 0:
        raise ValueError("maximum_horizon must be nonnegative")
    if sweeps < 1:
        raise ValueError("sweeps must be positive")
    if numeric_cap <= 0.0:
        raise ValueError("numeric_cap must be positive")

    affine_defect = map_step(anchor) - anchor
    displacement = torch.zeros_like(anchor)
    raw_current = anchor.clone()
    corrections = [torch.zeros_like(anchor) for _ in range(sweeps)]
    defect_maxima = [0.0] * sweeps
    correction_maxima = [0.0] * sweeps
    final_rows = [anchor.clone()]
    reached = 0

    if stop_when is not None and stop_when(0, final_rows[0]):
        diagnostics = [
            {
                "sweep": index + 1,
                "reached_horizon": 0,
                "maximum_uncorrected_defect_norm": 0.0,
                "maximum_correction_norm": 0.0,
                "hvp_calls": 0,
            }
            for index in range(sweeps)
        ]
        return torch.stack(final_rows), diagnostics

    for step in range(maximum_horizon):
        displacement = anchor_jacobian_vector(displacement) + affine_defect
        raw_next = anchor + displacement
        reference_current = raw_current
        reference_next = raw_next
        next_corrections = []
        for sweep in range(sweeps):
            correction_current = corrections[sweep]
            defect = map_step(reference_current) - reference_next
            next_correction = (
                jacobian_vector(reference_current, correction_current) + defect
            )
            correction_norm = float(torch.linalg.vector_norm(next_correction))
            if (
                not bool(torch.isfinite(next_correction).all())
                or correction_norm > numeric_cap
            ):
                raise RuntimeError(
                    f"streaming recentering sweep {sweep + 1} failed at step {step + 1}"
                )
            defect_maxima[sweep] = max(
                defect_maxima[sweep], float(torch.linalg.vector_norm(defect))
            )
            correction_maxima[sweep] = max(
                correction_maxima[sweep], correction_norm
            )
            center_current = reference_current + correction_current
            center_next = reference_next + next_correction
            next_corrections.append(next_correction)
            reference_current = center_current
            reference_next = center_next
        corrections = next_corrections
        raw_current = raw_next
        final_rows.append(reference_next)
        reached = step + 1
        if stop_when is not None and stop_when(reached, reference_next):
            break

    diagnostics = [
        {
            "sweep": index + 1,
            "reached_horizon": reached,
            "maximum_uncorrected_defect_norm": defect_maxima[index],
            "maximum_correction_norm": correction_maxima[index],
            "hvp_calls": reached,
        }
        for index in range(sweeps)
    ]
    return torch.stack(final_rows), diagnostics


def build_streaming_transformer_centerline(
    config,
    template,
    spec,
    train_pairs: Tensor,
    train_labels: Tensor,
    parameter: Tensor,
    velocity: Tensor,
    *,
    maximum_horizon: int,
    sweeps: int = 4,
    numeric_cap: float = 1.0e6,
    stop_when: Callable[[int, Tensor], bool] | None = None,
) -> dict:
    """Streaming counterpart of the frozen Transformer centerline builder."""

    anchor = torch.cat((parameter, velocity))

    def map_step(state: Tensor) -> Tensor:
        return optimizer_map(
            state, train_pairs, train_labels, template, spec, config
        )

    def jvp(center: Tensor, direction: Tensor) -> Tensor:
        return optimizer_jvp(
            center,
            direction,
            train_pairs,
            train_labels,
            template,
            spec,
            config,
        )

    center, diagnostics = streaming_signed_recentered_reference(
        anchor,
        map_step,
        jvp,
        lambda direction: jvp(anchor, direction),
        maximum_horizon=maximum_horizon,
        sweeps=sweeps,
        numeric_cap=numeric_cap,
        stop_when=stop_when,
    )
    dimension = int(parameter.numel())
    scaled = to_scaled(center, dimension, config.learning_rate)
    return {
        "map_step": map_step,
        "center": center,
        "scaled_center": scaled,
        "diagnostics": diagnostics,
        "horizon_reached": len(center) - 1,
        "centerline_sha256": hashlib.sha256(
            scaled.detach().cpu().contiguous().numpy().tobytes(order="C")
        ).hexdigest().upper(),
    }
