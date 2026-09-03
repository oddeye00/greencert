#!/usr/bin/env python3
"""Prefix-local streaming construction of variationally recentered paths."""
from __future__ import annotations

import hashlib
from collections.abc import Callable

import torch
from torch import Tensor

from transformer_four_sweep_development_audit import to_scaled
from transformer_modal_forecast import optimizer_jvp, optimizer_map
from transformer_modal_forecast_v15 import (
    optimizer_map_and_jvp,
    replayable_anchor_optimizer,
)
from transformer_optimizer_probe_v15 import scaled_optimizer_map_jvp_quadratic


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
    map_and_jacobian_vector: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]
    | None = None,
    anchor_map_value: Tensor | None = None,
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

    affine_defect = (
        map_step(anchor) if anchor_map_value is None else anchor_map_value
    ) - anchor
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
            if map_and_jacobian_vector is None:
                mapped = map_step(reference_current)
                linear_image = jacobian_vector(reference_current, correction_current)
            else:
                mapped, linear_image = map_and_jacobian_vector(
                    reference_current, correction_current
                )
            defect = mapped - reference_next
            next_correction = linear_image + defect
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


@torch.no_grad()
def streaming_terminal_quadratic_reference(
    anchor: Tensor,
    map_step: Callable[[Tensor], Tensor],
    anchor_jacobian_vector: Callable[[Tensor], Tensor],
    map_and_jacobian_vector: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
    terminal_map_jvp_quadratic: Callable[
        [Tensor, Tensor], tuple[Tensor, Tensor, Tensor]
    ],
    *,
    maximum_horizon: int,
    sweeps: int,
    numeric_cap: float,
    anchor_map_value: Tensor | None = None,
    terminal_encode: Callable[[Tensor], Tensor] | None = None,
    terminal_decode: Callable[[Tensor], Tensor] | None = None,
) -> dict:
    """Pipeline all sweeps and release the terminal quadratic forcing.

    The final sweep's input path, signed correction, output path, and
    center-quadratic forcing are retained.  Its callback may use a linearly
    transformed coordinate system through ``terminal_encode`` and
    ``terminal_decode``; this preserves the scaled-momentum proof object used
    by the certificate.  At each time index the final sweep already evaluates
    the map and JVP at precisely the point/direction needed by the
    cancellation-safe quadratic surrogate, so one nested graph supplies all
    three quantities without a second traversal of the time window.
    """

    if maximum_horizon < 1:
        raise ValueError("maximum_horizon must be positive")
    if sweeps < 1:
        raise ValueError("sweeps must be positive")
    if numeric_cap <= 0.0:
        raise ValueError("numeric_cap must be positive")
    encode = terminal_encode if terminal_encode is not None else (lambda value: value)
    decode = terminal_decode if terminal_decode is not None else (lambda value: value)
    affine_defect = (
        map_step(anchor) if anchor_map_value is None else anchor_map_value
    ) - anchor
    displacement = torch.zeros_like(anchor)
    raw_current = anchor.clone()
    corrections = [torch.zeros_like(anchor) for _ in range(sweeps)]
    defect_maxima = [0.0] * sweeps
    correction_maxima = [0.0] * sweeps
    penultimate_rows = [anchor.clone()]
    final_rows = [anchor.clone()]
    terminal_final_rows = [encode(anchor).clone()]
    terminal_corrections = [torch.zeros_like(anchor)]
    quadratic_rows = []

    for step in range(maximum_horizon):
        displacement = anchor_jacobian_vector(displacement) + affine_defect
        raw_next = anchor + displacement
        reference_current = raw_current
        reference_next = raw_next
        next_corrections = []
        for sweep in range(sweeps):
            correction_current = corrections[sweep]
            if sweep == sweeps - 1:
                mapped, linear_image, quadratic = terminal_map_jvp_quadratic(
                    reference_current, correction_current
                )
                terminal_input_next = reference_next
                terminal_reference_next = encode(reference_next)
                quadratic_rows.append(quadratic)
                defect = mapped - terminal_reference_next
            else:
                mapped, linear_image = map_and_jacobian_vector(
                    reference_current, correction_current
                )
                defect = mapped - reference_next
            next_correction = linear_image + defect
            correction_norm = float(torch.linalg.vector_norm(next_correction))
            if (
                not bool(torch.isfinite(next_correction).all())
                or correction_norm > numeric_cap
            ):
                raise RuntimeError(
                    f"terminal quadratic sweep {sweep + 1} failed at step {step + 1}"
                )
            defect_maxima[sweep] = max(
                defect_maxima[sweep], float(torch.linalg.vector_norm(defect))
            )
            correction_maxima[sweep] = max(
                correction_maxima[sweep], correction_norm
            )
            next_corrections.append(next_correction)
            if sweep == sweeps - 1:
                center_next_terminal = terminal_reference_next + next_correction
                penultimate_rows.append(terminal_input_next)
                terminal_final_rows.append(center_next_terminal)
                final_rows.append(decode(center_next_terminal))
                terminal_corrections.append(next_correction)
            else:
                center_current = reference_current + correction_current
                center_next = reference_next + next_correction
                reference_current = center_current
                reference_next = center_next
        corrections = next_corrections
        raw_current = raw_next

    diagnostics = [
        {
            "sweep": index + 1,
            "reached_horizon": maximum_horizon,
            "maximum_uncorrected_defect_norm": defect_maxima[index],
            "maximum_correction_norm": correction_maxima[index],
            "hvp_calls": maximum_horizon,
        }
        for index in range(sweeps)
    ]
    return {
        "penultimate": torch.stack(penultimate_rows),
        "final": torch.stack(final_rows),
        "terminal_final": torch.stack(terminal_final_rows),
        "terminal_correction": torch.stack(terminal_corrections),
        "quadratic": torch.stack(quadratic_rows),
        "diagnostics": diagnostics,
    }


def build_streaming_transformer_terminal_quadratic(
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
) -> dict:
    """Transformer wrapper for the one-pass terminal-quadratic release."""

    anchor = torch.cat((parameter, velocity))
    dimension = int(parameter.numel())
    eta = float(config.learning_rate)

    def encode(state: Tensor) -> Tensor:
        return to_scaled(state, dimension, eta)

    def decode(state: Tensor) -> Tensor:
        return torch.cat((state[..., :dimension], state[..., dimension:] / eta), dim=-1)

    def map_step(state: Tensor) -> Tensor:
        return optimizer_map(
            state, train_pairs, train_labels, template, spec, config
        )

    def fused(center: Tensor, direction: Tensor) -> tuple[Tensor, Tensor]:
        return optimizer_map_and_jvp(
            center,
            direction,
            train_pairs,
            train_labels,
            template,
            spec,
            config,
        )

    def terminal(center: Tensor, direction: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return scaled_optimizer_map_jvp_quadratic(
            center,
            direction,
            train_pairs,
            train_labels,
            template,
            spec,
            config,
        )

    anchor_map, anchor_jvp = replayable_anchor_optimizer(
        anchor,
        train_pairs,
        train_labels,
        template,
        spec,
        config,
    )
    result = streaming_terminal_quadratic_reference(
        anchor,
        map_step,
        anchor_jvp,
        fused,
        terminal,
        maximum_horizon=maximum_horizon,
        sweeps=sweeps,
        numeric_cap=numeric_cap,
        anchor_map_value=anchor_map,
        terminal_encode=encode,
        terminal_decode=decode,
    )
    scaled_terminal_correction = result["terminal_correction"]
    scaled_final = result["terminal_final"]
    result.update(
        {
            "scaled_penultimate": to_scaled(
                result["penultimate"], dimension, config.learning_rate
            ),
            "scaled_final": scaled_final,
            "scaled_terminal_correction": scaled_terminal_correction,
            "terminal_correction": decode(scaled_terminal_correction),
        }
    )
    return result


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
    fused_derivatives: bool = False,
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

    fused = None
    anchor_map_value = None
    anchor_jvp = lambda direction: jvp(anchor, direction)
    if fused_derivatives:
        anchor_map_value, anchor_jvp = replayable_anchor_optimizer(
            anchor,
            train_pairs,
            train_labels,
            template,
            spec,
            config,
        )

        def fused(center: Tensor, direction: Tensor) -> tuple[Tensor, Tensor]:
            return optimizer_map_and_jvp(
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
        anchor_jvp,
        maximum_horizon=maximum_horizon,
        sweeps=sweeps,
        numeric_cap=numeric_cap,
        stop_when=stop_when,
        map_and_jacobian_vector=fused,
        anchor_map_value=anchor_map_value,
    )
    dimension = int(parameter.numel())
    scaled = to_scaled(center, dimension, config.learning_rate)
    return {
        "map_step": map_step,
        "center": center,
        "scaled_center": scaled,
        "diagnostics": diagnostics,
        "horizon_reached": len(center) - 1,
        "fused_derivatives": bool(fused_derivatives),
        "centerline_sha256": hashlib.sha256(
            scaled.detach().cpu().contiguous().numpy().tobytes(order="C")
        ).hexdigest().upper(),
    }
