#!/usr/bin/env python3
"""Stateful one-power-at-a-time Gaussian Gram bounds."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from probe_jacobian_bound import ProbeConfig


@dataclass
class OnlineGramState:
    """One committed probe block that can be resumed power by power."""

    vectors: Tensor
    initial_norms: Tensor
    calibration: float
    config: ProbeConfig
    seed: int
    power: int = 0
    cumulative_operator_seconds: float = 0.0

    @classmethod
    def initialize(
        cls,
        *,
        dimension: int,
        dtype: torch.dtype,
        device: torch.device,
        config: ProbeConfig,
        seed: int,
    ) -> "OnlineGramState":
        if dimension < 1:
            raise ValueError("dimension must be positive")
        generator = torch.Generator(device=device).manual_seed(int(seed))
        vectors = torch.stack(
            [
                torch.randn(
                    dimension,
                    generator=generator,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(config.probes)
            ]
        )
        return cls(
            vectors=vectors,
            initial_norms=torch.linalg.vector_norm(vectors, dim=1),
            calibration=config.c_delta(),
            config=config,
            seed=int(seed),
        )

    @torch.no_grad()
    def step(self, apply_gram: Callable[[Tensor], Tensor]) -> dict:
        if self.power >= self.config.power:
            raise RuntimeError("online Gram state exceeded its frozen maximum power")
        started = time.perf_counter()
        self.vectors = apply_gram(self.vectors)
        self.cumulative_operator_seconds += time.perf_counter() - started
        self.power += 1
        final_norms = torch.linalg.vector_norm(self.vectors, dim=1)
        best = float(final_norms.max())
        valid = (self.initial_norms > 0.0) & (final_norms > 0.0)
        if bool(valid.any()):
            ratios = final_norms[valid] / self.initial_norms[valid]
            lower = float(ratios.max() ** (1.0 / (2.0 * self.power)))
        else:
            lower = 0.0
        upper = (
            0.0
            if best <= 0.0
            else (best / self.calibration) ** (1.0 / (2.0 * self.power))
        )
        return {
            "power": self.power,
            "Y": best,
            "c_delta": self.calibration,
            "operator_norm_upper_bound": upper,
            "operator_norm_lower_estimate": lower,
            "logical_gram_applications": self.config.probes * self.power,
            "batched_gram_calls": self.power,
            "cumulative_operator_seconds": self.cumulative_operator_seconds,
        }

    @property
    def allocated_bytes(self) -> int:
        return self.vectors.numel() * self.vectors.element_size()
