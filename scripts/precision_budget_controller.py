#!/usr/bin/env python3
"""Local residual budgets for a certificate-preserving Gram norm cap."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


def _finite_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


@dataclass(frozen=True)
class OperatorCapBudget:
    """A deterministic supersolution budget for one committed Gram block.

    ``target_squared_norm`` is the desired cap on ``||T||^2``.  Residual
    allowances are stored after multiplication by their theorem weights, so
    every Gram step can be checked independently.
    """

    target_squared_norm: float
    calibration_lower: float
    terminal_allowance: float
    residual_contribution_allowances: tuple[float, ...]

    def __post_init__(self) -> None:
        target = _finite_nonnegative(
            "target_squared_norm", self.target_squared_norm
        )
        calibration = _finite_nonnegative(
            "calibration_lower", self.calibration_lower
        )
        terminal = _finite_nonnegative("terminal_allowance", self.terminal_allowance)
        contributions = tuple(
            _finite_nonnegative(f"residual allowance {index}", value)
            for index, value in enumerate(self.residual_contribution_allowances)
        )
        if target <= 0.0:
            raise ValueError("target_squared_norm must be positive")
        if calibration <= 0.0:
            raise ValueError("calibration_lower must be positive")
        if not contributions:
            raise ValueError("at least one residual allowance is required")
        object.__setattr__(self, "target_squared_norm", target)
        object.__setattr__(self, "calibration_lower", calibration)
        object.__setattr__(self, "terminal_allowance", terminal)
        object.__setattr__(self, "residual_contribution_allowances", contributions)
        if self.total_allocated > self.available_total * (1.0 + 8.0e-15):
            raise ValueError("allocated terminal/residual budget exceeds Gram cap")

    @property
    def power(self) -> int:
        return len(self.residual_contribution_allowances)

    @property
    def available_total(self) -> float:
        return self.calibration_lower * self.target_squared_norm**self.power

    @property
    def total_allocated(self) -> float:
        return self.terminal_allowance + sum(
            self.residual_contribution_allowances
        )

    @property
    def design_slack(self) -> float:
        return self.available_total - self.total_allocated

    def residual_norm_allowances(self) -> tuple[float, ...]:
        target = self.target_squared_norm
        q = self.power
        return tuple(
            contribution / target ** (q - 1 - index)
            for index, contribution in enumerate(
                self.residual_contribution_allowances
            )
        )

    def check(
        self,
        *,
        terminal_norm_upper: float,
        residual_norm_uppers: Iterable[float],
    ) -> dict:
        terminal = _finite_nonnegative("terminal_norm_upper", terminal_norm_upper)
        residuals = tuple(
            _finite_nonnegative(f"residual upper {index}", value)
            for index, value in enumerate(residual_norm_uppers)
        )
        if len(residuals) != self.power:
            raise ValueError("residual count must equal the inspected Gram power")
        allowances = self.residual_norm_allowances()
        terminal_margin = self.terminal_allowance - terminal
        residual_margins = tuple(
            allowance - residual
            for allowance, residual in zip(allowances, residuals)
        )
        passed = terminal_margin >= 0.0 and all(
            margin >= 0.0 for margin in residual_margins
        )
        return {
            "passed": passed,
            "certified_operator_norm_upper": (
                math.sqrt(self.target_squared_norm) if passed else None
            ),
            "terminal_margin": terminal_margin,
            "residual_margins": residual_margins,
            "design_slack": self.design_slack,
            "residual_norm_allowances": allowances,
        }


def equal_residual_contribution_budget(
    *,
    target_squared_norm: float,
    calibration_lower: float,
    terminal_allowance: float,
    power: int,
    spend_fraction: float = 0.99,
) -> OperatorCapBudget:
    """Allocate equal weighted residual contributions below a norm cap."""
    if int(power) != power or power < 1:
        raise ValueError("power must be a positive integer")
    if not math.isfinite(spend_fraction) or not 0.0 <= spend_fraction <= 1.0:
        raise ValueError("spend_fraction must lie in [0,1]")
    target = _finite_nonnegative("target_squared_norm", target_squared_norm)
    calibration = _finite_nonnegative("calibration_lower", calibration_lower)
    terminal = _finite_nonnegative("terminal_allowance", terminal_allowance)
    if target <= 0.0 or calibration <= 0.0:
        raise ValueError("target and calibration must be positive")
    remaining = calibration * target**int(power) - terminal
    if remaining < 0.0:
        raise ValueError("terminal allowance alone exceeds the target cap")
    each = spend_fraction * remaining / int(power)
    return OperatorCapBudget(
        target,
        calibration,
        terminal,
        tuple(each for _ in range(int(power))),
    )
