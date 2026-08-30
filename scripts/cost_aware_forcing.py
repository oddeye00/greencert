#!/usr/bin/env python3
"""Residual-aware forcing bounds for corrected-path Green closure."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


def _finite_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


@dataclass(frozen=True)
class CostAwareForcing:
    kappa: float
    surrogate_injection_norm: float
    surrogate_error_norm: float
    norm_only_response_upper: float
    direct_response_norm: float | None
    direct_response_recurrence_residual_norm: float | None
    response_aware_upper: float | None
    selected_response_upper: float
    selected_method: str

    def as_dict(self) -> dict:
        return asdict(self)


def cost_aware_forcing_upper(
    *,
    kappa: float,
    surrogate_injection_norm: float,
    surrogate_error_norm: float,
    direct_response_norm: float | None = None,
    direct_response_recurrence_residual_norm: float | None = None,
) -> CostAwareForcing:
    """Bound ``||K s||`` and choose the tighter available release.

    If ``||s-q|| <= sigma``, the norm-only release is

        ``kappa * (||q|| + sigma)``.

    If a computed response ``z`` has recurrence residual ``r`` relative to
    ``z = K q``, then ``||Kq-z|| <= kappa*||r||`` and the response-aware
    release is

        ``||z|| + kappa * (sigma + ||r||)``.

    The minimum of two valid upper bounds is valid.  Passing neither or only
    one of the two direct-response quantities is rejected.
    """

    kappa = _finite_nonnegative("kappa", kappa)
    q_norm = _finite_nonnegative(
        "surrogate_injection_norm", surrogate_injection_norm
    )
    error = _finite_nonnegative("surrogate_error_norm", surrogate_error_norm)
    norm_only = kappa * (q_norm + error)
    if not math.isfinite(norm_only):
        raise ValueError("norm-only forcing bound overflowed")

    if (direct_response_norm is None) != (
        direct_response_recurrence_residual_norm is None
    ):
        raise ValueError("direct response norm and residual must be supplied together")

    response_upper = None
    response_norm = None
    response_residual = None
    selected = norm_only
    method = "norm_only"
    if direct_response_norm is not None:
        response_norm = _finite_nonnegative(
            "direct_response_norm", direct_response_norm
        )
        response_residual = _finite_nonnegative(
            "direct_response_recurrence_residual_norm",
            direct_response_recurrence_residual_norm,
        )
        response_upper = response_norm + kappa * (error + response_residual)
        if not math.isfinite(response_upper):
            raise ValueError("response-aware forcing bound overflowed")
        if response_upper < selected:
            selected = response_upper
            method = "direct_response"

    return CostAwareForcing(
        kappa=kappa,
        surrogate_injection_norm=q_norm,
        surrogate_error_norm=error,
        norm_only_response_upper=norm_only,
        direct_response_norm=response_norm,
        direct_response_recurrence_residual_norm=response_residual,
        response_aware_upper=response_upper,
        selected_response_upper=selected,
        selected_method=method,
    )
