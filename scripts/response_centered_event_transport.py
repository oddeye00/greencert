#!/usr/bin/env python3
"""Response-centered observable and classification-margin error bounds.

The GreenCert state theorem encloses the unknown remainder ``e`` around the
known signed response ``z``.  Output transport should use the same center:
evaluate the observable at ``c + z`` and charge derivatives only for ``e``.
This module implements the resulting scalar bounds independently of any model.
"""

from __future__ import annotations

import math


def _finite_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative, got {value}")
    return value


def observable_remainder_radius(
    *,
    center_gradient_upper: float,
    hessian_upper: float,
    response_norm: float,
    remainder_radius: float,
) -> float:
    """Bound an observable around ``c + z`` using data centered at ``c``.

    If ``||Dm(c)|| <= A``, ``||D2m|| <= B`` on the containing ball,
    ``||z|| <= d``, and ``||e|| <= E``, then

        |m(c + z + e) - m(c + z)|
            <= (A + B d) E + (B/2) E^2.
    """

    gradient = _finite_nonnegative("center_gradient_upper", center_gradient_upper)
    hessian = _finite_nonnegative("hessian_upper", hessian_upper)
    response = _finite_nonnegative("response_norm", response_norm)
    remainder = _finite_nonnegative("remainder_radius", remainder_radius)
    return (gradient + hessian * response) * remainder + 0.5 * hessian * remainder**2


def classification_margin_remainder_radius(
    *,
    output_jacobian_upper: float,
    output_hessian_upper: float,
    response_norm: float,
    remainder_radius: float,
) -> float:
    """Bound a true-class-minus-competitor margin around ``c + z``.

    The supplied derivative bounds are for the vector-valued logit map.  The
    norm of a difference of two output coordinates contributes ``sqrt(2)``.
    """

    return math.sqrt(2.0) * observable_remainder_radius(
        center_gradient_upper=output_jacobian_upper,
        hessian_upper=output_hessian_upper,
        response_norm=response_norm,
        remainder_radius=remainder_radius,
    )


def classification_margin_origin_radius(
    *,
    output_jacobian_upper: float,
    output_hessian_upper: float,
    total_radius: float,
) -> float:
    """The matched origin-centered output-margin radius."""

    jacobian = _finite_nonnegative("output_jacobian_upper", output_jacobian_upper)
    hessian = _finite_nonnegative("output_hessian_upper", output_hessian_upper)
    radius = _finite_nonnegative("total_radius", total_radius)
    return math.sqrt(2.0) * (jacobian * radius + 0.5 * hessian * radius**2)


def response_centering_dominates_origin_radius(
    *,
    output_jacobian_upper: float,
    output_hessian_upper: float,
    response_norm: float,
    response_max_norm: float,
    remainder_radius: float,
) -> bool:
    """Check the analytic radius dominance when ``response_norm <= p``."""

    response = _finite_nonnegative("response_norm", response_norm)
    maximum = _finite_nonnegative("response_max_norm", response_max_norm)
    if response > maximum:
        raise ValueError("response_norm cannot exceed response_max_norm")
    recentered = classification_margin_remainder_radius(
        output_jacobian_upper=output_jacobian_upper,
        output_hessian_upper=output_hessian_upper,
        response_norm=response,
        remainder_radius=remainder_radius,
    )
    origin = classification_margin_origin_radius(
        output_jacobian_upper=output_jacobian_upper,
        output_hessian_upper=output_hessian_upper,
        total_radius=maximum + remainder_radius,
    )
    return recentered <= origin
