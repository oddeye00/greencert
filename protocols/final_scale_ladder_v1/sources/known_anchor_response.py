"""Encode a known scaled-momentum anchor offset without discarding its tail.

The result is only initial data for a separately verified homogeneous
variational response. Encoding alone never removes the original gamma term
from an existing certificate.
"""
from fractions import Fraction
import math

import numpy as np
from flint import arb,ctx


def require(condition,message):
    if not condition:
        raise ValueError(message)


def rational_norm_upper(square,precision_bits=256):
    require(isinstance(square,Fraction) and square >= 0,"nonnegative exact square required")
    require(type(precision_bits) is int and precision_bits >= 64,"at least64 bits required")
    if square == 0:
        return 0.0
    old=ctx.prec;ctx.prec=precision_bits
    try:
        value=(arb(square.numerator)/arb(square.denominator)).sqrt()
        upper=math.nextafter(float(value.upper()),math.inf)
        require(math.isfinite(upper),"anchor norm overflow")
        return upper
    finally:
        ctx.prec=old


def encode_velocity_offset(velocity,reference_scaled_velocity,*,learning_rate,momentum):
    """Return dbar approximating eta*v-wref and an exact-rational tail bound.

    Every coordinate is checked against an exact rational expression. An
    exactly zero tail is returned only if every residual is exactly zero.
    Subnormal/underflow cases remain nonzero when not representable.
    """
    v,w=velocity,reference_scaled_velocity
    require(isinstance(v,np.ndarray) and isinstance(w,np.ndarray) and v.dtype == w.dtype == np.float64 and
            v.ndim == w.ndim == 1 and v.shape == w.shape and len(v)>0 and
            np.isfinite(v).all() and np.isfinite(w).all(),"finite aligned binary64 velocities required")
    require(type(learning_rate) in (int,float) and math.isfinite(learning_rate) and learning_rate>0 and
            type(momentum) in (int,float) and math.isfinite(momentum) and momentum>=0,
            "invalid optimizer constants")
    eta,mu=Fraction(learning_rate),Fraction(momentum)
    encoded=np.empty_like(v)
    offset_square=tail_square=Fraction(0)
    nonzero=nonrepresentable=0
    for i,(actual,stored) in enumerate(zip(v,w)):
        delta=eta*Fraction(float(actual))-Fraction(float(stored))
        try: center=float(delta)
        except OverflowError: raise ValueError("anchor offset overflow") from None
        require(math.isfinite(center),"nonfinite anchor encoding")
        tail=delta-Fraction(center)
        encoded[i]=center
        offset_square+=delta*delta
        if tail: tail_square+=tail*tail
        nonzero+=int(delta != 0)
        nonrepresentable+=int(tail != 0)
    return {"encoded_velocity_offset":encoded,"nonzero_offset_coordinates":nonzero,
        "nonrepresentable_offset_coordinates":nonrepresentable,
        "all_offset_coordinates_exactly_encoded":nonrepresentable == 0,
        "original_offset_norm_upper":rational_norm_upper(offset_square),
        "unrepresented_velocity_offset_norm_upper":rational_norm_upper(tail_square),
        "original_first_injection_norm_upper":rational_norm_upper(2*mu*mu*offset_square),
        "unrepresented_first_injection_norm_upper":rational_norm_upper(2*mu*mu*tail_square),
        "offset_sign":"eta*original_unscaled_velocity - reference_scaled_velocity",
        "homogeneous_response_computed":False,"original_certificate_gamma_replaced":False,
        "event_certificate_issued":False}
