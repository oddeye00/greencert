"""Outward local JVP/VJP recurrence arithmetic for scaled momentum.

Hessian products are supplied as verified intervals at the exact reference.
Forward products require H*state_theta. Adjoint products require
H*(state_velocity-state_theta), with the subtraction itself enclosed.
These helpers do not call an unverified numerical HVP or infer its error.
"""
import math
import numpy as np
from flint import arb


def vector(values):
    array=np.asarray(values,dtype=np.float64)
    if array.ndim!=1 or len(array)==0 or len(array)%2 or not np.isfinite(array).all():
        raise ValueError("finite nonempty even-dimensional state required")
    return array


def norm_upper(values):
    value=sum((abs(v).upper()**2 for v in values),arb(0)).sqrt()
    if not value.is_finite():raise ValueError("nonfinite residual enclosure")
    return math.nextafter(float(value.upper()),math.inf)


def adjoint_hvp_direction(state):
    """Exact dyadic subtraction as Arb; never cast this result back to float."""
    state=vector(state);n=len(state)//2
    return [arb(float(state[n+k]))-arb(float(state[k])) for k in range(n)]


def momentum_image(state,hvp,*,learning_rate,momentum,transpose=False):
    state=vector(state);n=len(state)//2
    if len(hvp)!=n or not all(isinstance(v,arb) and v.is_finite() for v in hvp):
        raise ValueError("one finite verified HVP enclosure per parameter required")
    if not all(math.isfinite(float(v)) for v in (learning_rate,momentum)):
        raise ValueError("finite optimizer constants required")
    eta,mu=arb(float(learning_rate)),arb(float(momentum))
    left=[];right=[]
    for k in range(n):
        a,b=arb(float(state[k])),arb(float(state[n+k]))
        if transpose:
            left.append(a+eta*hvp[k]);right.append(mu*(b-a))
        else:
            dr=mu*b+eta*hvp[k]
            left.append(a-dr);right.append(dr)
    return left+right


def rounded_recurrence_step(state,injection,hvp,*,learning_rate,momentum,transpose=False):
    state=vector(state);injection=vector(injection)
    if injection.shape!=state.shape:raise ValueError("injection/state shape mismatch")
    images=momentum_image(state,hvp,learning_rate=learning_rate,momentum=momentum,transpose=transpose)
    images=[a+arb(float(u)) for a,u in zip(images,injection)]
    rounded=np.array([float(v.mid()) for v in images])
    if not np.isfinite(rounded).all():raise ValueError("rounded recurrence overflow")
    residual=[arb(float(v))-exact for v,exact in zip(rounded,images)]
    return {"next_state":rounded,"local_residual_norm_upper":norm_upper(residual),
            "next_state_norm_upper":norm_upper([arb(float(v)) for v in rounded])}


def supplied_recurrence_residual(state,injection,next_state,hvp,*,learning_rate,momentum,transpose=False):
    """Audit already-computed states, including their actual rounding error."""
    state=vector(state);injection=vector(injection);next_state=vector(next_state)
    if state.shape!=injection.shape or state.shape!=next_state.shape:raise ValueError("state shapes differ")
    image=momentum_image(state,hvp,learning_rate=learning_rate,momentum=momentum,transpose=transpose)
    return norm_upper([arb(float(z))-a-arb(float(u)) for z,a,u in zip(next_state,image,injection)])
