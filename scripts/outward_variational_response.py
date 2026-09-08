"""Dimension-independent outward step for the signed momentum response.

The gradient and HVP arguments must already enclose derivatives of the exact
objective at the reference parameter, in direction z_theta. The returned
float64 next response has a proved recurrence residual norm upper bound.
"""
import math
import numpy as np
from flint import arb


def upper_float(value):
    return math.nextafter(float(value.upper()),math.inf)


def norm_upper(balls):
    return upper_float(sum((abs(v).upper()**2 for v in balls),arb(0)).sqrt())


def signed_momentum_step(current,next_reference,response,gradient,hvp,*,learning_rate,momentum):
    current=np.asarray(current,dtype=np.float64)
    next_reference=np.asarray(next_reference,dtype=np.float64)
    response=np.asarray(response,dtype=np.float64)
    if current.ndim!=1 or current.size%2 or current.size==0:
        raise ValueError("even, positive scaled-state dimension required")
    n=current.size//2
    if next_reference.shape!=current.shape or response.shape!=current.shape:
        raise ValueError("reference/response shape mismatch")
    if len(gradient)!=n or len(hvp)!=n or not all(np.isfinite(v).all() for v in (current,next_reference,response)):
        raise ValueError("invalid state or derivative geometry")
    if not all(math.isfinite(v) for v in (learning_rate,momentum)):
        raise ValueError("finite optimizer scalars required")
    eta,mu=arb(float(learning_rate)),arb(float(momentum))
    next_balls=[];velocity_balls=[];defects=[];wdefects=[]
    for i in range(n):
        if not gradient[i].is_finite() or not hvp[i].is_finite():
            raise ValueError("nonfinite derivative enclosure")
        r=mu*arb(float(current[n+i]))+eta*gradient[i]
        dr=mu*arb(float(response[n+i]))+eta*hvp[i]
        st=arb(float(current[i]))-r-arb(float(next_reference[i]))
        sw=r-arb(float(next_reference[n+i]))
        next_balls.append(arb(float(response[i]))-dr+st)
        velocity_balls.append(dr+sw)
        defects.append(st);wdefects.append(sw)
    next_balls+=velocity_balls;defects+=wdefects
    rounded=np.array([float(v.mid()) for v in next_balls])
    residual=[arb(float(v))-ball for v,ball in zip(rounded,next_balls)]
    result={"next_response":rounded,"recurrence_error_upper":norm_upper(residual),
            "defect_norm_upper":norm_upper(defects),
            "response_norm_upper":norm_upper([arb(float(v)) for v in rounded]),
            "parameter_response_norm_upper":norm_upper([arb(float(v)) for v in rounded[:n]])}
    return result
