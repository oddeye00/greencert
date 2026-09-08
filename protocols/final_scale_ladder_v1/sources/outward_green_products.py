"""Causal Green forward/adjoint products with verified local residuals.

The callback hvp(step, enclosed_direction) must enclose the exact Hessian
product at reference step. It must preserve Arb direction intervals. No
Hessian or Green matrix is formed. State storage is O(H*d), not O((H*d)^2).
"""
import math
import numpy as np
from flint import arb,ctx
from outward_green_momentum import adjoint_hvp_direction,rounded_recurrence_step


def streaming_norm_upper(values,chunk_size=65536):
    array=np.asarray(values,dtype=np.float64).reshape(-1)
    if not np.isfinite(array).all():raise ValueError("finite array required")
    square=arb(0)
    for start in range(0,len(array),chunk_size):
        square+=sum((arb(float(x))**2 for x in array[start:start+chunk_size]),arb(0))
    return math.nextafter(float(square.sqrt().upper()),math.inf)


def sequence_upper(bounds):
    if not all(math.isfinite(x) and x>=0 for x in bounds):raise ValueError("invalid residual norm")
    return math.nextafter(float(sum((arb(float(x))**2 for x in bounds),arb(0)).sqrt().upper()),math.inf)


def forward(injections,hvp,*,learning_rate,momentum,precision_bits=128,progress=None):
    values=np.asarray(injections,dtype=np.float64)
    if values.ndim!=2 or min(values.shape)<1 or values.shape[1]%2 or not np.isfinite(values).all():
        raise ValueError("H by even-state-dimension finite injection array required")
    old=ctx.prec;ctx.prec=precision_bits
    try:
        H,d=values.shape;n=d//2;state=np.zeros(d);result=np.empty_like(values);residuals=[]
        for j in range(H):
            if j==0:
                # J0 times exact zero anchor, and assignment of dyadic u0.
                state=values[0].copy();error=0.
            else:
                h=hvp(j,[arb(float(v)) for v in state[:n]])
                row=rounded_recurrence_step(state,values[j],h,learning_rate=learning_rate,momentum=momentum)
                state=row["next_state"];error=row["local_residual_norm_upper"]
            result[j]=state;residuals.append(error)
            if progress is not None:progress(j,error)
        ctx.prec=max(256,precision_bits)
        return {"states":result,"local_residuals":residuals,"residual_sequence_upper":sequence_upper(residuals),
                "terminal_sequence_norm_upper":streaming_norm_upper(result),"hvp_calls":H-1}
    finally:ctx.prec=old


def adjoint(cotangents,hvp,*,learning_rate,momentum,precision_bits=128,progress=None):
    values=np.asarray(cotangents,dtype=np.float64)
    if values.ndim!=2 or min(values.shape)<1 or values.shape[1]%2 or not np.isfinite(values).all():
        raise ValueError("H by even-state-dimension finite cotangent array required")
    old=ctx.prec;ctx.prec=precision_bits
    try:
        H,d=values.shape;result=np.empty_like(values);errors=[0.]*H
        # w_(H-1)=y_H exactly, with no pointless final J0 transpose product.
        state=values[-1].copy();result[-1]=state
        if progress is not None:progress(H-1,0.)
        for j in range(H-2,-1,-1):
            h=hvp(j+1,adjoint_hvp_direction(state))
            row=rounded_recurrence_step(state,values[j],h,learning_rate=learning_rate,momentum=momentum,transpose=True)
            state=row["next_state"];result[j]=state;errors[j]=row["local_residual_norm_upper"]
            if progress is not None:progress(j,errors[j])
        ctx.prec=max(256,precision_bits)
        return {"states":result,"local_residuals":errors,"residual_sequence_upper":sequence_upper(errors),
                "terminal_sequence_norm_upper":streaming_norm_upper(result),"hvp_calls":H-1}
    finally:ctx.prec=old
