"""Outward Green norm solve from LOCAL forward/adjoint recurrence errors.

The unknown gain stays inside the polynomial: no unverified bootstrap gain.
Input bounds and Gaussian calibration must be independently established.
See GREEN_LOCAL_RESIDUAL_INTERFACE.md for the exact identities and indexing.
"""
import math
from flint import arb,ctx
from outward_inexact_anytime_gram import exact_arb


def checked(value,name,positive=False):
    if value is None:raise ValueError(f"missing {name}")
    result=float(value)
    if not math.isfinite(result) or result<0 or (positive and result==0):
        raise ValueError(f"invalid {name}")
    return result


def direct_image_upper(*,terminal_upper,calibration_lower,forward_residual_upper,precision_bits=256):
    y=checked(terminal_upper,"terminal norm")
    c=checked(calibration_lower,"calibration",True)
    d=checked(forward_residual_upper,"forward residual")
    old=ctx.prec;ctx.prec=precision_bits
    try:
        gap=exact_arb(c)-exact_arb(d)
        if not bool(gap>0):return {"enclosed":False,"reason":"forward_residual_exhausts_calibration"}
        if y==0:upper=0.
        else:upper=math.nextafter(float((exact_arb(y)/gap).upper()),math.inf)
        slack=gap*exact_arb(upper)-exact_arb(y)
        assert bool(slack>=0)
        return {"enclosed":True,"operator_upper":upper,"supersolution_slack":str(slack),
                "conditional_on_verified_inputs_and_projection_event":True}
    finally:ctx.prec=old


def local_residual_slack(operator,*,shift,terminal_upper,calibration_lower,
                         forward_residuals,adjoint_residuals):
    """At lambda=operator^2, c minus the decreasing residual right side."""
    k=exact_arb(operator);lam=k*k;mu=exact_arb(shift)
    q=len(forward_residuals)
    if q<1 or len(adjoint_residuals)!=q:raise ValueError("residual sequences mismatch")
    if not bool(lam>mu):raise ValueError("strictly positive spectral gap required")
    d=[exact_arb(x) for x in forward_residuals];e=[exact_arb(x) for x in adjoint_residuals]
    rhs=(exact_arb(terminal_upper)+lam*d[-1]+k*e[-1])/(lam**(q-1)*(lam-mu))
    for l in range(q-1):rhs+=d[l]/lam**l+e[l]/(k*lam**l)
    return exact_arb(calibration_lower)-rhs


def gram_operator_upper(*,shift,terminal_upper,calibration_lower,
                        forward_residuals,adjoint_residuals,precision_bits=256):
    mu=checked(shift,"shift");y=checked(terminal_upper,"terminal norm")
    c=checked(calibration_lower,"calibration",True)
    d=tuple(checked(x,"forward residual") for x in forward_residuals)
    e=tuple(checked(x,"adjoint residual") for x in adjoint_residuals)
    if not d or len(d)!=len(e):raise ValueError("one forward/adjoint bound per Gram application required")
    if d[0]>=c:return {"enclosed":False,"reason":"first_forward_residual_exhausts_calibration"}
    old=ctx.prec;ctx.prec=precision_bits
    try:
        # If no nonconstant term survives, the inequality is impossible above
        # mu. Return an outward sqrt(mu), including the exact zero case.
        if y==0 and not any(e) and not any(d[1:]) and (mu==0 or not any(d)):
            upper=0. if mu==0 else math.nextafter(math.sqrt(mu),math.inf)
            while not bool(exact_arb(upper)**2>=exact_arb(mu)):
                upper=math.nextafter(upper,math.inf)
            return {"enclosed":True,"operator_upper":upper,"reason":"no_nonconstant_term",
                    "conditional_on_verified_inputs_and_projection_event":True}
        def admissible(k):
            if not bool(exact_arb(k)**2>exact_arb(mu)):return False
            return bool(local_residual_slack(k,shift=mu,terminal_upper=y,calibration_lower=c,
                forward_residuals=d,adjoint_residuals=e)>=0)
        lo=0.;upper=max(1.,math.nextafter(math.sqrt(mu),math.inf))
        while not admissible(upper):
            upper*=2
            if not math.isfinite(upper):raise OverflowError("no finite supersolution")
        for _ in range(320):
            mid=lo+(upper-lo)/2
            if mid==lo or mid==upper:break
            if admissible(mid):upper=mid
            else:lo=mid
        slack=local_residual_slack(upper,shift=mu,terminal_upper=y,calibration_lower=c,
            forward_residuals=d,adjoint_residuals=e)
        assert bool(slack>=0)
        return {"enclosed":True,"operator_upper":upper,"supersolution_slack":str(slack),
                "conditional_on_verified_inputs_and_projection_event":True}
    finally:ctx.prec=old
