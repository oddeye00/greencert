"""Outward response-centered closure with explicit scaled-anchor conversion.

All supplied gain/derivative/response bounds require independent verification.
This scalar calculation alone never establishes a neural event certificate.
"""
import math
from flint import arb,ctx


def checked(value,name):
    if value is None:raise ValueError(f"missing {name}")
    result=float(value)
    if not math.isfinite(result) or result<0:raise ValueError(f"invalid {name}")
    return result


def solve(*,gain,drift_by_input,parameter_response_norms,response_residual,
          first_injection_error,domain,precision_bits=256):
    """H-1 drift bounds at1..H-1; H+1 response norms at0..H, starting at0."""
    k=checked(gain,"Green gain");tau=checked(response_residual,"response residual")
    gamma=checked(first_injection_error,"first anchor injection error")
    rho=checked(domain,"verified parameter domain")
    p=[checked(v,"parameter response norm") for v in parameter_response_norms]
    m=[checked(v,"input derivative drift") for v in drift_by_input]
    if len(p)<2 or len(m)!=len(p)-2:raise ValueError("complete horizon geometry required")
    if p[0]!=0:raise ValueError("parameter anchor must agree exactly")
    old=ctx.prec;ctx.prec=precision_bits
    try:
        K=arb(k);P=arb(max(p));M=arb(max(m,default=0.))
        cross=max((arb(a)*arb(b) for a,b in zip(m,p[1:-1])),default=arb(0)).upper()
        q2=sum(((arb(a)*arb(b)**2/2)**2 for a,b in zip(m,p[1:-1])),arb(0))
        alpha=(K*(arb(tau)+arb(gamma))).upper()
        Y=(alpha+K*q2.sqrt()).upper();b=(K*cross).upper();B=(K*M).upper()
        def as_upper(v):return math.nextafter(float(v.upper()),math.inf) if not bool(v==0) else 0.
        base={"conditional_scalar_only":True,"horizon":len(p)-1,
              "anchor_error_included":True,"first_injection_error":gamma,
              "alpha_upper":as_upper(alpha),"nonlinear_forcing_upper":as_upper(K*q2.sqrt()),
              "linear_remainder_coefficient_upper":as_upper(b),
              "quadratic_remainder_coefficient_upper":as_upper(B/2)}
        if bool(Y==0):E=0.
        else:
            gap=1-b;discriminant=gap*gap-2*B*Y
            if not bool(gap>0) or not bool(discriminant>0):
                return {**base,"closure":False,"radius":None,"reason":"nonpositive_or_unresolved_discriminant"}
            root=Y/gap if bool(B==0) else 2*Y/(gap+discriminant.sqrt())
            E=as_upper(root)
        def slack(e):
            v=arb(e);return v-Y-b*v-B*v*v/2
        for _ in range(32):
            if bool(slack(E)>=0):break
            E=math.nextafter(E,math.inf)
        passed=bool(slack(E)>=0) and bool(P+arb(E)<=arb(rho))
        return {**base,"closure":passed,"radius":E if passed else None,
                "total_parameter_radius_upper":as_upper(P+arb(E)) if passed else None,
                "supersolution_slack":str(slack(E)),
                "reason":None if passed else "domain_or_supersolution_failed"}
    finally:ctx.prec=old
