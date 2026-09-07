"""Response-centered scalar closure without lossy input coercion.

Accept exact Python int/Fraction and finite Python float (as its exact dyadic
value). Return a verified binary64 supersolution, or abstain. This is a new
conditional scalar interface, not a replacement for any frozen producer.
"""
from fractions import Fraction
import math

from flint import arb, ctx, fmpq


def exact_nonnegative(value, name):
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"nonfinite {name}")
        result = Fraction.from_float(value)
    elif type(value) in (int, Fraction):
        result = Fraction(value)
    else:
        raise ValueError(f"unsupported exact input type for {name}")
    if result < 0:
        raise ValueError(f"negative {name}")
    return result


def enclosed(value):
    return arb(fmpq(value.numerator, value.denominator))


def float_upper(value):
    if bool(value == 0):
        return 0.0
    try:
        upper = math.nextafter(float(value.upper()), math.inf)
    except OverflowError:
        return None
    return upper if math.isfinite(upper) else None


def solve(*, gain, drift_by_input, parameter_response_norms, response_residual,
          first_injection_error, domain, precision_bits=256):
    if type(precision_bits) is not int or precision_bits < 64:
        raise ValueError("integer precision >=64 required")
    if not isinstance(parameter_response_norms, (tuple, list)) or not isinstance(drift_by_input, (tuple, list)):
        raise ValueError("explicit finite horizon sequences required")
    k = exact_nonnegative(gain, "Green gain")
    tau = exact_nonnegative(response_residual, "response residual")
    gamma = exact_nonnegative(first_injection_error, "first anchor injection error")
    rho = exact_nonnegative(domain, "verified domain")
    p = [exact_nonnegative(v, "parameter response norm") for v in parameter_response_norms]
    m = [exact_nonnegative(v, "input derivative drift") for v in drift_by_input]
    if len(p) < 2 or len(m) != len(p) - 2 or p[0] != 0:
        raise ValueError("complete horizon with exact parameter anchor required")
    # Exact comparisons are necessary: overlapping Arb intervals have no
    # total order, so max of approximate intervals is not a certified maximum.
    cross = max((a*b for a, b in zip(m, p[1:-1], strict=True)), default=Fraction(0))
    previous = ctx.prec
    ctx.prec = precision_bits
    try:
        K = enclosed(k)
        P = enclosed(max(p))
        q_squared = sum((enclosed(a*b*b/2)**2 for a, b in zip(m, p[1:-1], strict=True)), arb(0))
        alpha = enclosed(k*(tau+gamma)).upper()
        forcing = (K*q_squared.sqrt()).upper()
        Y = (alpha+forcing).upper()
        b = enclosed(k*cross).upper()
        B = enclosed(k*max(m, default=Fraction(0))).upper()
        base = {
            "schema": "exact_input_scalar_closure_v1",
            "conditional_scalar_only": True,
            "artifact_authentication_performed": False,
            "event_certificate_issued": False,
            "input_semantics": "exact int/Fraction or exact dyadic value of finite float",
            "horizon": len(p)-1,
            "first_injection_error_exact": {
                "numerator_hex": hex(gamma.numerator),
                "denominator_hex": hex(gamma.denominator),
            },
            "coefficient_enclosures": {
                "alpha": str(alpha), "nonlinear_forcing": str(forcing),
                "Y": str(Y), "b": str(b), "B": str(B),
            },
        }
        if bool(Y == 0):
            E = 0.0
        else:
            gap = 1-b
            discriminant = gap*gap-2*B*Y
            if not bool(gap > 0) or not bool(discriminant > 0):
                return {**base, "closure": False, "radius": None,
                        "reason": "nonpositive_or_unresolved_discriminant"}
            root = Y/gap if bool(B == 0) else 2*Y/(gap+discriminant.sqrt())
            E = float_upper(root)
            if E is None:
                return {**base, "closure": False, "radius": None,
                        "reason": "no_finite_binary64_radius"}

        def slack(value):
            error = arb(value)  # Every candidate radius is a literal binary64 number.
            return error-Y-b*error-B*error*error/2

        for _ in range(32):
            if bool(slack(E) >= 0):
                break
            E = math.nextafter(E, math.inf)
            if not math.isfinite(E):
                return {**base, "closure": False, "radius": None,
                        "reason": "no_finite_binary64_radius"}
        passed = bool(slack(E) >= 0) and bool(P+arb(E) <= enclosed(rho))
        return {**base, "closure": passed, "radius": E if passed else None,
                "total_parameter_radius_upper": float_upper(P+arb(E)) if passed else None,
                "supersolution_slack": str(slack(E)),
                "reason": None if passed else "domain_or_supersolution_failed"}
    finally:
        ctx.prec = previous
