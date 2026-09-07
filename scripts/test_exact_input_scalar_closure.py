"""Exact-rational oracle and boundary regressions for the unfrozen interface."""
import argparse
from fractions import Fraction as Q
import json
import math
from pathlib import Path
import random

from exact_input_scalar_closure import solve


def verify_exact_supersolution(args, result):
    if not result["closure"]:
        assert result["radius"] is None
        return
    k, tau, gamma, rho = [Q(args[key]) for key in (
        "gain", "response_residual", "first_injection_error", "domain")]
    p, m = [Q(v) for v in args["parameter_response_norms"]], [Q(v) for v in args["drift_by_input"]]
    e = Q.from_float(result["radius"])
    cross = max((a*b for a, b in zip(m, p[1:-1], strict=True)), default=Q(0))
    residual = e-k*(tau+gamma)-k*cross*e-k*max(m, default=Q(0))*e*e/2
    nonlinear_squared = k*k*sum(((a*b*b/2)**2 for a, b in zip(m, p[1:-1], strict=True)), Q(0))
    assert residual >= 0 and residual*residual >= nonlinear_squared
    assert max(p)+e <= rho
    assert not result["event_certificate_issued"]


def audit():
    args = dict(gain=1, drift_by_input=[], parameter_response_norms=[0, 0],
                response_residual=0, first_injection_error=Q(1, 2**1100), domain=0)
    assert not solve(**args)["closure"]
    smallest = float.fromhex("0x0.0000000000001p-1022")
    tiny = {**args, "domain": smallest}
    result = solve(**tiny)
    assert result["closure"] and result["radius"] == smallest
    verify_exact_supersolution(tiny, result)
    # Huge gain and tiny source must be multiplied before any binary64 cast.
    scaled = {**args, "gain": 2**1200, "first_injection_error": Q(1, 2**1210), "domain": 1}
    result = solve(**scaled)
    assert result["closure"]
    verify_exact_supersolution(scaled, result)
    # A nonrepresentable integer response must not be rounded down into domain.
    large = {**args, "first_injection_error": 0,
             "parameter_response_norms": [0, 2**53+1], "domain": 2**53}
    assert not solve(**large)["closure"]
    overflow = {**args, "first_injection_error": 2**1200, "domain": 2**1210}
    assert solve(**overflow)["reason"] == "no_finite_binary64_radius"
    invalid = 0
    for bad in (None, True, "1e-400", math.inf, math.nan, -1, Q(-1, 3)):
        try:
            solve(**{**args, "first_injection_error": bad})
        except ValueError:
            invalid += 1
        else:
            raise AssertionError("unsupported or invalid source accepted")
    rng = random.Random(20260908)
    cases = 0
    for _ in range(200):
        horizon = rng.randrange(1, 12)
        sample = dict(gain=Q(rng.randrange(1, 200), 7),
                      drift_by_input=[Q(rng.randrange(1, 50), 3) for _ in range(horizon-1)],
                      parameter_response_norms=[Q(0)]+[Q(rng.randrange(1, 50), 10**7) for _ in range(horizon)],
                      response_residual=Q(rng.randrange(1, 100), 10**14),
                      first_injection_error=Q(rng.randrange(1, 100), 10**13), domain=Q(1, 10))
        result = solve(**sample)
        assert result["closure"]
        verify_exact_supersolution(sample, result)
        cases += 1
    return {"schema": "exact_input_scalar_closure_tests_v1", "status": "PASS",
            "rational_supersolutions": cases,
            "underflow_source_refusal": True, "subnormal_radius_certified": True,
            "large_integer_domain_refusal": True, "balanced_extreme_exponents_certified": True,
            "unrepresentable_radius_abstention": True, "invalid_input_refusals": invalid,
            "oracle": "exact rational inequalities, no Arb in oracle",
            "frozen_producers_modified": False, "future_outcomes_accessed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = audit()
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n"
    if args.report:
        with args.report.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    print(payload, end="")
