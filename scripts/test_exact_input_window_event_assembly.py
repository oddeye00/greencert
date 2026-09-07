"""Exact-input event assembly: legacy contracts and independent count oracle."""
import argparse
from contextlib import redirect_stdout
from dataclasses import replace
from fractions import Fraction as Q
import io
import itertools
import json
from pathlib import Path
import random

from flint import ctx
import exact_input_window_event_assembly as exact
import test_window_event_assembly as legacy_tests
from test_exact_input_scalar_closure import verify_exact_supersolution


def legacy_contracts():
    # Rebind only the test fixture's imported API; frozen implementation files
    # and module attributes are never changed.
    names = ("Identity", "DriftRow", "MarginBound", "OutputRow", "PointMargin",
             "AnchorOutputRow", "ResponseBound", "GreenBound", "assemble", "first_persistent")
    previous = {name: getattr(legacy_tests, name) for name in names}
    try:
        for name in names:
            setattr(legacy_tests, name, getattr(exact, name))
        with redirect_stdout(io.StringIO()):
            return legacy_tests.main()
    finally:
        for name, value in previous.items():
            setattr(legacy_tests, name, value)


def fixture(horizon=3):
    identity = exact.Identity(*("a"*64, "b"*64, "c"*64, "d"*64), 451008,
                              horizon, "scaled_momentum_euclidean")
    return dict(identity=identity, training_count=3, example_ids=("a",), domain=1,
        drift_rows=[exact.DriftRow(identity, j, 1, 3, 0, "outward")
                    for j in range(1, horizon)],
        output_rows=[exact.OutputRow(identity, j, 1,
            (exact.MarginBound("a", -1 if j == 0 else 1,
                               -1 if j == 0 else 1, 0, 0),), "outward")
                     for j in range(horizon+1)],
        response=exact.ResponseBound(identity, (0,)*(horizon+1), 0, 0, True, "outward"),
        green=exact.GreenBound(identity, 1, "outward", "deterministic", 0, 0, 0),
        target=1, persistence=1)


def rational_count_oracle(args, result):
    e = Q(result["state_closure"]["radius"])
    exact_lower, exact_upper = [], []
    for row in args["output_rows"]:
        r = Q(0) if row.step == 0 else Q(args["response"].parameter_norms[row.step])+e
        lo = hi = 0
        for margin in row.margins:
            # sqrt(2)*a is the margin uncertainty. Compare squares with the
            # requisite signs; no Arb, floating sum, or irrational approximation.
            a = Q(0) if row.step == 0 else (
                Q(margin.point_jacobian_upper)*r+Q(margin.ball_hessian_upper)*r*r/2)
            lower, upper = Q(margin.lower), Q(margin.upper)
            lo += int(lower > 0 and lower*lower > 2*a*a)
            hi += int(not (upper < 0 and upper*upper > 2*a*a))
        exact_lower.append(lo)
        exact_upper.append(hi)
    for got_lo, want_lo, want_hi, got_hi in zip(
            result["lower_counts"], exact_lower, exact_upper, result["upper_counts"], strict=True):
        assert 0 <= got_lo <= want_lo <= want_hi <= got_hi <= len(args["example_ids"])
    possibilities = [range(lo, hi+1) for lo, hi in zip(exact_lower, exact_upper)]
    paths = 0
    for counts in itertools.product(*possibilities):
        actual = exact.first_persistent(counts, args["target"], args["persistence"])
        if result["bracket"] is not None:
            left, right = result["bracket"]
            assert actual is not None and 0 < left <= actual <= right
        paths += 1
    return paths


def audit():
    prior_precision = ctx.prec
    inherited = legacy_contracts()
    assert ctx.prec == prior_precision
    boundary = []

    def refuses(name, args, reason):
        result = exact.assemble(**args)
        assert result["bracket"] is None and result["reason"] == reason, (name, result)
        assert not result["certificate_issued"] and not result["artifact_authentication_performed"]
        json.dumps(result, allow_nan=False)
        boundary.append(name)

    one = fixture(1)
    huge = 2**53
    refuses("integer_response_outside_exact_domain",
        {**one, "domain": huge, "output_rows": [replace(r, domain=huge) for r in one["output_rows"]],
         "response": replace(one["response"], parameter_norms=(0, huge+1))}, "state_closure_failed")
    tiny = Q(1, 2**1100)
    refuses("positive_source_cannot_fit_zero_domain",
        {**one, "domain": 0, "response": replace(one["response"], first_injection_error_upper=tiny)},
        "state_closure_failed")
    refuses("tiny_nonzero_parameter_anchor",
        {**one, "response": replace(one["response"], parameter_norms=(tiny, 0))},
        "nonzero_parameter_response_anchor")
    args = fixture()
    refuses("integer_drift_domain_does_not_cover",
        {**args, "domain": huge+1,
         "drift_rows": [replace(r, domain=huge) for r in args["drift_rows"]]},
        "insufficient_drift_domain")
    refuses("integer_output_domain_does_not_cover",
        {**one, "domain": huge+1,
         "output_rows": [replace(r, domain=huge) for r in one["output_rows"]]},
        "insufficient_output_domain")
    rows = list(one["output_rows"])
    rows[1] = replace(rows[1], margins=(exact.MarginBound("a", huge+1, huge, 0, 0),))
    refuses("integer_reversed_interval_cannot_collapse", {**one, "output_rows": rows},
        "reversed_margin_interval")
    refuses("tiny_negative_derivative",
        {**one, "output_rows": [one["output_rows"][0], replace(rows[1],
            margins=(exact.MarginBound("a", 1, 1, -tiny, 0),))]}, "negative_output_jacobian")
    refuses("gain_just_below_identity_floor",
        {**one, "green": replace(one["green"], gain_upper=1-tiny)}, "gain_below_identity_diagonal_floor")
    refuses("tiny_probability_cannot_be_deterministic",
        {**one, "green": replace(one["green"], failure_probability=tiny)}, "inconsistent_deterministic_scope")
    gaussian = replace(one["green"], probability_scope="ideal_gaussian_probes",
                       failure_probability=tiny, expected_probes=1, complete_probes=1)
    randomized = exact.assemble(**{**one, "green": gaussian})
    assert randomized["bracket"] == [1, 1] and randomized["failure_probability"] is None
    encoded = randomized["failure_probability_exact"]
    assert Q(int(encoded["numerator_hex"], 16), int(encoded["denominator_hex"], 16)) == tiny
    assert Q(randomized["failure_probability_upper"]) >= tiny
    json.dumps(randomized, allow_nan=False)
    refuses("probability_just_above_one",
        {**one, "green": replace(gaussian, failure_probability=1+tiny)}, "invalid_gaussian_scope")
    for bad in (True, "1", None):
        refuses("invalid_domain_"+type(bad).__name__, {**one, "domain": bad}, "missing_or_invalid_domain")

    # A positive rational below binary64's range remains a strict margin.
    rows = [one["output_rows"][0], replace(one["output_rows"][1],
            margins=(exact.MarginBound("a", tiny, tiny, 0, 0),))]
    result = exact.assemble(**{**one, "output_rows": rows})
    assert result["bracket"] == [1, 1]
    # Exponents are balanced in exact arithmetic before taking enclosures.
    result = exact.assemble(**{**one,
        "green": replace(one["green"], gain_upper=2**1200),
        "response": replace(one["response"], first_injection_error_upper=Q(1, 2**1210))})
    assert result["bracket"] == [1, 1]

    rng = random.Random(20260909)
    count_paths = issued = 0
    for _ in range(200):
        args = fixture(4)
        p = (0,)+tuple(Q(rng.randrange(1, 8), 1000) for _ in range(4))
        args["response"] = replace(args["response"], parameter_norms=p,
            first_injection_error_upper=Q(rng.randrange(1, 5), 10000))
        args["drift_rows"] = [replace(row, mean_upper=Q(rng.randrange(0, 9), 100))
                              for row in args["drift_rows"]]
        rows = []
        for row in args["output_rows"]:
            lower = Q(-1) if row.step == 0 else Q(rng.randrange(-20, 21), 200)
            rows.append(replace(row, margins=(exact.MarginBound("a", lower,
                lower+Q(rng.randrange(0, 4), 200),
                Q(rng.randrange(1, 6), 3), Q(rng.randrange(0, 5), 7)),)))
        args["output_rows"] = rows
        args["persistence"] = rng.randrange(1, 4)
        result = exact.assemble(**args)
        assert result["inputs_compatible"] and result["state_closure"]["closure"]
        verify_exact_supersolution(dict(gain=args["green"].gain_upper,
            drift_by_input=[row.mean_upper for row in args["drift_rows"]],
            parameter_response_norms=p, response_residual=args["response"].residual_upper,
            first_injection_error=args["response"].first_injection_error_upper, domain=args["domain"]),
            result["state_closure"])
        count_paths += rational_count_oracle(args, result)
        issued += int(result["bracket"] is not None)
        json.dumps(result, allow_nan=False)
    assert issued > 0 and ctx.prec == prior_precision
    return dict(schema="exact_input_event_assembly_tests_v1", status="PASS",
        legacy_configurations=inherited["configurations"],
        legacy_count_paths=inherited["enumerated_admissible_count_paths"],
        legacy_contract_refusals=inherited["contract_rejections"],
        exact_boundary_refusals=boundary, rational_output_cases=200,
        rational_oracle_count_paths=count_paths, conditional_brackets=issued,
        tiny_probability_preserved=True, tiny_positive_margin_preserved=True,
        extreme_exponents_preserved=True, precision_restored=True,
        oracle="exact rational sign-and-square comparisons, no Arb",
        frozen_producers_modified=False, future_outcomes_accessed=False,
        event_certificate_issued=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = audit()
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n"
    if args.report:
        with args.report.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    print(payload, end="")
