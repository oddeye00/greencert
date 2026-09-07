"""Exhaustive count-level brackets and adversarial assembly-contract tests."""
from dataclasses import replace
import hashlib
import itertools
import json
import math
import time

from window_event_assembly import (Identity, DriftRow, MarginBound, OutputRow,
    PointMargin, AnchorOutputRow, ResponseBound, GreenBound, assemble, first_persistent)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def fixture(parameters=451008, horizon=6):
    identity = Identity(sha("candidate"), sha("reference"), sha("train"), sha("eval"),
                        parameters, horizon, "scaled_momentum_euclidean")
    patterns = [(-2., -1.), (-.03, -.2), (-.01, .02), (.1, .2), (.2, .2), (.2, .3), (.5, .5)]
    output = [OutputRow(identity, j, 0. if j == 0 else .25,
                        tuple(MarginBound(name, value, value, 1., 0.)
                              for name, value in zip(("a", "b"), patterns[min(j, 6)])), "outward")
              for j in range(horizon + 1)]
    return dict(identity=identity, training_count=3, example_ids=("a", "b"), domain=.25,
                drift_rows=[DriftRow(identity, j, .25, 3, 0., "outward") for j in range(1, horizon)],
                output_rows=output,
                response=ResponseBound(identity, (0.,) * (horizon + 1), 0., .025, True, "outward"),
                green=GreenBound(identity, 1., "outward", "deterministic", 0., 0, 0),
                target=1, persistence=2)


def replaced_row(rows, index, **changes):
    result = list(rows)
    result[index] = replace(result[index], **changes)
    return result


def main():
    started = time.perf_counter()
    enumerated = configurations = 0
    for parameters in (1, 13792, 451008, 1008864):
        args = fixture(parameters)
        for target in (1, 2):
            for persistence in (1, 2, 3, 5):
                result = assemble(**{**args, "target": target, "persistence": persistence})
                assert result["inputs_compatible"]
                assert result["conditional_on_supplied_bounds"] and not result["certificate_issued"]
                possible = [range(lo, hi + 1) for lo, hi in zip(result["lower_counts"], result["upper_counts"])]
                for counts in itertools.product(*possible):
                    actual = first_persistent(counts, target, persistence)
                    if result["bracket"] is not None:
                        left, right = result["bracket"]
                        assert actual is not None and left <= actual <= right
                    enumerated += 1
                configurations += 1
    args = fixture()
    assert assemble(**args)["bracket"] == [1, 3]
    point_anchor = AnchorOutputRow(args["identity"], tuple(PointMargin(m.example_id,m.lower,m.upper)
                                   for m in args["output_rows"][0].margins), "outward")
    point_rows = [point_anchor]+args["output_rows"][1:]
    assert assemble(**{**args,"output_rows":point_rows}) == assemble(**args)
    # Explicit lower/upper intervals work with NO optional point-margin field.
    interval_rows = [replace(row, margins=tuple(replace(m, lower=m.lower-.001, upper=m.upper+.001)
                                               for m in row.margins)) for row in args["output_rows"]]
    assert assemble(**{**args, "output_rows": interval_rows})["bracket"] == [1, 3]
    failed = []

    def refuses(name, changes, expected):
        result = assemble(**{**args, **changes})
        assert result["bracket"] is None and not result["certificate_issued"]
        assert result["inputs_compatible"] is False and result["reason"] == expected, (name, result)
        failed.append(name)

    refuses("missing_green", {"green": None}, "missing_green")
    refuses("float_green", {"green": replace(args["green"], arithmetic="float64")}, "unverified_arithmetic")
    refuses("missing_response", {"response": None}, "missing_response")
    refuses("implicit_anchor_zero", {"response": replace(args["response"], first_injection_error_upper=None)},
            "missing_or_invalid_first_anchor_injection_error")
    refuses("false_parameter_anchor", {"response": replace(args["response"], parameter_anchor_matches=False)}, "unmatched_parameter_anchor")
    refuses("numeric_parameter_anchor_flag", {"response": replace(args["response"], parameter_anchor_matches=1)}, "unmatched_parameter_anchor")
    refuses("nonzero_parameter_response_anchor", {"response": replace(args["response"], parameter_norms=(.1,)+(0.,)*6)},
            "nonzero_parameter_response_anchor")
    refuses("missing_last_response", {"response": replace(args["response"], parameter_norms=(0.,)*6)}, "incomplete_response_window")
    refuses("missing_drift", {"drift_rows": args["drift_rows"][:-1]}, "incomplete_drift_window")
    refuses("duplicate_drift", {"drift_rows": args["drift_rows"]+[args["drift_rows"][0]]}, "duplicate_drift_step")
    refuses("drift_at_anchor", {"drift_rows": replaced_row(args["drift_rows"], 0, step=0)}, "invalid_drift_step")
    refuses("drift_past_window", {"drift_rows": replaced_row(args["drift_rows"], 0, step=6)}, "out_of_range_drift_step")
    refuses("partial_training_mean", {"drift_rows": replaced_row(args["drift_rows"], 0, verified_training_count=2)}, "incomplete_training_mean")
    refuses("changed_training_count", {"training_count": 4}, "incomplete_training_mean")
    refuses("small_drift_domain", {"drift_rows": replaced_row(args["drift_rows"], 0, domain=.1)}, "insufficient_drift_domain")
    refuses("float_drift", {"drift_rows": replaced_row(args["drift_rows"], 0, arithmetic="float64")}, "unverified_arithmetic")
    refuses("missing_output", {"output_rows": args["output_rows"][:-1]}, "incomplete_output_window")
    refuses("point_output_after_anchor", {"output_rows": [args["output_rows"][0],replace(point_anchor,step=1)]+args["output_rows"][2:]},
            "point_only_output_away_from_anchor")
    refuses("point_margins_in_ball_row", {"output_rows": replaced_row(args["output_rows"],1,margins=point_anchor.margins)},
            "invalid_margin_record")
    refuses("ball_margins_in_point_row", {"output_rows": [replace(point_anchor,margins=args["output_rows"][0].margins)]+args["output_rows"][1:]},
            "invalid_margin_record")
    refuses("point_anchor_without_exact_parameters", {"output_rows":point_rows,
            "response":replace(args["response"],parameter_anchor_matches=False)}, "unmatched_parameter_anchor")
    refuses("duplicate_output", {"output_rows": args["output_rows"]+[args["output_rows"][0]]}, "duplicate_output_step")
    refuses("negative_output_step", {"output_rows": replaced_row(args["output_rows"], 0, step=-1)}, "invalid_output_step")
    refuses("output_past_window", {"output_rows": replaced_row(args["output_rows"], 0, step=7)}, "out_of_range_output_step")
    refuses("small_output_domain", {"output_rows": replaced_row(args["output_rows"], 1, domain=.1)}, "insufficient_output_domain")
    refuses("float_output", {"output_rows": replaced_row(args["output_rows"], 1, arithmetic="float64")}, "unverified_arithmetic")
    refuses("duplicate_example_ids", {"example_ids": ("a", "a")}, "duplicate_evaluation_ids")
    refuses("rotated_eval_population", {"output_rows": replaced_row(args["output_rows"], 1,
            margins=tuple(reversed(args["output_rows"][1].margins)))}, "incompatible_evaluation_population")
    refuses("missing_eval_example", {"output_rows": replaced_row(args["output_rows"], 1,
            margins=args["output_rows"][1].margins[:1])}, "incompatible_evaluation_population")
    first_margin = args["output_rows"][1].margins[0]
    for name, field, value, reason in (
            ("nan_margin", "lower", math.nan, "nonfinite_margin_lower"),
            ("inf_margin", "upper", math.inf, "nonfinite_margin_upper"),
            ("reversed_margin", "lower", 1., "reversed_margin_interval"),
            ("negative_J", "point_jacobian_upper", -1., "negative_output_jacobian"),
            ("negative_H", "ball_hessian_upper", -1., "negative_output_hessian")):
        refuses(name, {"output_rows": replaced_row(args["output_rows"], 1,
            margins=(replace(first_margin, **{field: value}), args["output_rows"][1].margins[1]))}, reason)
    for name, value, reason in (("nan_gain", math.nan, "nonfinite_green_gain"),
                                ("inf_gain", math.inf, "nonfinite_green_gain"),
                                ("below_diagonal_floor", .99, "gain_below_identity_diagonal_floor")):
        refuses(name, {"green": replace(args["green"], gain_upper=value)}, reason)
    gaussian = replace(args["green"], probability_scope="ideal_gaussian_probes",
                       failure_probability=1e-7, expected_probes=4, complete_probes=4)
    valid = assemble(**{**args, "green": gaussian})
    assert valid["bracket"] == [1, 3] and valid["probability_scope"] == "ideal_gaussian_probes"
    assert valid["failure_probability"] == 1e-7
    refuses("partial_probe_family", {"green": replace(gaussian, complete_probes=3)}, "incomplete_probe_family")
    refuses("extra_probe_family", {"green": replace(gaussian, complete_probes=5)}, "incomplete_probe_family")
    refuses("zero_randomized_failure_probability", {"green": replace(gaussian, failure_probability=0.)}, "invalid_gaussian_scope")
    refuses("certainty_relabel", {"green": replace(gaussian, probability_scope="deterministic")}, "inconsistent_deterministic_scope")
    refuses("unknown_probability_model", {"green": replace(gaussian, probability_scope="arbitrary_prng")}, "unsupported_probability_scope")
    refuses("boolean_target", {"target": True}, "invalid_target")
    refuses("impossible_target", {"target": 3}, "unattainable_target")
    refuses("oversized_persistence", {"persistence": 8}, "insufficient_persistence_window")
    for field in ("candidate_sha256", "reference_sha256", "training_set_sha256", "evaluation_set_sha256"):
        foreign = replace(args["identity"], **{field: sha("different-"+field)})
        refuses("foreign_"+field, {"drift_rows": replaced_row(args["drift_rows"], 0, identity=foreign)}, "incompatible_identity")
    # Tiny positive margins must not be mistaken for ties or rounded zeros.
    perfect = replace(args["response"], first_injection_error_upper=0.)
    exact_rows = [replace(row, margins=tuple(replace(m, lower=-1. if row.step == 0 else 1e-300,
                                                    upper=-1. if row.step == 0 else 1e-300) for m in row.margins))
                  for row in args["output_rows"]]
    assert assemble(**{**args, "response": perfect, "output_rows": exact_rows})["bracket"] == [1, 1]
    tied = [replace(row, margins=tuple(replace(m, lower=0., upper=0.) for m in row.margins)) for row in exact_rows]
    result = assemble(**{**args, "response": perfect, "output_rows": tied})
    assert result["bracket"] is None and result["lower_counts"] == [0]*7 and result["upper_counts"] == [2]*7
    # The output at step0 needs no ball domain only when the parameter
    # anchor matches. Nonzero optimizer-coordinate error is not dropped.
    nonzero = assemble(**args)
    assert nonzero["state_closure"]["radius"] >= .025
    exact = assemble(**{**args, "response": perfect})
    assert exact["state_closure"]["radius"] == 0.
    # Larger proved domains and reordered time rows are legitimate, rather
    # than needlessly rejected as a change to the mathematical guarantee.
    broader = [replace(r, domain=1.) for r in args["drift_rows"]]
    assert assemble(**{**args, "drift_rows": list(reversed(broader)),
                       "output_rows": list(reversed(args["output_rows"]))})["bracket"] == [1, 3]
    # Nonzero known response is charged in each output ball. Omitting it
    # would incorrectly retain the earlier upper endpoint3.
    displaced = replace(args["response"], parameter_norms=(0., 0., 0., .2, 0., 0., 0.))
    assert assemble(**{**args, "response": displaced})["bracket"] == [1, 4]
    nonlinear = [replace(row, mean_upper=.01) for row in args["drift_rows"]]
    nonzero = assemble(**{**args, "response": displaced, "drift_rows": nonlinear})
    assert nonzero["bracket"] == [1, 4]
    assert nonzero["state_closure"]["radius"] > .025
    impossible = assemble(**{**args, "response": replace(args["response"], first_injection_error_upper=1.)})
    assert impossible["inputs_compatible"] and impossible["reason"] == "state_closure_failed"
    assert impossible["bracket"] is None
    refuses("unsupported_optimizer_geometry", {"identity": replace(args["identity"], state_metric="adamw_full_state")},
            "unsupported_state_metric")
    # H=1 has no nonlinear interior inputs; it still has two output states.
    one = fixture(horizon=1)
    one["response"] = replace(one["response"], first_injection_error_upper=0.)
    one["output_rows"][1] = replace(one["output_rows"][1], margins=(
        MarginBound("a", 1., 1., 1., 0.), MarginBound("b", 1., 1., 1., 0.)))
    assert assemble(**{**one, "persistence": 1})["bracket"] == [1, 1]
    assert assemble(**{**one, "persistence": 2})["bracket"] is None
    result = {"all_passed": True, "configurations": configurations,
              "enumerated_admissible_count_paths": enumerated,
              "contract_rejections": len(failed), "rejection_cases": failed,
              "interval_only_margins": True, "tiny_margin_and_tie_checks": True,
              "nonzero_anchor_injection_preserved": True, "probability_scope_preserved": True,
              "nonzero_signed_response_and_nonlinearity": True, "single_step_edge_case": True,
              "seconds": time.perf_counter()-started}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
