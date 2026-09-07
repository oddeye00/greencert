"""Exact-input v1 assembly of finite-window event bounds.

This separately versioned interface preserves int/Fraction inputs and the
exact dyadic value of finite floats through validation and Arb transport.
Frozen window_event_assembly.py is retained unchanged for recorded replay.

This layer validates coverage/identity and evaluates scalar inequalities in
Arb. It does NOT authenticate artifact files or prove supplied neural/Green
bounds. Results remain explicitly conditional; a file-backed verifier must
first establish every supplied bound under its recorded numerical scope.

The supported theorem has an exact parameter anchor, possibly a nonzero
scaled-momentum-state conversion error, and a causal Green operator with
identity diagonal blocks. Its nonlinear remainder depends only on parameter
displacement (the momentum coordinates enter affinely). Other optimizer
geometries need a separately justified adapter; they are not silently treated
as momentum. No model size or modular-task constant is built in.
"""
from dataclasses import dataclass
from fractions import Fraction
import math
import re

from flint import arb, ctx
from exact_input_scalar_closure import solve, enclosed, float_upper

ExactScalar = int | float | Fraction


class AssemblyError(ValueError):
    pass


@dataclass(frozen=True)
class Identity:
    candidate_sha256: str
    reference_sha256: str
    training_set_sha256: str
    evaluation_set_sha256: str
    parameters: int
    horizon: int
    state_metric: str


@dataclass(frozen=True)
class DriftRow:
    identity: Identity
    step: int
    domain: ExactScalar
    verified_training_count: int
    mean_upper: ExactScalar
    arithmetic: str


@dataclass(frozen=True)
class MarginBound:
    example_id: str
    lower: ExactScalar
    upper: ExactScalar
    point_jacobian_upper: ExactScalar
    ball_hessian_upper: ExactScalar


@dataclass(frozen=True)
class OutputRow:
    identity: Identity
    step: int
    domain: ExactScalar
    margins: tuple[MarginBound, ...]
    arithmetic: str


@dataclass(frozen=True)
class PointMargin:
    example_id: str
    lower: ExactScalar
    upper: ExactScalar


@dataclass(frozen=True)
class AnchorOutputRow:
    """Point-only evidence, usable only at the exact parameter anchor."""
    identity: Identity
    margins: tuple[PointMargin, ...]
    arithmetic: str
    step: int = 0


@dataclass(frozen=True)
class ResponseBound:
    identity: Identity
    parameter_norms: tuple[ExactScalar, ...]
    residual_upper: ExactScalar
    first_injection_error_upper: ExactScalar
    parameter_anchor_matches: bool
    arithmetic: str


@dataclass(frozen=True)
class GreenBound:
    identity: Identity
    gain_upper: ExactScalar
    arithmetic: str
    probability_scope: str
    failure_probability: ExactScalar
    expected_probes: int
    complete_probes: int


def _require(condition, reason):
    if not condition:
        raise AssemblyError(reason)


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, f"invalid_{name}")
    return value


def _number(value, name, nonnegative=True):
    _require(type(value) in (float, int, Fraction), f"missing_or_invalid_{name}")
    if type(value) is float:
        _require(math.isfinite(value), f"nonfinite_{name}")
        result = Fraction.from_float(value)
    else:
        result = Fraction(value)
    _require(not nonnegative or result >= 0, f"negative_{name}")
    return result


def _arb_number(value):
    return enclosed(_number(value, "transport_scalar", False))


def _probability_record(value):
    probability = _number(value, "failure_probability")
    rounded = float(probability)  # Validated to lie in [0,1].
    exact_float = rounded if Fraction.from_float(rounded) == probability else None
    return {
        "failure_probability": exact_float,
        "failure_probability_exact": {
            "numerator_hex": hex(probability.numerator),
            "denominator_hex": hex(probability.denominator),
        },
        "failure_probability_upper": min(1.0, float_upper(enclosed(probability))),
    }


def _identity(value):
    _require(isinstance(value, Identity), "invalid_identity")
    for key in ("candidate_sha256", "reference_sha256", "training_set_sha256", "evaluation_set_sha256"):
        sha = getattr(value, key)
        _require(isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha) is not None, f"invalid_{key}")
    _integer(value.parameters, "parameters", 1)
    _integer(value.horizon, "horizon", 1)
    _require(value.state_metric == "scaled_momentum_euclidean", "unsupported_state_metric")


def _binding(row, expected):
    _identity(row.identity)
    _require(row.identity == expected, "incompatible_identity")
    _require(row.arithmetic == "outward", "unverified_arithmetic")


def first_persistent(counts, target, persistence):
    return next((j for j in range(len(counts) - persistence + 1)
                 if all(v >= target for v in counts[j:j + persistence])), None)


def _prepare(*, identity, training_count, example_ids, domain, drift_rows,
             output_rows, response, green, target, persistence, precision_bits):
    _identity(identity)
    _integer(training_count, "training_count", 1)
    _integer(target, "target", 1)
    _integer(persistence, "persistence", 1)
    _integer(precision_bits, "precision_bits", 64)
    _require(isinstance(example_ids, tuple) and len(example_ids) > 0, "invalid_evaluation_ids")
    _require(all(isinstance(v, str) and v for v in example_ids), "invalid_evaluation_ids")
    _require(len(set(example_ids)) == len(example_ids), "duplicate_evaluation_ids")
    _require(target <= len(example_ids), "unattainable_target")
    _require(persistence <= identity.horizon + 1, "insufficient_persistence_window")
    radius = _number(domain, "domain")
    _require(isinstance(response, ResponseBound), "missing_response")
    _binding(response, identity)
    _require(response.parameter_anchor_matches is True, "unmatched_parameter_anchor")
    _require(isinstance(response.parameter_norms, tuple) and len(response.parameter_norms) == identity.horizon + 1,
             "incomplete_response_window")
    parameter_norms = [_number(v, "parameter_response_norm") for v in response.parameter_norms]
    _require(parameter_norms[0] == 0, "nonzero_parameter_response_anchor")
    _number(response.residual_upper, "response_residual")
    # An explicit zero is valid; None/missing is not converted into zero.
    _number(response.first_injection_error_upper, "first_anchor_injection_error")
    _require(isinstance(green, GreenBound), "missing_green")
    _binding(green, identity)
    _require(_number(green.gain_upper, "green_gain") >= 1, "gain_below_identity_diagonal_floor")
    probability = _number(green.failure_probability, "failure_probability")
    _integer(green.expected_probes, "expected_probes")
    _integer(green.complete_probes, "complete_probes")
    if green.probability_scope == "deterministic":
        _require(probability == 0 and green.expected_probes == green.complete_probes == 0,
                 "inconsistent_deterministic_scope")
    elif green.probability_scope == "ideal_gaussian_probes":
        _require(0 < probability < 1 and green.expected_probes > 0,
                 "invalid_gaussian_scope")
        _require(green.complete_probes == green.expected_probes, "incomplete_probe_family")
    else:
        raise AssemblyError("unsupported_probability_scope")
    _require(isinstance(drift_rows, (tuple, list)), "missing_drift_rows")
    drift = {}
    for row in drift_rows:
        _require(isinstance(row, DriftRow), "invalid_drift_row")
        _binding(row, identity)
        _integer(row.step, "drift_step", 1)
        _require(row.step < identity.horizon, "out_of_range_drift_step")
        _require(row.step not in drift, "duplicate_drift_step")
        _require(_number(row.domain, "drift_domain") >= radius, "insufficient_drift_domain")
        _integer(row.verified_training_count, "verified_training_count")
        _require(row.verified_training_count == training_count, "incomplete_training_mean")
        drift[row.step] = _number(row.mean_upper, "mean_drift_upper")
    _require(set(drift) == set(range(1, identity.horizon)), "incomplete_drift_window")
    _require(isinstance(output_rows, (tuple, list)), "missing_output_rows")
    outputs = {}
    for row in output_rows:
        _require(isinstance(row, (OutputRow, AnchorOutputRow)), "invalid_output_row")
        _binding(row, identity)
        _integer(row.step, "output_step")
        _require(row.step <= identity.horizon, "out_of_range_output_step")
        _require(row.step not in outputs, "duplicate_output_step")
        point_only = isinstance(row, AnchorOutputRow)
        if point_only:
            _require(row.step == 0, "point_only_output_away_from_anchor")
        else:
            row_domain = _number(row.domain, "output_domain")
            # Retain compatibility with existing numeric anchor rows.
            _require(row.step == 0 or row_domain >= radius, "insufficient_output_domain")
        _require(isinstance(row.margins, tuple), "invalid_margin_rows")
        margin_type = PointMargin if point_only else MarginBound
        _require(all(isinstance(v, margin_type) for v in row.margins), "invalid_margin_record")
        _require(tuple(v.example_id for v in row.margins) == example_ids, "incompatible_evaluation_population")
        for margin in row.margins:
            lo = _number(margin.lower, "margin_lower", False)
            hi = _number(margin.upper, "margin_upper", False)
            _require(lo <= hi, "reversed_margin_interval")
            if not point_only:
                _number(margin.point_jacobian_upper, "output_jacobian")
                _number(margin.ball_hessian_upper, "output_hessian")
        outputs[row.step] = row
    _require(set(outputs) == set(range(identity.horizon + 1)), "incomplete_output_window")
    return radius, parameter_norms, drift, outputs


def assemble(*, identity, training_count, example_ids, domain, drift_rows,
             output_rows, response, green, target, persistence, precision_bits=256):
    """Return a conditional first-passage bracket or an explicit refusal.

    File/hash authenticity and supplied bound validity are external proof
    obligations. Missing or incompatible input never becomes a certificate.
    Output centers are the original reference points, not rounded c+z.
    """
    base = {"schema": "exact_input_window_event_assembly_v1",
            "input_semantics": "exact int/Fraction or exact dyadic value of finite float",
            "conditional_on_supplied_bounds": True, "artifact_authentication_performed": False,
            "certificate_issued": False, "bracket": None}
    try:
        radius, p, drift, outputs = _prepare(identity=identity, training_count=training_count,
            example_ids=example_ids, domain=domain, drift_rows=drift_rows, output_rows=output_rows,
            response=response, green=green, target=target, persistence=persistence, precision_bits=precision_bits)
    except AssemblyError as error:
        return {**base, "inputs_compatible": False, "reason": str(error)}
    closure = solve(gain=green.gain_upper, drift_by_input=[drift[j] for j in range(1, identity.horizon)],
        parameter_response_norms=p, response_residual=response.residual_upper,
        first_injection_error=response.first_injection_error_upper, domain=radius, precision_bits=precision_bits)
    base |= {"inputs_compatible": True, "state_closure": closure,
             "probability_scope": green.probability_scope, **_probability_record(green.failure_probability)}
    if not closure["closure"]:
        return {**base, "reason": "state_closure_failed"}
    prior = ctx.prec
    ctx.prec = precision_bits
    try:
        lower_counts, upper_counts = [], []
        for j in range(identity.horizon + 1):
            # Keep the exact sum p_j+E in Arb, not a rounded float sum.
            r = arb(0) if j == 0 else enclosed(p[j]) + arb(closure["radius"])
            lo = hi = 0
            for margin in outputs[j].margins:
                if j == 0:
                    # No derivative value is evaluated or asserted here.
                    uncertainty = arb(0)
                else:
                    uncertainty = arb(2).sqrt() * (_arb_number(margin.point_jacobian_upper) * r +
                        _arb_number(margin.ball_hessian_upper) * r * r / 2)
                lower = _arb_number(margin.lower) - uncertainty
                upper = _arb_number(margin.upper) + uncertainty
                lo += int(bool(lower > 0))
                hi += int(not bool(upper < 0))
            lower_counts.append(lo)
            upper_counts.append(hi)
        left = first_persistent(upper_counts, target, persistence)
        right = first_persistent(lower_counts, target, persistence)
        bracket = [left, right] if left is not None and right is not None and 0 < left <= right else None
        return {**base, "bracket": bracket, "lower_counts": lower_counts, "upper_counts": upper_counts,
                "reason": None if bracket is not None else "persistent_event_not_enclosed"}
    finally:
        ctx.prec = prior
