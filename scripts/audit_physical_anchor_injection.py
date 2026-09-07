#!/usr/bin/env python3
"""Exact-rational regression audit for the physical-anchor corollary.

No neural artifacts, randomness claims, or future outcomes are read. These
finite examples check algebra and indexing; the manuscript supplies the proof.
"""
from __future__ import annotations

import argparse
import json
import random
from fractions import Fraction as Q
from pathlib import Path


def add(x, y):
    return tuple(a + b for a, b in zip(x, y, strict=True))


def sub(x, y):
    return tuple(a - b for a, b in zip(x, y, strict=True))


def mv(a, x):
    return tuple(sum((v * w for v, w in zip(row, x, strict=True)), Q(0)) for row in a)


def green(jacobians, forcing):
    state = tuple(Q(0) for _ in forcing[0])
    result = []
    for jac, source in zip(jacobians, forcing, strict=True):
        state = add(mv(jac, state), source)
        result.append(state)
    return result


def first_persistent(counts, required, persistence):
    return next((j for j in range(len(counts) - persistence + 1)
                 if all(x >= required for x in counts[j:j + persistence])), None)


def audit():
    rng = random.Random(20260907)
    identity_cases = 0
    changed_j0_cases = 0
    missing_injection_changes = 0
    residual_identity_cases = 0
    momentum_cases = 0
    fraction = lambda: Q(rng.randrange(-3, 4), 16)
    for horizon in range(1, 5):
        for _ in range(12):
            matrix = tuple(tuple(fraction() for _ in range(2)) for _ in range(2))
            bias = (fraction(), fraction())
            diagonal = (fraction(), fraction())
            cross = (fraction(), fraction())

            def map_step(x):
                return add(add(mv(matrix, x), bias),
                           tuple(diagonal[i] * x[i] ** 2 + cross[i] * x[0] * x[1]
                                 for i in range(2)))

            def jacobian(x):
                return tuple(tuple(matrix[i][k] + (2 * diagonal[i] * x[i] if i == k else 0)
                                   + cross[i] * x[1 - k]
                                   for k in range(2)) for i in range(2))

            centers = [(fraction(), fraction()) for _ in range(horizon + 1)]
            physical = add(centers[0], (Q(1, 64), Q(-1, 128)))
            jacobians = [jacobian(c) for c in centers[:-1]]
            defects = [sub(map_step(c), nxt) for c, nxt in zip(centers[:-1], centers[1:], strict=True)]
            injection = sub(map_step(physical), map_step(centers[0]))
            sources = list(defects)
            sources[0] = add(sources[0], injection)
            actual = []
            state = physical
            for center in centers[1:]:
                state = map_step(state)
                actual.append(sub(state, center))
            virtual_error = [(Q(0), Q(0))] + actual
            nonlinear = [(Q(0), Q(0))]
            for j in range(1, horizon):
                nonlinear.append(sub(sub(map_step(add(centers[j], virtual_error[j])), map_step(centers[j])),
                                     mv(jacobians[j], virtual_error[j])))
            linear_response = green(jacobians, sources)
            nonlinear_response = green(jacobians, nonlinear)
            assert actual == [add(z, n) for z, n in zip(linear_response, nonlinear_response, strict=True)]
            identity_cases += 1
            changed = list(jacobians)
            changed[0] = ((Q(999), Q(-777)), (Q(555), Q(-333)))
            assert green(changed, sources) == linear_response
            changed_j0_cases += 1
            if injection != (0, 0):
                assert green(jacobians, defects) != linear_response
                missing_injection_changes += 1

            # Independently perturb source, injection, and computed sweep.
            computed_source = [add(s, (Q(1, 1024), Q(-1, 2048))) for s in defects]
            computed_injection = add(injection, (Q(1, 4096), Q(0)))
            computed = [add(z, (Q(j + 1, 8192), Q(-1, 16384)))
                        for j, z in enumerate(linear_response)]
            residuals = []
            previous = (Q(0), Q(0))
            for j, z in enumerate(computed):
                source = add(computed_source[j], computed_injection) if j == 0 else computed_source[j]
                residuals.append(sub(sub(z, mv(jacobians[j], previous)), source))
                previous = z
            response_difference_source = []
            for j in range(horizon):
                row = sub(sub(defects[j], computed_source[j]), residuals[j])
                if j == 0:
                    row = add(row, sub(injection, computed_injection))
                response_difference_source.append(row)
            assert [sub(z, w) for z, w in zip(linear_response, computed, strict=True)] == green(jacobians, response_difference_source)
            residual_identity_cases += 1

    # A nonlinear initial displacement cannot in general be replaced by J0 delta.
    actual_source = Q(1, 8) ** 2
    linearized_source = Q(0)  # G(x)=x^2 at c0=0
    assert actual_source != linearized_source
    nonlinear_anchor_counterexamples = 1

    for _ in range(32):
        theta, velocity = fraction(), fraction()
        eta, mu = Q(3, 1000), Q(9, 10)
        physical_w = eta * velocity
        stored_w = Q.from_float(float(physical_w))
        grad = lambda t: t ** 3 + t / 7
        def momentum(t, w):
            next_w = mu * w + eta * grad(t)
            return (t - next_w, next_w)
        delta = physical_w - stored_w
        assert sub(momentum(theta, physical_w), momentum(theta, stored_w)) == (-mu * delta, mu * delta)
        momentum_cases += 1

    # A concrete nonzero-offset nonlinear example closes the unchanged theorem.
    # Upper bounds use rational l1/Frobenius relaxations, not floating square roots.
    horizon = 4
    jacobians = [((Q(1, 2),),)] * horizon
    physical = Q(1, 64)
    map_scalar = lambda t: t / 2 + t * t / 64
    sources = [(map_scalar(physical),)] + [(Q(0),)] * (horizon - 1)
    z = green(jacobians, sources)
    nonlinear = [(Q(0),)] + [(z[j - 1][0] ** 2 / 64,) for j in range(1, horizon)]
    beta = sum(abs(x[0]) for x in green(jacobians, nonlinear))
    # ||K||_2 <= sqrt(||K||_1 ||K||_infty) <= 2 for this recurrence.
    kappa, drift, radius = Q(2), Q(1, 32), Q(1, 16)
    p = max(abs(x[0]) for x in z)
    error_radius = Q(1, 100000)
    assert p + error_radius <= radius
    assert beta + kappa * drift * p * error_radius + kappa * drift * error_radius ** 2 / 2 <= error_radius
    state = physical
    errors = []
    for center in z:
        state = map_scalar(state)
        errors.append(state - center[0])
    assert sum(e * e for e in errors) <= error_radius ** 2
    # For H=1, there are no unknown nonlinear transition inputs whatsoever.
    assert green([((Q(999),),)], [(map_scalar(physical),)]) == [(map_scalar(physical),)]

    # A time-zero gate on c0 is not a gate on the physical checkpoint.
    physical_counts = [0, 1, 1]
    stored_anchor_counts = [1, 1, 1]
    assert first_persistent(physical_counts, 1, 2) == 1
    assert first_persistent(stored_anchor_counts, 1, 2) == 0
    return {
        "schema": "physical_anchor_injection_rational_audit_v1",
        "status": "PASS",
        "arithmetic": "Python Fraction (exact rational)",
        "identity_cases": identity_cases,
        "changed_initial_jacobian_cases": changed_j0_cases,
        "omitted_source_changes_response_cases": missing_injection_changes,
        "inexact_response_identity_cases": residual_identity_cases,
        "nonlinear_anchor_linearization_counterexamples": nonlinear_anchor_counterexamples,
        "scaled_momentum_identity_cases": momentum_cases,
        "nonlinear_closure_and_realized_tube_cases": 1,
        "single_transition_boundary_cases": 1,
        "incorrect_time_zero_gate_counterexamples": 1,
        "scope": "Algebra/indexing regressions, not an independent proof or new neural experiment.",
        "future_outcomes_accessed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = audit()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
