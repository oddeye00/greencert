#!/usr/bin/env python3
"""Tests for predictable and role-stratified confidence spending."""

from __future__ import annotations

import math

from predictable_failure_budget import p_series_delta, role_stratified_budget


def main() -> None:
    total = 1.0e-6
    partial = sum(p_series_delta(total, n) for n in range(1, 1_000_001))
    assert partial < total
    assert total - partial < 7.0e-13

    budgets = [
        role_stratified_budget(
            total_delta=total, candidate_count=19, horizon=horizon
        )
        for horizon in (26, 131, 270, 299)
    ]
    for budget in budgets:
        assert math.isclose(budget.total, budget.candidate_delta, rel_tol=2.0e-16)
        assert budget.green_delta > budget.output_delta
    assert math.isclose(sum(b.candidate_delta for b in budgets[:1]) * 19, total)
    print("PASS: predictable p-series and role-stratified budgets conserve FWER.")


if __name__ == "__main__":
    main()

