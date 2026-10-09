# SPDX-License-Identifier: Apache-2.0
"""Hand-worked descriptive cost and fold summaries, including every boundary."""

from __future__ import annotations

from typing import Any

import pytest

from signalquarry._internal.validation.diagnostics import cost_curve_break_even, fold_consistency


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        ([], None),
        ([(0, 0.1)], None),
        ([(0, 0.1), (1, 0.05), (4, 0.01)], None),
        ([(0, 0.1), (1, 0.05), (2, -0.05), (4, -0.2)], 1.5),
        ([(0, 0.1), (2, -0.2)], 0.666667),
        ([(0, -0.1), (1, 0.1), (2, -0.1)], None),
        ([(0, 0), (1, 0.1)], None),
        ([(0, 0.1), (1, 0), (2, -0.1)], 1.0),
        ([(0, 0.1), (1, -0.1), (2, 0.1), (4, -0.1)], 0.5),
    ],
)
def test_break_even(points: list[tuple[float, float]], expected: float | None) -> None:
    assert cost_curve_break_even(points) == expected


@pytest.mark.parametrize(
    ("folds", "expected"),
    [
        (
            [],
            {
                "folds": 0,
                "positive": 0,
                "share_positive": None,
                "ahead_of_benchmark": None,
                "best": None,
                "worst": None,
                "dispersion": None,
                "status": "insufficient",
            },
        ),
        (
            [{"total_return": 0}],
            {
                "folds": 1,
                "positive": 0,
                "share_positive": 0.0,
                "ahead_of_benchmark": None,
                "best": 0.0,
                "worst": 0.0,
                "dispersion": None,
                "status": "insufficient",
            },
        ),
        (
            [{"total_return": -0.1}, {"total_return": 0.1}],
            {
                "folds": 2,
                "positive": 1,
                "share_positive": 0.5,
                "ahead_of_benchmark": None,
                "best": 0.1,
                "worst": -0.1,
                "dispersion": 0.14142136,
                "status": "insufficient",
            },
        ),
        (
            [
                {"total_return": -0.1, "excess_return": 0.01},
                {"total_return": 0, "excess_return": 0},
                {"total_return": 0.1, "excess_return": -0.01},
            ],
            {
                "folds": 3,
                "positive": 1,
                "share_positive": 0.333333,
                "ahead_of_benchmark": 1,
                "best": 0.1,
                "worst": -0.1,
                "dispersion": 0.1,
                "status": "ok",
            },
        ),
        (
            [{"total_return": 0.123456789}],
            {
                "folds": 1,
                "positive": 1,
                "share_positive": 1.0,
                "ahead_of_benchmark": None,
                "best": 0.12345679,
                "worst": 0.12345679,
                "dispersion": None,
                "status": "insufficient",
            },
        ),
        (
            [{"total_return": 0.1}, {"total_return": 0.1}, {"total_return": 0.1}],
            {
                "folds": 3,
                "positive": 3,
                "share_positive": 1.0,
                "ahead_of_benchmark": None,
                "best": 0.1,
                "worst": 0.1,
                "dispersion": 0.0,
                "status": "ok",
            },
        ),
        (
            [{"total_return": 0.1}, {"total_return": 0.1, "excess_return": 0}],
            {
                "folds": 2,
                "positive": 2,
                "share_positive": 1.0,
                "ahead_of_benchmark": 0,
                "best": 0.1,
                "worst": 0.1,
                "dispersion": 0.0,
                "status": "insufficient",
            },
        ),
    ],
)
def test_consistency(folds: list[dict[str, Any]], expected: dict[str, Any]) -> None:
    assert fold_consistency(folds) == expected
