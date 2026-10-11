# SPDX-License-Identifier: Apache-2.0
"""Hand-worked reference cases for causal regime labels and performance slices."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from signalquarry._internal.validation.regimes import (
    UNKNOWN,
    calendar_regime,
    regime_table,
    trend_regime,
    volatility_regime,
)


def test_trend_warm_up_requires_exactly_window_prior_levels() -> None:
    assert trend_regime(np.array([1.0, 2.0, 3.0, 4.0]), window=3) == [
        UNKNOWN,
        UNKNOWN,
        UNKNOWN,
        "above",
    ]


def test_trend_below_and_equal_to_mean() -> None:
    assert trend_regime(np.array([3.0, 2.0, 1.0, 1.0, 1.0]), window=2) == [
        UNKNOWN,
        UNKNOWN,
        "below",
        "below",
        "below",
    ]


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_trend_non_finite_window_is_unknown_until_it_leaves(invalid: float) -> None:
    levels = np.array([1.0, invalid, 3.0, 4.0, 5.0])
    assert trend_regime(levels, window=2) == [UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, "above"]


@pytest.mark.parametrize("session", [0, 1, 3, 4, 6])
def test_trend_labels_do_not_look_at_current_or_future_levels(session: int) -> None:
    levels = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 8.0])
    original = trend_regime(levels, window=3)
    changed = levels.copy()
    changed[session:] = [np.nan if index % 2 else 1000.0 for index in range(len(levels) - session)]
    assert trend_regime(changed, window=3)[: session + 1] == original[: session + 1]


def test_trend_default_window_is_200() -> None:
    labels = trend_regime(np.arange(1.0, 202.0))
    assert labels == [UNKNOWN] * 200 + ["above"]


def test_trend_one_level_window_is_below() -> None:
    assert trend_regime(np.array([1.0, 2.0]), window=1) == [UNKNOWN, "below"]


def test_trend_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError, match="REGIME_WINDOW_INVALID"):
        trend_regime(np.array([1.0]), window=0)


def test_volatility_warm_up_includes_earlier_trailing_values() -> None:
    # Trailing standard deviations are 1, 2, 3, 4, 5 divided by sqrt(2).
    # At session 5 there are exactly 5 prior returns and 4 trailing values.
    assert volatility_regime(np.array([0.0, 1.0, 3.0, 6.0, 10.0, 15.0]), window=2, min_history=5) == [
        UNKNOWN,
        UNKNOWN,
        UNKNOWN,
        UNKNOWN,
        UNKNOWN,
        "high",
    ]


def test_volatility_needs_three_trailing_values_even_after_warm_up() -> None:
    assert volatility_regime(np.array([0.0, 1.0, 3.0, 6.0, 10.0]), window=2, min_history=1) == [
        UNKNOWN,
        UNKNOWN,
        UNKNOWN,
        UNKNOWN,
        "high",
    ]


@pytest.mark.parametrize(
    ("returns", "expected"),
    [
        ([0.0, 3.0, 5.0, 6.0, 7.0], "low"),
        ([0.0, 1.0, 4.0, 6.0, 7.0], "mid"),
        ([0.0, 1.0, 3.0, 6.0, 7.0], "high"),
    ],
)
def test_volatility_thirds_rank_includes_the_current_trailing_value(
    returns: list[float], expected: str
) -> None:
    # At session 4, the three differences rank 0, 1 or 2 respectively.
    assert volatility_regime(np.array(returns), window=2, min_history=4)[4] == expected


def test_volatility_ties_use_strictly_lower_rank() -> None:
    assert volatility_regime(np.zeros(7), window=2, min_history=4) == [UNKNOWN] * 4 + ["low"] * 3


def test_volatility_tie_at_lower_third_boundary_is_mid() -> None:
    # Differences [1, 2, 3, 4, 5, 3] give rank 2 out of 6, not rank 3.
    returns = np.array([0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 18.0, 0.0])
    assert volatility_regime(returns, window=2, min_history=7)[7] == "mid"


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_volatility_skips_non_finite_windows_in_expanding_history(invalid: float) -> None:
    returns = np.array([0.0, 1.0, invalid, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert volatility_regime(returns, window=2, min_history=2) == [UNKNOWN] * 6 + ["low", "low"]


@pytest.mark.parametrize("session", [0, 1, 3, 4, 7, 10])
def test_volatility_labels_do_not_look_at_current_or_future_returns(session: int) -> None:
    returns = np.array([0.01, -0.02, 0.03, 0.0, 0.05, -0.01, 0.0, 0.03, -0.04, 0.01, 0.02, 0.1])
    original = volatility_regime(returns, window=3, min_history=5)
    changed = returns.copy()
    changed[session:] = [np.inf if index % 2 else -100.0 for index in range(len(returns) - session)]
    assert volatility_regime(changed, window=3, min_history=5)[: session + 1] == original[: session + 1]


def test_volatility_default_warm_up_is_252_prior_returns() -> None:
    assert volatility_regime(np.zeros(253)) == [UNKNOWN] * 252 + ["low"]


def test_volatility_rejects_window_without_sample_deviation() -> None:
    with pytest.raises(ValueError, match="REGIME_WINDOW_INVALID"):
        volatility_regime(np.array([0.0]), window=1)


def test_calendar_regime_reports_year_in_session_order() -> None:
    assert calendar_regime([date(2024, 12, 31), date(2025, 1, 2), date(2024, 1, 2)]) == [
        "2024",
        "2025",
        "2024",
    ]


def test_empty_label_series() -> None:
    assert trend_regime(np.array([])) == []
    assert volatility_regime(np.array([])) == []
    assert calendar_regime([]) == []


def test_regime_table_reference_with_benchmark() -> None:
    # 1.1 * .9 * 1.1 * .8 = .8712; mean is -.025, sample std is .15.
    # Benchmark: 1.02**4 = 1.08243216. Excess is -.1288 - .08243216.
    assert regime_table(
        ["below"] * 4, np.array([0.1, -0.1, 0.1, -0.2]), np.full(4, 0.02), min_sessions=4
    ) == [
        {
            "regime": "below",
            "sessions": 4,
            "share": 1.0,
            "status": "ok",
            "total_return": -0.1288,
            "annual_return": -6.3,
            "annual_volatility": 2.38117618,
            "sharpe": -2.645751,
            "max_drawdown": 0.208,
            "benchmark_total_return": 0.08243216,
            "excess_return": -0.21123216,
        }
    ]


def test_regime_table_first_appearance_and_selected_session_compounding() -> None:
    rows = regime_table(["z", "a", "z", "a"], np.array([0.1, -0.1, 0.1, -0.2]))
    assert [row["regime"] for row in rows] == ["z", "a"]
    assert [row["sessions"] for row in rows] == [2, 2]
    assert [row["share"] for row in rows] == [0.5, 0.5]
    assert [row["total_return"] for row in rows] == [0.21, -0.28]
    assert [row["max_drawdown"] for row in rows] == [0.0, 0.28]


def test_regime_table_benchmark_is_sliced_on_the_same_sessions() -> None:
    rows = regime_table(["z", "a", "z"], np.array([0.1, 0.5, -0.1]), np.array([0.2, -0.5, -0.2]))
    assert [row["benchmark_total_return"] for row in rows] == [-0.04, -0.5]
    assert [row["excess_return"] for row in rows] == [0.03, 1.0]


def test_regime_table_without_benchmark() -> None:
    row = regime_table(["above", "above"], np.array([0.1, -0.1]), min_sessions=2)[0]
    assert row["benchmark_total_return"] is None
    assert row["excess_return"] is None


def test_regime_table_insufficient_row_keeps_counts_and_compounded_returns() -> None:
    assert regime_table(["above"] * 2, np.array([0.1, -0.1]), np.array([0.2, -0.2]), min_sessions=3) == [
        {
            "regime": "above",
            "sessions": 2,
            "share": 1.0,
            "status": "insufficient",
            "total_return": -0.01,
            "annual_return": None,
            "annual_volatility": None,
            "sharpe": None,
            "max_drawdown": 0.1,
            "benchmark_total_return": -0.04,
            "excess_return": 0.03,
        }
    ]


def test_regime_table_default_minimum_includes_twentieth_session() -> None:
    rows = regime_table(["enough"] * 20 + ["short"] * 19, np.zeros(39))
    assert [row["status"] for row in rows] == ["ok", "insufficient"]
    assert [row["annual_return"] for row in rows] == [0.0, None]
    assert [row["share"] for row in rows] == [0.512821, 0.487179]


def test_regime_table_zero_volatility_has_no_sharpe() -> None:
    row = regime_table(["above"] * 2, np.array([0.1, 0.1]), min_sessions=2)[0]
    assert row["annual_return"] == 25.2
    assert row["annual_volatility"] == 0.0
    assert row["sharpe"] is None


def test_regime_table_single_session_has_no_sample_volatility() -> None:
    row = regime_table(["below"], np.array([-0.1]), min_sessions=1)[0]
    assert row["status"] == "ok"
    assert row["annual_return"] == -25.2
    assert row["annual_volatility"] is None
    assert row["sharpe"] is None
    assert row["max_drawdown"] == 0.1


def test_regime_table_returns_round_to_eight_decimals() -> None:
    row = regime_table(["above"], np.array([0.0123456789]), np.array([0.00123456789]), min_sessions=1)[0]
    assert row["total_return"] == 0.01234568
    assert row["annual_return"] == 3.11111108
    assert row["benchmark_total_return"] == 0.00123457
    assert row["excess_return"] == 0.01111111


@pytest.mark.parametrize(
    ("labels", "returns", "benchmark"),
    [(["above"], [0.1, 0.2], None), (["above"], [0.1], []), ([], [], [0.1])],
)
def test_regime_table_rejects_misaligned_series(
    labels: list[str], returns: list[float], benchmark: list[float] | None
) -> None:
    with pytest.raises(ValueError, match="^REGIME_SERIES_NOT_ALIGNED$"):
        regime_table(labels, np.array(returns), None if benchmark is None else np.array(benchmark))


def test_regime_table_empty_input() -> None:
    assert regime_table([], np.array([])) == []
    assert regime_table([], np.array([]), np.array([])) == []
