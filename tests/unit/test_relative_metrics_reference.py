# SPDX-License-Identifier: Apache-2.0
"""Independent reference vectors for benchmark-relative, drawdown-episode and turnover metrics."""

from __future__ import annotations

import math
from datetime import date
from decimal import Decimal

import numpy as np
import pytest

from signalquarry._internal.validation.metrics import (
    MIN_RELATIVE_OBSERVATIONS,
    drawdown_episodes,
    relative,
    summarize,
    trading_activity,
)

# Twenty sessions: the benchmark alternates +1% and -1%; the strategy takes half of each move
# and adds 0.1% a day. Beta is therefore 0.5 and the daily alpha 0.1%, exactly.
BENCHMARK = np.array([0.01, -0.01] * 10)
STRATEGY = 0.5 * BENCHMARK + 0.001


def test_relative_reference_vector() -> None:
    tracking = 0.005 * math.sqrt(20 / 19) * math.sqrt(252)
    assert relative(STRATEGY, BENCHMARK) == {
        "observations": 20,
        "status": "ok",
        "excess_total_return": round((1.006 * 0.996) ** 10 - (1.01 * 0.99) ** 10, 8),
        "active_return": 0.252,
        "tracking_error": round(tracking, 8),
        "information_ratio": round(0.252 / tracking, 6),
        "beta": 0.5,
        "alpha": 0.252,
        "correlation": 1.0,
        "up_capture": 0.6,
        "down_capture": 0.4,
    }


def test_relative_needs_the_minimum_number_of_aligned_sessions() -> None:
    assert MIN_RELATIVE_OBSERVATIONS == 20
    short = relative(STRATEGY[:19], BENCHMARK[:19])
    assert short == {"observations": 19, "status": "insufficient"}
    assert relative(STRATEGY[:20], BENCHMARK[:20])["status"] == "ok"


def test_relative_rejects_misaligned_series() -> None:
    with pytest.raises(ValueError, match="RELATIVE_RETURNS_NOT_ALIGNED"):
        relative(STRATEGY, BENCHMARK[:-1])
    with pytest.raises(ValueError, match="RELATIVE_RETURNS_NOT_ALIGNED"):
        relative(STRATEGY.reshape(2, 10), BENCHMARK.reshape(2, 10))


def test_identical_series_have_no_tracking_error_and_no_information_ratio() -> None:
    same = relative(BENCHMARK, BENCHMARK)
    assert same["excess_total_return"] == 0.0
    assert same["active_return"] == 0.0
    assert same["tracking_error"] == 0.0
    assert same["information_ratio"] is None
    assert (same["beta"], same["alpha"], same["correlation"]) == (1.0, 0.0, 1.0)
    assert (same["up_capture"], same["down_capture"]) == (1.0, 1.0)


def test_constant_benchmark_has_no_beta_alpha_or_correlation() -> None:
    # 0.1 repeated does not sum to an exact multiple, so a variance test would see noise.
    flat = np.full(20, 0.1)
    result = relative(STRATEGY, flat)
    assert (result["beta"], result["alpha"], result["correlation"]) == (None, None, None)
    assert result["down_capture"] is None
    assert result["up_capture"] == round(float(np.mean(STRATEGY)) / 0.1, 6)
    assert result["tracking_error"] > 0

    constant_strategy = relative(flat, BENCHMARK)
    assert constant_strategy["beta"] == 0.0
    assert constant_strategy["correlation"] is None


def test_capture_is_missing_when_the_benchmark_never_moves_that_way() -> None:
    rising = np.linspace(0.001, 0.02, 20)
    result = relative(rising * 2, rising)
    assert result["down_capture"] is None
    assert result["up_capture"] == 2.0
    assert result["beta"] == 2.0

    falling = relative(-rising * 2, -rising)
    assert falling["up_capture"] is None
    assert falling["down_capture"] == 2.0


SESSIONS = [date(2025, 1, day) for day in (2, 3, 6, 7, 8, 9, 10, 13, 14)]
EQUITY = [Decimal(v) for v in ("110", "99", "104.5", "121", "108.9", "108.9", "96.8", "130", "117")]


def test_drawdown_episodes_reference_vector() -> None:
    # 110 -> 99 recovers at 121; 121 -> 96.8 recovers at 130; 130 -> 117 is still open.
    assert drawdown_episodes(SESSIONS, EQUITY, Decimal("100")) == [
        {
            "peak": "2025-01-07",
            "trough": "2025-01-10",
            "recovery": "2025-01-13",
            "depth": 0.2,
            "sessions_to_trough": 3,
            "sessions_to_recovery": 1,
        },
        {
            "peak": "2025-01-02",
            "trough": "2025-01-03",
            "recovery": "2025-01-07",
            "depth": 0.1,
            "sessions_to_trough": 1,
            "sessions_to_recovery": 2,
        },
        {
            "peak": "2025-01-13",
            "trough": "2025-01-14",
            "recovery": None,
            "depth": 0.1,
            "sessions_to_trough": 1,
            "sessions_to_recovery": None,
        },
    ]


def test_drawdown_episodes_keep_the_deepest_and_match_the_summary() -> None:
    top = drawdown_episodes(SESSIONS, EQUITY, Decimal("100"), top=1)
    assert [item["depth"] for item in top] == [0.2]
    assert top[0]["depth"] == summarize(SESSIONS, EQUITY, Decimal("100"))["max_drawdown"]
    assert drawdown_episodes(SESSIONS, EQUITY, Decimal("100"), top=0) == []


def test_drawdown_from_the_starting_capital_has_no_peak_session() -> None:
    sessions = [date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6)]
    equity = [Decimal("90"), Decimal("95"), Decimal("100")]
    assert drawdown_episodes(sessions, equity, Decimal("100")) == [
        {
            "peak": None,
            "trough": "2025-01-02",
            "recovery": "2025-01-06",
            "depth": 0.1,
            "sessions_to_trough": 1,
            "sessions_to_recovery": 2,
        }
    ]


def test_drawdown_peak_is_the_last_session_at_the_high() -> None:
    sessions = [date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6)]
    equity = [Decimal("110"), Decimal("110"), Decimal("99")]
    (episode,) = drawdown_episodes(sessions, equity, Decimal("100"))
    assert (episode["peak"], episode["sessions_to_trough"]) == ("2025-01-03", 1)


def test_no_drawdown_without_a_decline() -> None:
    sessions = [date(2025, 1, 2), date(2025, 1, 3)]
    assert drawdown_episodes(sessions, [Decimal("100"), Decimal("101")], Decimal("100")) == []
    assert drawdown_episodes([], [], Decimal("100")) == []


def test_trading_activity_reference_vector() -> None:
    # 365 days is 365/365.25 years; mean equity is 200; one-way turnover uses the lesser side.
    sessions = [date(2024, 1, 2), date(2025, 1, 1)]
    equity = [Decimal("100"), Decimal("300")]
    years = 365 / 365.25
    turnover = 300 / 200 / years
    assert trading_activity(
        sessions, equity, bought=Decimal("500"), sold=Decimal("300"), fees=Decimal("2")
    ) == {
        "status": "ok",
        "bought": "500",
        "sold": "300",
        "turnover": round(turnover, 6),
        "implied_holding_sessions": round(252 / turnover, 1),
        "fee_drag": round(2 / 200 / years, 8),
    }
    swapped = trading_activity(
        sessions, equity, bought=Decimal("300"), sold=Decimal("500"), fees=Decimal("2")
    )
    assert swapped["turnover"] == round(turnover, 6)


def test_trading_activity_boundaries() -> None:
    sessions = [date(2024, 1, 2), date(2025, 1, 1)]
    never_sold = trading_activity(
        sessions,
        [Decimal("100"), Decimal("300")],
        bought=Decimal("500"),
        sold=Decimal("0"),
        fees=Decimal("0"),
    )
    assert never_sold["turnover"] == 0.0
    assert never_sold["implied_holding_sessions"] is None
    assert never_sold["fee_drag"] == 0.0

    assert trading_activity([], [], bought=Decimal(0), sold=Decimal(0), fees=Decimal(0)) == {
        "status": "insufficient"
    }
    wiped = trading_activity(
        sessions, [Decimal("0"), Decimal("0")], bought=Decimal(1), sold=Decimal(1), fees=Decimal(0)
    )
    assert wiped == {"status": "insufficient"}

    # A single session is annualized over one day, like the summary's CAGR.
    one_day = trading_activity(
        [date(2025, 1, 2)], [Decimal("100")], bought=Decimal("10"), sold=Decimal("10"), fees=Decimal("1")
    )
    assert one_day["turnover"] == round(10 / 100 * 365.25, 6)
    assert one_day["fee_drag"] == round(1 / 100 * 365.25, 8)
