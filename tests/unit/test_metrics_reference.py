# SPDX-License-Identifier: Apache-2.0
"""Independent reference vectors for daily equity-curve metrics."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import numpy as np

from signalquarry._internal.validation.metrics import daily_returns, summarize


def test_daily_returns_include_the_two_point_boundary() -> None:
    single = daily_returns([Decimal("100"), Decimal("110")])
    assert single.shape == (1,)
    np.testing.assert_allclose(single, [0.1], rtol=0, atol=1e-12)

    path = daily_returns([Decimal("100"), Decimal("110"), Decimal("99"), Decimal("108.90")])
    assert path.shape == (3,)
    np.testing.assert_allclose(path, [0.1, -0.1, 0.1], rtol=0, atol=1e-12)


def test_summary_reference_vector_with_downside_and_drawdown() -> None:
    # Daily returns: +10%, -10%, +10%, -20%; sample standard deviation is 15%.
    # Peak equity is 110, and the final 87.12 is 20.8% below that peak.
    sessions = [date(2024, 1, 2), date(2024, 4, 2), date(2024, 9, 2), date(2025, 1, 1)]
    equity = [Decimal("110"), Decimal("99"), Decimal("108.90"), Decimal("87.12")]
    assert summarize(sessions, equity, Decimal("100")) == {
        "start": "2024-01-02",
        "end": "2025-01-01",
        "sessions": 4,
        "total_return": -0.1288,
        "cagr": -0.12888227,
        "annual_volatility": 2.38117618,
        "sharpe": -2.645751,
        "sortino": -2.50998,
        "max_drawdown": 0.208,
        "calmar": -0.619626,
        "fills": 0,
        "fees": "0",
        "final_equity": "87.12",
    }
    with_costs = summarize(sessions, equity, Decimal("100"), fills=2, fees=Decimal("1.25"))
    assert (with_costs["fills"], with_costs["fees"]) == (2, "1.25")


def test_short_and_zero_equity_boundaries() -> None:
    flat = summarize([date(2025, 1, 2)], [Decimal("100")], Decimal("100"))
    assert flat["total_return"] == 0.0
    assert flat["cagr"] == 0.0
    assert flat["annual_volatility"] is None
    assert flat["sharpe"] is None
    assert flat["sortino"] is None
    assert flat["max_drawdown"] == 0.0
    assert flat["calmar"] is None

    two = summarize([date(2025, 1, 2), date(2025, 1, 3)], [Decimal("110"), Decimal("121")], Decimal("100"))
    assert two["sessions"] == 2
    assert two["annual_volatility"] == 0.0
    assert two["sharpe"] is None

    zero = summarize([date(2025, 1, 2)], [Decimal("0")], Decimal("100"))
    assert zero["total_return"] == -1.0
    assert zero["cagr"] == -1.0
    assert zero["max_drawdown"] == 1.0


def test_cagr_handles_subunit_terminal_equity_and_one_session_annualization() -> None:
    single_session = summarize([date(2025, 1, 2)], [Decimal("101")], Decimal("100"))
    assert single_session["cagr"] == round(1.01**365.25 - 1.0, 8)

    sessions = [date(2024, 1, 2), date(2025, 1, 2)]
    near_zero = summarize(sessions, [Decimal("50"), Decimal("0.5")], Decimal("100"))
    years = (sessions[-1] - sessions[0]).days / 365.25
    assert near_zero["cagr"] == round((0.5 / 100) ** (1 / years) - 1.0, 8)


def test_summary_keeps_fixed_eight_decimal_metric_precision() -> None:
    result = summarize(
        [date(2024, 1, 2), date(2024, 1, 3)],
        [Decimal("110"), Decimal("96.41975321")],
        Decimal("100"),
    )

    assert result["total_return"] == -0.03580247
    assert result["annual_volatility"] == 2.50829624
    assert result["max_drawdown"] == 0.12345679


def test_zero_return_is_excluded_from_downside_deviation() -> None:
    # Return path [0%, -10%]; downside deviation includes only the -10% observation.
    result = summarize([date(2025, 1, 2), date(2025, 1, 3)], [Decimal("100"), Decimal("90")], Decimal("100"))
    assert result["sortino"] == -7.937254
