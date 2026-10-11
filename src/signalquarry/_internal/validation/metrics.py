# SPDX-License-Identifier: Apache-2.0
"""Performance metrics from a daily equity curve. Definitions are documented and fixed.

* Returns are simple daily returns of end-of-day equity.
* Annualization uses 252 sessions per year; the risk-free rate is 0.
* ``cagr`` uses calendar years between the first and last session.
* ``max_drawdown`` is the largest peak-to-trough decline (a positive fraction).
"""

from __future__ import annotations

import math
from datetime import date
from decimal import Decimal
from typing import Any

import numpy as np

SESSIONS_PER_YEAR = 252


def daily_returns(equity: list[Decimal]) -> np.ndarray:
    values = np.array([float(item) for item in equity])
    if len(values) < 2:
        return np.array([])
    return values[1:] / values[:-1] - 1.0


def summarize(
    sessions: list[date],
    equity: list[Decimal],
    initial: Decimal,
    *,
    fills: int = 0,
    fees: Decimal = Decimal(0),
) -> dict[str, Any]:
    if not equity:
        return {"sessions": 0}
    values = np.array([float(item) for item in equity])
    returns = daily_returns([initial, *equity])
    peak = np.maximum.accumulate(np.concatenate(([float(initial)], values)))
    drawdown = 1.0 - np.concatenate(([float(initial)], values)) / peak
    years = max((sessions[-1] - sessions[0]).days / 365.25, 1 / 365.25)
    total = values[-1] / float(initial) - 1.0
    vol = float(np.std(returns, ddof=1) * math.sqrt(SESSIONS_PER_YEAR)) if len(returns) > 1 else None
    mean = float(np.mean(returns)) if len(returns) else 0.0
    downside = returns[returns < 0]
    sharpe = (mean * SESSIONS_PER_YEAR) / vol if vol else None
    sortino_den = (
        float(np.sqrt(np.mean(np.square(downside))) * math.sqrt(SESSIONS_PER_YEAR)) if len(downside) else None
    )
    max_dd = float(drawdown.max())
    cagr = (values[-1] / float(initial)) ** (1 / years) - 1.0 if values[-1] > 0 else -1.0
    return {
        "start": sessions[0].isoformat(),
        "end": sessions[-1].isoformat(),
        "sessions": len(sessions),
        "total_return": round(float(total), 8),
        "cagr": round(float(cagr), 8),
        "annual_volatility": None if vol is None else round(vol, 8),
        "sharpe": None if sharpe is None else round(float(sharpe), 6),
        "sortino": None if not sortino_den else round(float(mean * SESSIONS_PER_YEAR / sortino_den), 6),
        "max_drawdown": round(max_dd, 8),
        "calmar": None if max_dd == 0 else round(float(cagr / max_dd), 6),
        "fills": fills,
        "fees": str(fees),
        "final_equity": str(equity[-1]),
    }


# Fewer aligned sessions than this and a relative statistic is reported as ``insufficient``.
MIN_RELATIVE_OBSERVATIONS = 20


def _ratio(numerator: float, denominator: float, digits: int = 6) -> float | None:
    if denominator == 0 or not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    return round(numerator / denominator, digits)


def relative(returns: np.ndarray, benchmark_returns: np.ndarray) -> dict[str, Any]:
    """Statistics of a return series against a benchmark over the same sessions.

    * ``excess_total_return`` is the difference of the two compounded returns.
    * ``active_return`` is the mean daily difference, annualized; ``tracking_error`` is the
      annualized sample standard deviation of that difference; their ratio is the
      ``information_ratio``.
    * ``beta`` and ``alpha`` come from the least-squares line of the returns on the
      benchmark returns (``alpha`` annualized, risk-free rate 0).
    * ``up_capture`` and ``down_capture`` divide the mean return by the mean benchmark
      return on the sessions where the benchmark rose or fell.

    Descriptive only: none of these enters a gate or a claim level.
    """
    if returns.shape != benchmark_returns.shape or returns.ndim != 1:
        raise ValueError("RELATIVE_RETURNS_NOT_ALIGNED")
    n = len(returns)
    if n < MIN_RELATIVE_OBSERVATIONS:
        return {"observations": n, "status": "insufficient"}
    active = returns - benchmark_returns
    # A constant series has exactly zero dispersion; the floating-point variance may not.
    benchmark_varies = bool(np.ptp(benchmark_returns) > 0)
    returns_vary = bool(np.ptp(returns) > 0)
    tracking = float(np.std(active, ddof=1) * math.sqrt(SESSIONS_PER_YEAR)) if np.ptp(active) > 0 else 0.0
    active_annual = float(np.mean(active) * SESSIONS_PER_YEAR)
    beta = (
        float(np.cov(returns, benchmark_returns, ddof=1)[0, 1] / np.var(benchmark_returns, ddof=1))
        if benchmark_varies
        else None
    )
    up, down = benchmark_returns > 0, benchmark_returns < 0
    return {
        "observations": n,
        "status": "ok",
        "excess_total_return": round(float(np.prod(1.0 + returns) - np.prod(1.0 + benchmark_returns)), 8),
        "active_return": round(active_annual, 8),
        "tracking_error": round(tracking, 8),
        "information_ratio": _ratio(active_annual, tracking),
        "beta": None if beta is None else round(beta, 6),
        "alpha": None
        if beta is None
        else round(float((np.mean(returns) - beta * np.mean(benchmark_returns)) * SESSIONS_PER_YEAR), 8),
        "correlation": round(float(np.corrcoef(returns, benchmark_returns)[0, 1]), 6)
        if benchmark_varies and returns_vary
        else None,
        "up_capture": _ratio(float(np.mean(returns[up])), float(np.mean(benchmark_returns[up])))
        if up.any()
        else None,
        "down_capture": _ratio(float(np.mean(returns[down])), float(np.mean(benchmark_returns[down])))
        if down.any()
        else None,
    }


def drawdown_episodes(
    sessions: list[date], equity: list[Decimal], initial: Decimal, *, top: int = 5
) -> list[dict[str, Any]]:
    """The ``top`` deepest peak-to-recovery declines, deepest first (earlier first on ties).

    ``peak`` is the last session at the high before the decline, or ``None`` when that high
    is the starting capital. ``recovery`` is the first session back at or above the peak, or
    ``None`` while the curve is still below it at the end.
    """
    values = [float(initial), *(float(item) for item in equity)]
    episodes: list[dict[str, Any]] = []
    peak, trough = 0, 0

    def day(index: int) -> str:
        return sessions[index - 1].isoformat()

    def close(recovery: int | None) -> None:
        episodes.append(
            {
                "peak": None if peak == 0 else day(peak),
                "trough": day(trough),
                "recovery": None if recovery is None else day(recovery),
                "depth": round(1.0 - values[trough] / values[peak], 8),
                "sessions_to_trough": trough - peak,
                "sessions_to_recovery": None if recovery is None else recovery - trough,
            }
        )

    falling = False
    for index in range(1, len(values)):
        if values[index] >= values[peak]:
            if falling:
                close(index)
                falling = False
            peak = index
        elif not falling:
            falling, trough = True, index
        elif values[index] < values[trough]:
            trough = index
    if falling:
        close(None)
    episodes.sort(key=lambda item: -item["depth"])
    return episodes[:top]


def trading_activity(
    sessions: list[date],
    equity: list[Decimal],
    *,
    bought: Decimal,
    sold: Decimal,
    fees: Decimal,
) -> dict[str, Any]:
    """Turnover and cost drag from the total notional bought and sold.

    ``turnover`` is one-way and annualized: the lesser of purchases and sales, divided by
    the mean end-of-day equity and by the calendar years covered. ``fee_drag`` is the fees
    paid per year as a fraction of the same mean equity.
    """
    if not equity:
        return {"status": "insufficient"}
    mean_equity = float(np.mean([float(item) for item in equity]))
    years = max((sessions[-1] - sessions[0]).days / 365.25, 1 / 365.25)
    if mean_equity <= 0:
        return {"status": "insufficient"}
    turnover = float(min(bought, sold)) / mean_equity / years
    return {
        "status": "ok",
        "bought": str(bought),
        "sold": str(sold),
        "turnover": round(turnover, 6),
        "implied_holding_sessions": None if turnover == 0 else round(SESSIONS_PER_YEAR / turnover, 1),
        "fee_drag": round(float(fees) / mean_equity / years, 8),
    }
