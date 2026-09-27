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
