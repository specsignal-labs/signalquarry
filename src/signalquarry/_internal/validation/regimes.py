# SPDX-License-Identifier: Apache-2.0
"""Causal session labels and descriptive performance slices, without I/O."""

from __future__ import annotations

import math
from bisect import bisect_left, insort
from collections.abc import Sequence
from datetime import date
from typing import Any

import numpy as np

from signalquarry._internal.validation.evaluate import max_drawdown
from signalquarry._internal.validation.metrics import SESSIONS_PER_YEAR

UNKNOWN = "unknown"


def trend_regime(levels: np.ndarray, *, window: int = 200) -> list[str]:
    """Compare the previous level with its trailing mean; equality is ``below``.

    Labels remain unknown until a complete, finite window of prior levels exists.
    """
    if window < 1:
        raise ValueError("REGIME_WINDOW_INVALID")
    values = np.asarray(levels, dtype=np.float64)
    labels = [UNKNOWN] * len(values)
    for session in range(window, len(values)):
        prior = values[session - window : session]
        if np.isfinite(prior).all():
            labels[session] = "above" if prior[-1] > prior.mean() else "below"
    return labels


def volatility_regime(returns: np.ndarray, *, window: int = 63, min_history: int = 252) -> list[str]:
    """Rank the previous trailing sample volatility in its expanding history.

    History includes all finite trailing windows, even during warm-up. The current
    trailing value is included; ties rank by the number of strictly lower values.
    """
    if window < 2:
        raise ValueError("REGIME_WINDOW_INVALID")
    values = np.asarray(returns, dtype=np.float64)
    labels = [UNKNOWN] * len(values)
    history: list[float] = []
    for session in range(window, len(values)):
        prior = values[session - window : session]
        if not np.isfinite(prior).all():
            continue
        volatility = float(prior.std(ddof=1))
        rank = bisect_left(history, volatility)
        insort(history, volatility)
        count = len(history)
        if session < min_history or count < 3:
            continue
        labels[session] = "low" if rank < count / 3 else "high" if rank >= 2 * count / 3 else "mid"
    return labels


def calendar_regime(sessions: Sequence[date]) -> list[str]:
    """Return each session's calendar year, which is known before the session."""
    return [str(session.year) for session in sessions]


def regime_table(
    labels: Sequence[str],
    returns: np.ndarray,
    benchmark_returns: np.ndarray | None = None,
    *,
    min_sessions: int = 20,
) -> list[dict[str, Any]]:
    """Summarize each label in first-appearance order, preserving session order.

    Drawdown and compounding concatenate the selected sessions. Annual return is
    the arithmetic daily mean times 252; volatility uses the sample deviation.
    """
    if len(labels) != len(returns) or (
        benchmark_returns is not None and len(benchmark_returns) != len(returns)
    ):
        raise ValueError("REGIME_SERIES_NOT_ALIGNED")
    groups: dict[str, list[int]] = {}
    for session, label in enumerate(labels):
        groups.setdefault(label, []).append(session)
    rows: list[dict[str, Any]] = []
    for label, indices in groups.items():
        selected = returns[indices]
        count = len(indices)
        sufficient = count >= min_sessions
        total = float(np.prod(1.0 + selected) - 1.0)
        annual = float(selected.mean() * SESSIONS_PER_YEAR) if sufficient else None
        volatility = (
            float(selected.std(ddof=1) * math.sqrt(SESSIONS_PER_YEAR)) if sufficient and count > 1 else None
        )
        sharpe = annual / volatility if annual is not None and volatility else None
        benchmark = (
            float(np.prod(1.0 + benchmark_returns[indices]) - 1.0) if benchmark_returns is not None else None
        )
        rows.append(
            {
                "regime": label,
                "sessions": count,
                "share": round(count / len(returns), 6),
                "status": "ok" if sufficient else "insufficient",
                "total_return": round(total, 8),
                "annual_return": None if annual is None else round(annual, 8),
                "annual_volatility": None if volatility is None else round(volatility, 8),
                "sharpe": None if sharpe is None else round(sharpe, 6),
                "max_drawdown": round(max_drawdown(selected), 8),
                "benchmark_total_return": None if benchmark is None else round(benchmark, 8),
                "excess_return": None if benchmark is None else round(total - benchmark, 8),
            }
        )
    return rows
