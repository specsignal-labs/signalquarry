# SPDX-License-Identifier: Apache-2.0
"""The declared benchmark as a buy-and-hold curve on a run's own sessions.

Everything here is descriptive. A benchmark comparison never enters a gate or a claim level,
and the reference run is not a trial.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any

import numpy as np

from signalquarry._internal.contracts.spec import StrategySpecV1
from signalquarry._internal.data.dataset import Dataset
from signalquarry._internal.engine.reference import buy_and_hold
from signalquarry._internal.validation.metrics import daily_returns, relative, summarize


@dataclass(frozen=True)
class BenchmarkCurve:
    symbol: str
    sessions: list[date]
    equity: list[Decimal]
    fills: int
    fees: Decimal
    invested_from: date | None

    def returns(self, initial: Decimal) -> np.ndarray:
        return daily_returns([initial, *self.equity])


def _only(dataset: Dataset, symbol: str) -> Dataset:
    """``dataset`` reduced to one symbol, so the reference run does not hash the whole panel."""
    if symbol not in dataset.series or len(dataset.series) == 1:
        return dataset
    return Dataset(
        dataset.sessions,
        {symbol: dataset.series[symbol]},
        tuple(split for split in dataset.splits if split.symbol == symbol),
        tuple(dividend for dividend in dataset.dividends if dividend.symbol == symbol),
        source=dataset.source,
    )


def benchmark_curve(
    spec: StrategySpecV1, dataset: Dataset, symbol: str, sessions: Sequence[date]
) -> BenchmarkCurve:
    """Hold ``symbol`` over ``sessions``, the sessions of the run it is compared with.

    A session the reference run does not have keeps the last known value, starting from the
    initial cash: before the benchmark can be bought it is simply uninvested.
    """
    run = buy_and_hold(spec, _only(dataset, symbol), symbol, start=sessions[0], end=sessions[-1])
    known = dict(zip(run.sessions, run.equity, strict=True))
    equity: list[Decimal] = []
    last = spec.account.initial_cash
    for session in sessions:
        last = known.get(session, last)
        equity.append(last)
    return BenchmarkCurve(
        symbol=symbol,
        sessions=list(sessions),
        equity=equity,
        fills=len(run.fills),
        fees=sum((fill.fee for fill in run.fills), Decimal(0)),
        invested_from=run.fills[0].session if run.fills else None,
    )


def benchmark_summary(curve: BenchmarkCurve, returns: np.ndarray, initial: Decimal) -> dict[str, Any]:
    """The benchmark's own metrics and the run's statistics relative to it."""
    return {
        "symbol": curve.symbol,
        "invested_from": None if curve.invested_from is None else curve.invested_from.isoformat(),
        "metrics": summarize(curve.sessions, curve.equity, initial, fills=curve.fills, fees=curve.fees),
        "relative": relative(returns, curve.returns(initial)),
    }
