# SPDX-License-Identifier: Apache-2.0
"""In-memory daily dataset aligned on one session calendar.

Prices are stored as int64 micro-units (1e-6) of the raw, unadjusted trade
prices; corporate actions are stored separately and applied point-in-time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

import numpy as np

from signalquarry._internal.canonical import canonical_hash

MICRO = 1_000_000
T_PLUS_ONE_FROM = date(2024, 5, 28)
FIELDS = ("open", "high", "low", "close")


@dataclass(frozen=True)
class Split:
    symbol: str
    ex_date: date
    ratio: Decimal  # new shares per old share


@dataclass(frozen=True)
class Dividend:
    symbol: str
    ex_date: date
    pay_date: date
    amount: Decimal  # cash per share held before the ex-date


@dataclass(frozen=True)
class SymbolSeries:
    micro: dict[str, np.ndarray]  # field -> int64[n]; valid only where present
    volume: np.ndarray  # float64[n]
    present: np.ndarray  # bool[n]


@dataclass
class Dataset:
    sessions: tuple[date, ...]
    series: dict[str, SymbolSeries]
    splits: tuple[Split, ...] = ()
    dividends: tuple[Dividend, ...] = ()
    source: str = "unknown"
    _cumulative_split: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if list(self.sessions) != sorted(set(self.sessions)):
            raise ValueError("DATASET_SESSIONS_NOT_STRICTLY_INCREASING")
        n = len(self.sessions)
        for symbol, item in self.series.items():
            if len(item.present) != n or any(len(item.micro[name]) != n for name in FIELDS):
                raise ValueError(f"DATASET_SERIES_LENGTH_MISMATCH:{symbol}")
        index = {session: i for i, session in enumerate(self.sessions)}
        for symbol in self.series:
            factor = np.ones(n)
            for split in self.splits:
                if split.symbol == symbol and split.ex_date in index:
                    factor[index[split.ex_date] :] *= float(split.ratio)
            self._cumulative_split[symbol] = factor

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self.series))

    def index_of(self, session: date) -> int:
        return self.sessions.index(session)

    def cumulative_split(self, symbol: str) -> np.ndarray:
        """F(t): product of split ratios with ex-date <= session t (float)."""
        return self._cumulative_split[symbol]

    def price(self, symbol: str, field_name: str, index: int) -> Decimal | None:
        item = self.series[symbol]
        if not item.present[index]:
            return None
        return Decimal(int(item.micro[field_name][index])).scaleb(-6)

    def settlement_index(self, trade_index: int) -> int:
        """Index of the session on which a trade on ``trade_index`` settles (T+2 before 2024-05-28, T+1 after)."""
        lag = 2 if self.sessions[trade_index] < T_PLUS_ONE_FROM else 1
        return trade_index + lag

    def identity(self) -> str:
        return canonical_hash(
            {
                "source": self.source,
                "sessions": [s.isoformat() for s in self.sessions],
                "series": {
                    symbol: {
                        **{name: item.micro[name].tolist() for name in FIELDS},
                        "volume": item.volume.tolist(),
                        "present": item.present.tolist(),
                    }
                    for symbol, item in sorted(self.series.items())
                },
                "splits": [[s.symbol, s.ex_date.isoformat(), s.ratio] for s in self.splits],
                "dividends": [
                    [d.symbol, d.ex_date.isoformat(), d.pay_date.isoformat(), d.amount]
                    for d in self.dividends
                ],
            }
        )


def with_pending_session(dataset: Dataset, session: date) -> Dataset:
    """Append ``session`` with no bars: the shape the engine decides on before that session's open."""
    if dataset.sessions and session <= dataset.sessions[-1]:
        raise ValueError("DATASET_PENDING_SESSION_NOT_AFTER_END")
    series = {
        symbol: SymbolSeries(
            micro={name: np.append(item.micro[name], np.int64(0)) for name in FIELDS},
            volume=np.append(item.volume, 0.0),
            present=np.append(item.present, False),
        )
        for symbol, item in dataset.series.items()
    }
    return Dataset(
        (*dataset.sessions, session), series, dataset.splits, dataset.dividends, source=dataset.source
    )


def truncated(dataset: Dataset, stop: int) -> Dataset:
    """The first ``stop`` sessions of ``dataset``; the engine ignores actions dated after them."""
    sessions = dataset.sessions[:stop]
    series = {
        symbol: SymbolSeries(
            micro={name: item.micro[name][:stop].copy() for name in FIELDS},
            volume=item.volume[:stop].copy(),
            present=item.present[:stop].copy(),
        )
        for symbol, item in dataset.series.items()
    }
    return Dataset(sessions, series, dataset.splits, dataset.dividends, source=dataset.source)
