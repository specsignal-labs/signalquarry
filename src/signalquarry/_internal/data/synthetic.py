# SPDX-License-Identifier: Apache-2.0
"""Deterministic synthetic market data for demos and tests. Needs no credentials.

Symbols: ``SYNA`` (steady trend with quarterly dividends), ``SYNB`` (volatile,
2-for-1 split), ``SYNC`` (mean-reverting), ``SYNX`` (bear then recovery).
Prices are regime-switching geometric random walks, rounded to cents. Any other
symbol (e.g. ``SPY`` in ``sqy check``) gets a deterministic profile chosen by its name;
its data is synthetic and says nothing about the real instrument.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import numpy as np

from signalquarry._internal.canonical import canonical_hash
from signalquarry._internal.data.dataset import FIELDS, MICRO, Dataset, Dividend, Split, SymbolSeries

SYMBOLS = ("SYNA", "SYNB", "SYNC", "SYNX")
_PROFILE = {
    "SYNA": {"start": 100.0, "drifts": (0.0006, -0.0002), "vol": 0.010},
    "SYNB": {"start": 250.0, "drifts": (0.0009, -0.0012), "vol": 0.025},
    "SYNC": {"start": 50.0, "drifts": (0.0001, -0.0001), "vol": 0.012},
    "SYNX": {"start": 80.0, "drifts": (0.0007, -0.0015), "vol": 0.018},
}


def weekday_sessions(start: date, end: date) -> tuple[date, ...]:
    days, current = [], start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return tuple(days)


def _rng(symbol: str, seed: int) -> np.random.Generator:
    digest = canonical_hash({"symbol": symbol, "seed": seed})
    return np.random.Generator(np.random.PCG64(int(digest[7:23], 16)))


def synthetic_dataset(
    start: date = date(2014, 1, 2),
    end: date = date(2025, 12, 31),
    *,
    seed: int = 7,
    symbols: tuple[str, ...] = SYMBOLS,
) -> Dataset:
    sessions = weekday_sessions(start, end)
    n = len(sessions)
    series: dict[str, SymbolSeries] = {}
    splits: list[Split] = []
    dividends: list[Dividend] = []
    for symbol in symbols:
        profile_name = (
            symbol
            if symbol in _PROFILE
            else SYMBOLS[int(canonical_hash({"profile": symbol})[7:15], 16) % len(SYMBOLS)]
        )
        profile = _PROFILE[profile_name]
        rng = _rng(symbol, seed)
        regime = np.zeros(n, dtype=int)
        for i in range(1, n):
            regime[i] = regime[i - 1] if rng.random() > 0.01 else 1 - regime[i - 1]
        drift = np.where(regime == 0, profile["drifts"][0], profile["drifts"][1])
        shocks = rng.normal(0.0, profile["vol"], n)
        if profile_name == "SYNC":
            log_close = np.empty(n)
            log_close[0] = np.log(profile["start"])
            for i in range(1, n):
                log_close[i] = (
                    log_close[i - 1] + 0.05 * (np.log(profile["start"]) - log_close[i - 1]) + shocks[i]
                )
        else:
            log_close = np.log(profile["start"]) + np.cumsum(drift + shocks)
        if profile_name == "SYNX":
            half = n // 2
            log_close[:half] -= np.linspace(0, 0.8, half)
            log_close[half:] -= 0.8 - np.linspace(0, 0.9, n - half)
        close = np.exp(log_close)
        gap = rng.normal(0.0, profile["vol"] / 3, n)
        open_ = np.concatenate(([close[0]], close[:-1] * np.exp(gap[1:])))
        span = np.abs(rng.normal(0.0, profile["vol"], n))
        high = np.maximum(open_, close) * (1 + span / 2)
        low = np.minimum(open_, close) * (1 - span / 2)
        if profile_name == "SYNB":
            split_index = n // 2
            splits.append(Split(symbol, sessions[split_index], Decimal(2)))
            for array in (open_, high, low, close):
                array[split_index:] /= 2.0
        if profile_name == "SYNA":
            for i in range(60, n, 63):
                pay = min(i + 10, n - 1)
                dividends.append(Dividend(symbol, sessions[i], sessions[pay], Decimal("0.35")))
        cents = {
            name: np.round(values, 2) for name, values in zip(FIELDS, (open_, high, low, close), strict=True)
        }
        series[symbol] = SymbolSeries(
            micro={name: np.rint(values * MICRO).astype(np.int64) for name, values in cents.items()},
            volume=np.round(rng.lognormal(13.0, 0.4, n)),
            present=np.ones(n, dtype=bool),
        )
    return Dataset(sessions, series, tuple(splits), tuple(dividends), source=f"synthetic:seed={seed}")
