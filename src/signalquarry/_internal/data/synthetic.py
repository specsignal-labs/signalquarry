# SPDX-License-Identifier: Apache-2.0
"""Deterministic synthetic market data for demos and tests. Needs no credentials.

Symbols: ``SYNA`` (steady trend with quarterly dividends), ``SYNB`` (volatile,
2-for-1 split), ``SYNC`` (mean-reverting), ``SYNX`` (bear then recovery).
Prices are regime-switching geometric random walks, rounded to cents. Any other
symbol (e.g. ``SPY`` in ``sqy check``) gets a deterministic profile chosen by its name;
its data is synthetic and says nothing about the real instrument.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class SyntheticPanel:
    """A daily research panel; signal columns follow ``symbols``.

    The characteristic is defined even for absent bars, but is observable only
    where the dataset's ``present`` mask is true.
    """

    dataset: Dataset
    symbols: tuple[str, ...]
    sectors: dict[str, str]
    listed: dict[str, tuple[date, date | None]]
    signal: np.ndarray


def _panel_integer(name: str, value: int, minimum: int, maximum: int | None = None) -> None:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"SYNTHETIC_PANEL_ARGUMENT_INVALID:{name}")


def _panel_fraction(name: str, value: float, maximum: float = 1.0) -> None:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not 0.0 <= value <= maximum
    ):
        raise ValueError(f"SYNTHETIC_PANEL_ARGUMENT_INVALID:{name}")


def synthetic_panel(
    start: date = date(2016, 1, 4),
    end: date = date(2025, 12, 31),
    *,
    n_symbols: int = 200,
    n_sectors: int = 8,
    seed: int = 7,
    planted_ic: float = 0.0,
    late_listing_fraction: float = 0.10,
    delisting_fraction: float = 0.05,
    missing_rate: float = 0.002,
    split_fraction: float = 0.02,
    dividend_fraction: float = 0.30,
) -> SyntheticPanel:
    """Generate correlated raw bars and an optional next-session volume signal.

    There must be at least three weekday sessions and 1..n_symbols sectors.
    Event cohorts contain floor(fraction * n_symbols) symbols, selected without
    replacement independently for each event type. Listing boundaries are always
    observed; interior bars are independently missing with ``missing_rate``.
    Volumes have close base levels and multiplicative exp(0.6 * signal) shocks.
    Dividends recur every 63 sessions within each selected symbol's listed life.
    ``planted_ic`` is the signal loading in the idiosyncratic log return, rather
    than an exact rank IC after sector factors, volatility and cent rounding.
    """
    if type(start) is not date:
        raise ValueError("SYNTHETIC_PANEL_ARGUMENT_INVALID:start")
    if type(end) is not date or end < start or end == date.max:
        raise ValueError("SYNTHETIC_PANEL_ARGUMENT_INVALID:end")
    _panel_integer("n_symbols", n_symbols, 2, 3000)
    _panel_integer("n_sectors", n_sectors, 1, n_symbols)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("SYNTHETIC_PANEL_ARGUMENT_INVALID:seed")
    _panel_fraction("planted_ic", planted_ic, 0.5)
    for name, value in (
        ("late_listing_fraction", late_listing_fraction),
        ("delisting_fraction", delisting_fraction),
        ("missing_rate", missing_rate),
        ("split_fraction", split_fraction),
        ("dividend_fraction", dividend_fraction),
    ):
        _panel_fraction(name, value)

    sessions = weekday_sessions(start, end)
    n = len(sessions)
    if n < 3:
        raise ValueError("SYNTHETIC_PANEL_ARGUMENT_INVALID:end")
    seed = int(seed)
    symbols = tuple(f"S{i + 1:04d}" for i in range(n_symbols))
    sector_index = np.arange(n_symbols) % n_sectors
    sectors = {symbol: f"SEC{sector_index[i]}" for i, symbol in enumerate(symbols)}
    parameters = _rng("synthetic-panel:parameters", seed)
    sigma = parameters.uniform(0.01, 0.03, n_symbols)
    market_beta = parameters.uniform(0.8, 1.2, n_symbols)
    sector_beta = parameters.uniform(0.7, 1.3, n_symbols)
    initial_price = parameters.uniform(40.0, 120.0, n_symbols)
    base_volume = parameters.uniform(950_000.0, 1_050_000.0, n_symbols)

    signal = _rng("synthetic-panel:signal", seed).standard_normal((n, n_symbols))
    returns = _rng("synthetic-panel:noise", seed).standard_normal((n - 1, n_symbols))
    returns *= np.sqrt(1.0 - planted_ic**2)
    returns += planted_ic * signal[:-1]
    returns *= sigma
    market = _rng("synthetic-panel:market", seed).normal(0.0002, 0.006, n - 1)
    sector = _rng("synthetic-panel:sectors", seed).normal(0.0, 0.004, (n - 1, n_sectors))
    returns += market[:, None] * market_beta
    returns += sector[:, sector_index] * sector_beta
    close = np.empty((n, n_symbols))
    close[0] = initial_price
    close[1:] = initial_price * np.exp(np.cumsum(returns, axis=0))
    del returns
    bar_rng = _rng("synthetic-panel:bars", seed)
    open_ = np.empty_like(close)
    open_[0] = initial_price
    open_[1:] = close[:-1] * np.exp(bar_rng.normal(size=(n - 1, n_symbols)) * sigma / 3)
    span = np.abs(bar_rng.normal(size=close.shape)) * sigma / 2
    high = np.maximum(open_, close) * np.exp(span)
    low = np.minimum(open_, close) * np.exp(-span)
    volume = np.maximum(1.0, np.rint(base_volume * np.exp(0.6 * signal)))

    lifecycle = _rng("synthetic-panel:lifecycle", seed)
    first = np.zeros(n_symbols, dtype=int)
    stop = np.full(n_symbols, n, dtype=int)
    late = lifecycle.permutation(n_symbols)[: int(late_listing_fraction * n_symbols)]
    gone = lifecycle.permutation(n_symbols)[: int(delisting_fraction * n_symbols)]
    first[late] = lifecycle.integers(1, max(2, n // 3 + 1), size=len(late))
    stop[gone] = lifecycle.integers(max(2, 2 * n // 3), n, size=len(gone))
    rows = np.arange(n)[:, None]
    present = (rows >= first) & (rows < stop)
    present &= _rng("synthetic-panel:missing", seed).random((n, n_symbols)) >= missing_rate
    # Keep lifecycle dates observable even for very short or sparse panels.
    columns = np.arange(n_symbols)
    present[first, columns] = True
    present[stop - 1, columns] = True
    listed = {
        symbol: (sessions[first[i]], sessions[stop[i] - 1] if stop[i] < n else None)
        for i, symbol in enumerate(symbols)
    }

    actions = _rng("synthetic-panel:actions", seed)
    split_columns = actions.permutation(n_symbols)[: int(split_fraction * n_symbols)]
    dividend_columns = actions.permutation(n_symbols)[: int(dividend_fraction * n_symbols)]
    splits: list[Split] = []
    dividends: list[Dividend] = []
    for column in sorted(split_columns):
        # Prefer an ex-date with an earlier listed session; a one-bar life is
        # still allowed and has its split on that sole session.
        ex = int(actions.integers(min(first[column] + 1, stop[column] - 1), stop[column]))
        splits.append(Split(symbols[column], sessions[ex], Decimal(2)))
        for values in (open_, high, low, close):
            values[ex:, column] /= 2.0
    for column in sorted(dividend_columns):
        first_ex = min(first[column] + 60, stop[column] - 1)
        for ex in range(first_ex, stop[column], 63):
            pay = min(ex + 10, stop[column] - 1)
            dividends.append(Dividend(symbols[column], sessions[ex], sessions[pay], Decimal("0.10")))

    series: dict[str, SymbolSeries] = {}
    for column, symbol in enumerate(symbols):
        mask = present[:, column].copy()
        micro = {
            name: np.where(mask, np.maximum(1, np.rint(values[:, column] * 100)), 0).astype(np.int64)
            * (MICRO // 100)
            for name, values in zip(FIELDS, (open_, high, low, close), strict=True)
        }
        series[symbol] = SymbolSeries(micro, np.where(mask, volume[:, column], 0.0), mask)
    signal.setflags(write=False)
    dataset = Dataset(
        sessions,
        series,
        tuple(splits),
        tuple(dividends),
        source=f"synthetic-panel:seed={seed}:n={n_symbols}:ic={planted_ic}",
    )
    return SyntheticPanel(dataset, symbols, sectors, listed, signal)


def synthetic_memberships(
    panel: SyntheticPanel, *, every: int = 21, min_history: int = 20
) -> tuple[tuple[date, tuple[str, ...]], ...]:
    """Select using only prior observations, starting at session min_history.

    Membership decisions are spaced in sessions, including empty decisions.
    A symbol's last observed bar can qualify it for the next decision; it drops
    out once the preceding session is absent.
    """
    if not isinstance(panel, SyntheticPanel):
        raise ValueError("SYNTHETIC_PANEL_ARGUMENT_INVALID:panel")
    _panel_integer("every", every, 1)
    _panel_integer("min_history", min_history, 1)
    decisions = range(min_history, len(panel.dataset.sessions), every)
    members: list[list[str]] = [[] for _ in decisions]
    indices = np.fromiter(decisions, dtype=int)
    for symbol in sorted(panel.symbols):
        present = panel.dataset.series[symbol].present
        prior_count = np.cumsum(present, dtype=np.int64)[indices - 1]
        eligible = present[indices - 1] & (prior_count >= min_history)
        for decision in np.flatnonzero(eligible):
            members[decision].append(symbol)
    return tuple(
        (panel.dataset.sessions[index], tuple(cohort)) for index, cohort in zip(indices, members, strict=True)
    )
