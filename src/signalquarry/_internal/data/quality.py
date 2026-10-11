# SPDX-License-Identifier: Apache-2.0
"""Pure daily-dataset diagnostics without disclosing recorded market values."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np

from signalquarry._internal.calendar.nyse import MAX_YEAR, MIN_YEAR, is_holiday
from signalquarry._internal.data.dataset import FIELDS, Dataset

QUALITY_SCHEMA = "signalquarry.data-quality/v1"


@dataclass(frozen=True)
class QualityThresholds:
    max_gap_sessions: int = 3
    stale_close_sessions: int = 5
    extreme_move: float = 0.40


def _session_report(dataset: Dataset, *, check_calendar: bool) -> dict[str, Any]:
    sessions = dataset.sessions
    tracked = sum(MIN_YEAR <= session.year <= MAX_YEAR for session in sessions)
    calendar = "weekdays_only" if not tracked else "nyse" if tracked == len(sessions) else "partial"
    if not check_calendar:
        calendar = "not_checked"
    missing = []
    if check_calendar and sessions:
        # Bound the scan to the verified calendar, even for very long datasets.
        start = max(sessions[0].toordinal(), date(MIN_YEAR, 1, 1).toordinal())
        stop = min(sessions[-1].toordinal(), date(MAX_YEAR, 12, 31).toordinal())
        recorded = set(sessions)
        for ordinal in range(start, stop + 1):
            session = date.fromordinal(ordinal)
            if session not in recorded and not is_holiday(session):
                missing.append(session.isoformat())
    return {
        "count": len(sessions),
        "first": sessions[0].isoformat() if sessions else None,
        "last": sessions[-1].isoformat() if sessions else None,
        "calendar": calendar,
        "weekend_sessions": [session.isoformat() for session in sessions if session.weekday() >= 5],
        "holiday_sessions": [
            session.isoformat()
            for session in sessions
            if check_calendar and session.weekday() < 5 and is_holiday(session)
        ],
        "missing_trading_days": missing,
    }


def _symbol_report(dataset: Dataset, symbol: str, thresholds: QualityThresholds) -> dict[str, Any]:
    item = dataset.series[symbol]
    indices = np.flatnonzero(item.present)
    count = len(indices)
    sessions = dataset.sessions
    first, last = (int(indices[0]), int(indices[-1])) if count else (0, 0)
    gaps = []
    stale = []
    moves = []
    if count:
        distances = np.diff(indices)
        for position in np.flatnonzero(distances > 1):
            start, end = int(indices[position] + 1), int(indices[position + 1] - 1)
            gaps.append(
                {
                    "start": sessions[start].isoformat(),
                    "end": sessions[end].isoformat(),
                    "sessions": end - start + 1,
                }
            )

        closes = item.micro["close"][indices]
        # Missing bars break a stale run; identical raw closes across a split still count.
        starts = np.flatnonzero(np.r_[True, (distances > 1) | (closes[1:] != closes[:-1])])
        stops = np.r_[starts[1:], count]
        for run in np.flatnonzero(stops - starts >= thresholds.stale_close_sessions):
            start, stop = int(starts[run]), int(stops[run])
            stale.append(
                {
                    "start": sessions[int(indices[start])].isoformat(),
                    "end": sessions[int(indices[stop - 1])].isoformat(),
                    "sessions": stop - start,
                }
            )

        # Multiplying raw closes by F(t) removes all recorded intervening splits.
        adjusted = closes.astype(np.float64) * dataset.cumulative_split(symbol)[indices]
        previous, later = adjusted[:-1], adjusted[1:]
        delta = later - previous
        # A return is undefined for a nonpositive close; the price check reports it.
        valid = (closes[:-1] > 0) & (closes[1:] > 0)
        extreme = valid & (np.abs(delta) >= thresholds.extreme_move * previous)
        for position in np.flatnonzero(extreme):
            moves.append(
                {
                    "session": sessions[int(indices[position + 1])].isoformat(),
                    "direction": "up" if delta[position] > 0 else "down",
                }
            )

    opened, high, low, close = (item.micro[name][indices] for name in FIELDS)
    inconsistent = int(
        np.count_nonzero((low > opened) | (opened > high) | (low > close) | (close > high) | (low > high))
    )
    nonpositive = int(np.count_nonzero((opened <= 0) | (high <= 0) | (low <= 0) | (close <= 0)))
    volume = item.volume[indices]
    zero_volume = int(np.count_nonzero((volume <= 0) | np.isnan(volume)))
    longest_gap = max((gap["sessions"] for gap in gaps), default=0)
    findings = sorted(
        code
        for condition, code in (
            (not count, "QUALITY_SYMBOL_EMPTY"),
            (longest_gap > thresholds.max_gap_sessions, "QUALITY_GAP"),
            (inconsistent, "QUALITY_OHLC_INCONSISTENT"),
            (nonpositive, "QUALITY_PRICE_NONPOSITIVE"),
            (zero_volume, "QUALITY_ZERO_VOLUME"),
            (stale, "QUALITY_STALE_CLOSE"),
            (moves, "QUALITY_UNEXPLAINED_MOVE"),
        )
        if condition
    )
    return {
        "present": count,
        "first": sessions[first].isoformat() if count else None,
        "last": sessions[last].isoformat() if count else None,
        "coverage": round(count / (last - first + 1), 6) if count else None,
        "leading_missing": first if count else len(sessions),
        "trailing_missing": len(sessions) - last - 1 if count else len(sessions),
        "gaps": gaps,
        "longest_gap": longest_gap,
        "ohlc_violations": inconsistent,
        "nonpositive_prices": nonpositive,
        "zero_volume_sessions": zero_volume,
        "stale_close_runs": stale,
        "unexplained_moves": moves,
        "splits": 0,
        "dividends": 0,
        "findings": findings,
    }


def assess(
    dataset: Dataset,
    *,
    symbols: tuple[str, ...] | None = None,
    check_calendar: bool = True,
    thresholds: QualityThresholds = QualityThresholds(),  # noqa: B008 -- frozen, immutable default
) -> dict[str, Any]:
    """Inspect recorded sessions and present bars without changing the dataset."""
    selected = dataset.symbols if symbols is None else tuple(sorted(set(symbols)))
    for symbol in selected:
        if symbol not in dataset.series:
            raise ValueError(f"QUALITY_SYMBOL_UNKNOWN:{symbol}")
    sessions = _session_report(dataset, check_calendar=check_calendar)
    reports = {symbol: _symbol_report(dataset, symbol, thresholds) for symbol in selected}
    splits = Counter(split.symbol for split in dataset.splits)
    dividends = Counter(dividend.symbol for dividend in dataset.dividends)
    for symbol, report in reports.items():
        report["splits"] = splits[symbol]
        report["dividends"] = dividends[symbol]

    common: dict[str, Any] = {"first": None, "last": None, "sessions": 0}
    if reports and all(report["present"] for report in reports.values()):
        first = max(report["first"] for report in reports.values())
        last = min(report["last"] for report in reports.values())
        if first <= last:
            common = {
                "first": first,
                "last": last,
                "sessions": sum(first <= session.isoformat() <= last for session in dataset.sessions),
            }

    findings = {code for report in reports.values() for code in report["findings"]}
    for key, code in (
        ("weekend_sessions", "QUALITY_SESSION_ON_WEEKEND"),
        ("holiday_sessions", "QUALITY_SESSION_ON_HOLIDAY"),
        ("missing_trading_days", "QUALITY_TRADING_DAY_MISSING"),
    ):
        if sessions[key]:
            findings.add(code)
    return {
        "schema": QUALITY_SCHEMA,
        "dataset_identity": dataset.identity(),
        "source": dataset.source,
        "thresholds": asdict(thresholds),
        "sessions": sessions,
        "symbols": reports,
        "common_window": common,
        "findings": sorted(findings),
        "ok": not findings,
    }
