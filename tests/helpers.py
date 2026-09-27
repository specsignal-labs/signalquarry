# SPDX-License-Identifier: Apache-2.0
"""Small hand-built datasets and specs for engine tests."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from typing import Any

import numpy as np

from signalquarry._internal.contracts.spec import StrategySpecV1
from signalquarry._internal.data.dataset import MICRO, Dataset, Dividend, Split, SymbolSeries


def weekdays(start: date, count: int) -> tuple[date, ...]:
    out, current = [], start
    while len(out) < count:
        if current.weekday() < 5:
            out.append(current)
        current += timedelta(days=1)
    return tuple(out)


def dataset(
    sessions: tuple[date, ...], prices: dict[str, dict[str, list[float]]], *, splits=(), dividends=()
) -> Dataset:
    series = {}
    for symbol, fields in prices.items():
        opens, closes = fields["open"], fields.get("close", fields["open"])
        present = np.array(fields.get("present", [True] * len(sessions)))
        micro = {
            "open": np.rint(np.array(opens) * MICRO).astype(np.int64),
            "high": np.rint(np.maximum(opens, closes) * np.array(1.0) * MICRO).astype(np.int64),
            "low": np.rint(np.minimum(opens, closes) * MICRO).astype(np.int64),
            "close": np.rint(np.array(closes) * MICRO).astype(np.int64),
        }
        series[symbol] = SymbolSeries(micro, np.full(len(sessions), 1e6), present)
    return Dataset(sessions, series, tuple(splits), tuple(dividends), source="test")


def spec(
    symbols: tuple[str, ...], codes: tuple[str, ...] = ("GO", "WAIT"), **overrides: Any
) -> StrategySpecV1:
    body: dict[str, Any] = {
        "schema": "signalquarry.strategy/v1",
        "id": "test-strategy",
        "family": "test-family",
        "version": "1.0.0",
        "hypothesis": {
            "statement": "A test hypothesis statement.",
            "falsification": "Fails if the test fails.",
        },
        "data": {"symbols": list(symbols), "feed": "synthetic"},
        "account": {"model": "cash", "initial_cash": "10000"},
        "execution": {"costs": {"bps": "10"}},
        "reason_codes": {code: f"{code} reason" for code in codes},
    }
    for key, value in overrides.items():
        body[key] = {**body.get(key, {}), **value} if isinstance(value, dict) else value
    return StrategySpecV1.model_validate(body)


__all__ = ["Decimal", "Dividend", "Split", "dataset", "spec", "weekdays"]
