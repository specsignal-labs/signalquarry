# SPDX-License-Identifier: Apache-2.0
"""Streaming dataset identity must equal the established canonical document."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import numpy as np
import pytest

from signalquarry._internal.canonical import canonical_hash
from signalquarry._internal.data.dataset import FIELDS, Dataset, Dividend, Split, SymbolSeries


def _series(length: int, offset: int) -> SymbolSeries:
    base = np.arange(length, dtype=np.int64) + offset
    return SymbolSeries(
        {name: base + index for index, name in enumerate(FIELDS)},
        np.arange(length, dtype=np.float64) / 3 + offset,
        np.arange(length) % 7 != 0,
    )


def _legacy_identity(dataset: Dataset) -> str:
    return canonical_hash(
        {
            "source": dataset.source,
            "sessions": [s.isoformat() for s in dataset.sessions],
            "series": {
                symbol: {
                    **{name: item.micro[name].tolist() for name in FIELDS},
                    "volume": item.volume.tolist(),
                    "present": item.present.tolist(),
                }
                for symbol, item in sorted(dataset.series.items())
            },
            "splits": [[s.symbol, s.ex_date.isoformat(), s.ratio] for s in dataset.splits],
            "dividends": [
                [d.symbol, d.ex_date.isoformat(), d.pay_date.isoformat(), d.amount] for d in dataset.dividends
            ],
        }
    )


def test_streamed_identity_matches_legacy_across_chunk_and_symbol_boundaries() -> None:
    sessions = tuple(date(2020, 1, 1) + timedelta(days=i) for i in range(4099))
    dataset = Dataset(
        sessions,
        {"SYNB": _series(len(sessions), 7), "SYNA": _series(len(sessions), 13)},
        splits=(Split("SYNB", sessions[4096], Decimal("0.5")),),
        dividends=(Dividend("SYNA", sessions[100], sessions[110], Decimal("0.35")),),
        source="synthetic:Δ",
    )
    expected = _legacy_identity(dataset)
    assert dataset.identity() == expected
    assert (
        Dataset(
            sessions,
            {"SYNA": dataset.series["SYNA"], "SYNB": dataset.series["SYNB"]},
            dataset.splits,
            dataset.dividends,
            source=dataset.source,
        ).identity()
        == expected
    )


def test_nonfinite_array_value_is_still_rejected() -> None:
    session = date(2024, 1, 2)
    series = _series(1, 1)
    series.volume[0] = float("nan")
    dataset = Dataset((session,), {"SYNA": series})
    with pytest.raises(ValueError, match="CANONICAL_FLOAT_NOT_FINITE"):
        dataset.identity()
