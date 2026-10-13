# SPDX-License-Identifier: Apache-2.0
"""Exact prefix evidence includes the source, missing bars and dated actions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal

import numpy as np
import pytest

from signalquarry._internal.data.dataset import Dataset, Dividend, Split, SymbolSeries, truncated
from signalquarry._internal.data.prefix import is_dataset_prefix


def _inputs():
    sessions = tuple(date(2024, 1, 2) + timedelta(days=i) for i in range(10))
    series = {
        symbol: SymbolSeries(
            {name: np.arange(10, dtype=np.int64) for name in ("open", "high", "low", "close")},
            np.arange(10, dtype=float),
            np.array([True, False, *([True] * 8)]),
        )
        for symbol in ("AAA", "BBB")
    }
    selected = Dataset(
        sessions,
        series,
        splits=(Split("AAA", sessions[2], Decimal(2)), Split("AAA", sessions[8], Decimal(3))),
        dividends=(
            Dividend("AAA", sessions[3], sessions[9], Decimal("0.50")),
            Dividend("BBB", sessions[3], sessions[9], Decimal("0.75")),
            Dividend("AAA", sessions[8], sessions[9], Decimal("0.25")),
        ),
        source="alpaca:sip",
    )
    earlier = truncated(selected, 6)
    earlier.series = {"AAA": earlier.series["AAA"]}
    earlier.splits = selected.splits[:1]
    earlier.dividends = selected.dividends[:1]
    return earlier, selected


def _matches(earlier, selected, **span):
    return is_dataset_prefix(
        earlier, selected, **{"start": earlier.sessions[0], "end": earlier.sessions[-1], **span}
    )


def test_dataset_prefix_is_exact_on_original_symbols_and_keeps_inputs_unchanged():
    earlier, selected = _inputs()
    before = earlier.identity(), selected.identity()
    assert _matches(earlier, selected)
    assert before == (earlier.identity(), selected.identity())
    assert _matches(selected, selected)
    # Other symbols and bars/actions strictly beyond the earlier span cannot influence the hash.
    selected.series["BBB"].micro["close"][:] = 0
    for item in selected.series.values():
        for array in (*item.micro.values(), item.volume, item.present):
            array[6:] = 0
    assert _matches(earlier, selected)


@pytest.mark.parametrize("field", ["open", "high", "low", "close", "volume", "present", "absent-value"])
def test_dataset_prefix_refuses_any_array_change_inside_span(field):
    earlier, selected = _inputs()
    item = selected.series["AAA"]
    if field == "present":
        item.present[1] = True
    elif field == "volume":
        item.volume[1] += 1
    else:
        item.micro["close" if field == "absent-value" else field][1] += 1
    assert not _matches(earlier, selected)


@pytest.mark.parametrize(
    "change",
    ["source", "missing-symbol", "short", "shift", "gap", "extra-session", "span-start", "span-end"],
)
def test_dataset_prefix_refuses_different_calendar_symbol_or_source(change):
    earlier, selected = _inputs()
    span = {}
    if change == "source":
        selected.source = "alpaca:iex"
    elif change == "missing-symbol":
        del selected.series["AAA"]
    elif change == "short":
        selected = truncated(selected, 5)
    elif change == "shift":
        selected.sessions = tuple(session + timedelta(days=1) for session in selected.sessions)
    elif change == "gap":
        selected.sessions = (*selected.sessions[:2], *selected.sessions[3:])
    elif change == "extra-session":
        selected.sessions = (*selected.sessions[:6], selected.sessions[5], *selected.sessions[6:])
    elif change == "span-start":
        span["start"] = earlier.sessions[1]
    else:
        span["end"] = earlier.sessions[-2]
    assert not _matches(earlier, selected, **span)


@pytest.mark.parametrize("change", ["split", "dividend", "pay-date", "late-action", "removed-action"])
def test_dataset_prefix_refuses_revised_or_later_recorded_in_span_actions(change):
    earlier, selected = _inputs()
    if change == "split":
        selected.splits = (replace(selected.splits[0], ratio=Decimal(4)), *selected.splits[1:])
    elif change == "dividend":
        selected.dividends = (replace(selected.dividends[0], amount=Decimal(1)), *selected.dividends[1:])
    elif change == "pay-date":
        selected.dividends = (
            replace(selected.dividends[0], pay_date=selected.sessions[8]),
            *selected.dividends[1:],
        )
    elif change == "late-action":
        selected.splits = (*selected.splits, Split("AAA", earlier.sessions[4], Decimal(2)))
    else:
        selected.dividends = selected.dividends[1:]
    assert not _matches(earlier, selected)


def test_dataset_prefix_action_span_includes_dates_without_bars_and_later_pay_dates():
    earlier, selected = _inputs()
    start = earlier.sessions[0] - timedelta(days=1)
    end = earlier.sessions[-1] + timedelta(days=1)
    # A no-bar action date at the requested boundary still belongs to the identity.
    selected.sessions = (*selected.sessions[:6], *selected.sessions[7:])
    split = Split("AAA", start, Decimal(2))
    dividend = Dividend("AAA", end, date(2025, 1, 1), Decimal(1))
    selected.splits = (*selected.splits, split)
    earlier.splits = (*earlier.splits, split)
    selected.dividends = (*selected.dividends, dividend)
    earlier.dividends = (*earlier.dividends, dividend)
    assert _matches(earlier, selected, start=start, end=end)
    assert not _matches(earlier, selected)


def test_dataset_prefix_empty_input_and_action_order_are_not_relaxed():
    earlier, selected = _inputs()
    assert not is_dataset_prefix(truncated(earlier, 0), selected, start=date.min, end=date.max)
    other = Split("AAA", earlier.sessions[4], Decimal(3))
    earlier.splits = (*earlier.splits, other)
    selected.splits = (other, *selected.splits)
    assert not _matches(earlier, selected)
    assert earlier.identity() == deepcopy(earlier).identity()
