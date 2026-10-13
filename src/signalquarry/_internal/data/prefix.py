# SPDX-License-Identifier: Apache-2.0
"""Pure, exact dataset-prefix evidence for replayed universe inputs."""

from __future__ import annotations

from datetime import date

from signalquarry._internal.data.dataset import Dataset, truncated


def is_dataset_prefix(earlier: Dataset, selected: Dataset, *, start: date, end: date) -> bool:
    """Require the earlier identity on the initial sessions, symbols and action span.

    Both datasets must already have been replay-verified by the caller. The
    original request dates bound actions even on dates without a bar session.
    Preserve the selected source and every array value, including absent-bar
    placeholders; never replace either with the earlier dataset's values.
    """
    if (
        not earlier.sessions
        or not start <= earlier.sessions[0] <= earlier.sessions[-1] <= end
        or selected.sessions[: len(earlier.sessions)] != earlier.sessions
        or (
            len(selected.sessions) > len(earlier.sessions) and selected.sessions[len(earlier.sessions)] <= end
        )
        or not set(earlier.symbols) <= selected.series.keys()
    ):
        return False
    prefix = truncated(selected, len(earlier.sessions))
    prefix.series = {symbol: prefix.series[symbol] for symbol in earlier.symbols}
    prefix.splits = tuple(
        action
        for action in selected.splits
        if action.symbol in prefix.series and start <= action.ex_date <= end
    )
    prefix.dividends = tuple(
        action
        for action in selected.dividends
        if action.symbol in prefix.series and start <= action.ex_date <= end
    )
    return prefix.identity() == earlier.identity()
