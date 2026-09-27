# SPDX-License-Identifier: Apache-2.0
"""The read-only view a strategy receives. It never includes the current session's bar."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from types import MappingProxyType
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Bars:
    """Completed daily bars for one symbol, oldest first; index ``-1`` is the last completed session.

    Arrays are read-only float64 views, point-in-time split-adjusted as of the
    information cutoff. Missing sessions are NaN.
    """

    symbol: str
    sessions: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def __len__(self) -> int:
        return len(self.sessions)


@dataclass(frozen=True)
class Ctx:
    """Everything a strategy may use for one decision."""

    decision_session: date
    _bars: Mapping[str, Bars]
    positions: Mapping[str, Decimal]
    weights: Mapping[str, Decimal]
    cash: Decimal
    equity: Decimal
    state: Mapping[str, Any]

    def bars(self, symbol: str) -> Bars:
        try:
            return self._bars[symbol.upper()]
        except KeyError:
            raise KeyError(f"SYMBOL_NOT_DECLARED:{symbol}") from None

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(self._bars)


def freeze_state(state: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(state))
