# SPDX-License-Identifier: Apache-2.0
"""Pure, deterministic position transitions for fully resolved corporate actions.

The legacy symbol-keyed backtest and paper runner do not call this module yet.
Provider decoding and broker activity reconciliation must be verified before
the current fail-closed guard is relaxed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from types import MappingProxyType

from signalquarry._internal.data.identity import AssetKey
from signalquarry._internal.data.lifecycle import LifecycleError, LifecycleEvent

ZERO = Decimal(0)


@dataclass(frozen=True)
class CashReceivable:
    event_id: str
    pay_date: date
    amount: Decimal


@dataclass(frozen=True)
class LifecycleState:
    """Asset-keyed holdings, aliases and unpaid consideration after a transition."""

    positions: Mapping[AssetKey, Decimal]
    symbols: Mapping[AssetKey, str]
    receivables: tuple[CashReceivable, ...] = ()
    applied_event_ids: frozenset[str] = frozenset()
    last_event_key: tuple[date, int] | None = None

    def __post_init__(self) -> None:
        positions = dict(self.positions)
        symbols = dict(self.symbols)
        if any(
            not isinstance(quantity, Decimal) or not quantity.is_finite() or quantity < 0
            for quantity in positions.values()
        ):
            raise LifecycleError("invalid held quantity")
        if set(positions) - set(symbols):
            raise LifecycleError("held asset missing symbol")
        if len(set(symbols.values())) != len(symbols):
            raise LifecycleError("two assets share a current symbol")
        if any(
            not isinstance(item.amount, Decimal) or not item.amount.is_finite() or item.amount < 0
            for item in self.receivables
        ):
            raise LifecycleError("invalid cash receivable")
        object.__setattr__(self, "positions", MappingProxyType(positions))
        object.__setattr__(self, "symbols", MappingProxyType(symbols))


def _check_fraction(quantity: Decimal, event: LifecycleEvent) -> None:
    if event.fraction_policy == "reject_noninteger" and quantity != quantity.to_integral_value():
        raise LifecycleError(f"fractional shares for {event.event_id}")


def _set_alias(symbols: dict[AssetKey, str], asset: AssetKey, symbol: str) -> None:
    if any(key != asset and value == symbol for key, value in symbols.items()):
        raise LifecycleError(f"ambiguous successor symbol {symbol}")
    old = symbols.get(asset)
    if old is not None and old != symbol:
        raise LifecycleError(f"successor asset alias mismatch {symbol}")
    symbols[asset] = symbol


def apply_lifecycle_event(state: LifecycleState, event: LifecycleEvent) -> LifecycleState:
    """Apply one event once, with explicit order, terms and asset identity."""
    key = (event.effective_date, event.sequence)
    if event.event_id in state.applied_event_ids or (
        state.last_event_key is not None and key <= state.last_event_key
    ):
        raise LifecycleError(f"duplicate or out-of-order event {event.event_id}")
    if state.symbols.get(event.source) != event.source_symbol:
        raise LifecycleError(f"source asset alias mismatch {event.event_id}")

    positions = dict(state.positions)
    symbols = dict(state.symbols)
    receivables = list(state.receivables)
    held = positions.get(event.source, ZERO)

    if event.kind == "rename":
        assert event.target_symbol is not None
        symbols.pop(event.source)
        _set_alias(symbols, event.source, event.target_symbol)
    elif event.kind == "split":
        assert event.share_ratio is not None
        changed = held * event.share_ratio
        _check_fraction(changed, event)
        if held:
            positions[event.source] = changed
        if event.target_symbol is not None and event.target_symbol != event.source_symbol:
            symbols.pop(event.source)
            _set_alias(symbols, event.source, event.target_symbol)
    else:
        positions.pop(event.source, None)
        symbols.pop(event.source)
        if event.kind in ("stock_merger", "mixed_merger"):
            assert (
                event.target is not None and event.target_symbol is not None and event.share_ratio is not None
            )
            _set_alias(symbols, event.target, event.target_symbol)
            delivered = held * event.share_ratio
            _check_fraction(delivered, event)
            if delivered:
                positions[event.target] = positions.get(event.target, ZERO) + delivered
        if event.kind in ("cash_merger", "mixed_merger"):
            assert event.cash_per_old_share is not None and event.cash_pay_date is not None
            amount = held * event.cash_per_old_share
            if amount:
                receivables.append(CashReceivable(event.event_id, event.cash_pay_date, amount))

    return LifecycleState(
        positions,
        symbols,
        tuple(receivables),
        state.applied_event_ids | {event.event_id},
        key,
    )


def apply_lifecycle_events(state: LifecycleState, events: Sequence[LifecycleEvent]) -> LifecycleState:
    """Apply a batch in explicit effective-date and sequence order."""
    current = state
    for event in sorted(events, key=lambda item: (item.effective_date, item.sequence, item.event_id)):
        current = apply_lifecycle_event(current, event)
    return current
