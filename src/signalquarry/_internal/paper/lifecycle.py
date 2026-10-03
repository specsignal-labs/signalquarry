# SPDX-License-Identifier: Apache-2.0
"""Pure paper reconciliation for verified, asset-keyed corporate-action effects.

These records are normalized evidence, not Alpaca wire responses. The provider
adapter does not produce them yet, and the paper runner does not call this
module. In particular, a successful check here does not authorize an order.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from signalquarry._internal.data.identity import AssetKey
from signalquarry._internal.data.lifecycle import LifecycleEvent
from signalquarry._internal.engine.lifecycle import LifecycleState, apply_lifecycle_event
from signalquarry._internal.paper.models import PaperError

ZERO = Decimal(0)


@dataclass(frozen=True)
class BrokerAssetPosition:
    """A broker position whose asset identity and symbol were both verified."""

    asset: AssetKey
    symbol: str
    quantity: Decimal


@dataclass(frozen=True)
class PositionActivity:
    """One verified broker activity linked to a resolved lifecycle event."""

    activity_id: str
    event_id: str
    occurred: date
    asset: AssetKey
    symbol_before: str | None
    symbol_after: str | None
    quantity_delta: Decimal


@dataclass(frozen=True)
class CashActivity:
    """One verified cash credit linked to a resolved lifecycle event."""

    activity_id: str
    event_id: str
    occurred: date
    amount: Decimal


def _unreconciled(detail: str) -> PaperError:
    return PaperError("PAPER_CORPORATE_ACTION_UNRECONCILED", "blocked", detail)


def _positions(positions: Sequence[BrokerAssetPosition]) -> dict[AssetKey, BrokerAssetPosition]:
    observed: dict[AssetKey, BrokerAssetPosition] = {}
    symbols: set[str] = set()
    for item in positions:
        if (
            item.asset in observed
            or item.symbol in symbols
            or not item.symbol
            or not isinstance(item.quantity, Decimal)
            or not item.quantity.is_finite()
            or item.quantity <= 0
        ):
            raise _unreconciled("duplicate or invalid broker position")
        observed[item.asset] = item
        symbols.add(item.symbol)
    return observed


def _expected_positions(state: LifecycleState) -> dict[AssetKey, BrokerAssetPosition]:
    return {
        asset: BrokerAssetPosition(asset, state.symbols[asset], quantity)
        for asset, quantity in state.positions.items()
        if quantity
    }


def reconcile_position_transition(
    before: LifecycleState,
    event: LifecycleEvent,
    activities: Sequence[PositionActivity],
    broker_positions: Sequence[BrokerAssetPosition],
) -> LifecycleState:
    """Verify the event's exact share and alias effects against broker evidence.

    ``before`` must already match the broker before this event. The caller must
    verify the activity-to-event link and asset mapping against source records.
    Every nonzero holding effect, including a pure rename, requires an activity.
    No cash receivable is credited here; reconcile its payment separately.
    """
    after = apply_lifecycle_event(before, event)
    observed = _positions(broker_positions)
    expected = _expected_positions(after)
    changed = {
        asset: (
            after.positions.get(asset, ZERO) - before.positions.get(asset, ZERO),
            before.symbols.get(asset),
            after.symbols.get(asset),
        )
        for asset in set(before.positions) | set(after.positions)
        if before.positions.get(asset, ZERO) != after.positions.get(asset, ZERO)
        or (before.positions.get(asset, ZERO) > 0 and before.symbols.get(asset) != after.symbols.get(asset))
    }
    if not activities and changed and observed == _expected_positions(before):
        raise PaperError("PAPER_CORPORATE_ACTION_PENDING", "busy", event.event_id)

    totals: dict[AssetKey, Decimal] = defaultdict(lambda: ZERO)
    seen: set[str] = set()
    for item in activities:
        if (
            not item.activity_id
            or item.activity_id in seen
            or item.event_id != event.event_id
            or item.occurred != event.effective_date
            or item.asset not in changed
            or not isinstance(item.quantity_delta, Decimal)
            or not item.quantity_delta.is_finite()
        ):
            raise _unreconciled("invalid or unrelated position activity")
        delta, old_symbol, new_symbol = changed[item.asset]
        if (item.symbol_before, item.symbol_after) != (old_symbol, new_symbol):
            raise _unreconciled("position activity asset alias mismatch")
        seen.add(item.activity_id)
        totals[item.asset] += item.quantity_delta
    if set(totals) != set(changed) or any(totals[asset] != delta for asset, (delta, _, _) in changed.items()):
        raise _unreconciled("position activity amount mismatch")
    if observed != expected:
        raise _unreconciled("broker position or asset alias mismatch")
    return after


def reconcile_cash_payment(
    state: LifecycleState,
    event_id: str,
    activity: CashActivity | None,
    *,
    session: date,
    cash_before: Decimal,
    cash_after: Decimal,
) -> tuple[LifecycleState, Decimal]:
    """Verify and consume one dated receivable against a broker cash credit.

    The two cash balances must bracket only this credit; other account cash
    movements need their own journal reconciliation before this check.
    """
    matches = [item for item in state.receivables if item.event_id == event_id]
    if event_id not in state.applied_event_ids or len(matches) != 1:
        raise _unreconciled("unknown or already settled cash consideration")
    receivable = matches[0]
    if any(not isinstance(value, Decimal) or not value.is_finite() for value in (cash_before, cash_after)):
        raise _unreconciled("invalid broker cash balance")
    if session < receivable.pay_date:
        if activity is not None or cash_after != cash_before:
            raise _unreconciled("cash credited before verified payment date")
        raise PaperError("PAPER_CORPORATE_ACTION_PENDING", "busy", event_id)
    if activity is None and cash_after == cash_before:
        raise PaperError("PAPER_CORPORATE_ACTION_PENDING", "busy", event_id)
    if (
        activity is None
        or not activity.activity_id
        or activity.event_id != event_id
        or activity.occurred != receivable.pay_date
        or not isinstance(activity.amount, Decimal)
        or not activity.amount.is_finite()
        or activity.amount != receivable.amount
        or cash_after - cash_before != receivable.amount
    ):
        raise _unreconciled("cash activity or broker balance mismatch")
    settled = LifecycleState(
        state.positions,
        state.symbols,
        tuple(item for item in state.receivables if item is not receivable),
        state.applied_event_ids,
        state.last_event_key,
    )
    return settled, cash_after
