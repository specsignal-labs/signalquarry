# SPDX-License-Identifier: Apache-2.0
"""An in-memory options paper venue driven by a dataset, for tests and parity checks.

Quotes come from the simulator's own chain model (spot = the session's open before
noon, its close after), so paper and simulation see the same prices. Limit orders fill
at their limit when marketable, otherwise they rest. ``end_of_day`` expires legs:
in the money by at least $0.01 → assignment (OPASN), otherwise expiration (OPEXP).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import Literal
from zoneinfo import ZoneInfo

from signalquarry._internal.data.dataset import Dataset
from signalquarry._internal.engine.backtest import _Views
from signalquarry._internal.options.chains import ChainModel, realized_sigma
from signalquarry._internal.options.contracts import Quote, parse_occ
from signalquarry._internal.options.resolver import Candidate
from signalquarry._internal.paper.models import (
    OPEN_ORDER_STATUSES,
    Activity,
    BrokerAccount,
    BrokerClock,
    BrokerOrder,
    BrokerPosition,
    OrderRequest,
    PaperError,
)

NEW_YORK = ZoneInfo("America/New_York")
Fault = Literal["error_before_accept", "error_after_accept", "reject", "rest"]


def at(session: date, moment: time) -> datetime:
    return datetime.combine(session, moment, tzinfo=NEW_YORK).astimezone(UTC)


@dataclass
class FakeOptionsVenue:
    dataset: Dataset
    cash: Decimal
    tick: Decimal = Decimal("0.01")
    account_id: str = "fake-options-account"
    now: datetime = field(default_factory=lambda: datetime.now(UTC))
    positions_: dict[str, Decimal] = field(default_factory=dict)
    orders: dict[str, BrokerOrder] = field(default_factory=dict)
    activities_: list[Activity] = field(default_factory=list)
    faults: list[Fault] = field(default_factory=list)
    refuse_cancel: bool = False
    paper_only: bool = field(default=True, init=False)
    name: str = field(default="fake-options", init=False)
    _ids: itertools.count = field(default_factory=itertools.count, init=False, repr=False)

    def __post_init__(self) -> None:
        self._views = _Views(self.dataset, self.dataset.symbols)

    # -- pricing -------------------------------------------------------------------------
    def _index(self) -> int:
        return self.dataset.index_of(self.now.astimezone(NEW_YORK).date())

    def _spot(self, underlying: str) -> Decimal:
        index = self._index()
        field_name = "open" if self.now.astimezone(NEW_YORK).time() < time(12, 0) else "close"
        price = self.dataset.price(underlying, field_name, index)
        assert price is not None
        return price

    def _chain(self, underlying: str) -> ChainModel:
        index = self._index()
        closes = self._views.bars(underlying, max(0, index - 25), index).close
        return ChainModel(
            underlying,
            spot=self._spot(underlying),
            session=self.dataset.sessions[index],
            at=self.now,
            sigma=realized_sigma(closes),
            tick=self.tick,
        )

    def stock_quote(self, symbol: str) -> Quote | None:
        spot = self._spot(symbol)
        return Quote(spot - Decimal("0.01"), spot + Decimal("0.01"), self.now)

    def option_quote(self, symbol: str) -> Quote | None:
        contract = parse_occ(symbol)
        return self._chain(contract.underlying).quote(contract)

    def option_candidates(
        self, underlying: str, right: str, min_dte: int, max_dte: int, target: Decimal
    ) -> list[Candidate]:
        return self._chain(underlying).candidates(right, min_dte, max_dte, target)

    # -- account ---------------------------------------------------------------------------
    def account(self) -> BrokerAccount:
        equity = self.cash
        for symbol, quantity in self.positions_.items():
            if len(symbol) > 6:
                quote = self.option_quote(symbol)
                equity += quantity * (quote.ask if quote else Decimal(0)) * 100
            elif symbol in self.dataset.series:
                equity += quantity * self._spot(symbol)
        return BrokerAccount(self.account_id, "ACTIVE", self.cash, equity, self.cash, False)

    def clock(self) -> BrokerClock:
        local = self.now.astimezone(NEW_YORK)
        is_session = local.date() in self.dataset.sessions
        is_open = is_session and time(9, 30) <= local.time() < time(16, 0)
        following = next(
            (s for s in self.dataset.sessions if s > local.date()), local.date() + timedelta(days=1)
        )
        next_open = (
            at(local.date(), time(9, 30))
            if is_session and local.time() < time(9, 30)
            else at(following, time(9, 30))
        )
        return BrokerClock(self.now, is_open, next_open, at(local.date(), time(16, 0)))

    def positions(self) -> list[BrokerPosition]:
        return [BrokerPosition(s, q) for s, q in sorted(self.positions_.items()) if q]

    def open_orders(self) -> list[BrokerOrder]:
        return [o for o in self.orders.values() if o.status in OPEN_ORDER_STATUSES]

    def order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        return next((o for o in self.orders.values() if o.client_order_id == client_order_id), None)

    def activities(self, after: str | None) -> list[Activity]:
        return list(self.activities_)

    # -- orders ----------------------------------------------------------------------------
    def _fill(self, order: BrokerOrder) -> BrokerOrder:
        price = order.filled_average_price or Decimal(0)
        sign = -1 if order.side == "sell" else 1
        self.positions_[order.symbol] = self.positions_.get(order.symbol, Decimal(0)) + sign * order.quantity
        if not self.positions_[order.symbol]:
            del self.positions_[order.symbol]
        self.cash -= sign * price * 100 * order.quantity
        return order

    def submit(self, request: OrderRequest) -> BrokerOrder:
        fault = self.faults.pop(0) if self.faults else None
        if fault == "error_before_accept":
            raise PaperError("BROKER_UNAVAILABLE", "unavailable", "injected")
        if fault == "reject":
            raise PaperError("BROKER_ORDER_REJECTED", "blocked", "injected")
        existing = self.order_by_client_id(request.client_order_id)
        if existing is not None:
            return existing
        quote = self.option_quote(request.symbol)
        limit = request.limit_price
        assert quote is not None and limit is not None
        marketable = limit <= quote.bid if request.side == "sell" else limit >= quote.ask
        order = BrokerOrder(
            f"opt-{next(self._ids)}",
            request.client_order_id,
            request.symbol,
            request.side,
            request.quantity,
            "accepted",
            Decimal(0),
            None,
        )
        if marketable and fault != "rest":
            order = replace(
                order, status="filled", filled_quantity=request.quantity, filled_average_price=limit
            )
            self._fill(order)
        self.orders[order.order_id] = order
        if fault == "error_after_accept":
            raise PaperError("BROKER_UNAVAILABLE", "unavailable", "injected after accept")
        return order

    def cancel(self, order_id: str) -> None:
        order = self.orders.get(order_id)
        if order is not None and order.status in OPEN_ORDER_STATUSES and not self.refuse_cancel:
            self.orders[order_id] = replace(order, status="canceled")

    # -- lifecycle -------------------------------------------------------------------------
    def end_of_day(self, session: date) -> None:
        index = self.dataset.index_of(session)
        for symbol, quantity in list(self.positions_.items()):
            if len(symbol) <= 6:
                continue
            contract = parse_occ(symbol)
            if contract.expiration > session:
                continue
            close = self.dataset.price(contract.underlying, "close", index)
            assert close is not None
            itm = (
                (contract.strike - close) >= Decimal("0.01")
                if contract.right == "PUT"
                else (close - contract.strike) >= Decimal("0.01")
            )
            if itm:
                self._assign(symbol, contract, session)
            else:
                del self.positions_[symbol]
                self.activities_.append(
                    Activity(f"act-{next(self._ids)}", "OPEXP", symbol, quantity, session)
                )

    def assign_early(self, symbol: str) -> None:
        contract = parse_occ(symbol)
        self._assign(symbol, contract, self.now.astimezone(NEW_YORK).date())

    def _assign(self, symbol: str, contract, session: date) -> None:  # noqa: ANN001
        del self.positions_[symbol]
        shares = Decimal(100) if contract.right == "PUT" else Decimal(-100)
        self.positions_[contract.underlying] = self.positions_.get(contract.underlying, Decimal(0)) + shares
        if not self.positions_[contract.underlying]:
            del self.positions_[contract.underlying]
        self.cash -= shares * contract.strike
        self.activities_.append(Activity(f"act-{next(self._ids)}", "OPASN", symbol, Decimal(-1), session))
