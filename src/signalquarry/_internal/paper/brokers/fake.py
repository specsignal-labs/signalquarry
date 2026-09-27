# SPDX-License-Identifier: Apache-2.0
"""An in-memory paper broker driven by a dataset. Used by tests and the parity check.

Behaves like Alpaca paper for daily ``opg`` orders: splits adjust positions before
the open, market-on-open orders fill at the session's raw open, sale proceeds are
credited immediately (settlement is tracked by the kernel), no commissions and no
dividends. Faults can be injected per submission.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import Literal
from zoneinfo import ZoneInfo

from signalquarry._internal.data.dataset import Dataset
from signalquarry._internal.paper.models import (
    OPEN_ORDER_STATUSES,
    BrokerAccount,
    BrokerClock,
    BrokerOrder,
    BrokerPosition,
    OrderRequest,
    PaperError,
)

EXCHANGE = ZoneInfo("America/New_York")
Fault = Literal["error_before_accept", "error_after_accept", "reject"]


def exchange_time(session: date, at: time) -> datetime:
    return datetime.combine(session, at, tzinfo=EXCHANGE).astimezone(UTC)


@dataclass
class FakeBroker:
    dataset: Dataset
    cash: Decimal
    account_id: str = "fake-paper-account"
    now: datetime = field(default_factory=lambda: datetime.now(UTC))
    skew: timedelta = timedelta(0)
    positions_: dict[str, Decimal] = field(default_factory=dict)
    orders: dict[str, BrokerOrder] = field(default_factory=dict)
    faults: list[Fault] = field(default_factory=list)
    submissions: int = 0
    paper_only: bool = field(default=True, init=False)
    name: str = field(default="fake", init=False)
    _ids: itertools.count = field(default_factory=itertools.count, init=False, repr=False)
    _applied_splits: set[tuple[str, date]] = field(default_factory=set, init=False, repr=False)

    # -- test controls -------------------------------------------------------
    def pre_open(self, session: date, at: time = time(9, 10)) -> None:
        """Move the clock to ``session`` before the open and apply that day's splits."""
        self.now = exchange_time(session, at)
        for split in self.dataset.splits:
            key = (split.symbol, split.ex_date)
            if split.ex_date <= session and key not in self._applied_splits:
                self._applied_splits.add(key)
                if split.symbol in self.positions_:
                    self.positions_[split.symbol] *= split.ratio

    def open_session(self, session: date) -> None:
        """Fill (or cancel) every open ``opg`` order at ``session``'s raw open; sells first."""
        self.now = exchange_time(session, time(9, 31))
        index = self.dataset.index_of(session)
        pending = [o for o in self.orders.values() if o.status in OPEN_ORDER_STATUSES]
        for order in sorted(pending, key=lambda o: (o.side != "sell", o.symbol)):
            price = self.dataset.price(order.symbol, "open", index)
            cost = (price or Decimal(0)) * order.quantity
            held = self.positions_.get(order.symbol, Decimal(0))
            if (
                price is None
                or (order.side == "buy" and cost > self.cash)
                or (order.side == "sell" and order.quantity > held)
            ):
                self.orders[order.order_id] = replace(order, status="canceled")
                continue
            sign = 1 if order.side == "buy" else -1
            self.cash -= sign * cost
            self.positions_[order.symbol] = held + sign * order.quantity
            if not self.positions_[order.symbol]:
                del self.positions_[order.symbol]
            self.orders[order.order_id] = replace(
                order, status="filled", filled_quantity=order.quantity, filled_average_price=price
            )

    def next_session_after(self, session: date) -> date | None:
        return next((s for s in self.dataset.sessions if s > session), None)

    # -- PaperBroker ---------------------------------------------------------
    def account(self) -> BrokerAccount:
        local = self.now.astimezone(EXCHANGE)
        index = max(i for i, s in enumerate(self.dataset.sessions) if s <= local.date())
        closed = self.dataset.sessions[index] < local.date() or local.time() >= time(16, 0)
        stop = index + 1 if closed else index
        equity = self.cash
        for symbol, quantity in self.positions_.items():
            if symbol not in self.dataset.series:
                continue
            last = max((i for i in range(stop) if self.dataset.series[symbol].present[i]), default=None)
            if last is not None:
                close = self.dataset.price(symbol, "close", last)
                assert close is not None
                ratio = Decimal(1)
                for split in self.dataset.splits:
                    if split.symbol == symbol and self.dataset.sessions[last] < split.ex_date <= local.date():
                        ratio *= split.ratio
                equity += quantity * close / ratio
        return BrokerAccount(self.account_id, "ACTIVE", self.cash, equity, self.cash, False)

    def clock(self) -> BrokerClock:
        local = self.now.astimezone(EXCHANGE)
        today = local.date()
        sessions = self.dataset.sessions
        is_session = today in sessions
        is_open = is_session and time(9, 30) <= local.time() < time(16, 0)
        if is_session and local.time() < time(9, 30):
            next_session = today
        else:
            next_session = next((s for s in sessions if s > today), today + timedelta(days=1))
        return BrokerClock(
            timestamp=self.now + self.skew,
            is_open=is_open,
            next_open=exchange_time(next_session, time(9, 30)),
            next_close=exchange_time(today if is_open else next_session, time(16, 0)),
        )

    def positions(self) -> list[BrokerPosition]:
        return [BrokerPosition(s, q) for s, q in sorted(self.positions_.items()) if q]

    def open_orders(self) -> list[BrokerOrder]:
        return [o for o in self.orders.values() if o.status in OPEN_ORDER_STATUSES]

    def order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        return next((o for o in self.orders.values() if o.client_order_id == client_order_id), None)

    def submit(self, request: OrderRequest) -> BrokerOrder:
        self.submissions += 1
        fault = self.faults.pop(0) if self.faults else None
        if fault == "error_before_accept":
            raise PaperError("BROKER_UNAVAILABLE", "unavailable", "injected")
        if fault == "reject":
            raise PaperError("BROKER_ORDER_REJECTED", "blocked", "injected")
        existing = self.order_by_client_id(request.client_order_id)
        if existing is not None:
            return existing
        order = BrokerOrder(
            order_id=f"fake-{next(self._ids)}",
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            status="accepted",
            filled_quantity=Decimal(0),
            filled_average_price=None,
        )
        self.orders[order.order_id] = order
        if fault == "error_after_accept":
            raise PaperError("BROKER_UNAVAILABLE", "unavailable", "injected after accept")
        return order

    def cancel(self, order_id: str) -> None:
        order = self.orders.get(order_id)
        if order is not None and order.status in OPEN_ORDER_STATUSES:
            self.orders[order_id] = replace(order, status="canceled")
