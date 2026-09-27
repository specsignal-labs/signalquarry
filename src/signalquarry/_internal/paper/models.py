# SPDX-License-Identifier: Apache-2.0
"""Broker-neutral paper-trading records and the kernel's error type."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Literal, Protocol

# Envelope status for each kernel outcome; the CLI maps status to the exit code.
Status = Literal["blocked", "busy", "disabled", "invalid", "unavailable", "error"]

OPEN_ORDER_STATUSES = frozenset(
    {"new", "accepted", "pending_new", "accepted_for_bidding", "partially_filled", "held", "pending_cancel"}
)
TERMINAL_ORDER_STATUSES = frozenset({"filled", "canceled", "expired", "rejected", "done_for_day", "stopped"})


class PaperError(RuntimeError):
    """The kernel refused or could not complete an action. ``code`` is a reason code."""

    def __init__(self, code: str, status: Status, detail: str = "") -> None:
        super().__init__(f"{code}:{detail}" if detail else code)
        self.code = code
        self.status: Status = status
        self.detail = detail


@dataclass(frozen=True)
class BrokerAccount:
    account_id: str
    status: str
    cash: Decimal
    equity: Decimal
    buying_power: Decimal
    trading_blocked: bool


@dataclass(frozen=True)
class BrokerPosition:
    symbol: str
    quantity: Decimal


@dataclass(frozen=True)
class BrokerOrder:
    order_id: str
    client_order_id: str
    symbol: str
    side: Literal["buy", "sell"]
    quantity: Decimal
    status: str
    filled_quantity: Decimal
    filled_average_price: Decimal | None


@dataclass(frozen=True)
class OrderRequest:
    client_order_id: str
    symbol: str
    side: Literal["buy", "sell"]
    quantity: Decimal
    time_in_force: Literal["opg", "day"] = "opg"
    limit_price: Decimal | None = None  # None: market order


@dataclass(frozen=True)
class Activity:
    """A non-trade account activity: option assignment (OPASN), expiration (OPEXP) or exercise (OPEXC)."""

    activity_id: str
    kind: Literal["OPASN", "OPEXP", "OPEXC"]
    symbol: str
    quantity: Decimal
    occurred: date


@dataclass(frozen=True)
class BrokerClock:
    timestamp: datetime
    is_open: bool
    next_open: datetime
    next_close: datetime


class PaperBroker(Protocol):
    """A paper-only broker. Implementations must set ``paper_only = True``."""

    paper_only: bool
    name: str

    def account(self) -> BrokerAccount: ...

    def clock(self) -> BrokerClock: ...

    def positions(self) -> list[BrokerPosition]: ...

    def open_orders(self) -> list[BrokerOrder]: ...

    def order_by_client_id(self, client_order_id: str) -> BrokerOrder | None: ...

    def submit(self, request: OrderRequest) -> BrokerOrder: ...

    def cancel(self, order_id: str) -> None: ...
