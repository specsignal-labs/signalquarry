# SPDX-License-Identifier: Apache-2.0
"""The Alpaca options venue: the paper broker for orders and activities, the data API for quotes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from signalquarry._internal.data.alpaca import (
    AlpacaDataClient,
    ProviderError,
    latest_stock_quote,
    option_snapshots,
)
from signalquarry._internal.options.contracts import OptionsError, Quote, parse_occ
from signalquarry._internal.options.resolver import Candidate
from signalquarry._internal.paper.brokers.alpaca_paper import AlpacaPaperBroker
from signalquarry._internal.paper.models import (
    Activity,
    BrokerAccount,
    BrokerClock,
    BrokerOrder,
    BrokerPosition,
    OrderRequest,
    PaperError,
)

NEW_YORK = ZoneInfo("America/New_York")


@dataclass
class AlpacaOptionsVenue:
    broker: AlpacaPaperBroker
    data: AlpacaDataClient
    stock_feed: str = "iex"
    option_feed: str = "indicative"
    paper_only: bool = True
    name: str = "alpaca-paper-options"

    def account(self) -> BrokerAccount:
        return self.broker.account()

    def clock(self) -> BrokerClock:
        return self.broker.clock()

    def positions(self) -> list[BrokerPosition]:
        return self.broker.positions()

    def open_orders(self) -> list[BrokerOrder]:
        return self.broker.open_orders()

    def order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        return self.broker.order_by_client_id(client_order_id)

    def submit(self, request: OrderRequest) -> BrokerOrder:
        return self.broker.submit(request)

    def cancel(self, order_id: str) -> None:
        self.broker.cancel(order_id)

    def activities(self, after: str | None) -> list[Activity]:
        return self.broker.activities(after)

    def stock_quote(self, symbol: str) -> Quote | None:
        try:
            found = latest_stock_quote(self.data, symbol, self.stock_feed)
        except ProviderError as exc:
            raise PaperError("PAPER_DATA_UNAVAILABLE", "unavailable", exc.code) from exc
        return Quote(*found) if found else None

    def _snapshots(
        self, underlying: str, right: str, first: date, last: date, low: Decimal, high: Decimal
    ) -> dict[str, Quote]:
        try:
            raw = option_snapshots(
                self.data,
                underlying,
                right=right,
                expiration_from=first,
                expiration_to=last,
                strike_from=low,
                strike_to=high,
                feed=self.option_feed,
            )
        except ProviderError as exc:
            raise PaperError("PAPER_DATA_UNAVAILABLE", "unavailable", exc.code) from exc
        return {symbol: Quote(*values) for symbol, values in raw.items()}

    def option_quote(self, symbol: str) -> Quote | None:
        contract = parse_occ(symbol)
        quotes = self._snapshots(
            contract.underlying,
            contract.right,
            contract.expiration,
            contract.expiration,
            contract.strike,
            contract.strike,
        )
        return quotes.get(contract.symbol)

    def option_candidates(
        self, underlying: str, right: str, min_dte: int, max_dte: int, target: Decimal
    ) -> list[Candidate]:
        today = datetime.now(UTC).astimezone(NEW_YORK).date()
        quotes = self._snapshots(
            underlying,
            right,
            today + timedelta(days=min_dte),
            today + timedelta(days=max_dte),
            target * Decimal("0.85"),
            target * Decimal("1.15"),
        )
        out = []
        for symbol, quote in quotes.items():
            try:
                out.append(Candidate(parse_occ(symbol), quote))
            except OptionsError:
                continue
        return out
