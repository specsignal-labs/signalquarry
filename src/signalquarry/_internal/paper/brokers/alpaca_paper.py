# SPDX-License-Identifier: Apache-2.0
"""Alpaca **paper** trading adapter. The origin is hard-coded; no other host is accepted.

Reads are retried on 429/5xx. Order submission is never retried here: an
ambiguous failure is reported as ``BROKER_UNAVAILABLE`` and the kernel resolves it
by looking the order up by its client order id.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

from signalquarry._internal.data.alpaca import HttpResponse, offline
from signalquarry._internal.data.universe import ASSETS_PARAMS, ASSETS_PATH, AssetPage
from signalquarry._internal.paper.models import (
    Activity,
    BrokerAccount,
    BrokerClock,
    BrokerOrder,
    BrokerPosition,
    OrderRequest,
    PaperError,
)

PAPER_ORIGIN = "https://paper-api.alpaca.markets"


class BrokerTransport(Protocol):
    def request(
        self, method: str, url: str, headers: Mapping[str, str], body: bytes | None
    ) -> HttpResponse: ...


@dataclass(frozen=True)
class UrllibBrokerTransport:
    timeout_seconds: float = 15.0

    def request(self, method: str, url: str, headers: Mapping[str, str], body: bytes | None) -> HttpResponse:
        if not url.startswith(PAPER_ORIGIN + "/"):
            raise PaperError("BROKER_ORIGIN_NOT_PAPER", "invalid")
        if offline():
            raise PaperError(
                "BROKER_UNAVAILABLE", "unavailable", "SIGNALQUARRY_OFFLINE=1 forbids network access"
            )
        request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310 - fixed paper origin
                return HttpResponse(
                    int(response.status), {k.lower(): v for k, v in response.headers.items()}, response.read()
                )
        except urllib.error.HTTPError as exc:
            return HttpResponse(int(exc.code), {k.lower(): v for k, v in exc.headers.items()}, exc.read())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise PaperError("BROKER_UNAVAILABLE", "unavailable", type(exc).__name__) from exc


def _decimal(value: Any, name: str) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise PaperError("BROKER_RESPONSE_INVALID", "error", name) from exc


def _time(value: Any, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise PaperError("BROKER_RESPONSE_INVALID", "error", name) from exc
    if parsed.tzinfo is None:
        raise PaperError("BROKER_RESPONSE_INVALID", "error", name)
    return parsed


def _order(item: dict[str, Any]) -> BrokerOrder:
    try:
        side = str(item["side"])
        if side not in ("buy", "sell"):
            raise PaperError("BROKER_RESPONSE_INVALID", "error", "order side")
        average = item.get("filled_avg_price")
        return BrokerOrder(
            order_id=str(item["id"]),
            client_order_id=str(item.get("client_order_id") or ""),
            symbol=str(item["symbol"]).upper(),
            side=side,  # type: ignore[arg-type]
            quantity=_decimal(item.get("qty") or 0, "qty"),
            status=str(item["status"]),
            filled_quantity=_decimal(item.get("filled_qty") or 0, "filled_qty"),
            filled_average_price=_decimal(average, "filled_avg_price") if average is not None else None,
        )
    except KeyError as exc:
        raise PaperError("BROKER_RESPONSE_INVALID", "error", f"order field {exc}") from exc


@dataclass
class AlpacaPaperBroker:
    key_id: str = field(repr=False)
    secret_key: str = field(repr=False)
    transport: BrokerTransport = field(default_factory=UrllibBrokerTransport)
    origin: str = PAPER_ORIGIN
    max_retries: int = 3
    sleep: Callable[[float], None] = time.sleep
    paper_only: bool = field(default=True, init=False)
    name: str = field(default="alpaca-paper", init=False)

    def __post_init__(self) -> None:
        if self.origin != PAPER_ORIGIN:
            raise PaperError(
                "BROKER_ORIGIN_NOT_PAPER", "invalid", "only the Alpaca paper origin is supported"
            )

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.key_id,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _call(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        body: dict[str, Any] | None = None,
    ) -> HttpResponse:
        url = f"{PAPER_ORIGIN}{path}" + (
            f"?{urllib.parse.urlencode(sorted(params.items()))}" if params else ""
        )
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        attempts = self.max_retries + 1 if method == "GET" else 1
        for attempt in range(attempts):
            response = self.transport.request(method, url, self._headers(), payload)
            if response.status in (401, 403) and not (method == "POST" and response.status == 403):
                raise PaperError("BROKER_CREDENTIALS_REJECTED", "blocked", f"HTTP {response.status}")
            if response.status == 429 or response.status >= 500:
                if attempt + 1 < attempts:
                    wait = float(response.headers.get("retry-after", 2**attempt))
                    self.sleep(min(max(wait, 1.0), 30.0))
                    continue
                raise PaperError("BROKER_UNAVAILABLE", "unavailable", f"HTTP {response.status} {path}")
            return response
        raise PaperError("BROKER_UNAVAILABLE", "unavailable", path)  # pragma: no cover

    def _json(self, response: HttpResponse, path: str) -> Any:
        if response.status != 200:
            raise PaperError("BROKER_REQUEST_FAILED", "error", f"HTTP {response.status} {path}")
        try:
            return json.loads(response.body)
        except ValueError as exc:
            raise PaperError("BROKER_RESPONSE_INVALID", "error", path) from exc

    def account(self) -> BrokerAccount:
        item = self._json(self._call("GET", "/v2/account"), "/v2/account")
        try:
            return BrokerAccount(
                account_id=str(item["id"]),
                status=str(item["status"]),
                cash=_decimal(item["cash"], "cash"),
                equity=_decimal(item["equity"], "equity"),
                buying_power=_decimal(item["buying_power"], "buying_power"),
                trading_blocked=bool(item.get("trading_blocked") or item.get("account_blocked")),
            )
        except KeyError as exc:
            raise PaperError("BROKER_RESPONSE_INVALID", "error", f"account field {exc}") from exc

    def clock(self) -> BrokerClock:
        item = self._json(self._call("GET", "/v2/clock"), "/v2/clock")
        return BrokerClock(
            timestamp=_time(item.get("timestamp"), "timestamp"),
            is_open=bool(item.get("is_open")),
            next_open=_time(item.get("next_open"), "next_open"),
            next_close=_time(item.get("next_close"), "next_close"),
        )

    def asset_snapshot(self) -> AssetPage:
        """Read the complete US-equity master list, including inactive assets.

        This GET has no order or account effect. Omitting ``status`` is required
        because the provider's default includes every status.
        """
        response = self._call("GET", ASSETS_PATH, params=ASSETS_PARAMS)
        items = self._json(response, ASSETS_PATH)
        if not isinstance(items, list):
            raise PaperError("BROKER_RESPONSE_INVALID", "error", "asset list")
        return AssetPage(
            ASSETS_PATH,
            dict(ASSETS_PARAMS),
            response.body,
            hashlib.sha256(response.body).hexdigest(),
        )

    def calendar(self, start: date, end: date) -> list[date]:
        """Exchange sessions from ``start`` through ``end`` (trading API calendar)."""
        items = self._json(
            self._call("GET", "/v2/calendar", params={"start": start.isoformat(), "end": end.isoformat()}),
            "/v2/calendar",
        )
        try:
            return sorted(date.fromisoformat(str(item["date"])) for item in items)
        except (KeyError, TypeError, ValueError) as exc:
            raise PaperError("BROKER_RESPONSE_INVALID", "error", "calendar") from exc

    def next_session_after(self, session: date) -> date | None:
        from datetime import timedelta

        later = [
            s for s in self.calendar(session + timedelta(days=1), session + timedelta(days=10)) if s > session
        ]
        return later[0] if later else None

    def option_contracts(
        self, underlying: str, first: date, last: date, *, status: str = "inactive"
    ) -> list[str]:
        """OCC symbols of contracts expiring between ``first`` and ``last`` (read-only, trading API)."""
        items = self._json(
            self._call(
                "GET",
                "/v2/options/contracts",
                params={
                    "underlying_symbols": underlying,
                    "status": status,
                    "expiration_date_gte": first.isoformat(),
                    "expiration_date_lte": last.isoformat(),
                    "limit": "100",
                },
            ),
            "/v2/options/contracts",
        )
        return [str(item["symbol"]) for item in (items.get("option_contracts") or []) if "symbol" in item]

    def positions(self) -> list[BrokerPosition]:
        items = self._json(self._call("GET", "/v2/positions"), "/v2/positions")
        try:
            return [BrokerPosition(str(i["symbol"]).upper(), _decimal(i["qty"], "qty")) for i in items]
        except (KeyError, TypeError) as exc:
            raise PaperError("BROKER_RESPONSE_INVALID", "error", "positions") from exc

    def open_orders(self) -> list[BrokerOrder]:
        items = self._json(
            self._call(
                "GET",
                "/v2/orders",
                params={"status": "open", "limit": "500", "direction": "asc", "nested": "false"},
            ),
            "/v2/orders",
        )
        return [_order(item) for item in items]

    def order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        response = self._call(
            "GET", "/v2/orders:by_client_order_id", params={"client_order_id": client_order_id}
        )
        if response.status == 404:
            return None
        return _order(self._json(response, "/v2/orders:by_client_order_id"))

    def submit(self, request: OrderRequest) -> BrokerOrder:
        response = self._call(
            "POST",
            "/v2/orders",
            body={
                "symbol": request.symbol,
                "qty": format(request.quantity, "f"),
                "side": request.side,
                "type": "market" if request.limit_price is None else "limit",
                "time_in_force": request.time_in_force,
                "client_order_id": request.client_order_id,
                **({} if request.limit_price is None else {"limit_price": format(request.limit_price, "f")}),
            },
        )
        if response.status in (403, 422):
            existing = self.order_by_client_id(request.client_order_id)
            if existing is not None:
                return existing  # duplicate client order id: the first submission landed
            raise PaperError("BROKER_ORDER_REJECTED", "blocked", f"HTTP {response.status} {request.symbol}")
        return _order(self._json(response, "/v2/orders"))

    def activities(self, after: str | None) -> list[Activity]:
        """Option assignments, expirations and exercises (oldest first)."""
        params = {"activity_types": "OPASN,OPEXP,OPEXC", "direction": "asc", "page_size": "100"}
        if after:
            params["page_token"] = after
        items = self._json(
            self._call("GET", "/v2/account/activities", params=params), "/v2/account/activities"
        )
        out = []
        for item in items:
            try:
                out.append(
                    Activity(
                        str(item["id"]),
                        str(item["activity_type"]),  # type: ignore[arg-type]
                        str(item["symbol"]).upper(),
                        _decimal(item.get("qty") or 0, "qty"),
                        date.fromisoformat(str(item["date"])[:10]),
                    )
                )
            except (KeyError, ValueError) as exc:
                raise PaperError("BROKER_RESPONSE_INVALID", "error", "activity") from exc
        return out

    def cancel(self, order_id: str) -> None:
        path = f"/v2/orders/{urllib.parse.quote(order_id, safe='')}"
        response = self._call("DELETE", path)
        if response.status not in (200, 204, 404, 422):  # 422: no longer cancelable (filled or done)
            raise PaperError("BROKER_REQUEST_FAILED", "error", f"HTTP {response.status} cancel")
