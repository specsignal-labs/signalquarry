# SPDX-License-Identifier: Apache-2.0
"""Read-only Alpaca market-data provider (stock daily bars and corporate actions).

GET requests only, to ``https://data.alpaca.markets``. Every response page is
kept byte-for-byte (content-addressed by SHA-256) so a dataset can be verified
later. Bars are requested with ``adjustment=raw``; splits and dividends come
from the corporate-actions endpoint and are applied point-in-time by the engine.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Protocol

from signalquarry._internal.contracts import progress
from signalquarry._internal.data.dataset import Dividend, Split

DATA_ORIGIN = "https://data.alpaca.markets"
BARS_PATH = "/v2/stocks/bars"
CORPORATE_ACTIONS_PATH = "/v1/corporate-actions"
MAX_PAGES = 5000


def offline() -> bool:
    """``SIGNALQUARRY_OFFLINE=1`` forbids every network call (CI, demos, evals)."""
    return os.environ.get("SIGNALQUARRY_OFFLINE", "").strip() == "1"


class ProviderError(RuntimeError):
    """A provider request failed or returned unusable data. ``code`` is a reason code."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}:{detail}" if detail else code)
        self.code = code


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


class Transport(Protocol):
    def get(self, url: str, headers: Mapping[str, str]) -> HttpResponse: ...


@dataclass(frozen=True)
class UrllibTransport:
    timeout_seconds: float = 30.0

    def get(self, url: str, headers: Mapping[str, str]) -> HttpResponse:
        if offline():
            raise ProviderError("PROVIDER_UNAVAILABLE", "SIGNALQUARRY_OFFLINE=1 forbids network access")
        request = urllib.request.Request(url, headers=dict(headers), method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310 - fixed https origin
                return HttpResponse(
                    int(response.status), {k.lower(): v for k, v in response.headers.items()}, response.read()
                )
        except urllib.error.HTTPError as exc:
            return HttpResponse(int(exc.code), {k.lower(): v for k, v in exc.headers.items()}, exc.read())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError("PROVIDER_UNAVAILABLE", type(exc).__name__) from exc


@dataclass(frozen=True)
class RawPage:
    endpoint: str
    params: dict[str, str]
    body: bytes
    sha256: str
    payload: dict[str, Any]


@dataclass
class RateLimiter:
    """Token bucket: at most ``per_minute`` requests in any rolling minute (free plan: 200)."""

    per_minute: int = 180
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    _stamps: list[float] = field(default_factory=list)

    def acquire(self) -> None:
        now = self.clock()
        self._stamps = [stamp for stamp in self._stamps if now - stamp < 60.0]
        if len(self._stamps) >= self.per_minute:
            self.sleep(60.0 - (now - self._stamps[0]))
            now = self.clock()
        self._stamps.append(now)


@dataclass
class AlpacaDataClient:
    key_id: str = field(repr=False)
    secret_key: str = field(repr=False)
    transport: Transport = field(default_factory=UrllibTransport)
    limiter: RateLimiter = field(default_factory=RateLimiter)
    max_retries: int = 4
    sleep: Callable[[float], None] = time.sleep

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.key_id,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Accept": "application/json",
        }

    def get_page(self, path: str, params: dict[str, str]) -> RawPage:
        url = f"{DATA_ORIGIN}{path}?{urllib.parse.urlencode(sorted(params.items()))}"
        for attempt in range(self.max_retries + 1):
            self.limiter.acquire()
            response = self.transport.get(url, self._headers())
            if response.status == 429 or response.status >= 500:
                if attempt == self.max_retries:
                    break
                wait = float(response.headers.get("retry-after", 2**attempt))
                self.sleep(min(max(wait, 1.0), 60.0))
                continue
            if response.status in (401, 403):
                raise ProviderError("DATA_CREDENTIALS_REJECTED", f"HTTP {response.status}")
            if response.status != 200:
                raise ProviderError("PROVIDER_REQUEST_FAILED", f"HTTP {response.status} {path}")
            try:
                payload = json.loads(response.body)
            except ValueError as exc:
                raise ProviderError("PROVIDER_RESPONSE_INVALID", path) from exc
            if not isinstance(payload, dict):
                raise ProviderError("PROVIDER_RESPONSE_INVALID", path)
            return RawPage(
                path,
                dict(sorted(params.items())),
                response.body,
                hashlib.sha256(response.body).hexdigest(),
                payload,
            )
        raise ProviderError("PROVIDER_UNAVAILABLE", f"retries exhausted {path}")

    def paginate(self, path: str, params: dict[str, str]) -> list[RawPage]:
        pages: list[RawPage] = []
        seen: set[str] = set()
        token: str | None = None
        for _ in range(MAX_PAGES):
            request = dict(params, **({"page_token": token} if token else {}))
            page = self.get_page(path, request)
            pages.append(page)
            progress.emit("fetch", endpoint=path, pages=len(pages))
            token = page.payload.get("next_page_token")
            if not token:
                return pages
            if token in seen:
                raise ProviderError("PROVIDER_PAGINATION_LOOP", path)
            seen.add(token)
        raise ProviderError("PROVIDER_PAGINATION_LIMIT", path)

    def daily_bars(self, symbols: tuple[str, ...], start: date, end: date, feed: str) -> list[RawPage]:
        return self.paginate(
            BARS_PATH,
            {
                "symbols": ",".join(symbols),
                "timeframe": "1Day",
                "start": start.isoformat(),
                "end": end.isoformat(),
                "adjustment": "raw",
                "feed": feed,
                "limit": "10000",
                "sort": "asc",
            },
        )

    def corporate_actions(self, symbols: tuple[str, ...], start: date, end: date) -> list[RawPage]:
        return self.paginate(
            CORPORATE_ACTIONS_PATH,
            {
                "symbols": ",".join(symbols),
                "types": "forward_split,reverse_split,cash_dividend",
                "start": start.isoformat(),
                "end": end.isoformat(),
                "limit": "1000",
            },
        )


def latest_stock_quote(
    client: AlpacaDataClient, symbol: str, feed: str
) -> tuple[Decimal, Decimal, datetime] | None:
    """(bid, ask, timestamp) of the latest quote, or None."""
    page = client.get_page(f"/v2/stocks/{urllib.parse.quote(symbol)}/quotes/latest", {"feed": feed})
    quote = page.payload.get("quote") or {}
    try:
        return (
            Decimal(str(quote["bp"])),
            Decimal(str(quote["ap"])),
            datetime.fromisoformat(str(quote["t"]).replace("Z", "+00:00")),
        )
    except (KeyError, ValueError, ArithmeticError):
        return None


def option_snapshots(
    client: AlpacaDataClient,
    underlying: str,
    *,
    right: str,
    expiration_from: date,
    expiration_to: date,
    strike_from: Decimal,
    strike_to: Decimal,
    feed: str = "indicative",
) -> dict[str, tuple[Decimal, Decimal, datetime]]:
    """Latest option quotes by OCC symbol for one underlying (paginated)."""
    pages = option_snapshot_pages(
        client,
        underlying,
        right=right,
        expiration_from=expiration_from,
        expiration_to=expiration_to,
        strike_from=strike_from,
        strike_to=strike_to,
        feed=feed,
    )
    return quotes_from_snapshot_pages(pages)


def option_snapshot_pages(
    client: AlpacaDataClient,
    underlying: str,
    *,
    right: str,
    expiration_from: date,
    expiration_to: date,
    strike_from: Decimal,
    strike_to: Decimal,
    feed: str = "indicative",
) -> list[RawPage]:
    """The raw option-chain snapshot pages (kept byte-for-byte when recording chains)."""
    return client.paginate(
        f"/v1beta1/options/snapshots/{urllib.parse.quote(underlying)}",
        {
            "feed": feed,
            "type": right.lower(),
            "expiration_date_gte": expiration_from.isoformat(),
            "expiration_date_lte": expiration_to.isoformat(),
            "strike_price_gte": format(strike_from, "f"),
            "strike_price_lte": format(strike_to, "f"),
            "limit": "1000",
        },
    )


def quotes_from_snapshot_pages(pages: list[RawPage]) -> dict[str, tuple[Decimal, Decimal, datetime]]:
    out: dict[str, tuple[Decimal, Decimal, datetime]] = {}
    for page in pages:
        for symbol, snapshot in (page.payload.get("snapshots") or {}).items():
            quote = (snapshot or {}).get("latestQuote") or {}
            try:
                out[symbol] = (
                    Decimal(str(quote["bp"])),
                    Decimal(str(quote["ap"])),
                    datetime.fromisoformat(str(quote["t"]).replace("Z", "+00:00")),
                )
            except (KeyError, ValueError, ArithmeticError):
                continue
    return out


def option_daily_bar_count(client: AlpacaDataClient, symbol: str, start: date, end: date) -> int:
    """How many daily bars the data API returns for one (possibly expired) option contract."""
    pages = client.paginate(
        "/v1beta1/options/bars",
        {
            "symbols": symbol,
            "timeframe": "1Day",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": "1000",
        },
    )
    return sum(len((page.payload.get("bars") or {}).get(symbol, [])) for page in pages)


def bars_from_pages(pages: list[RawPage]) -> dict[str, list[dict[str, Any]]]:
    """Merge bar pages into ``{symbol: [{"session", "open", ...}]}`` sorted by session."""
    merged: dict[str, dict[date, dict[str, Any]]] = {}
    for page in pages:
        for symbol, rows in (page.payload.get("bars") or {}).items():
            for row in rows:
                try:
                    session = datetime.fromisoformat(str(row["t"]).replace("Z", "+00:00")).date()
                    values = {
                        "session": session,
                        **{
                            name: Decimal(str(row[key]))
                            for name, key in (("open", "o"), ("high", "h"), ("low", "l"), ("close", "c"))
                        },
                        "volume": float(row["v"]),
                    }
                except (KeyError, ValueError, ArithmeticError) as exc:
                    raise ProviderError("PROVIDER_RESPONSE_INVALID", f"bar {symbol}") from exc
                if (
                    min(values["open"], values["high"], values["low"], values["close"]) <= 0
                    or values["high"] < values["low"]
                ):
                    raise ProviderError("PROVIDER_RESPONSE_INVALID", f"bar values {symbol} {session}")
                existing = merged.setdefault(symbol.upper(), {})
                if session in existing and existing[session] != values:
                    raise ProviderError("PROVIDER_RESPONSE_INVALID", f"conflicting bars {symbol} {session}")
                existing[session] = values
    return {symbol: [rows[s] for s in sorted(rows)] for symbol, rows in merged.items()}


def actions_from_pages(pages: list[RawPage]) -> tuple[tuple[Split, ...], tuple[Dividend, ...]]:
    splits: dict[tuple[str, date], Split] = {}
    dividends: dict[tuple[str, date], Dividend] = {}
    for page in pages:
        actions = page.payload.get("corporate_actions") or {}
        try:
            for kind in ("forward_splits", "reverse_splits"):
                for item in actions.get(kind, []):
                    ratio = Decimal(str(item["new_rate"])) / Decimal(str(item["old_rate"]))
                    key = (str(item["symbol"]).upper(), date.fromisoformat(item["ex_date"]))
                    splits[key] = Split(key[0], key[1], ratio)
            for item in actions.get("cash_dividends", []):
                key = (str(item["symbol"]).upper(), date.fromisoformat(item["ex_date"]))
                pay = date.fromisoformat(item.get("payable_date") or item["ex_date"])
                dividends[key] = Dividend(key[0], key[1], pay, Decimal(str(item["rate"])))
        except (KeyError, ValueError, ArithmeticError) as exc:
            raise ProviderError("PROVIDER_RESPONSE_INVALID", "corporate action") from exc
    return tuple(splits[k] for k in sorted(splits)), tuple(dividends[k] for k in sorted(dividends))
