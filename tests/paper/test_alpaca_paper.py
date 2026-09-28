# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import pytest

from signalquarry._internal.data.alpaca import HttpResponse
from signalquarry._internal.paper.brokers.alpaca_paper import (
    PAPER_ORIGIN,
    AlpacaPaperBroker,
    UrllibBrokerTransport,
)
from signalquarry._internal.paper.models import OrderRequest, PaperError

ORDER = {
    "id": "o-1",
    "client_order_id": "sq-demo-abc-0",
    "symbol": "spy",
    "side": "buy",
    "qty": "3",
    "status": "accepted",
    "filled_qty": "0",
    "filled_avg_price": None,
}


@dataclass
class Script:
    responses: list[tuple[int, object]]
    calls: list[tuple[str, str, bytes | None]] = field(default_factory=list)

    def request(self, method: str, url: str, headers: Mapping[str, str], body: bytes | None) -> HttpResponse:
        assert url.startswith(PAPER_ORIGIN + "/")
        assert headers["APCA-API-KEY-ID"] == "key" and headers["APCA-API-SECRET-KEY"] == "secret"
        self.calls.append((method, url, body))
        status, payload = self.responses.pop(0)
        return HttpResponse(status, {"retry-after": "0"}, json.dumps(payload).encode())


def _broker(*responses: tuple[int, object]) -> tuple[AlpacaPaperBroker, Script]:
    script = Script(list(responses))
    return AlpacaPaperBroker("key", "secret", transport=script, sleep=lambda _: None), script


def test_only_the_paper_origin_is_accepted() -> None:
    with pytest.raises(PaperError) as info:
        AlpacaPaperBroker("key", "secret", origin="https://api.alpaca.markets")
    assert info.value.code == "BROKER_ORIGIN_NOT_PAPER"
    with pytest.raises(PaperError):
        UrllibBrokerTransport().request("GET", "https://api.alpaca.markets/v2/account", {}, None)
    broker, _ = _broker()
    assert "secret" not in repr(broker) and broker.paper_only is True


def test_asset_snapshot_reads_all_statuses_without_orders() -> None:
    import urllib.parse

    broker, script = _broker((200, [{"id": "asset-1", "status": "inactive"}]))
    page = broker.asset_snapshot()
    assert page.endpoint == "/v2/assets" and page.params == {"asset_class": "us_equity"}
    method, url, body = script.calls[0]
    query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    assert method == "GET" and body is None
    assert query == {"asset_class": ["us_equity"]}
    assert b'"inactive"' in page.body


def test_account_clock_positions_and_orders_are_parsed() -> None:
    broker, script = _broker(
        (200, {"id": "acct", "status": "ACTIVE", "cash": "1000.5", "equity": "1200", "buying_power": "2000"}),
        (503, {}),
        (
            200,
            {
                "timestamp": "2026-09-25T09:05:00-04:00",
                "is_open": False,
                "next_open": "2026-09-25T09:30:00-04:00",
                "next_close": "2026-09-25T16:00:00-04:00",
            },
        ),
        (200, [{"symbol": "spy", "qty": "4"}]),
        (200, [ORDER]),
        (404, {"message": "not found"}),
    )
    account = broker.account()
    assert (account.cash, account.equity, account.trading_blocked) == (
        Decimal("1000.5"),
        Decimal(1200),
        False,
    )
    assert broker.clock().is_open is False  # the 503 was retried
    assert broker.positions()[0].quantity == Decimal(4)
    assert broker.open_orders()[0].symbol == "SPY"
    assert broker.order_by_client_id("missing") is None
    assert "status=open" in script.calls[4][1]


def test_submission_is_never_retried_and_duplicates_resolve_by_lookup() -> None:
    broker, script = _broker((503, {}))
    with pytest.raises(PaperError) as info:
        broker.submit(OrderRequest("sq-demo-abc-0", "SPY", "buy", Decimal(3)))
    assert info.value.code == "BROKER_UNAVAILABLE" and len(script.calls) == 1
    body = json.loads(script.calls[0][2] or b"{}")
    assert body == {
        "symbol": "SPY",
        "qty": "3",
        "side": "buy",
        "type": "market",
        "time_in_force": "opg",
        "client_order_id": "sq-demo-abc-0",
    }

    broker, _ = _broker((422, {"message": "client_order_id must be unique"}), (200, ORDER))
    assert broker.submit(OrderRequest("sq-demo-abc-0", "SPY", "buy", Decimal(3))).order_id == "o-1"

    broker, _ = _broker((403, {"message": "insufficient buying power"}), (404, {}))
    with pytest.raises(PaperError) as info:
        broker.submit(OrderRequest("sq-demo-abc-0", "SPY", "buy", Decimal(3)))
    assert info.value.code == "BROKER_ORDER_REJECTED"


def test_rejected_credentials_do_not_leak() -> None:
    broker, _ = _broker((401, {"message": "unauthorized"}))
    with pytest.raises(PaperError) as info:
        broker.account()
    assert info.value.code == "BROKER_CREDENTIALS_REJECTED" and "secret" not in str(info.value)


def test_no_live_trading_host_anywhere_in_the_package() -> None:
    source = Path(__file__).parents[2] / "src"
    offenders = [
        str(path)
        for path in source.rglob("*.py")
        if re.search(r"(?<![-.\w])api\.alpaca\.markets", path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_calendar_cancel_and_invalid_responses() -> None:
    from datetime import date

    broker, script = _broker(
        (200, [{"date": "2026-09-24"}, {"date": "2026-09-25"}]),
        (204, {}),
        (422, {"message": "not cancelable"}),
        (500, {}),
        (500, {}),
        (500, {}),
        (500, {}),
        (200, {"id": "a"}),
        (200, [{"symbol": "SPY"}]),
        (200, [{"date": "not-a-date"}]),
        (400, {"message": "bad"}),
    )
    assert broker.calendar(date(2026, 9, 24), date(2026, 9, 25)) == [date(2026, 9, 24), date(2026, 9, 25)]
    broker.cancel("o-1")
    broker.cancel("o-2")
    with pytest.raises(PaperError) as info:
        broker.clock()
    assert info.value.code == "BROKER_UNAVAILABLE"
    for action in (
        broker.account,
        broker.positions,
        lambda: broker.calendar(date(2026, 1, 1), date(2026, 1, 2)),
    ):
        with pytest.raises(PaperError) as info:
            action()
        assert info.value.code == "BROKER_RESPONSE_INVALID"
    with pytest.raises(PaperError) as info:
        broker.cancel("o-3")
    assert info.value.code == "BROKER_REQUEST_FAILED"


def test_urllib_transport_maps_network_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    import urllib.error

    def refuse(request, timeout):
        raise urllib.error.URLError("down")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    with pytest.raises(PaperError) as info:
        UrllibBrokerTransport().request("GET", PAPER_ORIGIN + "/v2/clock", {}, None)
    assert info.value.code == "BROKER_UNAVAILABLE"

    def http_error(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 404, "nf", {"X-A": "1"}, io.BytesIO(b"{}"))

    monkeypatch.setattr("urllib.request.urlopen", http_error)
    response = UrllibBrokerTransport().request("GET", PAPER_ORIGIN + "/v2/orders", {}, None)
    assert response.status == 404 and response.headers["x-a"] == "1"


def test_limit_orders_and_activities() -> None:
    from datetime import date

    broker, script = _broker(
        (200, ORDER),
        (
            200,
            [
                {
                    "id": "a1",
                    "activity_type": "OPASN",
                    "symbol": "QQQ261002P00480000",
                    "qty": "-1",
                    "date": "2026-10-02",
                }
            ],
        ),
        (200, [{"id": "a2"}]),
    )
    broker.submit(
        OrderRequest("sq-demo-abc-0", "QQQ261002P00480000", "sell", Decimal(1), "day", Decimal("1.23"))
    )
    body = json.loads(script.calls[0][2] or b"{}")
    assert body["type"] == "limit" and body["limit_price"] == "1.23" and body["time_in_force"] == "day"
    activity = broker.activities(None)[0]
    assert (activity.kind, activity.symbol, activity.occurred) == (
        "OPASN",
        "QQQ261002P00480000",
        date(2026, 10, 2),
    )
    with pytest.raises(PaperError):
        broker.activities("token")


def test_options_venue_quotes_through_the_data_api() -> None:
    from signalquarry._internal.data.alpaca import AlpacaDataClient, RateLimiter
    from signalquarry._internal.paper.brokers.alpaca_options import AlpacaOptionsVenue

    @dataclass
    class DataScript:
        responses: list
        urls: list = field(default_factory=list)

        def get(self, url, headers):
            self.urls.append(url)
            return HttpResponse(200, {}, json.dumps(self.responses.pop(0)).encode())

    quote = {"bp": 1.1, "ap": 1.2, "t": "2026-09-28T14:00:00Z"}
    data = DataScript(
        [
            {"quote": {"bp": 480.1, "ap": 480.2, "t": "2026-09-28T14:00:00Z"}},
            {
                "snapshots": {
                    "QQQ261002P00475000": {"latestQuote": quote},
                    "BAD": {"latestQuote": quote},
                    "QQQ261002P00470000": {},
                }
            },
            {"snapshots": {"QQQ261002P00475000": {"latestQuote": quote}}},
        ]
    )
    client = AlpacaDataClient(
        "k", "s", transport=data, limiter=RateLimiter(per_minute=1000), sleep=lambda _: None
    )
    broker, _ = _broker()
    venue = AlpacaOptionsVenue(broker, client)
    assert venue.stock_quote("QQQ").mid == Decimal("480.15")
    candidates = venue.option_candidates("QQQ", "PUT", 1, 10, Decimal(475))
    assert [c.contract.symbol for c in candidates] == ["QQQ261002P00475000"]
    assert venue.option_quote("QQQ261002P00475000").bid == Decimal("1.1")
    assert "/v1beta1/options/snapshots/QQQ" in data.urls[1] and "type=put" in data.urls[1]


def test_option_contracts_lists_expired_symbols() -> None:
    from datetime import date

    broker, script = _broker(
        (200, {"option_contracts": [{"symbol": "QQQ240315P00400000"}, {"id": "no-symbol"}]})
    )
    assert broker.option_contracts("QQQ", date(2024, 3, 1), date(2024, 3, 31)) == ["QQQ240315P00400000"]
    method, url, _ = script.calls[0]
    assert method == "GET" and "/v2/options/contracts?" in url and "status=inactive" in url
