# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from signalquarry._internal.data.alpaca import (
    DATA_ORIGIN,
    AlpacaDataClient,
    HttpResponse,
    RateLimiter,
)
from signalquarry._internal.paper.brokers.alpaca_paper import PAPER_ORIGIN, AlpacaPaperBroker
from signalquarry._internal.paper.models import OrderRequest
from tools.alpaca_cassette import (
    AlpacaReadOnlyTransport,
    Cassette,
    CassetteEntry,
    CassetteError,
    RecordingAlpacaTransport,
    ReplayTransport,
)


@dataclass
class Script:
    responses: list[HttpResponse]
    calls: list[tuple[str, dict[str, str]]] = field(default_factory=list)

    def get(self, url: str, headers: Mapping[str, str]) -> HttpResponse:
        self.calls.append((url, dict(headers)))
        return self.responses.pop(0)


@dataclass
class BrokerScript:
    responses: list[HttpResponse]
    calls: list[tuple[str, str, bytes | None]] = field(default_factory=list)

    def request(self, method: str, url: str, headers: Mapping[str, str], body: bytes | None) -> HttpResponse:
        self.calls.append((method, url, body))
        return self.responses.pop(0)


def _client(transport: object) -> AlpacaDataClient:
    return AlpacaDataClient(
        "fixture-key-id-123",
        "local-redaction-probe-456",
        transport=transport,  # type: ignore[arg-type]
        limiter=RateLimiter(sleep=lambda _: None),
        sleep=lambda _: None,
    )


def test_data_client_records_private_redacted_responses_and_replays_offline(tmp_path: Path) -> None:
    secret = "local-redaction-probe-456"
    responses = [
        HttpResponse(
            200,
            {"content-type": "application/json", "x-provider-request-id": "not-retained"},
            json.dumps(
                {
                    "bars": {
                        "SPY": [{"t": "2025-01-02T05:00:00Z", "o": 10, "h": 11, "l": 9, "c": 10, "v": 5}]
                    },
                    "next_page_token": "synthetic-next-page",
                    "debug": secret,
                }
            ).encode(),
        ),
        HttpResponse(
            200,
            {"content-type": "application/json"},
            b'{"bars":{"SPY":[{"t":"2025-01-03T05:00:00Z","o":10,"h":12,"l":9,"c":11,"v":6}]},"next_page_token":null}',
        ),
        HttpResponse(
            200,
            {"content-type": "application/json"},
            b'{"corporate_actions":{"forward_splits":[],"reverse_splits":[],"cash_dividends":[]},"next_page_token":null}',
        ),
    ]
    source = Script(responses.copy())
    path = tmp_path / "private" / "synthetic-data.json"
    with RecordingAlpacaTransport(
        source,
        path,
        recording_kind="synthetic",
        credentials=("fixture-key-id-123", secret),
    ) as recorder:
        bars = _client(recorder).daily_bars(("SPY",), date(2025, 1, 2), date(2025, 1, 3), "iex")
        actions = _client(recorder).corporate_actions(("SPY",), date(2025, 1, 2), date(2025, 1, 3))

    saved = path.read_text(encoding="utf-8")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert "fixture-key-id-123" not in saved and secret not in saved
    assert "APCA-API-KEY-ID" not in saved and "x-provider-request-id" not in saved
    assert len(source.calls) == 3
    assert all(url.startswith(DATA_ORIGIN + "/") for url, _ in source.calls)
    assert all(headers["APCA-API-SECRET-KEY"] == secret for _, headers in source.calls)

    cassette = Cassette.read(path)
    assert len(cassette.entries) == 3
    assert cassette.recording_kind == "synthetic"
    encoded = cassette.dumps()
    assert Cassette.loads(encoded).dumps() == encoded
    replay = ReplayTransport(cassette)
    replayed_bars = _client(replay).daily_bars(("SPY",), date(2025, 1, 2), date(2025, 1, 3), "iex")
    replayed_actions = _client(replay).corporate_actions(("SPY",), date(2025, 1, 2), date(2025, 1, 3))
    replay.assert_complete()

    assert [item.payload["bars"] for item in replayed_bars] == [item.payload["bars"] for item in bars]
    assert replayed_bars[0].payload["debug"] == "[REDACTED]"
    assert [item.payload for item in replayed_actions] == [item.payload for item in actions]


def test_recorder_rejects_repository_paths_and_non_data_origins(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    with pytest.raises(CassetteError, match="outside Git worktrees"):
        RecordingAlpacaTransport(
            Script([]),
            repository / "tests" / "must-not-record.json",
            recording_kind="synthetic",
        )

    source = Script([])
    recorder = RecordingAlpacaTransport(
        source,
        tmp_path / "private" / "no-network.json",
        recording_kind="synthetic",
    )
    with pytest.raises(CassetteError, match="fixed Alpaca"):
        recorder.get("https://example.invalid/v2/stocks/bars", {})
    assert source.calls == []

    with pytest.raises(CassetteError, match="read-only Alpaca allowlist"):
        recorder.request("GET", f"{PAPER_ORIGIN}/v2/account", {}, None)
    with pytest.raises(CassetteError, match="read-only"):
        recorder.request("POST", f"{PAPER_ORIGIN}/v2/orders", {}, b"{}")
    assert source.calls == []


def test_recorder_rejects_credentials_in_request_url_before_network(tmp_path: Path) -> None:
    source = Script([])
    credential = "fixture/key+id-123"
    recorder = RecordingAlpacaTransport(
        source,
        tmp_path / "private" / "no-credential-url.json",
        recording_kind="private-provider",
        credentials=(credential, "local-redaction-probe-456"),
    )

    with pytest.raises(CassetteError, match="request URL containing credentials"):
        recorder.get(f"{DATA_ORIGIN}/v2/stocks/bars?api_key=fixture%2Fkey%2Bid-123", {})

    assert source.calls == []


@pytest.mark.parametrize(
    "origin,path",
    [
        (DATA_ORIGIN, "/v1/corporate-actions"),
        (DATA_ORIGIN, "/v1beta1/options/bars"),
        (DATA_ORIGIN, "/v1beta1/options/snapshots/SPY"),
        (DATA_ORIGIN, "/v2/stocks/bars"),
        (DATA_ORIGIN, "/v2/stocks/SPY/quotes/latest"),
        (PAPER_ORIGIN, "/v2/assets"),
        (PAPER_ORIGIN, "/v2/calendar"),
        (PAPER_ORIGIN, "/v2/clock"),
        (PAPER_ORIGIN, "/v2/options/contracts"),
    ],
)
def test_recorder_allowlist_accepts_documented_read_endpoints(origin: str, path: str, tmp_path: Path) -> None:
    response = HttpResponse(200, {"content-type": "application/json"}, b"{}")
    data_source = Script([response])
    paper_source = BrokerScript([response])
    path_out = tmp_path / "private" / "allowlisted.json"
    with RecordingAlpacaTransport(
        AlpacaReadOnlyTransport(data_source, paper_source), path_out, recording_kind="private-provider"
    ) as recorder:
        url = f"{origin}{path}"
        if origin == DATA_ORIGIN:
            recorder.get(url, {})
        else:
            recorder.request("GET", url, {}, None)
    assert Cassette.read(path_out).entries[0].url == url


def test_paper_calendar_can_be_recorded_and_replayed_but_account_data_is_blocked(
    tmp_path: Path,
) -> None:
    source = BrokerScript(
        [HttpResponse(200, {"content-type": "application/json"}, b'[{"date":"2026-10-01"}]')]
    )
    path = tmp_path / "private" / "synthetic-calendar.json"
    with RecordingAlpacaTransport(source, path, recording_kind="synthetic") as recorder:
        broker = AlpacaPaperBroker("fixture-paper-key", "local-paper-probe", transport=recorder)
        sessions = broker.calendar(date(2026, 10, 1), date(2026, 10, 2))
    assert sessions == [date(2026, 10, 1)]
    assert len(source.calls) == 1 and source.calls[0][0] == "GET"

    replay = ReplayTransport.read(path)
    broker = AlpacaPaperBroker("unused-key", "unused-secret", transport=replay)
    assert broker.calendar(date(2026, 10, 1), date(2026, 10, 2)) == sessions
    replay.assert_complete()


def test_one_cassette_replays_ordered_data_and_paper_reads(tmp_path: Path) -> None:
    data_source = Script(
        [
            HttpResponse(
                200,
                {"content-type": "application/json"},
                b'{"bars":{"SPY":[]},"next_page_token":null}',
            )
        ]
    )
    paper_source = BrokerScript(
        [HttpResponse(200, {"content-type": "application/json"}, b'[{"date":"2026-10-01"}]')]
    )
    path = tmp_path / "private" / "synthetic-mixed.json"
    with RecordingAlpacaTransport(
        AlpacaReadOnlyTransport(data_source, paper_source), path, recording_kind="synthetic"
    ) as recorder:
        data_client = _client(recorder)
        data_client.daily_bars(("SPY",), date(2026, 10, 1), date(2026, 10, 1), "iex")
        paper_broker = AlpacaPaperBroker("fixture-paper-key", "local-paper-probe", transport=recorder)
        paper_broker.calendar(date(2026, 10, 1), date(2026, 10, 1))

    replay = ReplayTransport.read(path)
    _client(replay).daily_bars(("SPY",), date(2026, 10, 1), date(2026, 10, 1), "iex")
    paper_broker = AlpacaPaperBroker("unused-key", "unused-token", transport=replay)
    assert paper_broker.calendar(date(2026, 10, 1), date(2026, 10, 1)) == [date(2026, 10, 1)]
    replay.assert_complete()


def test_replay_drives_paper_broker_calendar_and_submit_without_network() -> None:
    calendar_url = f"{PAPER_ORIGIN}/v2/calendar?end=2026-10-02&start=2026-10-01"
    calendar = CassetteEntry.create(
        "GET",
        calendar_url,
        None,
        HttpResponse(200, {"content-type": "application/json"}, b'[{"date":"2026-10-01"}]'),
    )
    order_body = json.dumps(
        {
            "symbol": "SPY",
            "qty": "1",
            "side": "buy",
            "type": "market",
            "time_in_force": "opg",
            "client_order_id": "sqy-synthetic-cassette-0",
        }
    ).encode()
    order_response = (
        b'{"id":"synthetic-order-id","client_order_id":"sqy-synthetic-cassette-0",'
        b'"symbol":"SPY","side":"buy","qty":"1","status":"accepted","filled_qty":"0",'
        b'"filled_avg_price":null}'
    )
    submit = CassetteEntry.create(
        "POST",
        f"{PAPER_ORIGIN}/v2/orders",
        order_body,
        HttpResponse(200, {"content-type": "application/json"}, order_response),
    )
    replay = ReplayTransport(Cassette((calendar, submit), "synthetic"))
    broker = AlpacaPaperBroker(
        "unused-key",
        "unused-token",
        transport=replay,
        sleep=lambda _: None,
    )

    assert broker.calendar(date(2026, 10, 1), date(2026, 10, 2)) == [date(2026, 10, 1)]
    order = broker.submit(OrderRequest("sqy-synthetic-cassette-0", "SPY", "buy", Decimal(1)))
    assert order.order_id == "synthetic-order-id"
    replay.assert_complete()


def test_private_provider_cassette_rejects_synthetic_order_entries() -> None:
    order = CassetteEntry.create(
        "POST",
        f"{PAPER_ORIGIN}/v2/orders",
        b"{}",
        HttpResponse(200, {"content-type": "application/json"}, b'{"status":"accepted"}'),
    )

    with pytest.raises(CassetteError, match="private-provider cassettes may contain only GET"):
        Cassette((order,), "private-provider")


def test_replay_rejects_mismatched_requests_and_unused_entries() -> None:
    response = HttpResponse(200, {"content-type": "application/json"}, b"{}")
    entry = CassetteEntry.create("GET", f"{DATA_ORIGIN}/v2/one", None, response)
    replay = ReplayTransport(Cassette((entry,), "synthetic"))
    with pytest.raises(CassetteError, match="request mismatch"):
        replay.get(f"{DATA_ORIGIN}/v2/two", {})
    with pytest.raises(CassetteError, match="1 cassette entry unused"):
        replay.assert_complete()


def test_cassette_loader_rejects_response_body_tampering() -> None:
    entry = CassetteEntry.create(
        "GET",
        f"{DATA_ORIGIN}/v2/test",
        None,
        HttpResponse(200, {"content-type": "application/json"}, b'{"ok":true}'),
    )
    document = json.loads(Cassette((entry,), "synthetic").dumps())
    document["entries"][0]["body_sha256"] = "0" * 64
    with pytest.raises(CassetteError, match="body hash mismatch"):
        Cassette.loads(json.dumps(document).encode())


def test_cassette_loader_rejects_metadata_tampering() -> None:
    entry = CassetteEntry.create(
        "GET",
        f"{DATA_ORIGIN}/v2/stocks/bars",
        None,
        HttpResponse(200, {"content-type": "application/json"}, b"{}"),
    )
    raw = Cassette((entry,), "synthetic").dumps()
    document = json.loads(raw)
    document["entries"][0]["status"] = 201

    with pytest.raises(CassetteError, match="cassette checksum mismatch"):
        Cassette.loads(json.dumps(document).encode())


def test_private_cassette_loader_rejects_group_readable_files(tmp_path: Path) -> None:
    entry = CassetteEntry.create(
        "GET",
        f"{DATA_ORIGIN}/v2/stocks/bars",
        None,
        HttpResponse(200, {"content-type": "application/json"}, b"{}"),
    )
    path = tmp_path / "private" / "cassette.json"
    path.parent.mkdir(mode=0o700)
    path.write_bytes(Cassette((entry,), "synthetic").dumps())
    path.chmod(0o644)
    with pytest.raises(CassetteError, match="owner-only"):
        Cassette.read(path)
