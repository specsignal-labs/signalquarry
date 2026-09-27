# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date

from signalquarry import api
from signalquarry._internal.data.alpaca import AlpacaDataClient, HttpResponse, RateLimiter
from signalquarry.cli.main import main


class Contracts:
    def __init__(self, symbols):
        self.symbols = symbols
        self.calls = []

    def option_contracts(self, underlying, first, last, status="inactive"):
        self.calls.append((underlying, first, last))
        return self.symbols


@dataclass
class Bars:
    counts: dict
    urls: list = field(default_factory=list)

    def get(self, url, headers):
        self.urls.append(url)
        symbol = url.split("symbols=")[1].split("&")[0]
        body = {"bars": {symbol: [{"t": "x"}] * self.counts.get(symbol, 0)}}
        return HttpResponse(200, {}, json.dumps(body).encode())


def _client(counts):
    return AlpacaDataClient(
        "k", "s", transport=Bars(counts), limiter=RateLimiter(per_minute=1000), sleep=lambda _: None
    )


def test_probe_reports_coverage() -> None:
    broker = Contracts(["QQQ240315P00400000", "QQQ240315C00450000"])
    envelope = api.data_probe_options(
        "qqq", "2024-03", broker=broker, client=_client({"QQQ240315P00400000": 12})
    )
    assert envelope.status == "ok" and envelope.data["historical_option_bars"] is True
    assert broker.calls == [("QQQ", date(2024, 3, 1), date(2024, 3, 31))]
    empty = api.data_probe_options("QQQ", "2023-12", broker=Contracts([]), client=_client({}))
    assert empty.data["historical_option_bars"] is False and "OPTIONS_HISTORY_UNAVAILABLE" in empty.warnings


def test_probe_usage_and_missing_credentials(capsys, monkeypatch, tmp_path) -> None:
    assert api.data_probe_options("QQQ", "March").status == "usage"
    monkeypatch.setenv("SIGNALQUARRY_CONFIG_DIR", str(tmp_path))
    for name in (
        "APCA_API_KEY_ID",
        "APCA_API_SECRET_KEY",
        "SIGNALQUARRY_PAPER_KEY_ID",
        "SIGNALQUARRY_PAPER_SECRET_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    code = main(["--json", "data", "probe", "options-coverage", "--month", "2024-03"])
    payload = json.loads(capsys.readouterr().out)
    assert (code, payload["reason_codes"]) == (69, ["DATA_CREDENTIALS_MISSING"])
