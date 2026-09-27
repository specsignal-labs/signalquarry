# SPDX-License-Identifier: Apache-2.0
"""`sqy data record options`: raw chain pages cached, a hash-only record committed, verify re-hashes."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

from signalquarry import api
from signalquarry._internal.data.alpaca import AlpacaDataClient, HttpResponse, RateLimiter

NOW = datetime(2026, 9, 25, 15, 0, tzinfo=UTC)


@dataclass
class Venue:
    chain: bool = True
    urls: list = field(default_factory=list)

    def get(self, url, headers):
        self.urls.append(url)
        if "/v2/stocks/" in url:
            body = {"quote": {"bp": 100.0, "ap": 100.2, "t": "2026-09-25T14:59:59Z"}}
        else:
            right = "P" if "type=put" in url else "C"
            symbol = f"QQQ261016{right}00100000"
            quote = {"bp": 1.2, "ap": 1.3, "t": "2026-09-25T14:59:58Z"}
            body = {"snapshots": {symbol: {"latestQuote": quote}} if self.chain else {}}
        return HttpResponse(200, {}, json.dumps(body).encode())


def _client(venue: Venue) -> AlpacaDataClient:
    return AlpacaDataClient(
        "k", "s", transport=venue, limiter=RateLimiter(per_minute=1000), sleep=lambda _: None
    )


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SIGNALQUARRY_CACHE_DIR", str(tmp_path / "cache"))
    root = tmp_path / "rec"
    assert api.init(root, demo=True, package="rec_lab").status == "ok"
    return root


def test_record_verify_and_tamper(project: Path, tmp_path: Path) -> None:
    venue = Venue()
    envelope = api.data_record_options("qqq", project=project, client=_client(venue), now=NOW)
    assert envelope.status == "ok" and envelope.data["contracts"] == 2 and envelope.data["pages"] == 2
    record = json.loads((project / envelope.data["path"]).read_text())
    assert record["schema"] == "signalquarry.options-record/v1" and record["redistributable"] is False
    assert "1.2" not in json.dumps(record)  # prices stay in the cached pages, not the record
    assert any("strike_price_gte=80.08" in url for url in venue.urls)  # 20% below the mid of 100.1
    assert api.data_verify(project=project).status == "ok"
    listed = api.data_ls(project=project)
    assert listed.data["option_chains"]["QQQ"]["records"] == 1 and "QQQ" in listed.summary
    page = tmp_path / "cache" / "pages" / f"{record['pages'][0]['sha256']}.json.gz"
    page.write_bytes(gzip.compress(b'{"snapshots": {}}'))
    verified = api.data_verify(project=project)
    assert verified.status == "blocked" and any(not r["ok"] for r in verified.data["datasets"])


def test_empty_chain_usage_and_credentials(
    project: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    empty = api.data_record_options("QQQ", project=project, client=_client(Venue(chain=False)), now=NOW)
    assert empty.status == "ok" and "OPTIONS_CHAIN_EMPTY" in empty.warnings
    assert api.data_record_options("QQQ1", project=project).status == "usage"
    assert api.data_record_options("QQQ", width=2, project=project).status == "usage"
    monkeypatch.setenv("SIGNALQUARRY_CONFIG_DIR", str(tmp_path / "none"))
    for name in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    missing = api.data_record_options("QQQ", project=project)
    assert (missing.status, missing.reason_codes) == ("unavailable", ["DATA_CREDENTIALS_MISSING"])


def test_cli_record_needs_credentials(
    project: Path, capsys, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from signalquarry.cli.main import main

    monkeypatch.setenv("SIGNALQUARRY_CONFIG_DIR", str(tmp_path / "none"))
    for name in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    code = main(["--json", "data", "record", "options", "--underlying", "QQQ", "--project", str(project)])
    assert code == 69 and json.loads(capsys.readouterr().out)["reason_codes"] == ["DATA_CREDENTIALS_MISSING"]


def test_recorded_chains_load_by_new_york_session(project: Path) -> None:
    from datetime import date
    from decimal import Decimal

    from signalquarry.api.data import load_recorded_chains
    from signalquarry.api.resolve import resolve

    api.data_record_options("QQQ", project=project, client=_client(Venue()), now=NOW)
    chains = load_recorded_chains(project)
    assert list(chains) == [("QQQ", date(2026, 9, 25))]
    assert chains[("QQQ", date(2026, 9, 25))]["QQQ261016P00100000"] == (Decimal("1.2"), Decimal("1.3"))
    resolved = resolve("backtest", "sma-trend", project)
    assert not isinstance(resolved, api.Envelope) and resolved.recorded_chains() is None  # synthetic, equity
