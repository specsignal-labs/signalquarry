# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import gzip
import json
import os
from datetime import date
from pathlib import Path

import pytest

from signalquarry._internal.data.alpaca import AlpacaDataClient, ProviderError, RateLimiter
from signalquarry._internal.data.credentials import load_data_credentials
from signalquarry._internal.data.library import build_dataset
from signalquarry._internal.data.synthetic import synthetic_dataset
from signalquarry._internal.engine.backtest import run_backtest
from signalquarry.api import backtest, data_fetch, data_ls, data_verify, init
from signalquarry.sdk import definition_of
from tests.fakes import FakeAlpaca
from tests.helpers import spec
from tests.unit.test_engine import SmaP, sma_trend

START, END = date(2019, 1, 2), date(2021, 12, 31)


def _client(fake: FakeAlpaca) -> AlpacaDataClient:
    return AlpacaDataClient(
        "key", "secret", transport=fake, limiter=RateLimiter(sleep=lambda s: None), sleep=lambda s: None
    )


def test_round_trip_through_alpaca_format_preserves_engine_results() -> None:
    source = synthetic_dataset(START, END, symbols=("SYNA", "SYNB"))
    fake = FakeAlpaca(source, page_size=250)
    client = _client(fake)
    bar_pages = client.daily_bars(("SYNA", "SYNB"), START, END, "sip")
    action_pages = client.corporate_actions(("SYNA", "SYNB"), START, END)
    assert len(bar_pages) > 2  # paginated
    rebuilt = build_dataset(bar_pages, action_pages, source="alpaca:sip")
    assert rebuilt.sessions == source.sessions
    assert (rebuilt.splits, rebuilt.dividends) == (source.splits, source.dividends)
    original = run_backtest(spec(("SYNA",)), definition_of(sma_trend), SmaP(), source)
    replayed = run_backtest(spec(("SYNA",)), definition_of(sma_trend), SmaP(), rebuilt)
    assert original.decisions == replayed.decisions and original.fills == replayed.fills


def test_retries_throttling_and_rejects_bad_credentials() -> None:
    source = synthetic_dataset(START, date(2019, 3, 1), symbols=("SYNA",))
    assert _client(FakeAlpaca(source, fail_first=429)).daily_bars(("SYNA",), START, END, "sip")
    with pytest.raises(ProviderError) as exc:
        _client(FakeAlpaca(source, fail_first=401)).daily_bars(("SYNA",), START, END, "sip")
    assert exc.value.code == "DATA_CREDENTIALS_REJECTED"
    with pytest.raises(ProviderError) as loop:
        _client(FakeAlpaca(source, page_size=5, loop_tokens=True)).daily_bars(("SYNA",), START, END, "sip")
    assert loop.value.code == "PROVIDER_PAGINATION_LOOP"


def test_fetch_manifest_verify_and_historical_backtest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SIGNALQUARRY_CACHE_DIR", str(tmp_path / "cache"))
    project = tmp_path / "real-lab"
    assert init(project, package="real_lab_data").status == "ok"
    fake = FakeAlpaca(synthetic_dataset(date(2016, 1, 4), date(2024, 12, 31), symbols=("SPY",)))
    fetched = data_fetch(
        strategy_id="sma-trend", project=project, client=_client(fake), end=date(2024, 12, 31)
    )
    assert fetched.status == "ok", fetched.as_dict()
    manifest_path = project / fetched.data["manifest"]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["redistributable"] is False and "open" not in json.dumps(manifest["coverage"])
    assert data_ls(project=project).data["datasets"][0]["dataset_id"] == fetched.data["dataset_id"]
    assert data_verify(project=project).status == "ok"

    result = backtest("sma-trend", project=project)
    assert result.status == "ok", result.as_dict()
    assert result.evidence["grade"] == "historical" and result.evidence["claim_level"] == "in_sample"
    assert result.data["dataset_id"] == fetched.data["dataset_id"]

    page = next((tmp_path / "cache" / "pages").glob("*.json.gz"))
    page.write_bytes(gzip.compress(b'{"bars": {}}'))
    verify = data_verify(project=project)
    assert verify.status == "blocked" and verify.reason_codes == ["DATA_PAGE_CORRUPT"]
    assert backtest("sma-trend", project=project).reason_codes == ["DATA_PAGE_CORRUPT"]


def test_backtest_asks_for_a_fetch_when_no_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIGNALQUARRY_CACHE_DIR", str(tmp_path / "cache"))
    project = tmp_path / "empty-lab"
    init(project, package="empty_lab_data")
    result = backtest("sma-trend", project=project)
    assert (
        result.status == "unavailable"
        and result.next_actions[0]["command"] == "sqy data fetch --strategy sma-trend"
    )


def test_credentials_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIGNALQUARRY_CONFIG_DIR", str(tmp_path))
    assert load_data_credentials({"APCA_API_KEY_ID": "k", "APCA_API_SECRET_KEY": "s"}).source == "environment"
    assert load_data_credentials({}) is None
    path = tmp_path / "credentials.toml"
    path.write_text('[data]\nkey_id = "k"\nsecret_key = "s"\n')
    os.chmod(path, 0o644)
    with pytest.raises(PermissionError):
        load_data_credentials({})
    os.chmod(path, 0o600)
    assert load_data_credentials({}).source == str(path)
