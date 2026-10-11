# SPDX-License-Identifier: Apache-2.0
"""Data-quality commands against temporary synthetic and recorded projects."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from signalquarry._internal.calendar.nyse import is_holiday
from signalquarry._internal.data.alpaca import AlpacaDataClient, RateLimiter
from signalquarry._internal.data.library import dataset_from_manifest
from signalquarry._internal.data.quality import assess
from signalquarry.api import data_fetch, data_quality, explain, init, spec_freeze
from signalquarry.api.data import library_for
from signalquarry.api.resolve import Resolved, resolve
from signalquarry.cli.main import main
from tests.fakes import FakeAlpaca
from tests.helpers import dataset, weekdays


def _assert_artifact(project: Path, payload: dict[str, Any], expected: dict[str, Any]) -> bytes:
    artifact = payload["artifacts"][0]
    content = (project / artifact["path"]).read_bytes()
    assert artifact["kind"] == "data_quality"
    assert artifact["sha256"] == hashlib.sha256(content).hexdigest()
    assert content == (json.dumps(expected, indent=2, sort_keys=True) + "\n").encode()

    def check_keys(value: Any) -> None:
        if isinstance(value, dict):
            assert not {"open", "high", "low", "close", "volume", "return", "returns"} & value.keys()
            for child in value.values():
                check_keys(child)
        elif isinstance(value, list):
            for child in value:
                check_keys(child)

    check_keys(json.loads(content))
    assert set(payload["data"]) == {
        "dataset_id",
        "dataset_identity",
        "ok",
        "findings",
        "sessions",
        "common_window",
        "symbols",
        *({"sufficiency"} if "sufficiency" in payload["data"] else set()),
    }
    for row in payload["data"]["symbols"].values():
        assert set(row) == {"present", "first", "last", "coverage", "longest_gap", "findings"}
    return content


def test_demo_quality_and_hand_counted_sufficiency(tmp_path: Path) -> None:
    project = tmp_path / "demo"
    assert init(project, demo=True, package="quality_demo_history").status == "ok"
    result = data_quality(strategy_id="sma-trend", project=project)
    assert result.status == "ok" and result.command == "data quality"
    assert result.data["ok"] is True and result.data["findings"] == []
    assert result.warnings == []
    assert result.summary == "synthetic: 1 symbols, no findings"
    assert result.data["sessions"]["calendar"] == "not_checked"
    assert result.artifacts[0]["path"] == "data/quality/synthetic.json"
    # 2014-01-02 through 2024-12-31 is 4016 days. Burn-in ends
    # 2017-01-02; full six-month folds end July 1 / January 1.
    # The 15th ends 2024-07-01; the 16th ends after pre-holdout data.
    assert result.data["sufficiency"] == {
        "years": round(4016 / 365.25, 2),
        "g1_years_ok": True,
        "walk_forward_folds": 15,
        "g2_folds_ok": True,
        "holdout_start": "2025-01-01",
        "warm_up_sessions": 200,
    }
    resolved = resolve("data quality", "sma-trend", project)
    assert isinstance(resolved, Resolved)
    expected = assess(resolved.dataset, symbols=("SYNA",), check_calendar=False)
    first = _assert_artifact(project, result.as_dict(), expected)
    second = data_quality(strategy_id="sma-trend", project=project)
    assert _assert_artifact(project, second.as_dict(), expected) == first


@pytest.fixture
def recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> tuple[Path, str]:
    monkeypatch.setenv("SIGNALQUARRY_CACHE_DIR", str(tmp_path / "cache"))
    package = "quality_" + re.sub(r"\W", "_", request.node.name)
    project = tmp_path / "recorded"
    assert init(project, package=package).status == "ok"
    spec_path = project / f"src/{package}/sma_trend/strategy.yaml"
    spec_path.write_text(
        spec_path.read_text().replace("benchmark: SPY", "benchmark: QQQ").replace("months: 12", "months: 0")
    )
    sessions = tuple(day for day in weekdays(date(2024, 2, 5), 20) if not is_holiday(day))
    count = len(sessions)
    opens = list(range(100, 100 + count))
    closes = opens.copy()
    closes[8:13] = [108] * 5
    source = dataset(
        sessions,
        {
            "SPY": {
                "open": opens,
                "close": closes,
                "present": [True] * 3 + [False] * 4 + [True] * (count - 7),
            },
            "QQQ": {"open": opens, "present": [False] * 10 + [True] * (count - 10)},
            "IWM": {"open": opens},
        },
    )
    client = AlpacaDataClient(
        "k", "s", transport=FakeAlpaca(source), limiter=RateLimiter(sleep=lambda _: None)
    )
    fetched = data_fetch(
        strategy_id="sma-trend",
        symbols=("IWM",),
        start=sessions[0],
        end=sessions[-1],
        project=project,
        client=client,
    )
    assert fetched.status == "ok"
    return project, fetched.data["dataset_id"]


@pytest.mark.parametrize("selection", ["--dataset-id", "--strategy"])
def test_recorded_quality_cli(
    recorded: tuple[Path, str], selection: str, capsys: pytest.CaptureFixture[str]
) -> None:
    project, dataset_id = recorded
    value = dataset_id if selection == "--dataset-id" else "sma-trend"
    args = ["data", "quality", selection, value, "--project", str(project), "--json"]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["command"] == "data quality" and result["status"] == "ok"
    assert result["reason_codes"] == []
    assert result["warnings"] == ["DATA_QUALITY_FINDINGS"]
    assert result["data"]["ok"] is False
    assert result["data"]["findings"] == ["QUALITY_GAP", "QUALITY_STALE_CLOSE"]
    assert result["data"]["sessions"]["calendar"] == "nyse"
    selected = ("IWM", "QQQ", "SPY") if selection == "--dataset-id" else ("QQQ", "SPY")
    assert (
        result["summary"]
        == f"{dataset_id}: {len(selected)} symbols, 2 findings (QUALITY_GAP, QUALITY_STALE_CLOSE)"
    )
    assert tuple(result["data"]["symbols"]) == selected
    assert result["data"]["common_window"]["first"] == "2024-02-20"
    if selection == "--strategy":
        assert result["data"]["sufficiency"] == {
            "years": round(25 / 365.25, 2),
            "g1_years_ok": False,
            "walk_forward_folds": 0,
            "g2_folds_ok": False,
            "holdout_start": None,
            "warm_up_sessions": 200,
        }
    else:
        assert "sufficiency" not in result["data"]
    library = library_for(project)
    source = dataset_from_manifest(library, library.manifests()[0])
    expected = assess(source, symbols=selected)
    first = _assert_artifact(project, result, expected)
    assert main(args) == 0
    assert _assert_artifact(project, json.loads(capsys.readouterr().out), expected) == first


def test_duplicate_and_corrupt_manifests(recorded: tuple[Path, str]) -> None:
    project, dataset_id = recorded
    path = project / "data" / "manifests" / f"{dataset_id}.json"
    duplicate = path.with_name("duplicate.json")
    duplicate.write_bytes(path.read_bytes())
    result = data_quality(dataset_id=dataset_id, project=project)
    assert result.status == "invalid" and result.reason_codes == ["DATA_MANIFEST_INVALID"]
    duplicate.unlink()
    manifest = json.loads(path.read_text())
    manifest["dataset_identity"] = "incorrect"
    path.write_text(json.dumps(manifest))
    result = data_quality(dataset_id=dataset_id, project=project)
    assert result.status == "invalid" and result.reason_codes == ["DATA_MANIFEST_INVALID"]
    assert result.artifacts == []


def test_empty_common_window_and_absent_benchmark(recorded: tuple[Path, str]) -> None:
    project, _ = recorded
    spec_path = next((project / "src").glob("*/sma_trend/strategy.yaml"))
    spec_path.write_text(
        spec_path.read_text()
        .replace("symbols: [SPY]", "symbols: [SPY, IWM]")
        .replace("benchmark: QQQ", "benchmark: ABC")
    )
    sessions = weekdays(date(2024, 3, 4), 6)
    prices = list(range(100, 106))
    source = dataset(
        sessions,
        {
            "SPY": {"open": prices, "present": [True] * 3 + [False] * 3},
            "IWM": {"open": prices, "present": [False] * 3 + [True] * 3},
        },
    )
    client = AlpacaDataClient(
        "k", "s", transport=FakeAlpaca(source), limiter=RateLimiter(sleep=lambda _: None)
    )
    assert (
        data_fetch(
            strategy_id="sma-trend", start=sessions[0], end=sessions[-1], project=project, client=client
        ).status
        == "ok"
    )
    result = data_quality(strategy_id="sma-trend", project=project)
    assert result.status == "ok"
    assert set(result.data["symbols"]) == {"SPY", "IWM"}
    assert result.data["common_window"] == {"first": None, "last": None, "sessions": 0}
    assert result.data["sufficiency"] == {
        "years": 0.0,
        "g1_years_ok": False,
        "walk_forward_folds": 0,
        "g2_folds_ok": False,
        "holdout_start": None,
        "warm_up_sessions": 200,
    }


def test_sealed_family_and_no_holdout(tmp_path: Path) -> None:
    project = tmp_path / "sealed"
    package = "quality_sealed_family"
    assert init(project, demo=True, package=package).status == "ok"
    spec_path = project / f"src/{package}/sma_trend/strategy.yaml"
    assert spec_freeze("sma-trend", project=project).status == "ok"
    spec_path.write_text(spec_path.read_text().replace("months: 12", "months: 24"))
    result = data_quality(strategy_id="sma-trend", project=project)
    assert result.data["sufficiency"]["holdout_start"] == "2025-01-01"
    assert result.data["sufficiency"]["years"] == round(4016 / 365.25, 2)
    assert result.data["sufficiency"]["walk_forward_folds"] == 15

    unsealed = tmp_path / "unsealed"
    assert init(unsealed, demo=True, package="quality_without_holdout").status == "ok"
    spec_path = unsealed / "src/quality_without_holdout/sma_trend/strategy.yaml"
    spec_path.write_text(spec_path.read_text().replace("months: 12", "months: 0"))
    result = data_quality(strategy_id="sma-trend", project=unsealed)
    assert result.data["sufficiency"]["holdout_start"] is None
    assert result.data["sufficiency"]["years"] == round(
        (date(2025, 12, 31) - date(2014, 1, 2)).days / 365.25, 2
    )
    assert result.data["sufficiency"]["walk_forward_folds"] == 17


def test_options_have_no_holdout(tmp_path: Path) -> None:
    project = tmp_path / "options"
    assert init(project, demo=True, kind="options", package="quality_options_history").status == "ok"
    result = data_quality(strategy_id="wheel", project=project)
    assert result.status == "ok"
    assert result.data["sufficiency"]["holdout_start"] is None
    assert result.data["sufficiency"]["warm_up_sessions"] == 50


def test_usage_and_not_found(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    project = tmp_path / "errors"
    assert init(project, demo=True, package="quality_usage_errors").status == "ok"
    for selection in ({}, {"strategy_id": "sma-trend", "dataset_id": "absent"}):
        result = data_quality(project=project, **selection)
        assert result.status == "usage" and result.reason_codes == ["USAGE_INVALID"]
    result = data_quality(dataset_id="absent", project=project)
    assert result.status == "invalid" and result.reason_codes == ["DATA_MANIFEST_INVALID"]
    result = data_quality(strategy_id="absent", project=project)
    expected = resolve("data quality", "absent", project)
    assert result.status == expected.status == "invalid"
    assert result.reason_codes == expected.reason_codes == ["STRATEGY_NOT_FOUND"]
    assert result.summary == expected.summary and result.data == expected.data
    for selection_args in ([], ["--strategy", "sma-trend", "--dataset-id", "absent"]):
        assert main(["data", "quality", *selection_args, "--project", str(project), "--json"]) == 64
        assert json.loads(capsys.readouterr().out)["reason_codes"] == ["USAGE_INVALID"]


def test_unfetched_strategy_returns_resolution_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SIGNALQUARRY_CACHE_DIR", str(tmp_path / "cache"))
    project = tmp_path / "unfetched"
    assert init(project, package="quality_unfetched_strategy").status == "ok"
    result = data_quality(strategy_id="sma-trend", project=project)
    expected = resolve("data quality", "sma-trend", project)
    assert result.status == expected.status == "unavailable"
    assert result.reason_codes == expected.reason_codes == ["PROVIDER_UNAVAILABLE"]
    assert result.next_actions == expected.next_actions


def test_quality_codes_can_be_explained() -> None:
    codes = (
        "DATA_QUALITY_FINDINGS",
        "QUALITY_SESSION_ON_WEEKEND",
        "QUALITY_SESSION_ON_HOLIDAY",
        "QUALITY_TRADING_DAY_MISSING",
        "QUALITY_SYMBOL_EMPTY",
        "QUALITY_GAP",
        "QUALITY_OHLC_INCONSISTENT",
        "QUALITY_PRICE_NONPOSITIVE",
        "QUALITY_ZERO_VOLUME",
        "QUALITY_STALE_CLOSE",
        "QUALITY_UNEXPLAINED_MOVE",
    )
    for code in codes:
        result = explain(code)
        assert result.status == "ok"
