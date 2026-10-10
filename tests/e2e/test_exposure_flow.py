# SPDX-License-Identifier: Apache-2.0
"""Optional passive exposures are separate artifacts and never mutate a run or ledger."""

from __future__ import annotations

import asyncio
import json
from datetime import date
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from signalquarry import api
from signalquarry._internal.data.alpaca import AlpacaDataClient, RateLimiter
from signalquarry._internal.engine.backtest import EngineError
from signalquarry._internal.evidence.runs import result_document, result_hash_ok
from signalquarry.api import runs
from signalquarry.api.envelope import Envelope
from signalquarry.cli.main import main
from tests.e2e.test_benchmark_flow import END, _market
from tests.fakes import FakeAlpaca


@pytest.fixture
def lab(tmp_path: Path) -> tuple[Path, str]:
    project = tmp_path / "exposure-lab"
    assert api.init(project, demo=True, package="exp_" + uuid4().hex).status == "ok"
    run = api.backtest("sma-trend", project=project)
    assert run.status == "ok", run
    return project, run.data["run_id"]


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }


def _document(project: Path, envelope: Envelope) -> dict[str, Any]:
    assert envelope.status == "ok", envelope
    document = json.loads((project / envelope.artifacts[0]["path"]).read_text())
    assert result_hash_ok(document)
    return document


def test_opt_in_order_dates_no_run_or_ledger_changes_and_report(lab: tuple[Path, str]) -> None:
    project, run_id = lab
    run_dir = project / ".signalquarry/runs" / run_id
    run_before = _snapshot(run_dir)
    evidence_before = _snapshot(project / "evidence")
    plain = api.diagnose("sma-trend", project=project)
    assert "exposure" not in _document(project, plain) and "exposure" not in plain.data
    assert plain.data["available"] == ["regimes", "costs"]
    assert plain.warnings == []
    diagnosed = api.diagnose("sma-trend", exposures=["synb", "SYNC"], project=project)
    document = _document(project, diagnosed)
    exposure = document["exposure"]
    assert exposure["status"] == "ok"
    assert exposure["references_requested"] == ["SYNB", "SYNC"]
    assert [row["name"] for row in exposure["regression"]["references"]] == ["SYNB", "SYNC"]
    assert exposure["rolling"]["window"] == 126 and exposure["rolling"]["step"] == 21
    rows = exposure["rolling"]["rows"]
    assert rows and all(date.fromisoformat(row["end"]) for row in rows)
    series = runs._series(run_dir, json.loads((run_dir / "result.json").read_text()))
    assert [row["end"] for row in rows] == [day.isoformat() for day in series.sessions[125::21]]
    regression = exposure["regression"]
    assert diagnosed.data["exposure"] == {
        "status": "ok",
        "r_squared": regression["r_squared"],
        "alpha_annual": regression["alpha_annual"],
        "betas": {row["name"]: row["beta"] for row in regression["references"]},
    }
    assert diagnosed.data["available"] == ["regimes", "costs", "exposure"]
    assert diagnosed.evidence == plain.evidence and diagnosed.warnings == []
    assert _snapshot(run_dir) == run_before
    assert _snapshot(project / "evidence") == evidence_before
    assert not list((project / "evidence").glob("*.jsonl"))
    reported = api.report("sma-trend", run_id=run_id, project=project)
    text = (project / reported.artifacts[0]["path"]).read_text()
    assert "## Exposure to reference series" in text
    assert text.index("## Cost sensitivity") < text.index("## Exposure to reference series")
    assert "SYNB" in text and "Rolling beta SYNC:" in text and exposure["note"] in text
    # A later opt-out document also produces no exposure section in the report.
    again = api.diagnose("sma-trend", project=project)
    assert "exposure" not in _document(project, again)
    reported = api.report("sma-trend", project=project)
    assert "## Exposure" not in (project / reported.artifacts[0]["path"]).read_text()


def test_own_symbol_exposure_is_partial_passive_association(lab: tuple[Path, str]) -> None:
    project, _ = lab
    diagnosed = api.diagnose("sma-trend", exposures=["SYNA"], project=project)
    regression = _document(project, diagnosed)["exposure"]["regression"]
    assert 0 < regression["references"][0]["beta"] < 1.2
    assert regression["r_squared"] > 0
    print(f"Own-symbol exposure: beta={regression['references'][0]['beta']}, R²={regression['r_squared']}")


@pytest.mark.parametrize("symbols", [["SYNA", "syna"], [""], ["SYNA "], ["A-B"], list("ABCDEFGHI")])
def test_invalid_symbols_write_nothing(lab: tuple[Path, str], symbols: list[str]) -> None:
    project, _ = lab
    before = _snapshot(project)
    diagnosed = api.diagnose("sma-trend", exposures=symbols, project=project)
    assert diagnosed.status == "usage" and diagnosed.reason_codes == ["USAGE_INVALID"]
    assert diagnosed.artifacts == [] and _snapshot(project) == before


def test_changed_strategy_and_unresolvable_project_are_unavailable(lab: tuple[Path, str]) -> None:
    project, _ = lab
    source = next((project / "src").rglob("strategy.py"))
    original = source.read_text()
    source.write_text(original + "\n# Changed after the run.\n")
    document = _document(project, api.diagnose("sma-trend", exposures=["SYNA"], project=project))
    assert document["exposure"] == {
        "status": "unavailable",
        "reason": "The project no longer reproduces that run.",
    }
    source.unlink()
    diagnosed = api.diagnose("sma-trend", exposures=["SYNA"], project=project)
    assert _document(project, diagnosed)["exposure"]["status"] == "unavailable"
    assert diagnosed.data["exposure"] == {"status": "unavailable"}
    assert "exposure" not in diagnosed.data["available"]


def test_recorded_params_window_and_dataset_identity(lab: tuple[Path, str]) -> None:
    project, _ = lab
    run = api.backtest(
        "sma-trend", params=["period=100"], start=date(2018, 1, 1), end=date(2020, 12, 31), project=project
    )
    diagnosed = api.diagnose("sma-trend", run_id=run.data["run_id"], exposures=["SYNA"], project=project)
    exposure = _document(project, diagnosed)["exposure"]
    assert exposure["status"] == "ok"
    assert exposure["regression"]["observations"] == run.metrics["sessions"]
    path = project / ".signalquarry/runs" / run.data["run_id"] / "result.json"
    document = json.loads(path.read_text())
    fields = {key: value for key, value in document.items() if key not in ("created_at", "result_hash")}
    fields["dataset_identity"] = "sha256:" + "0" * 64
    path.write_text(result_document(**fields))
    diagnosed = api.diagnose("sma-trend", run_id=run.data["run_id"], exposures=["SYNA"], project=project)
    assert _document(project, diagnosed)["exposure"]["status"] == "unavailable"
    fields["params"] = {"period": "invalid"}
    path.write_text(result_document(**fields))
    diagnosed = api.diagnose("sma-trend", run_id=run.data["run_id"], exposures=["SYNA"], project=project)
    assert _document(project, diagnosed)["exposure"]["status"] == "unavailable"


def test_engine_failures_name_every_missing_reference(
    lab: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    project, _ = lab

    def fail(*args: Any, **kwargs: Any) -> None:
        raise EngineError("DATA_INVALID", "Cannot hold the symbol.")

    monkeypatch.setattr(runs, "benchmark_curve", fail)
    diagnosed = api.diagnose("sma-trend", exposures=["SYNB", "SYNC"], project=project)
    exposure = _document(project, diagnosed)["exposure"]
    assert exposure["status"] == "unavailable" and "SYNB, SYNC" in exposure["reason"]
    assert diagnosed.warnings == ["EXPOSURE_DATA_MISSING"]


def test_insufficient_and_collinear_keep_regression_and_empty_rolling(
    lab: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    project, _ = lab
    short = api.backtest("sma-trend", start=date(2019, 1, 1), end=date(2019, 1, 20), project=project)
    diagnosed = api.diagnose("sma-trend", run_id=short.data["run_id"], exposures=["SYNA"], project=project)
    exposure = _document(project, diagnosed)["exposure"]
    assert exposure["status"] == exposure["regression"]["status"] == "insufficient"
    assert exposure["rolling"]["rows"] == [] and diagnosed.data["exposure"] == {"status": "insufficient"}
    original = runs.benchmark_curve
    resolved = runs.resolve("diagnose", "sma-trend", project)
    assert not isinstance(resolved, Envelope)

    def repeated(spec, dataset, symbol, sessions):
        return original(spec, resolved.dataset, "SYNA", sessions)

    monkeypatch.setattr(runs, "benchmark_curve", repeated)
    diagnosed = api.diagnose("sma-trend", run_id=lab[1], exposures=["SYNA", "SYNB"], project=project)
    exposure = _document(project, diagnosed)["exposure"]
    assert exposure["status"] == exposure["regression"]["status"] == "collinear"
    assert exposure["rolling"]["rows"] == [] and diagnosed.data["exposure"] == {"status": "collinear"}
    assert "exposure" not in diagnosed.data["available"]


def test_recorded_missing_symbol_warns_and_other_dataset_can_supply_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SIGNALQUARRY_CACHE_DIR", str(tmp_path / "cache"))
    project = tmp_path / "exposure-recorded"
    assert api.init(project, package="exp_real_" + uuid4().hex).status == "ok"
    client = AlpacaDataClient(
        "k", "s", transport=FakeAlpaca(_market(), page_size=5000), limiter=RateLimiter(sleep=lambda s: None)
    )
    assert api.data_fetch(symbols=("SPY",), project=project, client=client, end=END).status == "ok"
    run = api.backtest("sma-trend", project=project)
    before = _snapshot(project / ".signalquarry/runs")
    diagnosed = api.diagnose("sma-trend", exposures=["QQQ", "TLT"], project=project)
    exposure = _document(project, diagnosed)["exposure"]
    assert exposure == {
        "status": "unavailable",
        "reason": "Passive return data is unavailable for: QQQ, TLT.",
    }
    assert diagnosed.warnings == ["EXPOSURE_DATA_MISSING"]
    assert _snapshot(project / ".signalquarry/runs") == before
    assert api.data_fetch(symbols=("QQQ",), project=project, client=client, end=END).status == "ok"
    diagnosed = api.diagnose("sma-trend", run_id=run.data["run_id"], exposures=["QQQ"], project=project)
    assert _document(project, diagnosed)["exposure"]["status"] == "ok" and diagnosed.warnings == []


def test_cli_exposure_reaches_api(lab: tuple[Path, str], capsys: pytest.CaptureFixture[str]) -> None:
    project, _ = lab
    assert (
        main(
            [
                "--json",
                "diagnose",
                "--strategy",
                "sma-trend",
                "--project",
                str(project),
                "--exposure",
                "SYNB",
                "--exposure",
                "SYNC",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["data"]["exposure"]["status"] == "ok"
    assert list(payload["data"]["exposure"]["betas"]) == ["SYNB", "SYNC"]
    before = _snapshot(project)
    assert (
        main(["--json", "diagnose", "--strategy", "sma-trend", "--project", str(project), "--exposure", ""])
        == 64
    )
    assert json.loads(capsys.readouterr().out)["reason_codes"] == ["USAGE_INVALID"]
    assert _snapshot(project) == before


def test_mcp_optional_exposures_reach_api(lab: tuple[Path, str]) -> None:
    pytest.importorskip("mcp")
    from mcp import Client

    from signalquarry.mcp.server import mcp

    project, run_id = lab

    async def exercise() -> None:
        async with Client(mcp) as client:
            result = await client.call_tool(
                "sqy_diagnose",
                {
                    "strategy_id": "sma-trend",
                    "project": str(project),
                    "run_id": run_id,
                    "exposures": ["SYNB", "SYNC"],
                },
            )
            assert not result.is_error
            assert result.structured_content["data"]["exposure"]["status"] == "ok"
            assert list(result.structured_content["data"]["exposure"]["betas"]) == ["SYNB", "SYNC"]
            result = await client.call_tool(
                "sqy_diagnose", {"strategy_id": "sma-trend", "project": str(project), "run_id": run_id}
            )
            assert "exposure" not in result.structured_content["data"]

    asyncio.run(exercise())
