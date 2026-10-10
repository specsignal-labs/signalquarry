# SPDX-License-Identifier: Apache-2.0
"""Diagnostics describe recorded runs without changing runs or evidence state."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from signalquarry import api
from signalquarry._internal.evidence.runs import result_document, result_hash_ok
from signalquarry._internal.validation.regimes import calendar_regime, regime_table
from signalquarry.api import runs
from signalquarry.api.envelope import Envelope
from signalquarry.cli.main import main


@pytest.fixture
def lab(tmp_path: Path) -> tuple[Path, str]:
    project = tmp_path / "diagnose-lab"
    assert api.init(project, demo=True, package="diag_" + uuid4().hex).status == "ok"
    run = api.backtest("sma-trend", project=project)
    assert run.status == "ok", run
    return project, run.data["run_id"]


def _document(project: Path, envelope: Envelope) -> dict[str, Any]:
    assert envelope.status == "ok", envelope
    document = json.loads((project / envelope.artifacts[0]["path"]).read_text())
    assert result_hash_ok(document)
    return document


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*") if path.is_file()
    }


def _report(project: Path, run_id: str) -> str:
    envelope = api.report("sma-trend", run_id=run_id, project=project)
    assert envelope.status == "ok", envelope
    return (project / envelope.artifacts[0]["path"]).read_text()


def test_diagnose_backtest_cli_and_report(lab: tuple[Path, str], capsys: pytest.CaptureFixture[str]) -> None:
    project, run_id = lab
    run_dir = project / ".signalquarry/runs" / run_id
    before = _snapshot(run_dir)
    evidence_before = _snapshot(project / "evidence")
    baseline = _report(project, run_id)
    assert "## Regimes" not in baseline
    assert main(["--json", "diagnose", "--strategy", "sma-trend", "--project", str(project)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "diagnose" and payload["data"]["run_id"] == run_id
    assert payload["data"]["available"] == ["regimes", "costs"]
    assert payload["next_actions"] == []
    document = json.loads((project / payload["artifacts"][0]["path"]).read_text())
    assert result_hash_ok(document) and document["schema"] == "signalquarry.diagnostics/v1"
    result = json.loads((run_dir / "result.json").read_text())
    assert document["evidence"] == payload["evidence"] == result["evidence"]
    assert document["configuration_hash"] == result["configuration_hash"]
    assert document["dataset_identity"] == result["dataset_identity"]
    regimes = document["regimes"]
    assert regimes["status"] == "ok"
    assert all(regimes[kind]["status"] == "ok" for kind in ("calendar", "trend", "volatility"))
    series = runs._series(run_dir, result)
    assert [row["regime"] for row in regimes["calendar"]["rows"]] == [str(year) for year in range(2014, 2026)]
    assert (
        regimes["calendar"]["rows"][0]["total_return"]
        == regime_table(calendar_regime(series.sessions), series.returns)[0]["total_return"]
    )
    costs = document["costs"]
    assert [point["multiplier"] for point in costs["points"]] == [0, 1, 2, 4]
    for key in ("total_return", "sharpe", "max_drawdown"):
        assert costs["points"][1][key] == result["metrics"][key]
    assert document["folds"]["status"] == document["parameters"]["status"] == "unavailable"
    assert payload["data"]["worst_regime"]["status"] == "ok"
    assert _snapshot(run_dir) == before and _snapshot(project / "evidence") == evidence_before
    assert not list(project.rglob("*trials*.jsonl"))
    text = _report(project, run_id)
    assert "## Regimes" in text and "### Trend" in text and "### Volatility" in text
    assert "## Cost sensitivity" in text
    if costs["break_even"] is not None:
        assert f"Break-even at {costs['break_even']:.2f}× costs" in text
    elif costs["points"][0]["total_return"] <= 0:
        assert "The return is not positive even at zero cost, so there is no break-even." in text
    else:
        assert "No break-even within 4× costs: the return stays positive." in text
    assert "Unavailable diagnostics: folds:" in text and "parameters:" in text
    assert "## Fold consistency" not in text and "## Parameter sensitivity" not in text
    assert text.index("## Backtest") < text.index("## Regimes") < text.index("## Identity")
    again = api.diagnose("sma-trend", run_id=run_id, project=project)
    assert again.data["diagnostics_id"] == run_id + "-2"
    second = _document(project, again)
    assert second["costs"] == costs and _snapshot(run_dir) == before
    # A damaged newest diagnostic is skipped, leaving the prior valid artifact in use.
    path = project / again.artifacts[0]["path"]
    second["costs"]["points"][1]["total_return"] = 9
    path.write_text(json.dumps(second))
    assert _report(project, run_id) == text
    # With both diagnostic hashes invalid the original report is byte-identical.
    first_path = project / payload["artifacts"][0]["path"]
    document["costs"]["break_even"] = 123
    first_path.write_text(json.dumps(document))
    assert _report(project, run_id) == baseline


def test_evaluation_and_sweep_are_matched_by_configuration(lab: tuple[Path, str]) -> None:
    project, run_id = lab
    evaluated = api.evaluate_command("sma-trend", project=project)
    assert any(artifact["kind"] == "evaluation" for artifact in evaluated.artifacts), evaluated
    evaluation = json.loads(
        (project / next(a["path"] for a in evaluated.artifacts if a["kind"] == "evaluation")).read_text()
    )
    swept = api.sweep("sma-trend", ["period=150,200,250"], summary_only=True, project=project)
    assert swept.status == "ok", swept
    # A later sweep that excludes the base configuration must not replace its match.
    assert api.sweep("sma-trend", ["period=100,120"], summary_only=True, project=project).status == "ok"
    before = _snapshot(project / "evidence")
    diagnosed = api.diagnose("sma-trend", run_id=run_id, project=project)
    doc = _document(project, diagnosed)
    assert diagnosed.data["available"] == ["regimes", "costs", "folds", "parameters"]
    assert doc["folds"]["evaluation_id"] == evaluated.data["run_id"]
    assert doc["folds"]["consistency"]["folds"] == len(evaluation["folds"])
    assert doc["folds"]["consistency"]["status"] == "ok"
    assert diagnosed.data["share_positive"] == doc["folds"]["consistency"]["share_positive"]
    assert diagnosed.data["share_ahead_of_benchmark"] is not None
    assert doc["parameters"]["sweep_id"] == swept.data["sweep_id"]
    assert doc["parameters"]["best"] is not None and doc["parameters"]["neighbour_median"] is not None
    assert diagnosed.data["plateau"] == doc["parameters"]["plateau"]
    assert _snapshot(project / "evidence") == before
    text = _report(project, run_id)
    for heading in ("Regimes", "Cost sensitivity", "Fold consistency", "Parameter sensitivity"):
        assert f"## {heading}" in text
    assert "neighbour median" in text and "plateau" in text and "Descriptive only" in text
    # Newest matching artifacts, even when the fold count is insufficient, remain descriptive.
    fields = {key: value for key, value in evaluation.items() if key not in ("created_at", "result_hash")}
    fields["folds"] = fields["folds"][:2]
    target = project / ".signalquarry/runs/newest-evaluation"
    target.mkdir()
    (target / "evaluation.json").write_text(result_document(**fields))
    doc = _document(project, api.diagnose("sma-trend", run_id=run_id, project=project))
    assert doc["folds"]["evaluation_id"] == "newest-evaluation"
    assert doc["folds"]["status"] == "ok" and doc["folds"]["consistency"]["status"] == "insufficient"


def test_missing_wrong_strategy_and_tampered_runs(lab: tuple[Path, str], tmp_path: Path) -> None:
    project, run_id = lab
    assert api.diagnose("sma-trend", project=tmp_path).reason_codes == ["PROJECT_NOT_FOUND"]
    assert api.diagnose("absent", project=project).reason_codes == ["REPORT_NO_RUNS"]
    for missing in ("unknown", "../" + run_id):
        assert api.diagnose("sma-trend", run_id=missing, project=project).reason_codes == ["RUN_NOT_FOUND"]
    # A valid run of another strategy is refused before any project resolution.
    wrong = api.diagnose("another-strategy", run_id=run_id, project=project)
    assert wrong.reason_codes == ["RUN_ARTIFACT_INVALID"]
    path = project / ".signalquarry/runs" / run_id / "result.json"
    doc = json.loads(path.read_text())
    doc["metrics"]["total_return"] = 99
    path.write_text(json.dumps(doc))
    for selection in (run_id, None):
        assert api.diagnose("sma-trend", run_id=selection, project=project).reason_codes == [
            "RUN_ARTIFACT_INVALID"
        ]
    assert not (project / ".signalquarry/diagnostics").exists()


def test_changed_strategy_makes_costs_unavailable(lab: tuple[Path, str]) -> None:
    project, run_id = lab
    source = next((project / "src").rglob("strategy.py"))
    source.write_text(source.read_text() + "\n# Changed since the recorded run.\n")
    doc = _document(project, api.diagnose("sma-trend", run_id=run_id, project=project))
    assert doc["costs"] == {"status": "unavailable", "reason": "The project no longer reproduces that run."}
    assert "## Cost sensitivity" not in _report(project, run_id)
    assert "costs: The project no longer reproduces that run." in _report(project, run_id)


def test_recorded_variant_window_is_reproduced(lab: tuple[Path, str]) -> None:
    project, _ = lab
    run = api.backtest(
        "sma-trend", params=["period=100"], start=date(2018, 1, 1), end=date(2020, 12, 31), project=project
    )
    doc = _document(project, api.diagnose("sma-trend", project=project))
    assert doc["run_id"] == run.data["run_id"]
    assert doc["costs"]["status"] == "ok"
    assert doc["costs"]["points"][1]["total_return"] == run.metrics["total_return"]
    assert [row["regime"] for row in doc["regimes"]["calendar"]["rows"]] == ["2018", "2019", "2020"]


def test_no_benchmark_and_missing_equity(lab: tuple[Path, str]) -> None:
    project, run_id = lab
    directory = project / ".signalquarry/runs" / run_id
    (directory / "benchmark.csv").unlink()
    doc = _document(project, api.diagnose("sma-trend", project=project))
    assert doc["regimes"]["calendar"]["status"] == "ok"
    assert doc["regimes"]["trend"]["status"] == doc["regimes"]["volatility"]["status"] == "unavailable"
    text = _report(project, run_id)
    assert "### Calendar" in text and "### Trend" not in text
    assert "trend: The run has no benchmark.csv." in text
    (directory / "equity.csv").unlink()
    assert api.diagnose("sma-trend", project=project).reason_codes == ["RUN_ARTIFACT_INVALID"]


def test_bad_benchmark_is_refused(lab: tuple[Path, str]) -> None:
    project, run_id = lab
    benchmark = project / ".signalquarry/runs" / run_id / "benchmark.csv"
    content = benchmark.read_text()
    benchmark.write_text("\n".join(content.splitlines()[:-1]) + "\n")
    assert api.diagnose("sma-trend", project=project).reason_codes == ["RUN_ARTIFACT_INVALID"]
    benchmark.write_text("invalid")
    assert api.diagnose("sma-trend", project=project).reason_codes == ["RUN_ARTIFACT_INVALID"]


def test_cost_replay_failure_and_changed_dataset(
    lab: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    project, run_id = lab
    original = runs.summarize

    def changed(*args: Any, **kwargs: Any) -> dict[str, Any]:
        metrics = original(*args, **kwargs)
        metrics["total_return"] += 0.01
        return metrics

    with monkeypatch.context() as patch:
        patch.setattr(runs, "summarize", changed)
        doc = _document(project, api.diagnose("sma-trend", run_id=run_id, project=project))
        assert doc["costs"]["status"] == "unavailable" and "multiplier-1" in doc["costs"]["reason"]
        assert "points" not in doc["costs"]
    path = project / ".signalquarry/runs" / run_id / "result.json"
    doc = json.loads(path.read_text())
    fields = {key: value for key, value in doc.items() if key not in ("created_at", "result_hash")}
    fields["dataset_identity"] = "sha256:" + "0" * 64
    path.write_text(result_document(**fields))
    diagnosed = _document(project, api.diagnose("sma-trend", run_id=run_id, project=project))
    assert diagnosed["costs"]["status"] == "unavailable"


def test_options_costs_unavailable(tmp_path: Path) -> None:
    project = tmp_path / "diagnose-options"
    assert api.init(project, demo=True, kind="options", package="diag_opt_" + uuid4().hex).status == "ok"
    run = api.backtest("wheel", project=project)
    assert run.status == "ok", run
    doc = _document(project, api.diagnose("wheel", project=project))
    assert doc["costs"] == {
        "status": "unavailable",
        "reason": "Options strategies use a different cost model.",
    }
