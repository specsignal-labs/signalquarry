# SPDX-License-Identifier: Apache-2.0
"""Optional positions artifacts and verified, descriptive attribution through every facade."""

import asyncio
import csv
import io
import json
from datetime import date
from decimal import Decimal as D
from uuid import uuid4

import pytest

from signalquarry import api, research
from signalquarry._internal.canonical import file_sha256
from signalquarry._internal.evidence.runs import result_document, result_hash_ok
from signalquarry.api import runs
from signalquarry.api.resolve import resolve
from signalquarry.cli.main import main


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "positions-demo"
    assert api.init(root, demo=True, package="pos_" + uuid4().hex).status == "ok"
    return root


def _run(project, positions=False):
    result = api.backtest("sma-trend", project=project, positions=positions)
    assert result.status == "ok", result
    path = project / ".signalquarry/runs" / result.data["run_id"]
    return result, path, json.loads((path / "result.json").read_text())


def _diagnose(project, run_id):
    result = api.diagnose("sma-trend", project=project, run_id=run_id)
    assert result.status == "ok", result
    doc = json.loads((project / result.artifacts[0]["path"]).read_text())
    assert result_hash_ok(doc)
    return result, doc


def _snapshot(project):
    return {
        str(path.relative_to(project)): path.read_bytes()
        for path in project.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and ".signalquarry" not in path.parts
    }


def test_positions_backtest_attribution_report_research_and_unchanged_default(project):
    plain, plain_path, plain_doc = _run(project)
    snapshot = _snapshot(project)
    marked, path, doc = _run(project, True)
    assert _snapshot(project) == snapshot
    assert marked.data["ledger_hash"] == plain.data["ledger_hash"]
    assert marked.metrics == plain.metrics
    assert plain_doc.get("positions") is None and not (plain_path / "positions.csv").exists()
    assert research.load_run(plain.data["run_id"], project).positions is None
    assert doc["positions"] is True and result_hash_ok(doc)
    artifact = next(item for item in marked.artifacts if item["kind"] == "positions")
    assert file_sha256(project / artifact["path"]) == doc["positions_accounting"]["positions_sha256"]
    for name in ("equity.csv", "fills.csv", "decisions.jsonl", "benchmark.csv"):
        assert (path / name).read_bytes() == (plain_path / name).read_bytes()
    loaded = research.load_run(marked.data["run_id"], project)
    table = loaded.positions
    assert table is not None and table.num_rows > 0
    assert table.column_names == ["session", "symbol", "quantity", "mark", "value"]
    first = next(csv.DictReader(io.StringIO((path / "positions.csv").read_text())))
    assert table.to_pylist()[0]["quantity"] == float(first["quantity"])
    diagnosed, diagnostics = _diagnose(project, marked.data["run_id"])
    assert "attribution" in diagnosed.data["available"]
    block = diagnostics["attribution"]
    assert block["status"] == "ok"
    parts = block["reconciliation"]
    total = D(doc["metrics"]["final_equity"]) - D(doc["spec"]["account"]["initial_cash"])
    assert D(parts["total"]) == total == D(parts["symbols"]) + D(parts["fees"]) + D(parts["remainder"])
    for key in ("total", "symbols", "fees", "remainder"):
        assert D(parts[key]) == sum((D(row[key]) for row in block["by_year"]), D(0))
    plain_diagnosed, plain_diagnostics = _diagnose(project, plain.data["run_id"])
    assert "attribution" not in plain_diagnostics and "attribution" not in plain_diagnosed.data["available"]
    assert plain_diagnosed.next_actions[0]["command"] == "sqy backtest --strategy sma-trend --positions"
    report = api.report("sma-trend", run_id=marked.data["run_id"], project=project)
    assert report.status == "ok"
    text = (project / report.artifacts[0]["path"]).read_text()
    assert "## Attribution" in text and "SYNA" in text and "| Year | Symbol P&L |" in text
    assert "Descriptive and in-sample" in text and "Herfindahl" in text and "Dividends" in text
    assert text.index("## Cost sensitivity") < text.index("## Attribution") < text.index("## Identity")
    _, _, repeated = _run(project)
    for result_doc in (plain_doc, repeated):
        for key in ("run_id", "created_at", "result_hash"):
            result_doc.pop(key)
    assert plain_doc == repeated  # off remains byte-compatible after omitting existing varying fields


@pytest.mark.parametrize("filename", ["positions.csv", "equity.csv", "fills.csv"])
def test_positions_diagnose_detects_tampering(project, filename):
    run, path, _ = _run(project, True)
    target = path / filename
    target.write_text(target.read_text() + "\n")  # even a sum-preserving edit is detected by the binding hash
    result, doc = _diagnose(project, run.data["run_id"])
    assert "attribution" not in result.data["available"]
    assert doc["attribution"]["status"] == "unavailable"
    assert filename in doc["attribution"]["reason"]
    if filename == "positions.csv":
        with pytest.raises(research.ResearchError, match="positions.csv"):
            _ = research.load_run(run.data["run_id"], project).positions
    report = api.report("sma-trend", run_id=run.data["run_id"], project=project)
    text = (project / report.artifacts[0]["path"]).read_text()
    assert "## Attribution" not in text and "attribution:" in text


def test_positions_diagnose_detects_value_mismatch_even_with_updated_file_hash(project):
    run, path, document = _run(project, True)
    target = path / "positions.csv"
    rows = list(csv.DictReader(io.StringIO(target.read_text())))
    rows[0]["quantity"] = str(D(rows[0]["quantity"]) + 1)
    rows[0]["value"] = str(D(rows[0]["quantity"]) * D(rows[0]["mark"]))
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["session", "symbol", "quantity", "mark", "value"], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    document["positions_accounting"]["positions_sha256"] = file_sha256(target)
    document.pop("result_hash")
    (path / "result.json").write_text(result_document(**document))
    _, diagnostics = _diagnose(project, run.data["run_id"])
    assert diagnostics["attribution"]["status"] == "unavailable"
    assert "reconcile to equity" in diagnostics["attribution"]["reason"]


def test_positions_cli_and_mcp_argument_reach_api(project, capsys):
    assert (
        main(["--json", "backtest", "--strategy", "sma-trend", "--positions", "--project", str(project)]) == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert any(item["kind"] == "positions" for item in payload["artifacts"])
    pytest.importorskip("mcp")
    from mcp import Client

    from signalquarry.mcp.server import mcp

    async def exercise():
        async with Client(mcp) as client:
            tools = {item.name: item for item in (await client.list_tools()).tools}
            assert tools["sqy_backtest"].input_schema["properties"]["positions"]["default"] is False
            result = await client.call_tool(
                "sqy_backtest", {"strategy_id": "sma-trend", "project": str(project), "positions": True}
            )
            data = result.structured_content
            assert data["status"] == "ok", data
            assert any(item["kind"] == "positions" for item in data["artifacts"])
            assert data["data"]["ledger_hash"] == payload["data"]["ledger_hash"]

    asyncio.run(exercise())


def test_positions_summary_retains_fills_and_options_refuse_without_artifacts(project, tmp_path):
    resolved = resolve("backtest", "sma-trend", project)
    outcome = runs.execute_run(resolved, command="backtest", detail="summary", positions=True)
    kinds = {item["kind"] for item in outcome.artifacts}
    assert {"positions", "fills"} <= kinds and "decisions" not in kinds
    _, doc = _diagnose(project, outcome.run_id)
    assert doc["attribution"]["status"] == "ok"
    option_project = tmp_path / "options"
    assert api.init(option_project, kind="options", demo=True, package="posopt_" + uuid4().hex).status == "ok"
    outcome = api.backtest("wheel", positions=True, project=option_project)
    assert outcome.status != "ok" and outcome.reason_codes == ["STRATEGY_KIND_MISMATCH"]
    assert not list((option_project / ".signalquarry/runs").glob("*/result.json"))


def test_positions_sweep_and_study_keep_default_artifacts(project):
    swept = api.sweep("sma-trend", ["period=100,150"], project=project)
    assert swept.status == "ok", swept
    assert (
        api.study_init(
            "sma-trend", "positions-default", metric="max_drawdown", direction="lower", project=project
        ).status
        == "ok"
    )
    result = api.study_run("positions-default", project=project)
    assert result.status == "ok", result
    documents = list((project / ".signalquarry/runs").glob("*/result.json"))
    assert documents
    for path in documents:
        assert "positions" not in json.loads(path.read_text())
        assert not (path.parent / "positions.csv").exists()


def test_positions_empty_artifact_is_insufficient_and_missing_artifact_is_unavailable(project):
    result = api.backtest("sma-trend", positions=True, end=date(2014, 2, 28), project=project)
    assert result.status == "ok"
    path = project / ".signalquarry/runs" / result.data["run_id"] / "positions.csv"
    assert path.read_text() == "session,symbol,quantity,mark,value\n"
    table = research.load_run(result.data["run_id"], project).positions
    assert table is not None and table.num_rows == 0
    diagnosed, document = _diagnose(project, result.data["run_id"])
    assert "attribution" not in diagnosed.data["available"]
    assert document["attribution"] == {"status": "insufficient", "reason": "No positions rows."}
    path.unlink()
    _, document = _diagnose(project, result.data["run_id"])
    assert document["attribution"]["status"] == "unavailable"
    assert research.load_run(result.data["run_id"], project).positions is None
