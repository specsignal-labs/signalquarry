# SPDX-License-Identifier: Apache-2.0
"""Read-only notebook access to artifacts produced through the public API."""

from __future__ import annotations

import csv
import json
import sys
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from datetime import date
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pytest

from signalquarry import api, research
from signalquarry._internal.evidence.runs import result_document


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "research-lab"
    initialized = api.init(root, demo=True, package="research_" + uuid4().hex[:12])
    assert initialized.status == "ok", initialized
    return root


def _backtest(project: Path, *, label: str | None = None, period: int = 200) -> str:
    result = api.backtest("sma-trend", project=project, label=label, params=[f"period={period}"])
    assert result.status == "ok", result
    return result.data["run_id"]


@pytest.fixture
def run(project: Path) -> research.Run:
    return research.load_run(_backtest(project, label="base"), project)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _rewrite_result(run: research.Run, edit: Callable[[dict[str, Any]], None]) -> None:
    document = _json(run.path / "result.json")
    edit(document)
    document.pop("result_hash")
    (run.path / "result.json").write_text(result_document(**document), encoding="utf-8")


def test_listing_order_filter_latest_and_project_discovery(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert research.list_runs(project) == ()
    assert research.list_sweeps(project) == ()
    ids = [
        _backtest(project, label=label, period=period) for label, period in (("base", 200), ("variant", 100))
    ]
    # Directory names encode time only to the second. Recorded creation time decides order.
    for run_id, timestamp in zip(
        ids, ("2030-01-02T00:00:00+00:00", "2030-01-01T00:00:00+00:00"), strict=True
    ):
        path = project / ".signalquarry" / "runs" / run_id / "result.json"
        document = _json(path)
        document["created_at"] = timestamp  # excluded from the result hash
        path.write_text(json.dumps(document), encoding="utf-8")
    rows = research.list_runs(str(project / "src"))
    assert [row["run_id"] for row in rows] == ids
    assert [row["label"] for row in rows] == ["base", "variant"]
    assert set(rows[0]) == {
        "run_id",
        "strategy_id",
        "label",
        "command",
        "created_at",
        "configuration_hash",
        "params",
        "total_return",
        "sharpe",
        "max_drawdown",
    }
    assert rows == research.list_runs(project, strategy="sma-trend")
    assert research.list_runs(project, strategy="other") == ()
    for row in rows:
        document = _json(project / ".signalquarry" / "runs" / row["run_id"] / "result.json")
        assert row["params"] == document["params"]
        for metric in ("total_return", "sharpe", "max_drawdown"):
            assert row[metric] == document["metrics"][metric]
    assert research.latest_run("sma-trend", project).run_id == ids[0]
    monkeypatch.chdir(project / "src")
    assert research.list_runs() == rows
    assert research.load_run(ids[0]).run_id == ids[0]
    assert research.latest_run("sma-trend").run_id == ids[0]
    with pytest.raises(research.ResearchError) as exc:
        research.latest_run("other", project)
    assert exc.value.code == "RUN_NOT_FOUND"


def test_run_curves_document_properties_and_immutability(run: research.Run) -> None:
    document = _json(run.path / "result.json")
    rows = _csv(run.path / "equity.csv")
    equity = np.array([float(row["equity"]) for row in rows])
    assert run.sessions == tuple(date.fromisoformat(row["session"]) for row in rows)
    assert len(run.sessions) == document["metrics"]["sessions"]
    np.testing.assert_array_equal(run.equity, equity)
    expected = (
        equity / np.concatenate(([float(document["spec"]["account"]["initial_cash"])], equity[:-1])) - 1
    )
    np.testing.assert_array_equal(run.returns, expected)
    np.testing.assert_array_equal(
        run.benchmark, [float(row["equity"]) for row in _csv(run.path / "benchmark.csv")]
    )
    assert run.path.is_absolute()
    assert run.strategy_id == document["strategy_id"]
    assert run.label == "base"
    assert run.params == document["params"]
    assert run.metrics == document["metrics"]
    assert run.configuration_hash == document["configuration_hash"]
    assert run.grade == document["evidence"]["grade"]
    assert isinstance(run.document, MappingProxyType)
    assert run.document["spec"]["data"]["symbols"] == ("SYNA",)
    for mapping in (run.document, run.params, run.metrics, run.document["spec"]["account"]):
        with pytest.raises(TypeError):
            mapping["new"] = 1
    with pytest.raises(FrozenInstanceError):
        run.run_id = "other"
    for array in (run.equity, run.returns, run.benchmark):
        assert array is not None and array.dtype == np.float64 and not array.flags.writeable
        with pytest.raises(ValueError):
            array[0] = 0
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_full_fills_and_decisions(run: research.Run) -> None:
    fills = run.fills()
    rows = _csv(run.path / "fills.csv")
    assert fills.num_rows == len(rows) == run.metrics["fills"] > 0
    assert fills.column_names == ["session", "symbol", "side", "quantity", "price", "fee", "settle_session"]
    expected = [
        {
            "session": date.fromisoformat(row["session"]),
            "symbol": row["symbol"],
            "side": row["side"],
            **{key: float(row[key]) for key in ("quantity", "price", "fee")},
            "settle_session": date.fromisoformat(row["settle_session"]) if row["settle_session"] else None,
        }
        for row in rows
    ]
    assert fills.to_pylist() == expected
    decisions = list(run.decisions())
    assert decisions == [json.loads(line) for line in (run.path / "decisions.jsonl").read_text().splitlines()]
    assert decisions
    decisions[0]["extra"] = True
    assert "extra" not in next(run.decisions())


def test_arrow_schema_values_and_optional_benchmark(run: research.Run, project: Path) -> None:
    table = run.to_arrow()
    assert table.schema == pa.schema(
        [
            ("session", pa.date32()),
            ("equity", pa.float64()),
            ("returns", pa.float64()),
            ("benchmark", pa.float64()),
        ]
    )
    assert table["session"].to_pylist() == list(run.sessions)
    np.testing.assert_array_equal(table["equity"].to_numpy(), run.equity)
    np.testing.assert_array_equal(table["returns"].to_numpy(), run.returns)
    np.testing.assert_array_equal(table["benchmark"].to_numpy(), run.benchmark)
    (run.path / "benchmark.csv").unlink()
    without = research.load_run(run.run_id, project)
    assert without.benchmark is None
    assert without.to_arrow().schema == table.schema
    assert without.to_arrow()["benchmark"].to_pylist() == [None] * len(run.sessions)


def test_first_return_initial_cash_and_legacy_fallback(run: research.Run, project: Path) -> None:
    _rewrite_result(run, lambda doc: doc["spec"]["account"].update(initial_cash="125000"))
    changed = research.load_run(run.run_id, project)
    assert changed.returns[0] == pytest.approx(run.equity[0] / 125000 - 1)
    _rewrite_result(run, lambda doc: doc["spec"]["account"].pop("initial_cash"))
    legacy = research.load_run(run.run_id, project)
    assert legacy.returns[0] == 0.0
    np.testing.assert_array_equal(legacy.returns[1:], run.returns[1:])


def test_to_pandas_missing_dependency(run: research.Run, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pandas", None)
    with pytest.raises(research.ResearchError) as exc:
        run.to_pandas()
    assert exc.value.code == "PANDAS_NOT_INSTALLED"


def test_to_pandas_lazy_call_path(run: research.Run, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []
    sentinel = object()

    class FakeFrame:
        def __init__(self, data: dict[str, Any]) -> None:
            calls.append(data)

        def set_index(self, column: str) -> object:
            calls.append(column)
            return sentinel

    pandas = ModuleType("pandas")
    pandas.DataFrame = FakeFrame
    monkeypatch.setitem(sys.modules, "pandas", pandas)
    assert run.to_pandas() is sentinel
    assert calls == [run.to_arrow().to_pydict(), "session"]


def test_sweeps_points_runs_tables_and_summary_detail(project: Path) -> None:
    ids: list[str] = []
    full_schema = research.load_run(_backtest(project), project).fills().schema
    for summary in (False, True):
        result = api.sweep(
            "sma-trend", ["period=100,200", "weight=0.5"], project=project, summary_only=summary
        )
        assert result.status == "ok", result
        ids.append(result.data["sweep_id"])
        sweep = research.load_sweep(ids[-1], str(project / "src"))
        assert sweep.path.name == sweep.sweep_id == ids[-1]
        assert sweep.grid == {"period": (100, 200), "weight": (0.5,)}
        assert isinstance(sweep.points, tuple)
        assert sweep.points == tuple(result.data["points"])
        assert sweep.pbo == result.data["pbo"]
        with pytest.raises(TypeError):
            sweep.points[0]["params"]["period"] = 10
        with pytest.raises(TypeError):
            sweep.grid["period"] = (10,)
        runs = sweep.runs()
        assert [run.run_id for run in runs] == [point["run_id"] for point in sweep.points]
        assert [run.params["period"] for run in runs] == [100, 200]
        assert all(run.label is None for run in runs)
        table = sweep.to_arrow()
        assert table.column_names == [
            "period",
            "weight",
            "run_id",
            "configuration_hash",
            "total_return",
            "sharpe",
            "max_drawdown",
            "fills",
        ]
        for row, point in zip(table.to_pylist(), sweep.points, strict=True):
            assert row == {
                **point["params"],
                **{key: point[key] for key in table.column_names if key not in sweep.grid},
            }
        for run in runs:
            assert run.fills().schema == full_schema
            if summary:
                assert run.fills().num_rows == 0
                assert list(run.decisions()) == []
            else:
                assert run.fills().num_rows == run.metrics["fills"] > 0
                assert list(run.decisions())
    assert [doc["sweep_id"] for doc in research.list_sweeps(project)] == ids[::-1]
    with pytest.raises(TypeError):
        research.list_sweeps(project)[0]["summary_only"] = False


def test_comparison_and_no_writes_or_cached_artifacts(project: Path) -> None:
    ids = [_backtest(project, period=period) for period in (100, 200)]
    result = api.runs_compare(ids, project=project)
    assert result.status == "ok", result
    comparison_id = result.data["comparison_id"]
    before = {
        path.relative_to(project): (path.stat().st_mtime_ns, path.read_bytes())
        for path in project.rglob("*")
        if path.is_file()
    }
    comparison = research.load_comparison(comparison_id, project)
    document = _json(project / ".signalquarry" / "comparisons" / comparison_id / "comparison.json")
    assert comparison["schema"] == "signalquarry.comparison/v1"
    assert comparison["comparison_id"] == comparison_id
    for run_id, pair in document["pairs"].items():
        for key, value in pair.items():
            assert comparison["pairs"][run_id][key] == (tuple(value) if isinstance(value, list) else value)
    assert tuple(row["run_id"] for row in comparison["runs"]) == tuple(ids)
    with pytest.raises(TypeError):
        comparison["pairs"][ids[1]]["sessions"] = 0
    research.list_runs(project)
    research.list_sweeps(project)
    for run_id in ids:
        run = research.load_run(run_id, project)
        run.fills()
        list(run.decisions())
        run.to_arrow()
    after = {
        path.relative_to(project): (path.stat().st_mtime_ns, path.read_bytes())
        for path in project.rglob("*")
        if path.is_file()
    }
    assert after == before
    (run.path / "result.json").unlink()
    with pytest.raises(research.ResearchError) as exc:
        research.load_run(run.run_id, project)
    assert exc.value.code == "RUN_ARTIFACT_INVALID"


@pytest.mark.parametrize(
    "loader,identifier",
    [
        (research.load_run, "x"),
        (research.load_sweep, "x"),
        (research.load_comparison, "x"),
        (research.latest_run, "x"),
        (research.list_runs, None),
        (research.list_sweeps, None),
    ],
)
def test_project_not_found(loader: Callable[..., Any], identifier: str | None, tmp_path: Path) -> None:
    with pytest.raises(research.ResearchError) as exc:
        loader(identifier, project=tmp_path) if identifier is not None else loader(project=tmp_path)
    assert exc.value.code == "PROJECT_NOT_FOUND"


@pytest.mark.parametrize("loader", [research.load_run, research.load_sweep, research.load_comparison])
@pytest.mark.parametrize("identifier", ["missing", "", ".", "..", "../outside", "sub/path", "sub\\path"])
def test_missing_or_unsafe_ids(loader: Callable[..., Any], identifier: str, project: Path) -> None:
    with pytest.raises(research.ResearchError) as exc:
        loader(identifier, project)
    assert exc.value.code == "RUN_NOT_FOUND"


@pytest.mark.parametrize("damage", ["missing", "unreadable", "schema", "list", "tampered"])
def test_invalid_run_documents(run: research.Run, project: Path, damage: str) -> None:
    path = run.path / "result.json"
    if damage == "missing":
        path.unlink()
    elif damage == "unreadable":
        path.write_text("not json", encoding="utf-8")
    elif damage == "schema":
        path.write_text('{"schema":"other/v1"}', encoding="utf-8")
    elif damage == "list":
        path.write_text("[]", encoding="utf-8")
    else:
        document = _json(path)
        document["metrics"]["sharpe"] = 99.0
        path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(research.ResearchError) as exc:
        research.load_run(run.run_id, project)
    assert exc.value.code == "RUN_ARTIFACT_INVALID"


@pytest.mark.parametrize("damage", ["missing", "truncated", "malformed", "benchmark_sessions"])
def test_invalid_curves(run: research.Run, project: Path, damage: str) -> None:
    path = run.path / "equity.csv"
    if damage == "missing":
        path.unlink()
    elif damage == "truncated":
        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n", encoding="utf-8")
    elif damage == "malformed":
        path.write_text("session,equity\nnot-a-date,broken\n", encoding="utf-8")
    else:
        path = run.path / "benchmark.csv"
        path.write_text(
            path.read_text().replace(run.sessions[0].isoformat(), "1999-01-01", 1), encoding="utf-8"
        )
    with pytest.raises(research.ResearchError) as exc:
        research.load_run(run.run_id, project)
    assert exc.value.code == "RUN_ARTIFACT_INVALID"


@pytest.mark.parametrize(
    "kind,filename,loader",
    [
        ("sweeps", "sweep.json", research.load_sweep),
        ("comparisons", "comparison.json", research.load_comparison),
    ],
)
@pytest.mark.parametrize("content", [None, "not json", "[]", '{"schema":"other/v1"}'])
def test_invalid_sweep_and_comparison_documents(
    project: Path, kind: str, filename: str, loader: Callable[..., Any], content: str | None
) -> None:
    directory = project / ".signalquarry" / kind / "broken"
    directory.mkdir(parents=True)
    if content is not None:
        (directory / filename).write_text(content, encoding="utf-8")
    with pytest.raises(research.ResearchError) as exc:
        loader("broken", project)
    assert exc.value.code == "RUN_ARTIFACT_INVALID"


@pytest.mark.parametrize(
    "filename,content",
    [
        ("fills.csv", "session,symbol,side,quantity,price,fee,settle_session\nbroken,SYNA,buy,1,2,3,\n"),
        ("decisions.jsonl", "not json\n"),
        ("decisions.jsonl", "[]\n"),
    ],
)
def test_invalid_optional_artifacts(run: research.Run, filename: str, content: str) -> None:
    (run.path / filename).write_text(content, encoding="utf-8")
    with pytest.raises(research.ResearchError) as exc:
        run.fills() if filename == "fills.csv" else list(run.decisions())
    assert exc.value.code == "RUN_ARTIFACT_INVALID"


def test_study_results_are_listed_loaded_and_verified(tmp_path: Path) -> None:
    from signalquarry import research
    from signalquarry.api import init, study_init, study_run

    project = tmp_path / "facade-study"
    assert init(project, demo=True, package="facade_study_lab").status == "ok"
    assert research.list_studies(project) == ()
    with pytest.raises(research.ResearchError) as missing:
        research.latest_study("first", project)
    assert missing.value.code == "RUN_NOT_FOUND"

    assert study_init("sma-trend", "first", project=project).status == "ok"
    ran = study_run("first", project=project)
    again = study_run("first", project=project)
    rows = research.list_studies(project)
    assert [row["study_run_id"] for row in rows] == [again.data["study_run_id"], ran.data["study_run_id"]]
    assert {row["study_id"] for row in rows} == {"first"} and rows[0]["verdict"] == ran.data["verdict"][
        "outcome"
    ]

    latest = research.latest_study("first", project)
    assert latest["study_hash"] == ran.data["study_hash"]
    expected = ran.data["verdict"]
    assert latest["verdict"]["outcome"] == expected["outcome"]
    assert latest["verdict"]["difference"] == expected["difference"]
    assert latest["verdict"]["interval_90"] == tuple(expected["interval_90"])  # frozen: lists become tuples
    assert [arm["id"] for arm in latest["arms"]] == ["base", "costs-x2", "buy-and-hold"]
    with pytest.raises(TypeError):
        latest["verdict"]["outcome"] = "supported"  # type: ignore[index]
    assert research.load_study(ran.data["study_run_id"], project)["study_id"] == "first"

    path = project / ".signalquarry" / "studies" / again.data["study_run_id"] / "study.json"
    path.write_text(path.read_text().replace('"not_supported"', '"supported"'))
    with pytest.raises(research.ResearchError) as tampered:
        research.latest_study("first", project)
    assert tampered.value.code == "RUN_ARTIFACT_INVALID"
    with pytest.raises(research.ResearchError) as unknown:
        research.load_study("nope", project)
    assert unknown.value.code == "RUN_NOT_FOUND"
