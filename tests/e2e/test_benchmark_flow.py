# SPDX-License-Identifier: Apache-2.0
"""The declared benchmark in backtests, evaluations and reports."""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from signalquarry._internal.data.alpaca import AlpacaDataClient, RateLimiter
from signalquarry.api import backtest, data_fetch, evaluate_command, init, report
from tests.fakes import FakeAlpaca
from tests.helpers import dataset, weekdays

END = date(2024, 12, 31)


def _demo(tmp_path: Path, name: str, *, benchmark: str | None = "SYNA") -> Path:
    project = tmp_path / name
    package = name.replace("-", "_")
    assert init(project, demo=True, package=package).status == "ok"
    spec_path = project / f"src/{package}/sma_trend/strategy.yaml"
    text = spec_path.read_text()
    assert "benchmark: SYNA\n" in text
    spec_path.write_text(
        text.replace("benchmark: SYNA\n", "" if benchmark is None else f"benchmark: {benchmark}\n")
    )
    return project


def _result(project: Path, envelope) -> dict:
    path = next(a["path"] for a in envelope.artifacts if a["kind"] == "result")
    return json.loads((project / path).read_text())


def test_backtest_compares_with_the_declared_benchmark(tmp_path: Path) -> None:
    project = _demo(tmp_path, "bench-same")
    envelope = backtest("sma-trend", project=project)
    assert envelope.status == "ok" and "BENCHMARK_DATA_MISSING" not in envelope.warnings
    assert "SYNA buy-and-hold" in envelope.summary

    kinds = {a["kind"]: a["path"] for a in envelope.artifacts}
    assert set(kinds) == {"result", "equity", "benchmark", "fills", "decisions"}
    result = _result(project, envelope)
    block = result["benchmark"]
    assert block["symbol"] == "SYNA" and block["invested_from"] == result["metrics"]["start"]
    assert block["metrics"]["sessions"] == result["metrics"]["sessions"]
    assert block["metrics"]["fills"] >= 1
    assert block["relative"]["status"] == "ok"
    assert block["relative"]["observations"] == result["metrics"]["sessions"]
    assert 0 < block["relative"]["beta"] < 1  # in the market only part of the time

    for key in ("benchmark_total_return", "benchmark_max_drawdown", "excess_total_return", "beta"):
        assert key in envelope.metrics
    assert envelope.metrics["benchmark_total_return"] == block["metrics"]["total_return"]
    assert envelope.metrics["excess_total_return"] == pytest.approx(
        result["metrics"]["total_return"] - block["metrics"]["total_return"], abs=1e-6
    )
    # result.json keeps the strategy's own metrics exactly as before.
    assert "beta" not in result["metrics"] and "benchmark_total_return" not in result["metrics"]

    with (project / kinds["equity"]).open() as handle:
        equity = list(csv.DictReader(handle))
    with (project / kinds["benchmark"]).open() as handle:
        reference = list(csv.DictReader(handle))
    assert [row["session"] for row in reference] == [row["session"] for row in equity]
    assert set(reference[0]) == {"session", "equity"}

    assert result["drawdowns"][0]["depth"] == result["metrics"]["max_drawdown"]
    assert result["activity"]["status"] == "ok" and result["activity"]["turnover"] > 0


def test_the_comparison_does_not_change_the_run(tmp_path: Path) -> None:
    plain = backtest("sma-trend", project=_demo(tmp_path, "bench-none", benchmark=None))
    same = backtest("sma-trend", project=_demo(tmp_path, "bench-self"))
    other = backtest("sma-trend", project=_demo(tmp_path, "bench-other", benchmark="SYNB"))
    for envelope in (same, other):
        assert envelope.data["ledger_hash"] == plain.data["ledger_hash"]
        assert envelope.data["dataset_identity"] == plain.data["dataset_identity"]
        assert {k: v for k, v in envelope.metrics.items() if k in plain.metrics} == plain.metrics
    assert other.metrics["benchmark_total_return"] != same.metrics["benchmark_total_return"]


def test_no_benchmark_means_no_comparison_and_no_warning(tmp_path: Path) -> None:
    project = _demo(tmp_path, "bench-absent", benchmark=None)
    envelope = backtest("sma-trend", project=project)
    assert envelope.warnings.count("BENCHMARK_DATA_MISSING") == 0
    assert "buy-and-hold" not in envelope.summary
    assert "benchmark" not in {a["kind"] for a in envelope.artifacts}
    result = _result(project, envelope)
    assert "benchmark" not in result and "drawdowns" in result and "activity" in result
    assert not any(key.startswith("benchmark") for key in envelope.metrics)

    evaluation = evaluate_command("sma-trend", project=project)
    assert "benchmark" not in evaluation.data["oos"] and "relative" not in evaluation.data["oos"]
    assert "excess_return" not in evaluation.data["folds"][0]
    rendered = report("sma-trend", project=project)
    text = (project / next(a["path"] for a in rendered.artifacts if a["kind"] == "report")).read_text()
    assert "### Versus" not in text and "### Drawdowns" in text and "| Fold |" in text


def test_evaluation_adds_the_benchmark_without_touching_the_gates(tmp_path: Path) -> None:
    plain = evaluate_command("sma-trend", project=_demo(tmp_path, "bench-eval-none", benchmark=None))
    versus = evaluate_command("sma-trend", project=_demo(tmp_path, "bench-eval"))
    assert versus.data["gates"] == plain.data["gates"]
    assert versus.evidence["claim_level"] == plain.evidence["claim_level"]

    folds, oos = versus.data["folds"], versus.data["oos"]
    for fold, bare in zip(folds, plain.data["folds"], strict=True):
        assert {k: fold[k] for k in bare} == bare
        assert fold["excess_return"] == pytest.approx(fold["total_return"] - fold["benchmark_total_return"])
        assert 0 <= fold["benchmark_max_drawdown"] <= 1
    assert oos["benchmark"]["symbol"] == "SYNA"
    assert oos["benchmark"]["sessions"] == oos["sessions"] == oos["relative"]["observations"]
    assert oos["folds_ahead_of_benchmark"] == sum(1 for fold in folds if fold["excess_return"] > 0)
    assert {k: oos[k] for k in plain.data["oos"]} == plain.data["oos"]


def test_report_shows_the_comparison_and_the_folds(tmp_path: Path) -> None:
    project = _demo(tmp_path, "bench-report")
    backtest("sma-trend", project=project)
    evaluate_command("sma-trend", project=project)
    rendered = report("sma-trend", project=project)
    kinds = {a["kind"]: a["path"] for a in rendered.artifacts}
    text = (project / kinds["report"]).read_text()
    assert "### Versus SYNA buy-and-hold" in text
    assert "| Metric | Strategy | SYNA | Difference |" in text
    assert "### Drawdowns" in text and "### Trading" in text
    assert "| Fold | Sessions | Return | Sharpe | Max drawdown | SYNA return | Excess |" in text
    assert "Ahead of SYNA buy-and-hold in " in text
    svg = (project / kinds["equity"]).read_text()
    assert svg.count("<polyline") == 2 and "SYNA" in svg


def _market() -> object:
    sessions = weekdays(date(2015, 1, 5), 2600)
    rng = np.random.Generator(np.random.PCG64(3))
    prices = {}
    for symbol, drift in (("SPY", 0.0012), ("QQQ", 0.0006)):
        closes = 100 * np.cumprod(1 + drift + rng.normal(0, 0.004, len(sessions)))
        opens = np.concatenate(([100.0], closes[:-1]))
        prices[symbol] = {"open": list(np.round(opens, 2)), "close": list(np.round(closes, 2))}
    return dataset(sessions, prices)


def test_benchmark_from_another_recorded_dataset_or_reported_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SIGNALQUARRY_CACHE_DIR", str(tmp_path / "cache"))
    project = tmp_path / "bench-real"
    assert init(project, package="bench_real").status == "ok"
    spec_path = project / "src/bench_real/sma_trend/strategy.yaml"
    spec_path.write_text(spec_path.read_text().replace("benchmark: SPY", "benchmark: QQQ"))
    client = AlpacaDataClient(
        "k",
        "s",
        transport=FakeAlpaca(_market(), page_size=5000),
        limiter=RateLimiter(sleep=lambda s: None),
    )
    # Only the strategy's own symbol is recorded: the benchmark cannot be compared.
    assert data_fetch(symbols=("SPY",), project=project, client=client, end=END).status == "ok"
    missing = backtest("sma-trend", project=project)
    assert missing.status == "ok" and "BENCHMARK_DATA_MISSING" in missing.warnings
    assert missing.next_actions[0]["command"] == "sqy data fetch --strategy sma-trend"
    assert "benchmark" not in _result(project, missing)
    evaluation = evaluate_command("sma-trend", project=project)
    assert "BENCHMARK_DATA_MISSING" in evaluation.warnings and "benchmark" not in evaluation.data["oos"]

    # A second manifest that holds the benchmark over the same sessions is used for it,
    # while the strategy keeps running on the dataset it ran on before.
    assert data_fetch(symbols=("QQQ",), project=project, client=client, end=END).status == "ok"
    found = backtest("sma-trend", project=project)
    assert "BENCHMARK_DATA_MISSING" not in found.warnings
    assert found.data["dataset_identity"] == missing.data["dataset_identity"]
    assert found.data["ledger_hash"] == missing.data["ledger_hash"]
    assert _result(project, found)["benchmark"]["symbol"] == "QQQ"
