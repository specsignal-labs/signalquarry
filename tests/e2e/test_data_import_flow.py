# SPDX-License-Identifier: Apache-2.0
"""ADR 0016 local-import acceptance, through public API and CLI surfaces."""

from __future__ import annotations

import csv
import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from signalquarry import api, research
from signalquarry._internal.calendar.nyse import is_holiday
from signalquarry._internal.data.alpaca import AlpacaDataClient, RateLimiter
from signalquarry._internal.data.dataset import FIELDS, Dataset, Dividend, Split
from signalquarry._internal.data.library import dataset_from_manifest, select_manifest
from signalquarry._internal.data.synthetic import synthetic_dataset
from signalquarry._internal.paper.runner import Outcome
from signalquarry.api.resolve import Resolved, resolve
from signalquarry.cli.main import main
from tests.fakes import FakeAlpaca
from tests.helpers import dataset
from tests.unit.test_local_data_import import ACTION_HEADER, BAR, HEADER, snapshot


def _sessions(start: date, end: date) -> tuple[date, ...]:
    return tuple(
        date.fromordinal(day)
        for day in range(start.toordinal(), end.toordinal() + 1)
        if not is_holiday(date.fromordinal(day))
    )


def _market(*, actions: bool = False) -> Dataset:
    sessions = _sessions(date(2016, 1, 4), date(2024, 12, 31))
    rng = np.random.Generator(np.random.PCG64(3))
    closes = 123.456789 * np.cumprod(1 + 0.0012 + rng.normal(0, 0.004, len(sessions)))
    opens = np.concatenate(([123.456789], closes[:-1]))
    if actions:
        opens[800:] /= 2
        closes[800:] /= 2
    return dataset(
        sessions,
        {"SPY": {"open": list(np.round(opens, 6)), "close": list(np.round(closes, 6))}},
        splits=(Split("SPY", sessions[800], Decimal("2")),) if actions else (),
        dividends=(Dividend("SPY", sessions[1000], sessions[1003], Decimal("0.375")),) if actions else (),
    )


def _csvs(tmp_path: Path, market: Dataset) -> tuple[Path, Path]:
    bars, actions = tmp_path / "bars.csv", tmp_path / "actions.csv"
    with bars.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER.strip().split(","))
        for i in reversed(range(len(market.sessions))):
            for symbol in market.symbols:
                if market.series[symbol].present[i]:
                    writer.writerow(
                        [
                            market.sessions[i].isoformat(),
                            symbol,
                            *(format(market.price(symbol, name, i), "f") for name in FIELDS),
                            market.series[symbol].volume[i],
                        ]
                    )
    with actions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(ACTION_HEADER.strip().split(","))
        for item in market.splits:
            writer.writerow([item.symbol, "split", item.ex_date, "", item.ratio, ""])
        for dividend in market.dividends:
            writer.writerow(
                [dividend.symbol, "dividend", dividend.ex_date, dividend.pay_date, "", dividend.amount]
            )
    return bars, actions


def _project(tmp_path: Path, name: str, *, feed: str = "local") -> Path:
    project = tmp_path / name
    package = name.replace("-", "_").lower()
    assert api.init(project, package=package).status == "ok"
    spec = project / f"src/{package}/sma_trend/strategy.yaml"
    spec.write_text(
        spec.read_text()
        .replace("feed: sip", f"feed: {feed}")
        .replace("period: 200", "period: 50")
        .replace("sizing: whole_shares", "sizing: fractional\n  rebalance: every_decision")
    )
    return project


def _manifest(project: Path) -> dict:
    return api.data.library_for(project).manifests()[0]


def _artifact(project: Path, result: api.Envelope, name: str) -> Path:
    return project / next(item["path"] for item in result.artifacts if item["kind"] == name)


@pytest.fixture(autouse=True)
def _offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIGNALQUARRY_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SIGNALQUARRY_OFFLINE", "1")


def test_import_same_bytes_same_identity_and_ledger_in_separate_projects(tmp_path: Path) -> None:
    bars, actions = _csvs(tmp_path, _market())
    projects = [_project(tmp_path, f"local-replay-{i}") for i in range(2)]
    imports = [
        api.data_import(bars=bars, actions=actions, source="Source", adjustment="raw", project=project)
        for project in projects
    ]
    assert all(item.status == "ok" for item in imports)
    assert imports[0].data["dataset_identity"] == imports[1].data["dataset_identity"]
    runs = [api.backtest("sma-trend", project=project) for project in projects]
    assert all(item.status == "ok" for item in runs)
    assert runs[0].data["ledger_hash"] == runs[1].data["ledger_hash"]
    assert runs[0].evidence["provenance"] == "provenance supplied by the user"
    assert all(api.data_verify(project=project).status == "ok" for project in projects)


def test_import_and_fake_fetch_same_decisions_fills_equity_with_actions(tmp_path: Path) -> None:
    market = _market(actions=True)
    bars, actions = _csvs(tmp_path, market)
    local_project = _project(tmp_path, "local-parity")
    fetched_project = _project(tmp_path, "fetched-parity", feed="sip")
    client = AlpacaDataClient(
        "k", "s", transport=FakeAlpaca(market), limiter=RateLimiter(sleep=lambda _: None)
    )
    assert (
        api.data_fetch(
            strategy_id="sma-trend",
            start=market.sessions[0],
            end=market.sessions[-1],
            project=fetched_project,
            client=client,
        ).status
        == "ok"
    )
    assert (
        api.data_import(
            bars=bars, actions=actions, source="Source", adjustment="raw", project=local_project
        ).status
        == "ok"
    )
    imported, fetched = (
        api.backtest("sma-trend", project=project) for project in (local_project, fetched_project)
    )
    assert imported.status == fetched.status == "ok"
    # Source participates in dataset identity, and feed in configuration identity.
    assert imported.data["dataset_identity"] != fetched.data["dataset_identity"]
    assert imported.data["ledger_hash"] != fetched.data["ledger_hash"]
    for name in ("decisions", "fills", "equity"):
        assert (
            _artifact(local_project, imported, name).read_bytes()
            == _artifact(fetched_project, fetched, name).read_bytes()
        )
    rebuilt = dataset_from_manifest(api.data.library_for(local_project), _manifest(local_project))
    assert rebuilt.splits == market.splits and rebuilt.dividends == market.dividends


def test_import_adjusted_and_omitted_actions_cap_real_passing_evaluation(tmp_path: Path) -> None:
    bars, actions = _csvs(tmp_path, _market())
    cases = [
        ("raw", actions, "historical"),
        ("adjusted", None, "historical_adjusted"),
        ("raw", None, "historical_adjusted"),
    ]
    for index, (adjustment, action_path, grade) in enumerate(cases):
        project = _project(tmp_path, f"import-gates-{index}")
        imported = api.data_import(
            bars=bars,
            actions=action_path,
            no_actions=action_path is None,
            source="Source",
            adjustment=adjustment,
            project=project,
        )
        assert imported.status == "ok"
        assert api.spec_freeze("sma-trend", project=project).status == "ok"
        result = api.evaluate_command("sma-trend", project=project)
        assert result.status == "ok", result.data
        assert all(gate["ok"] for gate in result.data["gates"].values())
        assert result.evidence["grade"] == grade
        assert result.evidence["claim_level"] == ("walk_forward" if index == 0 else "in_sample")
        assert result.evidence["provenance"] == "provenance supplied by the user"
        assert api.trials_ls(project=project).data["summary"]["project_count"] == 1
        if index == 0:
            opened = api.evaluate_command("sma-trend", open_holdout=True, project=project)
            assert opened.status == "ok", opened.data
            assert all(gate["ok"] for gate in opened.data["gates"].values())
            assert opened.evidence["claim_level"] == "holdout_passed"
        else:
            assert "DATA_IMPORT_CLAIM_CAPPED" in result.warnings
            before = snapshot(project)
            refused = api.evaluate_command("sma-trend", open_holdout=True, project=project)
            assert refused.reason_codes == ["DATA_IMPORT_CLAIM_CAPPED"] and snapshot(project) == before
            assert api.holdout_status(project=project).data["families"][0]["state"] == "sealed"
        panel = research.load_panel("sma-trend", project=project)
        assert panel.grade == grade and panel.dataset_id == imported.data["dataset_id"]
        assert panel.sessions[-1] < panel.sealed_from
        rendered = api.report("sma-trend", project=project)
        text = _artifact(project, rendered, "report").read_text()
        assert "provenance supplied by the user" in text and "123.456789" not in text
        assert ("Claims are capped at `in_sample`" in text) == (index > 0)
        if index > 0:
            assert (
                "future corporate actions" if adjustment == "adjusted" else "no actions file was supplied"
            ) in text


@pytest.mark.parametrize("adjustment,use_actions", [("raw", True), ("raw", False), ("adjusted", False)])
def test_import_study_sweep_diagnose_and_default_deny_export(
    tmp_path: Path, adjustment: str, use_actions: bool
) -> None:
    bars, actions = _csvs(tmp_path, _market())
    project = _project(tmp_path, f"import-research-{adjustment}-{use_actions}")
    imported = api.data_import(
        bars=bars,
        actions=actions if use_actions else None,
        no_actions=not use_actions,
        source="Source",
        license="Redistribution allowed by supplier",
        adjustment=adjustment,
        project=project,
    )
    assert imported.status == "ok"
    grade = "historical" if use_actions else "historical_adjusted"
    assert api.spec_freeze("sma-trend", project=project).status == "ok"
    base = api.backtest("sma-trend", project=project)
    diagnostic = api.diagnose("sma-trend", project=project, exposures=("SPY",))
    assert diagnostic.status == "ok" and diagnostic.evidence == base.evidence
    assert diagnostic.evidence["grade"] == grade
    swept = api.sweep("sma-trend", ["period=40,60"], project=project)
    assert swept.status == "ok" and swept.evidence["grade"] == grade
    assert swept.evidence["provenance"] == "provenance supplied by the user"
    assert (
        api.study_init("sma-trend", "trend", metric="max_drawdown", direction="lower", project=project).status
        == "ok"
    )
    study = project / "studies/trend/study.yaml"
    text = study.read_text().replace("TODO", "Declared study hypothesis")
    study.write_text(text)
    studied = api.study_run("trend", project=project)
    assert studied.status == "ok", studied.summary
    assert studied.evidence["grade"] == grade and studied.evidence["claim_level"] == "in_sample"
    assert studied.evidence["provenance"] == "provenance supplied by the user"
    report = _artifact(project, studied, "comparison").read_text()
    assert "provenance supplied by the user" in report and "123.456789" not in report
    assert api.trials_ls(project=project).data["summary"]["project_count"] >= 3
    # Refusal is based on produced evidence, even after feed/declarations change.
    spec = next((project / "src").rglob("strategy.yaml"))
    spec.write_text(spec.read_text().replace("feed: local", "feed: sip"))
    target = tmp_path / "export"
    target.mkdir()
    (target / "keep").write_text("Existing output")
    before = snapshot(tmp_path)
    for tier in ("public", "nda"):
        refused = api.evidence_export("sma-trend", tier=tier, out=target, project=project)
        assert refused.reason_codes == ["DATA_IMPORT_EXPORT_REFUSED"]
        assert snapshot(tmp_path) == before
    assert "123.456789" not in json.dumps(_manifest(project))


def test_import_selects_newest_covering_local_manifest(tmp_path: Path) -> None:
    project = _project(tmp_path, "import-selection")
    bars, actions = _csvs(tmp_path, _market())
    first = api.data_import(bars=bars, actions=actions, source="First", adjustment="raw", project=project)
    # A more recent import can have a shorter end date: local selection is by import time.
    bars.write_text(HEADER + BAR + "2024-01-03,SPY,124,125,123,124,1000000\n")
    second = api.data_import(
        bars=bars, no_actions=True, source="Second", adjustment="adjusted", project=project
    )
    assert first.status == second.status == "ok"
    resolved = resolve("backtest", "sma-trend", project)
    assert isinstance(resolved, Resolved) and resolved.dataset_id == second.data["dataset_id"]
    assert resolved.evidence_grade == "historical_adjusted"
    assert resolved.symbol_source("SPY") == (resolved.dataset, "SPY")
    assert resolved.symbol_source("QQQ") is None
    assert select_manifest(api.data.library_for(project), ("QQQ",), "local") is None


def test_import_local_feed_without_a_manifest_points_to_import(tmp_path: Path) -> None:
    project = _project(tmp_path, "import-missing-manifest")
    before = snapshot(project)
    result = api.backtest("sma-trend", project=project)
    assert result.status == "unavailable" and result.reason_codes == ["PROVIDER_UNAVAILABLE"]
    assert result.next_actions[0]["command"] == "sqy data import --help"
    assert "no imported local dataset" in result.summary and snapshot(project) == before


def test_import_cli_help_usage_and_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["data", "import", "--help"]) == 0
    help_text = capsys.readouterr().out
    for flag in ("--bars", "--actions", "--no-actions", "--adjustment", "--source", "--license", "--project"):
        assert flag in help_text
    project = _project(tmp_path, "import-cli")
    bars = tmp_path / "bars.csv"
    bars.write_text(HEADER + BAR)
    base = [
        "--json",
        "data",
        "import",
        "--bars",
        str(bars),
        "--source",
        "A source",
        "--adjustment",
        "raw",
        "--project",
        str(project),
    ]
    for extra in ([], ["--actions", str(bars), "--no-actions"]):
        before = snapshot(tmp_path)
        assert main(base + extra) == 64
        assert json.loads(capsys.readouterr().out)["status"] == "usage"
        assert snapshot(tmp_path) == before
    before = snapshot(tmp_path)
    assert main([part if part != "raw" else "adjusted" for part in base] + ["--actions", str(bars)]) == 64
    assert json.loads(capsys.readouterr().out)["reason_codes"] == ["USAGE_INVALID"]
    assert snapshot(tmp_path) == before
    assert main(base + ["--no-actions"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["evidence"]["grade"] == "historical_adjusted"
    assert result["evidence"]["provenance"] == "provenance supplied by the user"


def test_import_cli_sparse_quality_stays_bounded_and_reimport_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _project(tmp_path, "import-cli-sparse")
    bars = tmp_path / "bars.csv"
    bars.write_text(
        HEADER + BAR.replace("2024-01-02", "2016-01-04") + BAR.replace("2024-01-02", "2035-12-31")
    )
    args = [
        "--json",
        "data",
        "import",
        "--bars",
        str(bars),
        "--source",
        "Sparse file",
        "--adjustment",
        "raw",
        "--no-actions",
        "--project",
        str(project),
    ]
    assert main(args) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["data"]["quality"]["missing_trading_days"] > 4000
    assert "QUALITY_TRADING_DAY_MISSING" in first["warnings"]
    assert [item["kind"] for item in first["artifacts"]] == ["dataset_manifest"]
    before = snapshot(tmp_path)
    assert main(args) == 0
    second = json.loads(capsys.readouterr().out)
    assert first["data"] == second["data"] and snapshot(tmp_path) == before


def test_import_existing_demo_hash_and_synthetic_identity_unchanged(tmp_path: Path) -> None:
    project = tmp_path / "import-demo-golden"
    assert api.init(project, demo=True, package="import_demo_golden").status == "ok"
    resolved = resolve("backtest", "sma-trend", project)
    assert isinstance(resolved, Resolved)
    assert (
        resolved.strategy.configuration_hash
        == "sha256:c24e29eb1448438e7ec228fb05e2f5c3f6d0d5c27166a3dc421bf250785b9235"
    )
    assert (
        synthetic_dataset(date(2014, 1, 2), date(2025, 12, 31), symbols=("SYNA",)).identity()
        == "sha256:6229c5b059545c501be654386fcc482441ec74bac8085c53bb87145489022947"
    )


def test_import_paper_preflight_refuses_capped_grade_without_writes(tmp_path: Path) -> None:
    project = _project(tmp_path, "import-paper-refusal")
    bars, _ = _csvs(tmp_path, _market())
    assert (
        api.data_import(bars=bars, no_actions=True, source="Source", adjustment="raw", project=project).status
        == "ok"
    )
    assert api.spec_freeze("sma-trend", project=project).status == "ok"
    before = snapshot(project)
    refused = api.paper_preflight("demo", project=project)
    assert refused.reason_codes == ["DATA_IMPORT_CLAIM_CAPPED"]
    assert refused.evidence["grade"] == "historical_adjusted" and snapshot(project) == before


def test_import_paper_preflight_grade_blocks_an_otherwise_ready_strategy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from signalquarry.api import paper

    calls = []

    def ready_kernel(context):
        calls.append(context)
        return SimpleNamespace(preflight=lambda: Outcome("ready", {"ready": True}))

    monkeypatch.setattr(paper, "kernel_for", ready_kernel)
    project = _project(tmp_path, "import-paper-ready")
    bars, actions = _csvs(tmp_path, _market())
    assert (
        api.data_import(bars=bars, actions=actions, source="Source", adjustment="raw", project=project).status
        == "ok"
    )
    assert api.spec_freeze("sma-trend", project=project).status == "ok"
    assert api.paper_preflight("demo", project=project).status == "ok"
    assert len(calls) == 1
    assert (
        api.data_import(
            bars=bars, no_actions=True, source="Source", adjustment="adjusted", project=project
        ).status
        == "ok"
    )
    before = snapshot(project)
    assert api.paper_preflight("demo", project=project).reason_codes == ["DATA_IMPORT_CLAIM_CAPPED"]
    assert len(calls) == 1 and snapshot(project) == before


def test_import_export_refuses_without_policy_and_preserves_existing_output(tmp_path: Path) -> None:
    project = _project(tmp_path, "import-export-refusal")
    bars, actions = _csvs(tmp_path, _market())
    assert (
        api.data_import(
            bars=bars,
            actions=actions,
            source="Source",
            license="Publication permitted",
            adjustment="raw",
            project=project,
        ).status
        == "ok"
    )
    assert api.backtest("sma-trend", project=project).status == "ok"
    target = tmp_path / "output"
    target.mkdir()
    (target / "keep").write_text("Keep existing output")
    before = snapshot(tmp_path)
    for tier in ("public", "nda"):
        refused = api.evidence_export("sma-trend", tier=tier, out=target, project=project)
        assert refused.reason_codes == ["DATA_IMPORT_EXPORT_REFUSED"]
        assert snapshot(tmp_path) == before


@pytest.mark.parametrize("adjustment", ["raw", "adjusted"])
def test_import_report_cannot_reuse_a_stronger_claim_from_another_dataset(
    tmp_path: Path, adjustment: str
) -> None:
    project = _project(tmp_path, f"import-report-selection-{adjustment}")
    bars, actions = _csvs(tmp_path, _market())
    assert (
        api.data_import(bars=bars, actions=actions, source="Source", adjustment="raw", project=project).status
        == "ok"
    )
    assert api.spec_freeze("sma-trend", project=project).status == "ok"
    raw_run = api.backtest("sma-trend", project=project)
    evaluation = api.evaluate_command("sma-trend", open_holdout=True, project=project)
    assert evaluation.evidence["claim_level"] == "holdout_passed"
    assert (
        api.data_import(
            bars=bars, no_actions=True, source="Source", adjustment=adjustment, project=project
        ).status
        == "ok"
    )
    adjusted_run = api.backtest("sma-trend", project=project)
    assert adjusted_run.evidence["grade"] == "historical_adjusted"
    # Rows and configuration can be identical; the declarations belong to a different manifest.
    assert raw_run.data["dataset_identity"] == adjusted_run.data["dataset_identity"]
    rendered = api.report("sma-trend", project=project)
    assert rendered.evidence["grade"] == "historical_adjusted"
    assert rendered.evidence["claim_level"] == "in_sample"
    assert rendered.data["evaluation_run"] is None
    text = _artifact(project, rendered, "report").read_text()
    assert "Claims are capped at `in_sample`" in text and "provenance supplied by the user" in text
    previous = api.report("sma-trend", run_id=raw_run.data["run_id"], project=project)
    assert previous.evidence["claim_level"] == "holdout_passed"
