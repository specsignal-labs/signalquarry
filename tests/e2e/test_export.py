# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from importlib import resources
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from signalquarry import api
from signalquarry._internal.paper.journal import Journal
from signalquarry.cli.main import main

SCHEMAS = {
    "strategies/": "strategy-profile",
    "results/": "study-result",
    "forward/": "forward-record",
}


def _sqy(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict]:
    code = main(["--json", *argv])
    return code, json.loads(capsys.readouterr().out)


def _schema_valid(bundle: Path) -> None:
    manifest = json.loads((bundle / "bundle.json").read_text())
    for entry in manifest["files"]:
        for prefix, name in SCHEMAS.items():
            if entry["path"].startswith(prefix) and entry["path"].endswith(".json"):
                schema = json.loads(
                    (
                        resources.files("signalquarry") / "schemas" / "v1" / "showcase" / f"{name}.json"
                    ).read_text()
                )
                errors = list(
                    Draft202012Validator(schema).iter_errors(json.loads((bundle / entry["path"]).read_text()))
                )
                assert not errors, (entry["path"], [e.message for e in errors])


RULES = """
    rules:
      how: Holds the symbol while its close is above the moving average, otherwise cash.
      thesis: Trend persistence may reduce drawdowns compared with holding throughout.
      entry: Close above the average.
      exit: Close below the average.
      risks: [Whipsaw in sideways markets costs money and time.]
      leverage: None
      execution: Next open, whole shares
      costs: 5 bps per side
"""


@pytest.fixture
def lab(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    project = tmp_path / "export-lab"
    _sqy(capsys, "init", str(project), "--demo", "--package", "export_lab")
    assert _sqy(capsys, "backtest", "--strategy", "sma-trend", "--project", str(project))[0] == 0
    _sqy(capsys, "evaluate", "--strategy", "sma-trend", "--project", str(project))
    return project


def _publication(project: Path, body: str) -> None:
    (project / "publication").mkdir(exist_ok=True)
    (project / "publication" / "sma-trend.publication.yaml").write_text(body)


def test_default_deny_and_caps(lab: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = _sqy(capsys, "evidence", "export", "--family", "sma-trend", "--project", str(lab))
    assert (code, payload["reason_codes"]) == (65, ["PUBLICATION_NOT_FOUND"])
    _publication(
        lab,
        "schema: signalquarry.publication/v1\nfamily: sma-trend\nfamily_label: Trend\ntier: rules\nstrategies: [{id: sma-trend, title: SMA trend demo, summary: A trend filter demo strategy used to test exports end to end., assets: SYNA or cash}]\n",
    )
    code, payload = _sqy(capsys, "evidence", "export", "--family", "sma-trend", "--project", str(lab))
    assert (
        payload["reason_codes"] == ["PUBLICATION_INVALID"]
        and "COMMERCIAL_REQUIRES_CATEGORY" in payload["summary"]
    )


def test_rules_tier_export_with_backtest_results(lab: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _publication(
        lab,
        "schema: signalquarry.publication/v1\nfamily: sma-trend\nfamily_label: Trend following\ncommercial: false\n"
        "tier: rules\nbacktest_results: allow\nstrategies:\n  - id: sma-trend\n    title: SMA trend demo\n"
        "    summary: A trend filter demo strategy used to test exports end to end.\n    assets: SYNA or cash\n"
        + RULES,
    )
    code, payload = _sqy(capsys, "evidence", "export", "--family", "sma-trend", "--project", str(lab))
    assert code == 0, payload
    bundle = Path(payload["data"]["path"])
    _schema_valid(bundle)
    profile = json.loads((bundle / "strategies/sma-trend/profile.json").read_text())
    assert profile["disclosure_tier"] == "rules" and profile["rules"]["exit"] == "Close below the average."
    assert profile["claim_level"] == "none" and profile["evidence_grade"] == "synthetic"
    results = list((bundle / "results").glob("*.json"))
    assert len(results) == 1
    result = json.loads(results[0].read_text())
    assert result["evidence"] == "diagnostic" and "bps" in result["assumptions"]  # synthetic data
    assert _sqy(capsys, "evidence", "verify", "--bundle", str(bundle))[0] == 0
    code, second = _sqy(capsys, "evidence", "export", "--family", "sma-trend", "--project", str(lab))
    assert second["data"]["supersedes"] == payload["data"]["bundle_hash"]


def test_commercial_export_publishes_lagged_returns_only(
    lab: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _publication(
        lab,
        "schema: signalquarry.publication/v1\nfamily: sma-trend\nfamily_label: Trend following\n"
        "forward: {resolution: weekly, lag_days: 14}\nstrategies:\n  - id: sma-trend\n    title: Trend strategy A\n"
        "    summary: A commercial trend strategy under forward paper test, published at the category tier.\n"
        "    assets: US equities\n    deployment: demo\n",
    )
    journal = Journal.open(lab / "paper" / "demo" / "journal.jsonl")
    start = date(2026, 6, 1)
    for offset in range(0, 70):
        day = start + timedelta(days=offset)
        if day.weekday() < 5:
            journal.append(
                "session_started",
                {"session": day, "equity": 100000 + offset * 50},
                now=datetime(2026, 9, 1, tzinfo=UTC),
            )
    today = date(2026, 8, 20)
    envelope = api.evidence_export("sma-trend", project=lab, today=today)
    assert envelope.status == "ok", envelope.as_dict()
    bundle = Path(envelope.data["path"])
    _schema_valid(bundle)
    record = json.loads((bundle / "forward/sma-trend.json").read_text())
    assert record["positions"] == record["fills"] == record["exposure"] == "withheld"
    assert record["weeks"] and all(
        date.fromisoformat(w["week_ending"]) <= today - timedelta(days=14) for w in record["weeks"]
    )
    assert not (bundle / "results").exists()
    profile = json.loads((bundle / "strategies/sma-trend/profile.json").read_text())
    assert (
        profile["commercial"] is True and profile["disclosure_tier"] == "category" and "rules" not in profile
    )
    assert profile["status"] == "forward_paper" and profile["evidence_grade"] == "paper"


def test_nda_tier_is_separated(lab: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _publication(
        lab,
        "schema: signalquarry.publication/v1\nfamily: sma-trend\nfamily_label: Trend following\n"
        "strategies: [{id: sma-trend, title: Trend strategy A, summary: A commercial trend strategy published at the category tier only., assets: US equities}]\n",
    )
    code, payload = _sqy(
        capsys, "evidence", "export", "--family", "sma-trend", "--tier", "nda", "--project", str(lab)
    )
    assert code == 0, payload
    bundle = Path(payload["data"]["path"])
    manifest = json.loads((bundle / "bundle.json").read_text())
    assert any(f["tier"] == "nda" and f["path"].startswith("nda/runs/") for f in manifest["files"])
    assert _sqy(capsys, "evidence", "verify", "--bundle", str(bundle))[0] == 2  # public maximum
    assert _sqy(capsys, "evidence", "verify", "--bundle", str(bundle), "--max-tier", "nda")[0] == 0


def test_perf_capture_chains_and_nda_carries_journal_and_captures(
    lab: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = _sqy(capsys, "perf", "capture", "--alias", "demo", "--project", str(lab))[1]
    second = _sqy(capsys, "perf", "capture", "--alias", "demo", "--project", str(lab))[1]
    assert first["status"] == "ok" and first["data"]["previous_snapshot_hash"] is None
    assert second["data"]["previous_snapshot_hash"] == first["data"]["snapshot_hash"]
    assert second["data"]["captures"] == 2
    journal = Journal.open(lab / "paper" / "demo" / "journal.jsonl")
    journal.append(
        "order_final",
        {"client_order_id": "sq-demo-x-0", "order_id": "broker-123", "orders": [{"order_id": "broker-456"}]},
        now=datetime(2026, 9, 1, tzinfo=UTC),
    )
    _publication(
        lab,
        "schema: signalquarry.publication/v1\nfamily: sma-trend\nfamily_label: Trend following\n"
        "strategies:\n  - id: sma-trend\n    title: Trend strategy A\n"
        "    summary: A commercial trend strategy under forward paper test, published at the category tier.\n"
        "    assets: US equities\n    deployment: demo\n",
    )
    code, payload = _sqy(
        capsys, "evidence", "export", "--family", "sma-trend", "--tier", "nda", "--project", str(lab)
    )
    assert code == 0, payload
    bundle = Path(payload["data"]["path"])
    exported = json.loads((bundle / "nda/paper/demo/journal.json").read_text())
    text = json.dumps(exported)
    assert "broker-123" not in text and "broker-456" not in text and "sq-demo-x-0" in text
    assert exported["head"] == Journal.open(lab / "paper" / "demo" / "journal.jsonl").head
    assert len(list((bundle / "nda/paper/demo/captures").glob("*.json"))) == 2
    assert _sqy(capsys, "evidence", "verify", "--bundle", str(bundle), "--max-tier", "nda")[0] == 0


def test_options_profiles_are_labelled_options(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from signalquarry._internal.publish import export

    project = tmp_path / "wheel-export"
    _sqy(capsys, "init", str(project), "--demo", "--kind", "options", "--package", "wheel_export_lab")
    assert _sqy(capsys, "backtest", "--strategy", "wheel", "--project", str(project))[0] == 0
    _publication_path = project / "publication"
    _publication_path.mkdir()
    (_publication_path / "wheel.publication.yaml").write_text(
        "schema: signalquarry.publication/v1\nfamily: wheel\nfamily_label: Options wheel\n"
        "strategies: [{id: wheel, title: Wheel strategy A, summary: A commercial options wheel published at the category tier only., assets: QQQ options}]\n"
    )
    code, payload = _sqy(capsys, "evidence", "export", "--family", "wheel", "--project", str(project))
    assert code == 0, payload
    profile = json.loads((Path(payload["data"]["path"]) / "strategies/wheel/profile.json").read_text())
    assert profile["asset_class"] == "options"  # the publication file said nothing; the spec decides
    _schema_valid(Path(payload["data"]["path"]))

    original = export._claim
    export._claim = lambda *args: ("walk_forward", "low_evidence_options")
    try:
        code, payload = _sqy(capsys, "evidence", "export", "--family", "wheel", "--project", str(project))
    finally:
        export._claim = original
    profile = json.loads((Path(payload["data"]["path"]) / "strategies/wheel/profile.json").read_text())
    assert profile["evidence_grade"] == "low_evidence_options" and profile["claim_level"] == "walk_forward"


def test_options_results_are_option_proxies(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from signalquarry._internal.contracts.spec import StrategySpecV1
    from signalquarry._internal.publish import export

    project = tmp_path / "wheel-results"
    _sqy(capsys, "init", str(project), "--demo", "--kind", "options", "--package", "wheel_results_lab")
    assert _sqy(capsys, "backtest", "--strategy", "wheel", "--project", str(project))[0] == 0
    spec_path = next(project.glob("src/*/wheel/strategy.yaml"))
    import yaml

    spec = StrategySpecV1.model_validate(yaml.safe_load(spec_path.read_text()))
    result = export._result(project, "wheel", spec)
    assert result is not None and result["evidence"] == "diagnostic"  # synthetic demo data
    run = next(project.glob(".signalquarry/runs/*/result.json"))
    document = json.loads(run.read_text())
    document["evidence"]["grade"] = "low_evidence_options"
    run.write_text(json.dumps(document))
    result = export._result(project, "wheel", spec)
    assert result is not None and result["evidence"] == "option_proxy"
    assert "/contract fee" in result["assumptions"] and "modelled option prices" in result["assumptions"]
