# SPDX-License-Identifier: Apache-2.0
"""Research studies (ADR 0015): the acceptance cases, end to end."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from signalquarry._internal.engine.backtest import EngineError
from signalquarry._internal.evidence.runs import iter_runs, read_curve
from signalquarry._internal.validation import ledger
from signalquarry.api import (
    evaluate_command,
    evidence_verify,
    init,
    spec_freeze,
    study_check,
    study_init,
    study_ls,
    study_run,
    study_show,
)
from signalquarry.cli.main import main

STUDY = """\
schema: signalquarry.study/v1
id: trend
hypothesis:
  statement: Holding SYNA only above its 200-session average lowers drawdown versus buy-and-hold.
  falsification: Max drawdown over the same sessions is not lower than buy-and-hold.
base: sma-trend
baselines:
  - {id: buy-and-hold, kind: benchmark}
  - {id: vol-matched, kind: benchmark_scaled}
variants:
  - {id: costs-x2, execution: {costs: {bps: "10"}}, role: sensitivity}
  - {id: half-weight, params: {weight: "0.5"}, role: ablation, note: Half the exposure.}
grid: {period: [100, 150]}
compare: {metric: max_drawdown, direction: lower, versus: buy-and-hold}
"""
ARMS = [
    "base",
    "costs-x2",
    "half-weight",
    "grid-period-100",
    "grid-period-150",
    "buy-and-hold",
    "vol-matched",
]
COUNTED = {"base", "half-weight", "grid-period-100", "grid-period-150"}


def _project(tmp_path: Path, name: str, study: str = STUDY) -> Path:
    project = tmp_path / name
    assert init(project, demo=True, package=name.replace("-", "_")).status == "ok"
    (project / "studies" / "trend").mkdir(parents=True)
    (project / "studies" / "trend" / "study.yaml").write_text(study)
    return project


@pytest.fixture
def historical(monkeypatch: pytest.MonkeyPatch) -> None:
    """Treat the demo dataset as recorded history, so trials, budgets and logs are live."""
    study_api = sys.modules["signalquarry.api.study"]
    real = study_api.resolve

    def resolve(command, strategy_id, project):
        resolved = real(command, strategy_id, project)
        return (
            resolved
            if not hasattr(resolved, "grade")
            else replace(resolved, grade="historical", dataset_id="test-dataset")
        )

    monkeypatch.setattr(study_api, "resolve", resolve)
    resolve_api = sys.modules["signalquarry.api.resolve"]
    plain = resolve_api.resolve

    def also(command, strategy_id, project):
        resolved = plain(command, strategy_id, project)
        return (
            resolved
            if not hasattr(resolved, "grade")
            else replace(resolved, grade="historical", dataset_id="test-dataset")
        )

    monkeypatch.setattr(resolve_api, "resolve", also)
    monkeypatch.setattr(sys.modules["signalquarry.api.evidence"], "resolve", also)


def test_check_lists_the_arms_and_runs_nothing(tmp_path: Path) -> None:
    project = _project(tmp_path, "study-check")
    envelope = study_check("trend", project=project)
    assert envelope.status == "ok", envelope.summary
    arms = envelope.data["arms"]
    assert [arm["id"] for arm in arms] == ARMS
    assert {arm["id"] for arm in arms if arm["counts"]} == COUNTED
    assert [arm["kind"] for arm in arms][-2:] == ["benchmark", "benchmark_scaled"]
    assert arms[-1]["symbol"] == "SYNA" and arms[0]["family"] == "sma-trend"
    assert len({arm["configuration_hash"] for arm in arms[:5]}) == 5
    assert all(arm["existing_run"] is None for arm in arms)
    assert envelope.data["trials"] == {} and "synthetic data records none" in envelope.summary
    assert envelope.data["compare"] == {
        "metric": "max_drawdown",
        "direction": "lower",
        "versus": "buy-and-hold",
    }
    assert not (project / ".signalquarry").exists() and not (project / "evidence" / "studies.jsonl").exists()


def test_run_compares_every_arm_on_the_same_sessions(tmp_path: Path) -> None:
    project = _project(tmp_path, "study-run")
    envelope = study_run("trend", project=project)
    assert envelope.status == "ok", envelope.summary
    data = envelope.data
    assert [arm["id"] for arm in data["arms"]] == ARMS
    assert data["comparable"] is True and envelope.warnings == []
    assert envelope.evidence == {"grade": "synthetic", "claim_level": "none"}

    # 1. Comparable by construction: one dataset, one set of sessions.
    runs = {document["run_id"]: (directory, document) for directory, document in iter_runs(project)}
    sessions = set()
    for arm in data["arms"][:5]:
        directory, document = runs[arm["run_id"]]
        assert document["command"] == "study" and document["label"] == arm["id"]
        assert document["dataset_identity"] == runs[data["arms"][0]["run_id"]][1]["dataset_identity"]
        sessions.add(tuple(read_curve(directory / "equity.csv")[0]))
    assert len(sessions) == 1

    # 7. The verdict follows the declared rule: here the filter's drawdown is not lower.
    outcome = data["verdict"]
    assert (outcome["metric"], outcome["direction"]) == ("max_drawdown", "lower")
    assert outcome["subject"] == data["arms"][0]["max_drawdown"]
    assert outcome["baseline"] == data["arms"][5]["max_drawdown"]
    assert outcome["difference"] > 0 and outcome["outcome"] == "not_supported"
    assert "not_supported" in envelope.summary
    assert set(data["pairs"]) == set(ARMS) - {"buy-and-hold"}
    assert data["pbo"]["configurations"] == 4  # the subject, the ablation and the two grid points

    directory = project / ".signalquarry" / "studies" / data["study_run_id"]
    assert {p.name for p in directory.iterdir()} == {"study.json", "comparison.md", "equity.svg"}
    result = json.loads((directory / "study.json").read_text())
    assert result["schema"] == "signalquarry.study-result/v1" and result["verdict"] == outcome
    assert result["arms"][1]["changes"] == {"params": {}, "spec": {"execution.costs.bps": ["5", "10"]}}
    assert result["arms"][2]["changes"]["params"] == {"weight": ["0.95", "0.5"]}
    assert result["arms"][2]["note"] == "Half the exposure."
    # The scaled benchmark has the subject's volatility and the benchmark's Sharpe ratio.
    scaled, plain, base = (
        result["arms"][6]["metrics"],
        result["arms"][5]["metrics"],
        result["arms"][0]["metrics"],
    )
    assert scaled["annual_volatility"] == pytest.approx(base["annual_volatility"], rel=1e-3)
    assert scaled["sharpe"] == pytest.approx(plain["sharpe"], abs=0.01)
    text = (directory / "comparison.md").read_text()
    assert "**Not supported.**" in text and "| costs-x2 | sensitivity | no |" in text
    assert "## What each variant changes" in text and "Half the exposure." in text
    assert "Probability of backtest overfitting across the 4 candidate" in text
    assert (directory / "equity.svg").read_text().count("<polyline") == 7
    # Synthetic data is not evidence: no trial and no study record.
    assert not (project / "evidence" / "trials.jsonl").exists()
    assert not (project / "evidence" / "studies.jsonl").exists()


def test_a_second_run_reuses_the_arms_and_rerun_reproduces_them(tmp_path: Path) -> None:
    project = _project(tmp_path, "study-resume")
    first = study_run("trend", project=project)
    count = len(iter_runs(project))
    second = study_run("trend", project=project)
    assert [arm["resumed"] for arm in second.data["arms"]] == [True] * 5 + [False, False]
    assert [arm["run_id"] for arm in second.data["arms"]] == [arm["run_id"] for arm in first.data["arms"]]
    assert second.data["comparison_hash"] == first.data["comparison_hash"]
    assert len(iter_runs(project)) == count
    assert second.data["study_run_id"] != first.data["study_run_id"]
    checked = study_check("trend", project=project)
    assert [arm["existing_run"] for arm in checked.data["arms"][:5]] == [
        arm["run_id"] for arm in first.data["arms"][:5]
    ]

    # 5. Determinism: recomputing every arm gives the ledgers recorded before.
    again = study_run("trend", rerun=True, project=project)
    assert again.status == "ok" and again.data["comparison_hash"] == first.data["comparison_hash"]
    assert [arm["resumed"] for arm in again.data["arms"]] == [False] * 7
    assert len(iter_runs(project)) == count + 5

    # A recorded ledger that the recomputation does not reproduce stops the study.
    directory, document = iter_runs(project)[-1]
    from signalquarry._internal.evidence.runs import result_document

    forged = {k: v for k, v in document.items() if k not in ("created_at", "result_hash", "schema")}
    forged["ledger_hash"] = "sha256:" + "0" * 64
    (directory / "result.json").write_text(result_document(**forged))
    failed = study_run("trend", rerun=True, project=project)
    assert (failed.status, failed.reason_codes) == ("blocked", ["STUDY_DETERMINISM_FAILED"])


def test_on_real_data_only_candidate_arms_are_trials_and_the_study_is_logged(
    tmp_path: Path, historical: None
) -> None:
    project = _project(tmp_path, "study-real")
    checked = study_check("trend", project=project)
    assert checked.data["trials"] == {"sma-trend": {"new": 4, "used": 0, "budget": 50}}
    assert "4 new trial(s)" in checked.summary and not (project / "evidence" / "trials.jsonl").exists()

    envelope = study_run("trend", project=project)
    assert envelope.status == "ok", envelope.summary
    assert envelope.evidence == {"grade": "historical", "claim_level": "in_sample"}
    # 2. One trial per candidate configuration; none for the sensitivity and benchmark arms.
    trials = [entry for entry in ledger.trials(project).entries() if entry.get("kind") == "trial"]
    hashes = {arm["id"]: arm["configuration_hash"] for arm in envelope.data["arms"]}
    assert {entry["configuration_hash"] for entry in trials} == {hashes[arm] for arm in COUNTED}
    assert len(trials) == 4 and {entry["command"] for entry in trials} == {"study"}
    runs = {document["run_id"]: document for _, document in iter_runs(project)}
    recorded = {arm["id"]: runs[arm["run_id"]]["trial_recorded"] for arm in envelope.data["arms"][:5]}
    assert recorded == {
        "base": True,
        "costs-x2": False,
        "half-weight": True,
        "grid-period-100": True,
        "grid-period-150": True,
    }

    log = ledger.studies(project).entries()
    assert [entry["kind"] for entry in log] == ["study_started"] + ["arm_completed"] * 7 + ["study_completed"]
    assert log[0]["study_hash"] == envelope.data["study_hash"] and len(log[0]["arms"]) == 7
    assert [entry["arm"] for entry in log[1:8]] == ARMS[:5] + ARMS[5:]
    assert log[-1]["verdict"] == "not_supported"
    assert log[-1]["comparison_hash"] == envelope.data["comparison_hash"]
    assert evidence_verify(project=project).status == "ok"

    # Running it again records nothing new in the trial ledger.
    study_run("trend", project=project)
    assert len(ledger.trials(project).entries()) == 4
    assert study_check("trend", project=project).data["trials"]["sma-trend"] == {
        "new": 0,
        "used": 4,
        "budget": 50,
    }

    # 8. An edited study log is detected.
    path = project / "evidence" / "studies.jsonl"
    path.write_text(path.read_text().replace("not_supported", "supported", 1))
    assert evidence_verify(project=project).status == "blocked"


def test_a_study_over_budget_is_refused_with_nothing_written(tmp_path: Path, historical: None) -> None:
    project = _project(tmp_path, "study-budget")
    spec = next(project.glob("src/*/sma_trend/strategy.yaml"))
    spec.write_text(spec.read_text().replace("trial_budget: 50", "trial_budget: 3"))
    checked = study_check("trend", project=project)
    assert (checked.status, checked.reason_codes) == ("blocked", ["TRIAL_BUDGET_EXHAUSTED"])
    assert checked.data["trials"] == {"sma-trend": {"new": 4, "used": 0, "budget": 3}}
    envelope = study_run("trend", project=project)
    assert (envelope.status, envelope.reason_codes) == ("blocked", ["TRIAL_BUDGET_EXHAUSTED"])
    assert "4 new trials would exceed family sma-trend's budget (0/3 used)" in envelope.summary
    assert not (project / ".signalquarry").exists()
    assert not (project / "evidence" / "trials.jsonl").exists()
    assert not (project / "evidence" / "studies.jsonl").exists()


def test_an_interrupted_study_resumes_without_counting_twice(
    tmp_path: Path, historical: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    whole = study_run("trend", project=_project(tmp_path, "study-whole"))
    project = _project(tmp_path, "study-crash")
    study_api = sys.modules["signalquarry.api.study"]
    real, calls = study_api.execute_run, []

    def failing(resolved, **options):
        calls.append(options["label"])
        if len(calls) == 3:
            raise EngineError("BACKTEST_RANGE_EMPTY")
        return real(resolved, **options)

    monkeypatch.setattr(study_api, "execute_run", failing)
    broken = study_run("trend", project=project)
    assert (broken.status, broken.reason_codes) == ("invalid", ["BACKTEST_RANGE_EMPTY"])
    assert "arm half-weight" in broken.summary
    log = ledger.studies(project).entries()
    assert [entry["kind"] for entry in log] == [
        "study_started",
        "arm_completed",
        "arm_completed",
        "arm_failed",
    ]
    assert (log[-1]["arm"], log[-1]["reason"]) == ("half-weight", "BACKTEST_RANGE_EMPTY")
    assert len(ledger.trials(project).entries()) == 1  # the subject; the sensitivity arm is not counted
    assert not (project / ".signalquarry" / "studies").exists()

    monkeypatch.setattr(study_api, "execute_run", real)
    resumed = study_run("trend", project=project)
    assert resumed.status == "ok"
    assert [arm["resumed"] for arm in resumed.data["arms"][:5]] == [True, True, False, False, False]
    assert len(ledger.trials(project).entries()) == 4
    # 4. The same answer as the study that was never interrupted.
    assert resumed.data["verdict"] == whole.data["verdict"]
    assert resumed.data["pairs"] == whole.data["pairs"] and resumed.data["pbo"] == whole.data["pbo"]
    assert evidence_verify(project=project).status == "ok"


def test_a_study_never_reaches_a_sealed_holdout_or_changes_a_claim(tmp_path: Path, historical: None) -> None:
    project = _project(tmp_path, "study-seal")
    frozen = spec_freeze("sma-trend", project=project)
    seal = date.fromisoformat(frozen.data["holdout_start"])
    before = evaluate_command("sma-trend", project=project)
    envelope = study_run("trend", project=project)
    assert envelope.status == "ok" and "HOLDOUT_CLIPPED" in envelope.warnings
    # 6. No arm's sessions reach the seal date.
    for directory, document in iter_runs(project):
        if document["command"] == "study":
            assert read_curve(directory / "equity.csv")[0][-1] < seal
            assert document["window"]["end"] < seal.isoformat()
    checked = study_check("trend", project=project)
    assert checked.data["holdout_clipped"] is True and checked.warnings == ["HOLDOUT_CLIPPED"]
    # 9. The strategy's claim is what it was.
    after = evaluate_command("sma-trend", project=project)
    assert after.evidence["claim_level"] == before.evidence["claim_level"]
    assert after.data["gates"]["G1_sample"] == before.data["gates"]["G1_sample"]
    assert after.data["gates"]["G3_stress"] == before.data["gates"]["G3_stress"]
    # The study's extra configurations are priced, as they should be: the same out-of-sample
    # record now has to clear a deflated Sharpe ratio computed over more trials.
    was, now = before.data["gates"]["G2_walk_forward"], after.data["gates"]["G2_walk_forward"]
    assert now["psr"] == was["psr"] and now["dsr"] < was["dsr"]
    assert after.data["oos"]["project_trials"] == before.data["oos"]["project_trials"] + 3
    assert ledger.holdout_opened(project, "sma-trend") is None


def test_a_strategy_baseline_runs_on_the_study_dataset_and_counts_in_its_own_family(
    tmp_path: Path, historical: None
) -> None:
    project = _project(tmp_path, "study-other")
    package = project / "src" / "study_other"
    shutil.copytree(package / "sma_trend", package / "sma_fast")
    spec = package / "sma_fast" / "strategy.yaml"
    spec.write_text(
        spec.read_text()
        .replace("id: sma-trend", "id: sma-fast")
        .replace("family: sma-trend", "family: sma-fast")
        .replace("period: 200", "period: 50")
    )
    config = project / "signalquarry.toml"
    config.write_text(
        config.read_text().replace(
            '"study_other.sma_trend.strategy"',
            '"study_other.sma_trend.strategy", "study_other.sma_fast.strategy"',
        )
    )
    study = project / "studies" / "trend" / "study.yaml"
    study.write_text(
        STUDY.replace(
            "  - {id: vol-matched, kind: benchmark_scaled}\n",
            "  - {id: faster, kind: strategy, strategy: sma-fast}\n",
        )
        .replace("versus: buy-and-hold", "versus: faster")
        .replace("metric: max_drawdown, direction: lower", "metric: sharpe, direction: higher")
    )
    checked = study_check("trend", project=project)
    assert checked.status == "ok", checked.summary
    assert checked.data["trials"] == {
        "sma-fast": {"new": 1, "used": 0, "budget": 50},
        "sma-trend": {"new": 4, "used": 0, "budget": 50},
    }
    envelope = study_run("trend", project=project)
    assert envelope.status == "ok", envelope.summary
    assert envelope.data["comparable"] is True
    assert len(ledger.trials(project, "sma-fast").entries()) == 5  # one log in the project layout
    families = [entry["family"] for entry in ledger.trials(project).entries()]
    assert families.count("sma-fast") == 1 and families.count("sma-trend") == 4
    assert envelope.data["verdict"]["metric"] == "sharpe"
    assert envelope.data["verdict"]["baseline"] == next(
        arm["sharpe"] for arm in envelope.data["arms"] if arm["id"] == "faster"
    )


@pytest.mark.parametrize(
    ("change", "code"),
    [
        (lambda text: text.replace("id: trend\n", "id: other\n"), "STUDY_SPEC_INVALID"),
        (lambda text: text.replace("versus: buy-and-hold", "versus: nobody"), "STUDY_SPEC_INVALID"),
        (lambda text: text + "surprise: 1\n", "STUDY_SPEC_INVALID"),
        (lambda text: "- not\n- a mapping\n", "STUDY_SPEC_INVALID"),
        (lambda text: "schema: [unclosed\n", "STUDY_SPEC_INVALID"),
        (lambda text: text.replace('weight: "0.5"', 'weight: "7"'), "STUDY_ARM_INVALID"),
        (lambda text: text.replace('params: {weight: "0.5"}', "params: {nope: 1}"), "STUDY_ARM_INVALID"),
        (lambda text: text.replace('bps: "10"', 'bps: "-1"'), "STUDY_ARM_INVALID"),
        (lambda text: text.replace("period: [100, 150]", "period: [100, 200]"), "STUDY_ARM_INVALID"),
        (lambda text: text.replace("base: sma-trend", "base: missing"), "STRATEGY_NOT_FOUND"),
        (
            lambda text: text.replace("base: sma-trend\n", "base: sma-trend\ndataset: another-dataset\n"),
            "STUDY_DATASET_MISMATCH",
        ),
        (
            lambda text: text.replace(
                "{id: vol-matched, kind: benchmark_scaled}", "{id: other, kind: strategy, strategy: missing}"
            ),
            "STRATEGY_NOT_FOUND",
        ),
    ],
)
def test_invalid_studies_are_refused_before_anything_runs(tmp_path: Path, change, code: str) -> None:
    project = _project(tmp_path, "study-bad", change(STUDY))
    for command in (study_check, study_run):
        envelope = command("trend", project=project)
        assert envelope.reason_codes == [code], envelope.summary
        assert envelope.status == "invalid"
    assert not (project / ".signalquarry").exists()


def test_a_benchmark_baseline_needs_a_symbol(tmp_path: Path) -> None:
    project = _project(tmp_path, "study-nobench")
    spec = next(project.glob("src/*/sma_trend/strategy.yaml"))
    spec.write_text(spec.read_text().replace("benchmark: SYNA\n", ""))
    envelope = study_check("trend", project=project)
    assert envelope.reason_codes == ["STUDY_ARM_INVALID"] and "buy-and-hold" in envelope.summary
    study = project / "studies" / "trend" / "study.yaml"
    study.write_text(
        STUDY.replace(
            "{id: buy-and-hold, kind: benchmark}", "{id: buy-and-hold, kind: benchmark, symbol: SYNB}"
        ).replace(
            "{id: vol-matched, kind: benchmark_scaled}",
            "{id: vol-matched, kind: benchmark_scaled, symbol: SYNB}",
        )
    )
    checked = study_check("trend", project=project)
    assert checked.status == "ok" and [arm["symbol"] for arm in checked.data["arms"][-2:]] == ["SYNB", "SYNB"]
    assert study_run("trend", project=project).status == "ok"


def test_init_ls_and_show(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    project = tmp_path / "study-cli"
    assert init(project, demo=True, package="study_cli").status == "ok"

    def sqy(*argv: str) -> tuple[int, dict]:
        code = main(["--json", *argv, "--project", str(project)])
        return code, json.loads(capsys.readouterr().out)

    code, listing = sqy("study", "ls")
    assert code == 0 and listing["data"]["studies"] == []
    code, missing = sqy("study", "show", "--study", "first")
    assert (code, missing["reason_codes"]) == (65, ["STUDY_NOT_FOUND"])
    code, missing = sqy("study", "run", "--study", "first")
    assert (code, missing["reason_codes"]) == (65, ["STUDY_NOT_FOUND"])

    code, created = sqy("study", "init", "--strategy", "sma-trend", "--id", "first")
    assert code == 0 and created["data"]["path"] == "studies/first/study.yaml"
    text = (project / "studies" / "first" / "study.yaml").read_text()
    assert "base: sma-trend" in text and "versus: buy-and-hold" in text
    code, again = sqy("study", "init", "--strategy", "sma-trend", "--id", "first")
    assert (code, again["reason_codes"]) == (65, ["STUDY_EXISTS"])
    code, bad = sqy("study", "init", "--strategy", "sma-trend", "--id", "Not A Slug")
    assert (code, bad["reason_codes"]) == (64, ["USAGE_INVALID"])
    code, unknown = sqy("study", "init", "--strategy", "nope", "--id", "second")
    assert (code, unknown["reason_codes"]) == (65, ["STRATEGY_NOT_FOUND"])

    code, unrun = sqy("study", "show", "--study", "first")
    assert (code, unrun["reason_codes"]) == (65, ["STUDY_NO_RESULT"])
    code, checked = sqy("study", "check", "--study", "first")
    assert code == 0 and [arm["id"] for arm in checked["data"]["arms"]] == [
        "base",
        "costs-x2",
        "buy-and-hold",
    ]
    code, ran = sqy("study", "run", "--study", "first")
    assert code == 0, ran
    code, shown = sqy("study", "show", "--study", "first")
    assert code == 0 and shown["data"]["study_run_id"] == ran["data"]["study_run_id"]
    assert shown["data"]["result"]["verdict"] == ran["data"]["verdict"] and shown["warnings"] == []
    assert {a["kind"] for a in shown["artifacts"]} == {"study", "comparison", "equity"}
    code, listing = sqy("study", "ls")
    (row,) = listing["data"]["studies"]
    assert (row["id"], row["valid"], row["runs"], row["arms"]) == ("first", True, 1, 3)
    assert row["verdict"] == ran["data"]["verdict"]["outcome"]

    # Editing the file makes a different study: the old result is shown as such, and not counted.
    path = project / "studies" / "first" / "study.yaml"
    path.write_text(path.read_text().replace("metric: sharpe", "metric: sortino"))
    code, shown = sqy("study", "show", "--study", "first")
    assert shown["warnings"] == ["STUDY_FILE_CHANGED"]
    code, listing = sqy("study", "ls")
    assert listing["data"]["studies"][0]["runs"] == 0 and listing["data"]["studies"][0]["verdict"] is None
    path.write_text("nonsense: true\n")
    code, listing = sqy("study", "ls")
    assert listing["data"]["studies"][0]["valid"] is False

    assert study_ls(project=tmp_path).reason_codes == ["PROJECT_NOT_FOUND"]
    assert study_show("x", project=tmp_path).reason_codes == ["PROJECT_NOT_FOUND"]
    assert study_init("sma-trend", "x", project=tmp_path).reason_codes == ["PROJECT_NOT_FOUND"]
    assert study_check("x", project=tmp_path).reason_codes == ["PROJECT_NOT_FOUND"]
