# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from signalquarry._internal.data.alpaca import AlpacaDataClient, RateLimiter
from signalquarry._internal.validation.evaluate import add_months
from signalquarry.api import (
    backtest,
    data_fetch,
    evaluate_command,
    holdout_status,
    init,
    spec_freeze,
    trials_extend,
    trials_ls,
)
from tests.fakes import FakeAlpaca
from tests.helpers import dataset, weekdays

END = date(2024, 12, 31)


def _trending_spy() -> object:
    sessions = weekdays(date(2015, 1, 5), 2600)
    rng = np.random.Generator(np.random.PCG64(3))
    closes = 100 * np.cumprod(1 + 0.0012 + rng.normal(0, 0.004, len(sessions)))
    opens = np.concatenate(([100.0], closes[:-1]))
    return dataset(sessions, {"SPY": {"open": list(np.round(opens, 2)), "close": list(np.round(closes, 2))}})


def _project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, *, budget: int = 50) -> Path:
    monkeypatch.setenv("SIGNALQUARRY_CACHE_DIR", str(tmp_path / "cache"))
    project = tmp_path / name
    assert init(project, package=name.replace("-", "_")).status == "ok"
    spec_path = project / f"src/{name.replace('-', '_')}/sma_trend/strategy.yaml"
    text = spec_path.read_text().replace(
        "sizing: whole_shares", "sizing: fractional\n  rebalance: every_decision"
    )
    spec_path.write_text(
        text.replace("trial_budget: 50", f"trial_budget: {budget}").replace("period: 200", "period: 50")
    )
    client = AlpacaDataClient(
        "k",
        "s",
        transport=FakeAlpaca(_trending_spy(), page_size=5000),
        limiter=RateLimiter(sleep=lambda s: None),
    )
    assert data_fetch(strategy_id="sma-trend", project=project, client=client, end=END).status == "ok"
    return project


def test_unfrozen_evaluation_is_capped_and_recorded_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path, monkeypatch, "eval-lab-one")
    first = evaluate_command("sma-trend", project=project)
    assert first.evidence["claim_level"] == "in_sample" and "FREEZE_REQUIRED" in first.warnings
    assert first.data["gates"]["G2_walk_forward"]["folds"] >= 6
    low, high = first.data["oos"]["sharpe_annual_90"]
    assert low < high  # a 90% bootstrap interval for the out-of-sample annual Sharpe
    evaluate_command("sma-trend", project=project)
    assert trials_ls(project=project).data["summary"]["project_count"] == 1  # same configuration counted once


def test_freeze_seal_clip_gates_and_single_holdout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _project(tmp_path, monkeypatch, "eval-lab-two")
    assert evaluate_command("sma-trend", project=project, open_holdout=True).reason_codes == [
        "FREEZE_REQUIRED"
    ]
    frozen = spec_freeze("sma-trend", project=project)
    seal = date.fromisoformat(frozen.data["holdout_start"])
    last_session = _trending_spy().sessions[-1]
    assert frozen.status == "ok" and seal == add_months(last_session, -12) + timedelta(days=1)
    run = backtest("sma-trend", project=project)
    assert "HOLDOUT_CLIPPED" in run.warnings and date.fromisoformat(run.metrics["end"]) < seal

    walk = evaluate_command("sma-trend", project=project)
    assert walk.status == "ok", walk.data["gates"]
    assert walk.evidence["claim_level"] == "walk_forward" and walk.evidence["holdout"] == "sealed"
    opened = evaluate_command("sma-trend", project=project, open_holdout=True)
    assert opened.status == "ok", opened.data["gates"]
    assert opened.evidence["claim_level"] == "holdout_passed" and opened.evidence["holdout"] == "opened"
    again = evaluate_command("sma-trend", project=project, open_holdout=True)
    assert again.status == "blocked" and again.reason_codes == ["HOLDOUT_REUSED"]
    assert holdout_status(project=project).data["families"][0]["state"] == "opened"


def test_trial_budget_blocks_new_configurations_until_a_human_extends_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path, monkeypatch, "eval-lab-three", budget=1)
    evaluate_command("sma-trend", project=project)
    spec_path = project / "src/eval_lab_three/sma_trend/strategy.yaml"
    spec_path.write_text(spec_path.read_text().replace("period: 50", "period: 60"))
    blocked = evaluate_command("sma-trend", project=project)
    assert blocked.status == "blocked" and blocked.reason_codes == ["TRIAL_BUDGET_EXHAUSTED"]
    assert (
        trials_extend(
            "sma-trend", "Human reviewed the first trial and approved more.", project=project
        ).status
        == "ok"
    )
    assert evaluate_command("sma-trend", project=project).reason_codes != ["TRIAL_BUDGET_EXHAUSTED"]
    assert trials_ls(project=project).data["summary"]["project_count"] == 2


def test_edited_ledger_is_detected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _project(tmp_path, monkeypatch, "eval-lab-four")
    evaluate_command("sma-trend", project=project)
    path = project / "evidence/trials.jsonl"
    entry = json.loads(path.read_text().splitlines()[0])
    entry["sharpe"] = 9.9
    path.write_text(json.dumps(entry) + "\n")
    assert trials_ls(project=project).reason_codes == ["EVIDENCE_LOG_CORRUPT"]


def test_synthetic_evaluation_never_raises_the_claim(tmp_path: Path) -> None:
    project = tmp_path / "demo-eval"
    init(project, demo=True, package="demo_eval_lab")
    spec_freeze("sma-trend", project=project)
    result = evaluate_command("sma-trend", project=project)
    assert result.evidence["grade"] == "synthetic" and result.evidence["claim_level"] == "none"
    assert trials_ls(project=project).data["summary"]["project_count"] == 0  # synthetic runs are not trials


def test_budget_warning_at_eighty_percent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _project(tmp_path, monkeypatch, "eval-lab-budget", budget=5)
    spec_path = project / "src/eval_lab_budget/sma_trend/strategy.yaml"
    seen = []
    for period in (50, 60, 70, 80):
        spec_path.write_text(spec_path.read_text().replace("period: 50", f"period: {period}"))
        run = backtest("sma-trend", project=project)
        seen.append([w for w in run.warnings if w.startswith("TRIAL_BUDGET_NEARLY_USED")])
        spec_path.write_text(spec_path.read_text().replace(f"period: {period}", "period: 50"))
    assert seen[:3] == [[], [], []] and seen[3] == ["TRIAL_BUDGET_NEARLY_USED:4/5"]
    walk = evaluate_command("sma-trend", project=project)
    assert "TRIAL_BUDGET_NEARLY_USED:4/5" in walk.warnings


def test_training_cutoff_moves_the_holdout_earlier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _project(tmp_path, monkeypatch, "eval-lab-cutoff")
    spec_path = project / "src/eval_lab_cutoff/sma_trend/strategy.yaml"
    last_session = _trending_spy().sessions[-1]
    cutoff = add_months(last_session, -30)
    text = spec_path.read_text()
    assert "holdout: {months: 12}" in text
    spec_path.write_text(
        text.replace(
            "holdout: {months: 12}", f"holdout: {{months: 12, training_cutoff: {cutoff.isoformat()}}}"
        )
    )
    frozen = spec_freeze("sma-trend", project=project)
    assert frozen.status == "ok", frozen.summary
    assert date.fromisoformat(frozen.data["holdout_start"]) == cutoff + timedelta(days=1)
