# SPDX-License-Identifier: Apache-2.0
"""``--jobs``: worker processes simulate, one process records, and the record is the same."""

from __future__ import annotations

import json
import sys
from concurrent.futures import Future
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from signalquarry import api
from signalquarry._internal.data.alpaca import AlpacaDataClient, RateLimiter
from signalquarry._internal.evidence.runs import iter_runs
from signalquarry._internal.validation import ledger
from signalquarry.api import data_fetch, evidence_verify, init, study_run, sweep
from signalquarry.api.envelope import Envelope
from signalquarry.api.runs import Simulated
from signalquarry.cli.main import main
from tests.fakes import FakeAlpaca
from tests.helpers import dataset, weekdays

END = date(2024, 12, 31)
STUDY = """\
schema: signalquarry.study/v1
id: trend
hypothesis:
  statement: Holding the symbol only above its 200-session average lowers drawdown versus buy-and-hold.
  falsification: Max drawdown over the same sessions is not lower than buy-and-hold.
base: sma-trend
baselines:
  - {id: buy-and-hold, kind: benchmark}
variants:
  - {id: costs-x2, execution: {costs: {bps: "10"}}, role: sensitivity}
grid: {period: [100, 150, 250]}
compare: {metric: max_drawdown, direction: lower, versus: buy-and-hold}
"""
ARMS = ["base", "costs-x2", "grid-period-100", "grid-period-150", "grid-period-250", "buy-and-hold"]
COUNTED = ["base", "grid-period-100", "grid-period-150", "grid-period-250"]
# A strategy whose 150-session variant fails in the engine, so one arm fails and the others do not.
FAILS_AT_150 = """\
    if p.period == 150:
        return Decision.target({}, "NOT_DECLARED")
    close = ctx.bars(p.symbol).close
"""


def _study(project: Path) -> Path:
    (project / "studies" / "trend").mkdir(parents=True)
    (project / "studies" / "trend" / "study.yaml").write_text(STUDY)
    return project


def _demo(tmp_path: Path, name: str, *, failing: bool = False) -> Path:
    project = tmp_path / name
    assert init(project, demo=True, package=name.replace("-", "_")).status == "ok"
    if failing:
        code = next(project.glob("src/*/sma_trend/strategy.py"))
        code.write_text(code.read_text().replace("    close = ctx.bars(p.symbol).close\n", FAILS_AT_150))
    return _study(project)


def _recorded(tmp_path: Path, name: str) -> Path:
    """A project on recorded data that a spawned worker reads from the files, as a user's would be."""
    sessions = weekdays(date(2015, 1, 5), 2600)
    closes = 100 * np.cumprod(1 + 0.0006 + np.random.Generator(np.random.PCG64(3)).normal(0, 0.006, 2600))
    opens = np.concatenate(([100.0], closes[:-1]))
    market = dataset(
        sessions, {"SPY": {"open": list(np.round(opens, 2)), "close": list(np.round(closes, 2))}}
    )
    project = tmp_path / name
    assert init(project, package=name.replace("-", "_")).status == "ok"
    client = AlpacaDataClient(
        "k", "s", transport=FakeAlpaca(market, page_size=5000), limiter=RateLimiter(sleep=lambda s: None)
    )
    assert data_fetch(strategy_id="sma-trend", project=project, client=client, end=END).status == "ok"
    return _study(project)


def _labels(project: Path) -> list[str]:
    return [document["label"] for _, document in iter_runs(project)]


def _ledgers(project: Path) -> dict[str, str]:
    return {document["label"]: document["ledger_hash"] for _, document in iter_runs(project)}


def _trials(project: Path) -> list[dict[str, Any]]:
    """The trial ledger without what differs between two projects: time, chain and code location."""
    volatile = {"at", "hash", "prev", "configuration_hash"}
    return [
        {key: value for key, value in entry.items() if key not in volatile}
        for entry in ledger.trials(project).entries()
    ]


def test_worker_processes_record_what_one_process_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SIGNALQUARRY_CACHE_DIR", str(tmp_path / "cache"))
    serial, pooled = _recorded(tmp_path, "jobs-serial"), _recorded(tmp_path, "jobs-pooled")
    one = study_run("trend", project=serial)
    many = study_run("trend", jobs=3, project=pooled)
    assert one.status == "ok" and many.status == "ok", (one.summary, many.summary)
    assert many.evidence == {"grade": "historical", "claim_level": "in_sample"}
    assert [arm["id"] for arm in many.data["arms"]] == ARMS
    assert not any(arm["resumed"] for arm in many.data["arms"])

    # The same answer, the same simulations and the same record, arm for arm.
    for key in ("verdict", "pairs", "pbo", "trials"):
        assert many.data[key] == one.data[key], key
    assert _ledgers(pooled) == _ledgers(serial) and list(_ledgers(pooled)) == ARMS[:5]
    assert _trials(pooled) == _trials(serial) and len(_trials(pooled)) == 4
    hashes = {arm["id"]: arm["configuration_hash"] for arm in many.data["arms"]}
    recorded = [entry["configuration_hash"] for entry in ledger.trials(pooled).entries()]
    assert recorded == [hashes[arm] for arm in COUNTED]  # appended in the declared order
    runs = {document["label"]: document for _, document in iter_runs(pooled)}
    assert {label: run["trial_recorded"] for label, run in runs.items()} == {
        arm: arm in COUNTED for arm in ARMS[:5]
    }
    log = ledger.studies(pooled).entries()
    assert [entry["kind"] for entry in log] == ["study_started"] + ["arm_completed"] * 6 + ["study_completed"]
    assert [entry["arm"] for entry in log[1:7]] == ARMS
    assert evidence_verify(project=pooled).status == "ok"

    # Nothing left to simulate: every arm is reused and no trial is added.
    again = study_run("trend", jobs=3, project=pooled)
    assert [arm["resumed"] for arm in again.data["arms"]] == [True] * 5 + [False]
    assert again.data["comparison_hash"] == many.data["comparison_hash"]
    # Recomputing in workers reproduces every ledger recorded before.
    redone = study_run("trend", rerun=True, jobs=2, project=pooled)
    assert redone.status == "ok" and redone.data["comparison_hash"] == many.data["comparison_hash"]
    assert len(ledger.trials(pooled).entries()) == 4 and len(iter_runs(pooled)) == 10
    assert evidence_verify(project=pooled).status == "ok"


def test_a_failing_arm_stops_the_study_where_one_process_would_stop(tmp_path: Path) -> None:
    serial = _demo(tmp_path, "jobs-fail-serial", failing=True)
    pooled = _demo(tmp_path, "jobs-fail-pooled", failing=True)
    one = study_run("trend", project=serial)
    many = study_run("trend", jobs=4, project=pooled)
    for envelope in (one, many):
        assert (envelope.status, envelope.reason_codes) == ("invalid", ["REASON_CODE_UNDECLARED"])
        assert "arm grid-period-150" in envelope.summary
    # The arms before the failing one are written; the one after it was simulated and discarded.
    assert _labels(pooled) == _labels(serial) == ["base", "costs-x2", "grid-period-100"]
    assert not (pooled / ".signalquarry" / "studies").exists()


def test_sweep_points_from_workers_match_the_serial_sweep(tmp_path: Path) -> None:
    project = _demo(tmp_path, "jobs-sweep")

    def points(envelope: Envelope) -> tuple[list[dict[str, Any]], Any]:
        path = next(item["path"] for item in envelope.artifacts if item["path"].endswith("sweep.json"))
        document = json.loads((project / path).read_text())
        return [{k: v for k, v in point.items() if k != "run_id"} for point in document["points"]], document[
            "pbo"
        ]

    one = sweep("sma-trend", ["period=100,150,200", "weight=0.5,0.9"], project=project)
    many = sweep("sma-trend", ["period=100,150,200", "weight=0.5,0.9"], jobs=3, project=project)
    assert one.status == "ok" and many.status == "ok", (one.summary, many.summary)
    assert points(many) == points(one) and len(points(many)[0]) == 6
    runs = iter_runs(project)
    assert len(runs) == 12
    assert [document["ledger_hash"] for _, document in runs[6:]] == [
        document["ledger_hash"] for _, document in runs[:6]
    ]

    failing = _demo(tmp_path, "jobs-sweep-fail", failing=True)
    broken = sweep("sma-trend", ["period=100,150,250"], jobs=3, project=failing)
    assert (broken.status, broken.reason_codes) == ("invalid", ["REASON_CODE_UNDECLARED"])
    assert len(iter_runs(failing)) == 1 and not (failing / ".signalquarry" / "sweeps").exists()


def test_a_study_worker_checks_the_project_it_rebuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _demo(tmp_path, "jobs-worker")
    study_api = sys.modules["signalquarry.api.study"]
    monkeypatch.setattr(study_api, "_worker_plans", {})
    plan = study_api._plan("study run", "trend", project)
    arm = next(arm for arm in plan.arms if arm.id == "grid-period-100")
    job = study_api._Job(
        study_id="trend",
        root=plan.root,
        study_hash=plan.study_hash,
        arm=arm.id,
        configuration_hash=arm.configuration_hash,
        dataset_identity=plan.subject.dataset.identity(),
        start=plan.start,
        end=plan.end,
        scratch=tmp_path / "scratch",
    )
    simulated = study_api._simulate_arm(job)
    assert isinstance(simulated, Simulated) and simulated.spool is not None
    assert {path.name for path in job.scratch.iterdir()} == {"decisions.jsonl", "fills.jsonl"}
    assert not (project / ".signalquarry").exists() and not list(project.glob("evidence/*.jsonl"))
    assert study_run("trend", project=project).status == "ok"
    assert _ledgers(project)["grid-period-100"] == simulated.result.ledger_hash

    for change, reason in (
        ({"study_hash": "sha256:" + "0" * 64}, "the study no longer resolves"),
        ({"study_id": "gone"}, "the study no longer resolves"),
        ({"arm": "nobody"}, "its configuration is not the one that was checked"),
        ({"configuration_hash": "sha256:" + "0" * 64}, "its configuration is not the one that was checked"),
        ({"dataset_identity": "sha256:" + "0" * 64}, "the dataset is not the one that was checked"),
    ):
        refused = study_api._simulate_arm(replace(job, scratch=tmp_path / "unused", **change))
        assert isinstance(refused, str) and refused.startswith(reason), change
    assert not (tmp_path / "unused").exists()


def _historical(monkeypatch: pytest.MonkeyPatch) -> None:
    """Treat the demo dataset as recorded history in this process (the workers are stubbed)."""
    for name in ("signalquarry.api.study", "signalquarry.api.sweep"):
        module = sys.modules[name]
        real = module.resolve

        def resolve(command: str, strategy_id: str, project: Path | None, real: Any = real) -> Any:
            resolved = real(command, strategy_id, project)
            return (
                resolved
                if isinstance(resolved, Envelope)
                else replace(resolved, grade="historical", dataset_id="test-dataset")
            )

        monkeypatch.setattr(module, "resolve", resolve)


def _answered(value: Simulated | str) -> Future[Simulated | str]:
    future: Future[Simulated | str] = Future()
    future.set_result(value)
    return future


def test_a_project_that_changed_under_the_workers_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _historical(monkeypatch)
    project = _demo(tmp_path, "jobs-changed")
    study_api = sys.modules["signalquarry.api.study"]
    monkeypatch.setattr(
        study_api,
        "_submit",
        lambda stack, plan, arms, jobs, identity: {
            arm.id: _answered("its configuration is not the one that was checked") for arm in arms
        },
    )
    refused = study_run("trend", jobs=2, project=project)
    assert (refused.status, refused.reason_codes) == ("blocked", ["PROJECT_CHANGED_DURING_RUN"])
    assert "base: its configuration is not the one that was checked" in refused.summary
    assert not (project / ".signalquarry").exists() and not (project / "evidence" / "trials.jsonl").exists()
    log = ledger.studies(project).entries()
    assert [entry["kind"] for entry in log] == ["study_started", "arm_failed"]
    assert (log[-1]["arm"], log[-1]["reason"]) == ("base", "PROJECT_CHANGED_DURING_RUN")

    class Pool:
        def submit(self, function: Any, job: Any) -> Future[Simulated | str]:
            return _answered("the dataset is not the one that was checked")

    sweep_api = sys.modules["signalquarry.api.sweep"]
    monkeypatch.setattr(sweep_api, "simulation_pool", lambda stack, workers: (Pool(), tmp_path / "scratch"))
    blocked = sweep("sma-trend", ["period=100,150"], jobs=2, project=project)
    assert (blocked.status, blocked.reason_codes) == ("blocked", ["PROJECT_CHANGED_DURING_RUN"])
    assert "the dataset is not the one that was checked" in blocked.summary
    assert not (project / ".signalquarry").exists() and not (project / "evidence" / "trials.jsonl").exists()


def test_a_sweep_worker_checks_the_project_it_rebuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _demo(tmp_path, "jobs-point")
    sweep_api = sys.modules["signalquarry.api.sweep"]
    monkeypatch.setattr(sweep_api, "_worker_resolved", {})
    resolved = sweep_api.resolve("sweep", "sma-trend", project)
    strategy = resolved.strategy
    params = strategy.definition.params.model_validate({**strategy.params.model_dump(), "period": 120})
    job = sweep_api._Job(
        strategy_id="sma-trend",
        root=resolved.root,
        point={"period": 120},
        configuration_hash=replace(strategy, params=params).configuration_hash,
        dataset_identity=resolved.dataset.identity(),
        end=None,
        scratch=tmp_path / "scratch",
    )
    simulated = sweep_api._simulate_point(job)
    assert isinstance(simulated, Simulated)
    assert not (project / ".signalquarry").exists()
    assert sweep("sma-trend", ["period=120"], project=project).status == "ok"
    assert iter_runs(project)[0][1]["ledger_hash"] == simulated.result.ledger_hash

    for change, reason in (
        ({"strategy_id": "nope"}, "the strategy no longer resolves"),
        ({"point": {"period": 5}}, "its parameters are no longer valid"),
        ({"point": {"period": 121}}, "its configuration is not the one that was checked"),
        ({"dataset_identity": "sha256:" + "0" * 64}, "the dataset is not the one that was checked"),
    ):
        refused = sweep_api._simulate_point(replace(job, scratch=tmp_path / "unused", **change))
        assert refused == reason, change
    assert not (tmp_path / "unused").exists()


def test_jobs_is_validated_and_reaches_the_api_from_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _demo(tmp_path, "jobs-usage")
    for envelope in (
        study_run("trend", jobs=0, project=project),
        sweep("sma-trend", ["period=100,150"], jobs=0, project=project),
    ):
        assert (envelope.status, envelope.reason_codes) == ("usage", ["USAGE_INVALID"])
        assert envelope.summary == "--jobs must be at least 1"
    assert not (project / ".signalquarry").exists()
    # One arm or one point left to simulate needs no worker.
    assert sweep("sma-trend", ["period=100"], jobs=4, project=project).status == "ok"

    seen: list[tuple[str, dict[str, Any]]] = []

    def record(name: str) -> Any:
        def call(*args: Any, **options: Any) -> Envelope:
            seen.append((name, options))
            return Envelope(command=name)

        return call

    monkeypatch.setattr(api, "study_run", record("study run"))
    monkeypatch.setattr(api, "sweep", record("sweep"))
    assert main(["study", "run", "--study", "trend", "--jobs", "3", "--project", str(project)]) == 0
    assert (
        main(["sweep", "--strategy", "sma-trend", "--param", "period=100,150", "--project", str(project)])
        == 0
    )
    assert main(["sweep", "--strategy", "sma-trend", "--param", "period=100", "--jobs", "2"]) == 0
    capsys.readouterr()
    assert [(name, options["jobs"]) for name, options in seen] == [
        ("study run", 3),
        ("sweep", 1),
        ("sweep", 2),
    ]
