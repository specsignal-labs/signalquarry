# SPDX-License-Identifier: Apache-2.0
"""Research studies (ADR 0015): ``sqy study init | check | run | ls | show``.

A study is ``studies/<id>/study.yaml``: a subject strategy, baselines, bounded variants and one
predeclared comparison rule. Every arm runs on the same dataset and window through the one run
writer, every candidate configuration on real data is a trial, the whole study is checked
against the trial budgets before anything is spent, and the verdict says only what the rule
and a paired interval allow. A study never opens a holdout and never raises a claim level.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from concurrent.futures import Future
from contextlib import ExitStack
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from pydantic import ValidationError

from signalquarry._internal.canonical import canonical_hash, file_sha256
from signalquarry._internal.contracts import progress
from signalquarry._internal.contracts.spec import SLUG_PATTERN, ExecutionSpec
from signalquarry._internal.contracts.study import Baseline, StudySpecV1, Variant, load_study
from signalquarry._internal.engine.backtest import EngineError
from signalquarry._internal.evidence.report import growth_svg, render_study
from signalquarry._internal.evidence.runs import (
    iter_runs,
    read_curve,
    result_document,
    result_hash_ok,
    unique_run_id,
    write_run,
)
from signalquarry._internal.project.project import ProjectError, find_root, load_config, load_strategies
from signalquarry._internal.validation import ledger
from signalquarry._internal.validation.benchmark import BenchmarkCurve, benchmark_curve
from signalquarry._internal.validation.compare import (
    RunSeries,
    comparability,
    compare_runs,
    configuration_diff,
    verdict,
)
from signalquarry._internal.validation.metrics import summarize
from signalquarry._internal.validation.stats import moments, pbo_cscv
from signalquarry.api.envelope import Envelope, Status
from signalquarry.api.evidence import (
    budget_shortfall,
    budget_warnings,
    family_seal,
    record_run_trial,
    trial_budget,
)
from signalquarry.api.resolve import Resolved, resolve
from signalquarry.api.runs import Simulated, commit_run, execute_run, simulate_run, simulation_pool

STUDY_RESULT_SCHEMA = "signalquarry.study-result/v1"
_STRATEGY_KINDS = ("subject", "variant", "strategy")


@dataclass(frozen=True)
class _Arm:
    id: str
    kind: str  # subject | variant | strategy | benchmark | benchmark_scaled
    role: str  # subject | candidate | ablation | sensitivity | baseline
    counts: bool  # a trial on real data
    resolved: Resolved | None = None  # the strategy-like arms
    symbol: str | None = None  # the benchmark arms
    note: str | None = None

    @property
    def configuration_hash(self) -> str | None:
        return None if self.resolved is None else self.resolved.strategy.configuration_hash

    @property
    def family(self) -> str | None:
        return None if self.resolved is None else self.resolved.strategy.spec.family


@dataclass(frozen=True)
class _Plan:
    root: Path
    path: Path
    study: StudySpecV1
    study_hash: str
    subject: Resolved
    arms: tuple[_Arm, ...]
    start: date | None
    end: date | None
    clipped: bool

    @property
    def window(self) -> dict[str, str | None]:
        return {
            "start": None if self.start is None else self.start.isoformat(),
            "end": None if self.end is None else self.end.isoformat(),
        }


@dataclass(frozen=True)
class _Done:
    arm: _Arm
    series: RunSeries
    equity: list[tuple[str, Decimal]]
    run_id: str | None
    result_hash: str | None
    resumed: bool


@dataclass(frozen=True)
class _Job:
    """What a worker process needs to rebuild one arm from the project and simulate it."""

    study_id: str
    root: Path
    study_hash: str
    arm: str
    configuration_hash: str
    dataset_identity: str
    start: date | None
    end: date | None
    scratch: Path


class _StudyError(Exception):
    def __init__(self, code: str, detail: str, *, status: Status = "invalid") -> None:
        super().__init__(detail)
        self.code, self.detail = code, detail
        self.status: Status = status


def _refusal(command: str, error: _StudyError) -> Envelope:
    return Envelope(command=command, status=error.status, reason_codes=[error.code], summary=error.detail)


def _study_paths(root: Path) -> dict[str, Path]:
    return {path.parent.name: path for path in sorted((root / "studies").glob("*/study.yaml"))}


def _read_study(root: Path, study_id: str) -> tuple[Path, StudySpecV1]:
    path = _study_paths(root).get(study_id)
    if path is None:
        raise _StudyError("STUDY_NOT_FOUND", study_id)
    try:
        study = load_study(path)
    except ValidationError as exc:
        error = exc.errors()[0]
        where = ".".join(str(part) for part in error["loc"])
        raise _StudyError(
            "STUDY_SPEC_INVALID", f"{study_id}: {where}: {error['msg']}".replace(": : ", ": ")
        ) from None
    except (ValueError, yaml.YAMLError, OSError) as exc:
        raise _StudyError("STUDY_SPEC_INVALID", f"{study_id}: {exc}") from None
    if study.id != study_id:
        raise _StudyError(
            "STUDY_SPEC_INVALID",
            f"{study_id}: the id in study.yaml is {study.id}; it must match the directory",
        )
    return path, study


def _merged(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """``base`` with ``override`` applied key by key through nested mappings."""
    result = dict(base)
    for key, value in override.items():
        current = result.get(key)
        if isinstance(value, Mapping) and isinstance(current, Mapping):
            result[key] = _merged(cast(Mapping[str, Any], current), cast(Mapping[str, Any], value))
        else:
            result[key] = value
    return result


def _variant(subject: Resolved, variant: Variant) -> Resolved:
    strategy = subject.strategy
    try:
        params = strategy.definition.params.model_validate({**strategy.params.model_dump(), **variant.params})
    except ValidationError as exc:
        error = exc.errors()[0]
        where = ".".join(str(part) for part in error["loc"])
        raise _StudyError("STUDY_ARM_INVALID", f"{variant.id}: params.{where}: {error['msg']}") from None
    spec = strategy.spec
    if variant.execution:
        try:
            execution = ExecutionSpec.model_validate(
                _merged(spec.execution.model_dump(mode="json"), variant.execution)
            )
        except ValidationError as exc:
            error = exc.errors()[0]
            where = ".".join(str(part) for part in error["loc"])
            raise _StudyError(
                "STUDY_ARM_INVALID", f"{variant.id}: execution.{where}: {error['msg']}"
            ) from None
        spec = spec.model_copy(update={"execution": execution})
    return replace(subject, strategy=replace(strategy, spec=spec, params=params))


def _baseline(command: str, subject: Resolved, baseline: Baseline, project: Path | None) -> _Arm | Envelope:
    if baseline.kind == "strategy":
        assert baseline.strategy is not None
        other = resolve(command, baseline.strategy, project)
        if isinstance(other, Envelope):
            return other
        missing = sorted(set(other.strategy.spec.data.symbols) - set(subject.dataset.series))
        if missing:
            raise _StudyError(
                "STUDY_ARM_INVALID",
                f"{baseline.id}: the study's dataset does not hold {', '.join(missing)}; "
                "record one dataset that covers every arm",
            )
        other = replace(other, dataset=subject.dataset, dataset_id=subject.dataset_id, grade=subject.grade)
        return _Arm(baseline.id, "strategy", "baseline", True, resolved=other)
    symbol = baseline.symbol or subject.strategy.spec.benchmark
    if symbol is None:
        raise _StudyError(
            "STUDY_ARM_INVALID",
            f"{baseline.id}: give the baseline a symbol or declare a benchmark in strategy.yaml",
        )
    if subject.symbol_source(symbol) is None:
        raise _StudyError(
            "BENCHMARK_DATA_MISSING",
            f"{baseline.id}: no recorded dataset covers {symbol}",
            status="unavailable",
        )
    return _Arm(baseline.id, baseline.kind, "baseline", False, symbol=symbol)


def _plan(command: str, study_id: str, project: Path | None) -> _Plan | Envelope:
    try:
        root = find_root(project)
    except ProjectError as exc:
        return Envelope(
            command=command, status="invalid", reason_codes=[exc.code], summary=exc.detail or exc.code
        )
    try:
        path, study = _read_study(root, study_id)
        subject = resolve(command, study.base, project)
        if isinstance(subject, Envelope):
            return subject
        if study.dataset is not None and study.dataset != subject.dataset_id:
            raise _StudyError(
                "STUDY_DATASET_MISMATCH",
                f"the study pins dataset {study.dataset}, but {study.base} resolves to {subject.dataset_id}",
            )
        arms = [_Arm("base", "subject", "subject", True, resolved=subject)]
        seen = {subject.strategy.configuration_hash: "base"}
        for variant in study.all_variants():
            resolved = _variant(subject, variant)
            other = seen.get(resolved.strategy.configuration_hash)
            if other is not None:
                raise _StudyError(
                    "STUDY_ARM_INVALID",
                    f"{variant.id}: the same configuration as {other}; it changes nothing",
                )
            seen[resolved.strategy.configuration_hash] = variant.id
            arms.append(
                _Arm(
                    variant.id,
                    "variant",
                    variant.role,
                    variant.role != "sensitivity",
                    resolved=resolved,
                    note=variant.note,
                )
            )
        for baseline in study.baselines:
            arm = _baseline(command, subject, baseline, project)
            if isinstance(arm, Envelope):
                return arm
            arms.append(arm)
    except _StudyError as exc:
        return _refusal(command, exc)
    start, end, clipped = study.window.start, study.window.end, False
    seals = [family_seal(root, family) for family in sorted({arm.family for arm in arms if arm.family})]
    earliest = min((seal for seal in seals if seal is not None), default=None)
    if earliest is not None and (end is None or end >= earliest):
        end, clipped = earliest - timedelta(days=1), True  # a study never reaches a sealed holdout
    return _Plan(
        root=root,
        path=path,
        study=study,
        study_hash=canonical_hash(study.study_document()),
        subject=subject,
        arms=tuple(arms),
        start=start,
        end=end,
        clipped=clipped,
    )


def _trials(plan: _Plan) -> tuple[dict[str, dict[str, int]], str | None]:
    """New trials per family against each budget, and why the study cannot be afforded."""
    if plan.subject.grade == "synthetic":
        return {}, None
    by_family: dict[str, list[_Arm]] = {}
    for arm in plan.arms:
        if arm.counts and arm.family is not None:
            by_family.setdefault(arm.family, []).append(arm)
    report: dict[str, dict[str, int]] = {}
    shortfall: str | None = None
    for family, members in sorted(by_family.items()):
        assert members[0].resolved is not None
        spec = members[0].resolved.strategy.spec
        known = {
            entry["configuration_hash"]
            for entry in ledger.trials(plan.root, family).entries()
            if entry.get("kind") == "trial"
        }
        hashes = {arm.configuration_hash for arm in members if arm.configuration_hash is not None}
        report[family] = {
            "new": len(hashes - known),
            "used": ledger.trial_summary(plan.root, family)["family_count"],
            "budget": trial_budget(plan.root, spec.evaluation.trial_budget, family),
        }
        shortfall = shortfall or budget_shortfall(plan.root, spec, hashes)
    return report, shortfall


def _existing(plan: _Plan, arm: _Arm, dataset_identity: str) -> tuple[Path, dict[str, Any]] | None:
    """The newest verified run of this arm's configuration on this dataset and window."""
    for directory, document in reversed(iter_runs(plan.root)):
        if (
            document.get("configuration_hash") == arm.configuration_hash
            and document.get("dataset_identity") == dataset_identity
            and document.get("window") == plan.window
            and result_hash_ok(document)
            and (directory / "equity.csv").is_file()
        ):
            return directory, document
    return None


def _from_files(
    directory: Path, document: dict[str, Any], initial: Decimal
) -> tuple[list[date], list[Decimal], np.ndarray]:
    sessions, equity = read_curve(directory / "equity.csv")
    metrics: dict[str, Any] = document.get("metrics") or {}
    if not sessions or len(sessions) != metrics.get("sessions"):
        raise _StudyError(
            "RUN_ARTIFACT_INVALID", f"{document.get('run_id')}: equity.csv does not match result.json"
        )
    values = np.array([float(initial), *(float(item) for item in equity)])
    return sessions, equity, values[1:] / values[:-1] - 1.0


def _grade(document: Mapping[str, Any]) -> Any:
    evidence: Mapping[str, Any] = document.get("evidence") or {}
    return evidence.get("grade")


def _strategy_document(arm: _Arm, document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_id": arm.id,
        "source_run": document.get("run_id"),
        "strategy_id": document.get("strategy_id"),
        "configuration_hash": document.get("configuration_hash"),
        "dataset_identity": document.get("dataset_identity"),
        "evidence": {"grade": _grade(document)},
        "metrics": document.get("metrics") or {},
        "params": document.get("params") or {},
        "spec": document.get("spec") or {},
    }


_worker_plans: dict[tuple[Path, str], tuple[_Plan, str] | None] = {}  # one per worker process


def _simulate_arm(job: _Job) -> Simulated | str:
    """Simulate one arm in a worker process; what changed, when the project is not the parent's.

    The worker rebuilds the study from the files, so it must arrive at the hashes the parent
    planned with. It appends nothing to the evidence and writes only into ``job.scratch``: the
    parent records and writes every arm, in the declared order.
    """
    key = (job.root, job.study_id)
    if key not in _worker_plans:
        plan = _plan("study run", job.study_id, job.root)
        _worker_plans[key] = None if isinstance(plan, Envelope) else (plan, plan.subject.dataset.identity())
    found = _worker_plans[key]
    if found is None or found[0].study_hash != job.study_hash:
        return "the study no longer resolves to the plan that was checked"
    plan, dataset_identity = found
    arm = next((arm for arm in plan.arms if arm.id == job.arm), None)
    if arm is None or arm.resolved is None or arm.configuration_hash != job.configuration_hash:
        return "its configuration is not the one that was checked"
    if dataset_identity != job.dataset_identity:
        return "the dataset is not the one that was checked"
    job.scratch.mkdir()
    return simulate_run(arm.resolved, start=job.start, end=job.end, scratch=job.scratch)


def _submit(
    stack: ExitStack, plan: _Plan, arms: list[_Arm], jobs: int, dataset_identity: str
) -> dict[str, Future[Simulated | str]]:
    """Start the simulations of ``arms`` in at most ``jobs`` worker processes."""
    pool, scratch = simulation_pool(stack, min(jobs, len(arms)))
    futures: dict[str, Future[Simulated | str]] = {}
    for number, arm in enumerate(arms):
        assert arm.configuration_hash is not None
        job = _Job(
            study_id=plan.study.id,
            root=plan.root,
            study_hash=plan.study_hash,
            arm=arm.id,
            configuration_hash=arm.configuration_hash,
            dataset_identity=dataset_identity,
            start=plan.start,
            end=plan.end,
            scratch=scratch / str(number),
        )
        futures[arm.id] = pool.submit(_simulate_arm, job)
    return futures


def _run_strategy_arm(
    plan: _Plan,
    arm: _Arm,
    dataset_identity: str,
    *,
    rerun: bool,
    benchmarks: dict[tuple[date, date, int], BenchmarkCurve | None],
    simulation: Future[Simulated | str] | None = None,
) -> _Done:
    assert arm.resolved is not None
    resolved = arm.resolved
    initial = resolved.strategy.spec.account.initial_cash
    found = _existing(plan, arm, dataset_identity)
    if found is not None and not rerun:
        directory, document = found
        sessions, equity, returns = _from_files(directory, document, initial)
        if arm.counts:
            # The run may have been made as a sensitivity arm, or the ledger may be new: a
            # counted arm is recorded here whatever produced the run. Recording is idempotent.
            m = moments(returns)
            record_run_trial(
                resolved,
                command="study",
                returns=returns,
                moments={"n": m.n, "sharpe": m.sharpe, "skew": m.skew, "kurtosis": m.kurtosis},
                window=(sessions[0], sessions[-1]),
            )
        return _Done(
            arm,
            RunSeries(_strategy_document(arm, document), tuple(sessions), returns),
            [(s.isoformat(), e) for s, e in zip(sessions, equity, strict=True)],
            str(document["run_id"]),
            str(document.get("result_hash")),
            resumed=True,
        )
    if simulation is None:
        run = execute_run(
            resolved,
            command="study",
            start=plan.start,
            end=plan.end,
            label=arm.id,
            benchmarks=benchmarks,
            record=arm.counts,
        )
    else:
        simulated = simulation.result()  # an EngineError in the worker is raised here
        if isinstance(simulated, str):
            raise _StudyError(
                "PROJECT_CHANGED_DURING_RUN",
                f"{arm.id}: {simulated}; the project changed while the study ran",
                status="blocked",
            )
        run = commit_run(
            resolved,
            simulated,
            command="study",
            start=plan.start,
            end=plan.end,
            label=arm.id,
            benchmarks=benchmarks,
            record=arm.counts,
        )
    if found is not None and found[1].get("ledger_hash") != run.ledger_hash:
        raise _StudyError(
            "STUDY_DETERMINISM_FAILED",
            f"{arm.id}: run {run.run_id} has ledger {run.ledger_hash}, "
            f"but {found[1].get('run_id')} recorded {found[1].get('ledger_hash')} for the same inputs",
            status="blocked",
        )
    directory = plan.root / ".signalquarry" / "runs" / run.run_id
    sessions, equity = read_curve(directory / "equity.csv")
    document = {
        "run_id": run.run_id,
        "strategy_id": resolved.strategy.spec.id,
        "configuration_hash": resolved.strategy.configuration_hash,
        "dataset_identity": run.dataset_identity,
        "evidence": run.evidence,
        "metrics": run.metrics,
        "params": resolved.strategy.params.model_dump(mode="json"),
        "spec": {k: v for k, v in resolved.strategy.spec.outcome_document().items() if k != "params"},
    }
    return _Done(
        arm,
        RunSeries(_strategy_document(arm, document), tuple(sessions), run.returns),
        [(s.isoformat(), e) for s, e in zip(sessions, equity, strict=True)],
        run.run_id,
        run.result_hash,
        resumed=False,
    )


def _benchmark_arm(plan: _Plan, arm: _Arm, subject: _Done, dataset_identity: str) -> _Done:
    assert arm.symbol is not None
    spec = plan.subject.strategy.spec
    initial = spec.account.initial_cash
    source = plan.subject.symbol_source(arm.symbol)
    assert source is not None  # checked when the study was planned
    sessions = list(subject.series.sessions)
    curve = benchmark_curve(spec, source[0], arm.symbol, sessions)
    equity, fills, fees = curve.equity, curve.fills, curve.fees
    returns = curve.returns(initial)
    if arm.kind == "benchmark_scaled":
        # The benchmark's own returns scaled to the subject's realized volatility over the same
        # sessions: "the same risk, held passively". Rescaling is frictionless, so no further
        # cost is charged; the scale uses the whole window and is descriptive.
        ours, theirs = float(np.std(subject.series.returns, ddof=1)), float(np.std(returns, ddof=1))
        scale = ours / theirs if theirs > 0 and math.isfinite(ours) else 1.0
        returns = returns * scale
        level = float(initial) * np.cumprod(1.0 + returns)
        equity = [Decimal(str(round(float(value), 6))) for value in level]
    document = {
        "run_id": arm.id,
        "source_run": None,
        "strategy_id": f"{arm.kind}:{arm.symbol}",
        "configuration_hash": None,
        "dataset_identity": dataset_identity,
        "evidence": {"grade": plan.subject.evidence_grade},
        "metrics": summarize(sessions, equity, initial, fills=fills, fees=fees),
    }
    return _Done(
        arm,
        RunSeries(document, tuple(sessions), returns),
        [(s.isoformat(), e) for s, e in zip(sessions, equity, strict=True)],
        None,
        None,
        resumed=False,
    )


def _arm_rows(plan: _Plan, done: list[_Done]) -> list[dict[str, Any]]:
    subject = done[0].series.document
    rows: list[dict[str, Any]] = []
    for item in done:
        arm, document = item.arm, item.series.document
        row: dict[str, Any] = {
            "id": arm.id,
            "kind": arm.kind,
            "role": arm.role,
            "counts": arm.counts,
            "strategy_id": document["strategy_id"],
            "configuration_hash": arm.configuration_hash,
            "run_id": item.run_id,
            "result_hash": item.result_hash,
            "resumed": item.resumed,
            "metrics": document["metrics"],
        }
        if arm.note is not None:
            row["note"] = arm.note
        if arm.kind == "variant":
            row["changes"] = {
                "params": configuration_diff(subject["params"], document["params"]),
                "spec": configuration_diff(subject["spec"], document["spec"]),
            }
        rows.append(row)
    return rows


def study_check(study_id: str, *, project: Path | None = None) -> Envelope:
    """Validate a study against the project and show what running it would cost. Runs nothing."""
    command = "study check"
    plan = _plan(command, study_id, project)
    if isinstance(plan, Envelope):
        return plan
    trials, shortfall = _trials(plan)
    dataset_identity = plan.subject.dataset.identity()
    arms: list[dict[str, Any]] = []
    for arm in plan.arms:
        found = _existing(plan, arm, dataset_identity) if arm.kind in _STRATEGY_KINDS else None
        arms.append(
            {
                "id": arm.id,
                "kind": arm.kind,
                "role": arm.role,
                "counts": arm.counts,
                "family": arm.family,
                "configuration_hash": arm.configuration_hash,
                "symbol": arm.symbol,
                "existing_run": None if found is None else found[1].get("run_id"),
            }
        )
    new = sum(row["new"] for row in trials.values())
    envelope = Envelope(
        command=command,
        summary=f"{study_id}: {len(arms)} arms, {new} new trial(s)"
        + ("" if plan.subject.grade != "synthetic" else " (synthetic data records none)"),
        data={
            "study_id": study_id,
            "study_hash": plan.study_hash,
            "base": plan.study.base,
            "dataset_id": plan.subject.dataset_id,
            "dataset_identity": dataset_identity,
            "grade": plan.subject.evidence_grade,
            "window": plan.window,
            "holdout_clipped": plan.clipped,
            "arms": arms,
            "trials": trials,
            "compare": plan.study.compare.model_dump(mode="json"),
        },
        warnings=["HOLDOUT_CLIPPED"] if plan.clipped else [],
        next_actions=[{"command": f"sqy study run --study {study_id}", "why": "Run every arm and compare."}],
    )
    if shortfall is not None:
        envelope.status, envelope.reason_codes = "blocked", ["TRIAL_BUDGET_EXHAUSTED"]
        envelope.summary = f"{study_id}: {shortfall}"
        envelope.next_actions = []
    return envelope


def study_run(study_id: str, *, rerun: bool = False, jobs: int = 1, project: Path | None = None) -> Envelope:
    """Run every arm of a study, compare them by its declared rule and record the result.

    Arms whose run already exists for the same configuration, dataset and window are reused;
    ``rerun`` recomputes them and requires the ledger hash they had. A study that would exceed
    a family's trial budget is refused with nothing written. ``jobs`` above 1 simulates the
    arms in that many worker processes; trials, runs and the study log are still written by
    this process in the declared order, so what is recorded does not depend on ``jobs``.
    """
    command = "study run"
    if jobs < 1:
        return Envelope(
            command=command,
            status="usage",
            reason_codes=["USAGE_INVALID"],
            summary="--jobs must be at least 1",
        )
    plan = _plan(command, study_id, project)
    if isinstance(plan, Envelope):
        return plan
    root, study = plan.root, plan.study
    real = plan.subject.grade != "synthetic"
    trials, shortfall = _trials(plan)
    if shortfall is not None:
        return Envelope(
            command=command,
            status="blocked",
            reason_codes=["TRIAL_BUDGET_EXHAUSTED"],
            summary=f"{study_id}: {shortfall}",
            data={"trials": trials},
        )
    family = plan.subject.strategy.spec.family
    dataset_identity = plan.subject.dataset.identity()
    common = {"study_id": study_id, "study_hash": plan.study_hash, "family": family}

    def log(kind: str, **fields: Any) -> None:
        if real:  # synthetic data is not evidence: nothing is recorded for it
            ledger.append(
                root, "studies", family, {"kind": kind, "at": datetime.now(UTC), **common, **fields}
            )

    log(
        "study_started",
        strategy_id=study.base,
        dataset_identity=dataset_identity,
        window=plan.window,
        arms=[
            {
                "id": arm.id,
                "kind": arm.kind,
                "role": arm.role,
                "counts": arm.counts,
                "configuration_hash": arm.configuration_hash,
            }
            for arm in plan.arms
        ],
    )
    done: list[_Done] = []
    benchmarks: dict[tuple[date, date, int], BenchmarkCurve | None] = {}
    ordered = [arm for arm in plan.arms if arm.kind in _STRATEGY_KINDS] + [
        arm for arm in plan.arms if arm.kind not in _STRATEGY_KINDS
    ]
    with ExitStack() as stack:
        pending = [
            arm
            for arm in ordered
            if arm.kind in _STRATEGY_KINDS and (rerun or _existing(plan, arm, dataset_identity) is None)
        ]
        simulations = (
            _submit(stack, plan, pending, jobs, dataset_identity) if jobs > 1 and len(pending) > 1 else {}
        )
        for number, arm in enumerate(ordered):
            progress.emit("study", arm=arm.id, done=number, total=len(ordered))
            try:
                if arm.kind in _STRATEGY_KINDS:
                    item = _run_strategy_arm(
                        plan,
                        arm,
                        dataset_identity,
                        rerun=rerun,
                        benchmarks=benchmarks,
                        simulation=simulations.get(arm.id),
                    )
                else:
                    item = _benchmark_arm(plan, arm, done[0], dataset_identity)
            except EngineError as exc:
                code = str(exc).split(":", 1)[0]
                log("arm_failed", arm=arm.id, reason=code)
                return Envelope(
                    command=command,
                    status="invalid",
                    reason_codes=[code],
                    summary=f"{study_id}: arm {arm.id}: {exc}",
                )
            except _StudyError as exc:
                log("arm_failed", arm=arm.id, reason=exc.code)
                return _refusal(command, exc)
            log(
                "arm_completed",
                arm=arm.id,
                run_id=item.run_id,
                result_hash=item.result_hash,
                resumed=item.resumed,
            )
            done.append(item)

    by_id = {item.arm.id: item for item in done}
    subject, versus = by_id["base"], by_id[study.compare.versus]
    others = [item for item in done if item.arm.id not in ("base", versus.arm.id)]
    comparison = compare_runs([versus.series, subject.series, *(item.series for item in others)])
    outcome = verdict(
        study.compare.metric,
        study.compare.direction,
        subject.series,
        versus.series,
        comparable=bool(comparability([subject.series, versus.series])["comparable"]),
    )
    candidates = [item for item in done if item.arm.kind in ("subject", "variant") and item.arm.counts]
    pbo = None
    if len(candidates) >= 2 and len({len(item.series.returns) for item in candidates}) == 1:
        pbo = pbo_cscv(np.column_stack([item.series.returns for item in candidates]))
    claim = "none" if not real else "in_sample"
    evidence = {"grade": plan.subject.evidence_grade, "claim_level": claim}
    result: dict[str, Any] = {
        "study_id": study_id,
        "study_hash": plan.study_hash,
        "study": study.study_document(),
        "base": study.base,
        "dataset_id": plan.subject.dataset_id,
        "dataset_identity": dataset_identity,
        "window": plan.window,
        "holdout_clipped": plan.clipped,
        "sessions": len(subject.series.sessions),
        "arms": _arm_rows(plan, done),
        "comparison": comparison,
        "verdict": outcome,
        "pbo": pbo,
        "trials": trials,
        "evidence": evidence,
    }
    chart = growth_svg([(item.arm.id, item.equity) for item in done])
    text = result_document(schema=STUDY_RESULT_SCHEMA, **result)
    study_run_id = unique_run_id(root, f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{study_id}", kind="studies")
    files: dict[str, Any] = {"study.json": text, "comparison.md": render_study(result, chart=bool(chart))}
    if chart:
        files["equity.svg"] = chart
    artifacts = write_run(root, study_run_id, files, kind="studies")
    comparison_hash = canonical_hash({"comparison": comparison, "verdict": outcome, "pbo": pbo})
    log(
        "study_completed",
        study_run_id=study_run_id,
        comparison_hash=comparison_hash,
        verdict=outcome["outcome"],
    )
    warnings = (["HOLDOUT_CLIPPED"] if plan.clipped else []) + (
        [] if comparison["comparable"] else ["RUNS_NOT_COMPARABLE"]
    )
    if real:
        warnings += budget_warnings(root, plan.subject.strategy.spec.evaluation.trial_budget, family)
    return Envelope(
        command=command,
        summary=(
            f"{study_id}: {outcome['outcome']} "
            f"({study.compare.metric} {study.compare.direction} than {study.compare.versus}: {outcome['reason']})"
        ),
        evidence=evidence,
        warnings=warnings,
        artifacts=artifacts,
        data={
            "study_id": study_id,
            "study_run_id": study_run_id,
            "study_hash": plan.study_hash,
            "comparison_hash": comparison_hash,
            "verdict": outcome,
            "comparable": comparison["comparable"],
            "reasons": comparison["reasons"],
            "arms": [
                {
                    key: row[key]
                    for key in ("id", "kind", "role", "counts", "run_id", "resumed", "configuration_hash")
                }
                | {
                    "total_return": row["metrics"].get("total_return"),
                    "sharpe": row["metrics"].get("sharpe"),
                    "max_drawdown": row["metrics"].get("max_drawdown"),
                }
                for row in result["arms"]
            ],
            "pairs": comparison["pairs"],
            "pbo": pbo,
            "trials": trials,
        },
        next_actions=[
            {
                "command": f"sqy evaluate --strategy {study.base}",
                "why": "A study compares in-sample runs; only walk-forward gates on a frozen "
                "configuration support a claim.",
            }
        ],
    )


def _results(root: Path, study_id: str | None = None) -> list[tuple[Path, dict[str, Any]]]:
    return [
        (directory, document)
        for directory, document in iter_runs(root, "study.json", kind="studies")
        if document.get("schema") == STUDY_RESULT_SCHEMA
        and (study_id is None or document.get("study_id") == study_id)
    ]


def study_ls(*, project: Path | None = None) -> Envelope:
    """The project's studies and the verdict of each one's latest run."""
    try:
        root = find_root(project)
    except ProjectError as exc:
        return Envelope(
            command="study ls", status="invalid", reason_codes=[exc.code], summary=exc.detail or exc.code
        )
    rows: list[dict[str, Any]] = []
    for study_id, path in _study_paths(root).items():
        row: dict[str, Any] = {"id": study_id, "path": str(path.relative_to(root))}
        try:
            _, study = _read_study(root, study_id)
        except _StudyError as exc:
            row.update(valid=False, error=exc.detail)
        else:
            results = [
                doc
                for _, doc in _results(root, study_id)
                if doc.get("study_hash") == canonical_hash(study.study_document())
            ]
            latest = results[-1] if results else None
            row.update(
                valid=True,
                base=study.base,
                hypothesis=study.hypothesis.statement,
                arms=study.arm_count() + sum(b.kind != "strategy" for b in study.baselines),
                runs=len(results),
                verdict=None if latest is None else latest["verdict"]["outcome"],
            )
        rows.append(row)
    return Envelope(command="study ls", summary=f"{len(rows)} studies", data={"studies": rows})


def study_show(study_id: str, *, project: Path | None = None) -> Envelope:
    """The latest recorded result of a study."""
    command = "study show"
    try:
        root = find_root(project)
    except ProjectError as exc:
        return Envelope(
            command=command, status="invalid", reason_codes=[exc.code], summary=exc.detail or exc.code
        )
    if study_id not in _study_paths(root):
        return Envelope(command=command, status="invalid", reason_codes=["STUDY_NOT_FOUND"], summary=study_id)
    results = _results(root, study_id)
    if not results:
        return Envelope(
            command=command,
            status="invalid",
            reason_codes=["STUDY_NO_RESULT"],
            summary=f"{study_id} has not been run",
            next_actions=[{"command": f"sqy study run --study {study_id}", "why": "Run the study first."}],
        )
    directory, document = results[-1]
    if not result_hash_ok(document):
        return Envelope(
            command=command,
            status="invalid",
            reason_codes=["RUN_ARTIFACT_INVALID"],
            summary=f"{directory.name}: study.json does not match its recorded hash",
        )
    current = None
    try:
        current = canonical_hash(_read_study(root, study_id)[1].study_document())
    except _StudyError:
        pass
    return Envelope(
        command=command,
        summary=f"{study_id}: {document['verdict']['outcome']} ({directory.name})",
        evidence=document.get("evidence"),
        warnings=[] if current == document.get("study_hash") else ["STUDY_FILE_CHANGED"],
        data={"study_run_id": directory.name, "result": document},
        artifacts=[
            {
                "path": str(path.relative_to(root)),
                "sha256": file_sha256(path),
                "kind": path.name.split(".")[0],
            }
            for path in sorted(directory.iterdir())
            if path.is_file()
        ],
    )


_TEMPLATE = """\
schema: signalquarry.study/v1
id: {study_id}
hypothesis:
  statement: {statement}
  falsification: {falsification}
base: {strategy_id}
# window: {{start: 2016-01-01, end: 2024-12-31}}
baselines:
  - {{id: buy-and-hold, kind: benchmark}}
# Bounded variants of the subject. On real data each candidate or ablation is a trial;
# a sensitivity variant changes execution or cost assumptions only and is not.
variants:
  - {{id: costs-x2, execution: {{costs: {{bps: "{double_bps}"}}}}, role: sensitivity}}
# grid: {{period: [100, 150, 200]}}
# Decide the rule before running: one metric, one direction, one baseline.
compare: {{metric: sharpe, direction: higher, versus: buy-and-hold}}
"""


def study_init(strategy_id: str, study_id: str, *, project: Path | None = None) -> Envelope:
    """Scaffold ``studies/<id>/study.yaml`` for a strategy. Nothing is run."""
    command = "study init"
    resolved_root: Path
    try:
        resolved_root = find_root(project)
    except ProjectError as exc:
        return Envelope(
            command=command, status="invalid", reason_codes=[exc.code], summary=exc.detail or exc.code
        )
    if not re.fullmatch(SLUG_PATTERN, study_id):
        return Envelope(
            command=command,
            status="usage",
            reason_codes=["USAGE_INVALID"],
            summary="study id: lower-case letters, digits and hyphens, 2 to 64 characters",
        )
    try:
        strategies = load_strategies(load_config(resolved_root))
    except ProjectError as exc:
        return Envelope(
            command=command, status="invalid", reason_codes=[exc.code], summary=exc.detail or exc.code
        )
    strategy = strategies.get(strategy_id)
    if strategy is None:
        return Envelope(
            command=command,
            status="invalid",
            reason_codes=["STRATEGY_NOT_FOUND"],
            summary=strategy_id,
            data={"strategies": sorted(strategies)},
        )
    path = resolved_root / "studies" / study_id / "study.yaml"
    if path.exists():
        return Envelope(
            command=command,
            status="invalid",
            reason_codes=["STUDY_EXISTS"],
            summary=str(path.relative_to(resolved_root)),
        )
    spec = strategy.spec
    path.parent.mkdir(parents=True)
    path.write_text(
        _TEMPLATE.format(
            study_id=study_id,
            strategy_id=strategy_id,
            statement=json.dumps(spec.hypothesis.statement, ensure_ascii=False),
            falsification=json.dumps(spec.hypothesis.falsification, ensure_ascii=False),
            double_bps=spec.execution.costs.bps * 2,
        ),
        encoding="utf-8",
    )
    load_study(path)  # the scaffold must be a valid study
    return Envelope(
        command=command,
        summary=f"created {path.relative_to(resolved_root)}",
        data={"study_id": study_id, "path": str(path.relative_to(resolved_root))},
        warnings=[] if spec.benchmark else ["STUDY_BENCHMARK_UNDECLARED"],
        next_actions=[
            {
                "command": f"edit {path.relative_to(resolved_root)}",
                "why": "State the hypothesis, the baseline and the comparison rule before any run.",
            },
            {
                "command": f"sqy study check --study {study_id}",
                "why": "See the arms and what they would cost.",
            },
        ],
    )
