# SPDX-License-Identifier: Apache-2.0
"""One way to execute a backtest and persist it as a run directory.

``sqy backtest`` and ``sqy sweep`` both go through :func:`execute_run`, so every run has the
same artifacts, the same trial accounting and the same descriptive context, and runs can be
compared with each other.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, cast

import numpy as np

from signalquarry._internal.canonical import file_sha256
from signalquarry._internal.engine.backtest import EngineError
from signalquarry._internal.engine.run import is_options, simulate, simulate_equity_ticks
from signalquarry._internal.evidence.report import growth_svg, render_comparison
from signalquarry._internal.evidence.run_spool import spool_equity_run
from signalquarry._internal.evidence.runs import (
    csv_chunks,
    iter_runs,
    jsonl_chunks,
    new_run_id,
    read_curve,
    result_document,
    result_hash_ok,
    unique_run_id,
    write_run,
)
from signalquarry._internal.project.project import ProjectError, find_root
from signalquarry._internal.validation.benchmark import BenchmarkCurve, benchmark_curve, benchmark_summary
from signalquarry._internal.validation.compare import RunSeries, compare_runs
from signalquarry._internal.validation.evaluate import equity_returns
from signalquarry._internal.validation.metrics import drawdown_episodes, summarize, trading_activity
from signalquarry._internal.validation.stats import moments
from signalquarry.api.envelope import Envelope
from signalquarry.api.evidence import holdout_state, record_run_trial, trial_evidence
from signalquarry.api.resolve import Resolved

Detail = Literal["full", "summary"]


@dataclass
class RunOutcome:
    run_id: str
    sessions: list[date]
    returns: np.ndarray
    metrics: dict[str, Any]
    context: dict[str, Any]  # drawdowns, activity and (when available) benchmark
    evidence: dict[str, Any]
    artifacts: list[dict[str, str]]
    ledger_hash: str
    dataset_identity: str
    fill_count: int
    engine_warnings: list[str]
    result_hash: str = ""
    warnings: list[str] = field(default_factory=list[str])
    next_actions: list[dict[str, str]] = field(default_factory=list[dict[str, str]])


def _benchmark(
    resolved: Resolved, sessions: list[date], cache: dict[tuple[date, date, int], BenchmarkCurve | None]
) -> BenchmarkCurve | None:
    """The benchmark curve for these sessions; runs over the same window share one."""
    key = (sessions[0], sessions[-1], len(sessions))
    if key not in cache:
        source = resolved.benchmark_source()
        curve = None
        if source is not None:
            try:
                curve = benchmark_curve(resolved.strategy.spec, source[0], source[1], sessions)
            except EngineError:
                curve = None
        cache[key] = curve
    return cache[key]


def execute_run(
    resolved: Resolved,
    *,
    command: str,
    start: date | None = None,
    end: date | None = None,
    label: str | None = None,
    detail: Detail = "full",
    benchmarks: dict[tuple[date, date, int], BenchmarkCurve | None] | None = None,
    record: bool = True,
) -> RunOutcome:
    """Simulate ``resolved.strategy``, record its trial and write its run directory.

    Raises :class:`EngineError` before anything is written when the simulation fails.
    ``detail="summary"`` keeps ``result.json`` and the equity curves and leaves out fills and
    decisions. ``benchmarks`` lets a caller that runs many variants share benchmark curves.
    ``record=False`` is for a run that is not a candidate (a study's sensitivity arm): it
    writes the run and appends no trial.
    """
    root, strategy = resolved.root, resolved.strategy
    spec = strategy.spec
    initial = spec.account.initial_cash
    scratch: TemporaryDirectory[str] | None = None
    traded: tuple[Decimal, Decimal] | None = None  # notional bought and sold (equity runs)
    fill_rows: Iterable[Iterable[Any]]
    decision_chunks: Iterable[str]
    try:
        if is_options(spec):
            result = simulate(
                spec,
                strategy.definition,
                strategy.params,
                resolved.dataset,
                start=start,
                end=end,
                recorded_chains=resolved.recorded_chains(),
            )
            fill_count = len(result.fills)
            fees = sum((fill.fee for fill in result.fills), Decimal(0))
            fill_rows = (
                [
                    f.session.isoformat(),
                    f.symbol,
                    f.side,
                    f.quantity,
                    f.price,
                    f.fee,
                    f.settle_session.isoformat() if f.settle_session else "",
                ]
                for f in result.fills
            )
            decision_chunks = jsonl_chunks(result.decisions)
        else:
            scratch = TemporaryDirectory(prefix="signalquarry-backtest-")
            spool = spool_equity_run(
                simulate_equity_ticks(
                    spec, strategy.definition, strategy.params, resolved.dataset, start=start, end=end
                ),
                Path(scratch.name),
            )
            result = spool.result
            fill_count = spool.fill_count
            fees = spool.fees
            traded = (spool.bought, spool.sold)
            fill_rows = spool.fills_csv_rows()
            decision_chunks = spool.decisions_chunks()
    except EngineError:
        if scratch is not None:
            scratch.cleanup()
        raise
    try:
        returns = equity_returns(result, initial)
        m = moments(returns)
        if record:
            record_run_trial(
                resolved,
                command=command,
                returns=returns,
                moments={"n": m.n, "sharpe": m.sharpe, "skew": m.skew, "kurtosis": m.kurtosis},
                window=(result.sessions[0], result.sessions[-1]),
            )
        metrics = summarize(result.sessions, result.equity, initial, fills=fill_count, fees=fees)
        # Descriptive context for the same run: none of it enters a gate or the claim level.
        context: dict[str, Any] = {"drawdowns": drawdown_episodes(result.sessions, result.equity, initial)}
        if traded is not None:
            context["activity"] = trading_activity(
                result.sessions, result.equity, bought=traded[0], sold=traded[1], fees=fees
            )
        warnings: list[str] = []
        next_actions: list[dict[str, str]] = []
        benchmark = _benchmark(resolved, result.sessions, {} if benchmarks is None else benchmarks)
        if benchmark is not None:
            context["benchmark"] = benchmark_summary(benchmark, returns, initial)
        elif spec.benchmark is not None:
            warnings.append("BENCHMARK_DATA_MISSING")
            next_actions.append(
                {
                    "command": f"sqy data fetch --strategy {spec.id}",
                    "why": f"No recorded dataset covers the benchmark {spec.benchmark}.",
                }
            )
        run_id = unique_run_id(root, new_run_id(strategy.configuration_hash, datetime.now(UTC)))
        evidence = {
            "grade": resolved.evidence_grade,
            "claim_level": "none" if resolved.grade == "synthetic" else "in_sample",
            "holdout": holdout_state(root, spec.family),
            "trial": None
            if resolved.grade == "synthetic"
            else trial_evidence(root, spec.family, strategy.configuration_hash),
        }
        document = result_document(
            run_id=run_id,
            strategy_id=spec.id,
            strategy_version=spec.version,
            configuration_hash=strategy.configuration_hash,
            code_tree_hash=strategy.code_tree_hash,
            dataset_id=resolved.dataset_id,
            dataset_identity=result.dataset_identity,
            ledger_hash=result.ledger_hash,
            evidence=evidence,
            metrics=metrics,
            warnings=result.warnings,
            # What produced the run, so that two runs can be told apart without the project.
            command=command,
            params=strategy.params.model_dump(mode="json"),
            # The declared parameters stay out of `spec`: `params` holds the ones used.
            spec={key: value for key, value in spec.outcome_document().items() if key != "params"},
            # The window that was asked for (after any holdout clip), not the sessions found.
            window={
                "start": None if start is None else start.isoformat(),
                "end": None if end is None else end.isoformat(),
            },
            trial_recorded=record and resolved.grade != "synthetic",
            **({} if label is None else {"label": label}),
            **context,
        )
        files: dict[str, str | bytes | Iterable[str | bytes]] = {
            "result.json": document,
            "equity.csv": csv_chunks(
                ["session", "equity", "settled_cash"],
                [
                    [s.isoformat(), e, c]
                    for s, e, c in zip(result.sessions, result.equity, result.cash, strict=True)
                ],
            ),
        }
        if benchmark is not None:
            files["benchmark.csv"] = csv_chunks(
                ["session", "equity"],
                [[s.isoformat(), e] for s, e in zip(benchmark.sessions, benchmark.equity, strict=True)],
            )
        if detail == "full":
            files["fills.csv"] = csv_chunks(
                ["session", "symbol", "side", "quantity", "price", "fee", "settle_session"], fill_rows
            )
            files["decisions.jsonl"] = decision_chunks
        artifacts = write_run(root, run_id, files)
    finally:
        if scratch is not None:
            scratch.cleanup()
    return RunOutcome(
        run_id=run_id,
        sessions=result.sessions,
        returns=returns,
        metrics=metrics,
        context=context,
        evidence=evidence,
        artifacts=artifacts,
        ledger_hash=result.ledger_hash,
        dataset_identity=result.dataset_identity,
        fill_count=fill_count,
        engine_warnings=result.warnings,
        result_hash=str(json.loads(document)["result_hash"]),
        warnings=warnings,
        next_actions=next_actions,
    )


class _RunError(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code, self.detail = code, detail


def _load(root: Path, run_id: str) -> tuple[Path, dict[str, Any]]:
    """One run directory and its verified result document."""
    directory = root / ".signalquarry" / "runs" / run_id
    if "/" in run_id or run_id in ("", ".", "..") or not directory.is_dir():
        raise _RunError("RUN_NOT_FOUND", run_id)
    try:
        document = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise _RunError("RUN_ARTIFACT_INVALID", f"{run_id}: result.json is missing or unreadable") from None
    typed = cast(dict[str, Any], document) if isinstance(document, dict) else {}
    if typed.get("schema") != "signalquarry.result/v1":
        raise _RunError("RUN_ARTIFACT_INVALID", f"{run_id}: not a backtest result")
    if not result_hash_ok(typed):
        raise _RunError("RUN_ARTIFACT_INVALID", f"{run_id}: result.json does not match its recorded hash")
    return directory, typed


def _row(document: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = document.get("metrics") or {}
    evidence: dict[str, Any] = document.get("evidence") or {}
    return {
        "run_id": document.get("run_id"),
        "strategy_id": document.get("strategy_id"),
        "command": document.get("command"),
        "label": document.get("label"),
        "created_at": document.get("created_at"),
        "configuration_hash": document.get("configuration_hash"),
        "dataset_id": document.get("dataset_id"),
        "grade": evidence.get("grade"),
        "params": document.get("params"),
        "start": metrics.get("start"),
        "end": metrics.get("end"),
        "total_return": metrics.get("total_return"),
        "sharpe": metrics.get("sharpe"),
        "max_drawdown": metrics.get("max_drawdown"),
    }


def _project_root(command: str, project: Path | None) -> Path | Envelope:
    try:
        return find_root(project)
    except ProjectError as exc:
        return Envelope(
            command=command, status="invalid", reason_codes=[exc.code], summary=exc.detail or exc.code
        )


def runs_ls(*, strategy_id: str | None = None, limit: int = 20, project: Path | None = None) -> Envelope:
    """The most recent backtest runs (including sweep points), newest first."""
    root = _project_root("runs ls", project)
    if isinstance(root, Envelope):
        return root
    if limit < 1:
        return Envelope(
            command="runs ls",
            status="usage",
            reason_codes=["USAGE_INVALID"],
            summary="--limit must be at least 1",
        )
    found = iter_runs(root, strategy_id=strategy_id)
    rows = [_row(document) for _, document in reversed(found[-limit:])]
    return Envelope(
        command="runs ls",
        summary=f"{len(rows)} of {len(found)} runs" + (f" of {strategy_id}" if strategy_id else ""),
        data={"runs": rows, "total": len(found)},
    )


def runs_show(run_id: str, *, project: Path | None = None) -> Envelope:
    """One run's result document and its artifacts."""
    root = _project_root("runs show", project)
    if isinstance(root, Envelope):
        return root
    try:
        directory, document = _load(root, run_id)
    except _RunError as exc:
        return Envelope(command="runs show", status="invalid", reason_codes=[exc.code], summary=exc.detail)
    metrics: dict[str, Any] = document.get("metrics") or {}
    return Envelope(
        command="runs show",
        summary=f"{run_id}: {document.get('strategy_id')}"
        + (f" [{document['label']}]" if document.get("label") else ""),
        metrics=metrics,
        evidence=document.get("evidence"),
        data={"result": document},
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


def _series(directory: Path, document: dict[str, Any]) -> RunSeries:
    try:
        sessions, equity = read_curve(directory / "equity.csv")
    except (OSError, KeyError, ValueError, ArithmeticError):
        raise _RunError(
            "RUN_ARTIFACT_INVALID", f"{document.get('run_id')}: equity.csv is missing or unreadable"
        ) from None
    metrics: dict[str, Any] = document.get("metrics") or {}
    if not sessions or len(sessions) != metrics.get("sessions"):
        raise _RunError(
            "RUN_ARTIFACT_INVALID", f"{document.get('run_id')}: equity.csv does not match result.json"
        )
    spec: dict[str, Any] = document.get("spec") or {}
    account: dict[str, Any] = spec.get("account") or {}
    # Runs written before the specification was recorded have no initial cash on file: their
    # first session then counts as a return of zero.
    initial = Decimal(str(account.get("initial_cash", equity[0])))
    values = np.array([float(initial), *(float(item) for item in equity)])
    return RunSeries(document, tuple(sessions), values[1:] / values[:-1] - 1.0)


_CLAIM_ORDER = ("none", "in_sample", "walk_forward", "holdout_passed", "paper_forward")


def _claim(document: Mapping[str, Any]) -> str:
    evidence: Mapping[str, Any] = document.get("evidence") or {}
    return str(evidence.get("claim_level", "none"))


def runs_compare(run_ids: Sequence[str], *, project: Path | None = None) -> Envelope:
    """Compare two or more runs against the first one and write the comparison.

    Runs are differenced only when they share dataset, sessions, account and evidence grade;
    otherwise they are listed side by side with the reasons and the warning
    ``RUNS_NOT_COMPARABLE``. A comparison is descriptive and never raises a claim level.
    """
    command = "runs compare"
    if len(run_ids) < 2 or len(set(run_ids)) != len(run_ids):
        return Envelope(
            command=command,
            status="usage",
            reason_codes=["USAGE_INVALID"],
            summary="give two or more different run ids; the first is the reference",
        )
    root = _project_root(command, project)
    if isinstance(root, Envelope):
        return root
    loaded: list[tuple[Path, RunSeries]] = []
    try:
        for run_id in run_ids:
            directory, document = _load(root, run_id)
            loaded.append((directory, _series(directory, document)))
    except _RunError as exc:
        return Envelope(command=command, status="invalid", reason_codes=[exc.code], summary=exc.detail)
    comparison = compare_runs([series for _, series in loaded])
    labels = {
        str(series.document["run_id"]): str(series.document.get("label") or series.document["run_id"])
        for _, series in loaded
    }
    chart = growth_svg(
        [
            (
                labels[str(series.document["run_id"])],
                [(s.isoformat(), e) for s, e in zip(*read_curve(directory / "equity.csv"), strict=True)],
            )
            for directory, series in loaded
        ]
    )
    comparison_id = unique_run_id(root, f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-compare", kind="comparisons")
    files: dict[str, str | bytes | Iterable[str | bytes]] = {
        "comparison.json": result_document(
            schema="signalquarry.comparison/v1", comparison_id=comparison_id, **comparison
        ),
        "comparison.md": render_comparison(comparison, labels=labels, chart=bool(chart)),
    }
    if chart:
        files["equity.svg"] = chart
    artifacts = write_run(root, comparison_id, files, kind="comparisons")
    grades = {str(run["grade"]) for run in comparison["runs"]}
    claims = [_claim(series.document) for _, series in loaded]
    reference = labels[comparison["reference"]]
    return Envelope(
        command=command,
        summary=(
            f"{len(run_ids)} runs against {reference}"
            if comparison["comparable"]
            else f"{len(run_ids)} runs are not comparable: {', '.join(comparison['reasons'])}"
        ),
        evidence={
            "grade": grades.pop() if len(grades) == 1 else "mixed",
            "claim_level": min(
                claims, key=lambda claim: _CLAIM_ORDER.index(claim) if claim in _CLAIM_ORDER else 0
            ),
        },
        warnings=[] if comparison["comparable"] else ["RUNS_NOT_COMPARABLE"],
        data={"comparison_id": comparison_id, **comparison},
        artifacts=artifacts,
    )
