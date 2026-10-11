# SPDX-License-Identifier: Apache-2.0
"""One way to execute a backtest and persist it as a run directory.

``sqy backtest`` and ``sqy sweep`` both go through :func:`execute_run`, so every run has the
same artifacts, the same trial accounting and the same descriptive context, and runs can be
compared with each other.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from decimal import Decimal
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, cast

import numpy as np
from pydantic import ValidationError

from signalquarry._internal.canonical import file_sha256
from signalquarry._internal.contracts.spec import SYMBOL_PATTERN
from signalquarry._internal.engine.backtest import BacktestResult, EngineError
from signalquarry._internal.engine.run import is_options, simulate, simulate_equity_ticks
from signalquarry._internal.evidence.report import growth_svg, render_comparison
from signalquarry._internal.evidence.run_spool import SpoolRun, SpoolSummary, spool_equity_run
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
from signalquarry._internal.validation.diagnostics import cost_curve_break_even, fold_consistency
from signalquarry._internal.validation.evaluate import equity_returns
from signalquarry._internal.validation.exposure import MAX_REFERENCES, exposures, rolling_exposures
from signalquarry._internal.validation.metrics import drawdown_episodes, summarize, trading_activity
from signalquarry._internal.validation.regimes import (
    calendar_regime,
    regime_table,
    trend_regime,
    volatility_regime,
)
from signalquarry._internal.validation.sensitivity import sensitivity
from signalquarry._internal.validation.stats import moments
from signalquarry.api.envelope import Envelope
from signalquarry.api.evidence import holdout_state, record_run_trial, trial_evidence
from signalquarry.api.resolve import Resolved, resolve

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


@dataclass(frozen=True)
class Simulated:
    """A finished simulation that has been neither recorded nor written.

    ``spool`` holds an equity run's fills and decisions in the caller's scratch directory;
    an options run keeps them on ``result``. It can cross a process boundary.
    """

    result: SpoolSummary | BacktestResult
    spool: SpoolRun | None = None


def simulate_run(
    resolved: Resolved, *, start: date | None = None, end: date | None = None, scratch: Path
) -> Simulated:
    """Simulate ``resolved.strategy``. Reads no evidence and writes only inside ``scratch``.

    Raises :class:`EngineError` when the simulation fails, leaving ``scratch`` empty.
    """
    strategy = resolved.strategy
    spec = strategy.spec
    if is_options(spec):
        return Simulated(
            simulate(
                spec,
                strategy.definition,
                strategy.params,
                resolved.dataset,
                start=start,
                end=end,
                recorded_chains=resolved.recorded_chains(),
            )
        )
    spool = spool_equity_run(
        simulate_equity_ticks(
            spec, strategy.definition, strategy.params, resolved.dataset, start=start, end=end
        ),
        scratch,
    )
    return Simulated(spool.result, spool)


def simulation_pool(stack: ExitStack, workers: int) -> tuple[ProcessPoolExecutor, Path]:
    """Worker processes for :func:`simulate_run`, and the scratch directory they spool into.

    Both live until ``stack`` closes; leaving early cancels what has not started. Workers are
    spawned, so on every platform they import the project afresh and share nothing with the
    caller but the files. A worker rebuilds its configuration from those files and must check
    it against the hashes the caller planned with; the caller commits the results in its own
    order with :func:`commit_run`, so what is recorded does not depend on the workers.
    """
    scratch = Path(stack.enter_context(TemporaryDirectory(prefix="signalquarry-jobs-")))
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
    stack.callback(pool.shutdown, wait=True, cancel_futures=True)
    return pool, scratch


def commit_run(
    resolved: Resolved,
    simulated: Simulated,
    *,
    command: str,
    start: date | None = None,
    end: date | None = None,
    label: str | None = None,
    detail: Detail = "full",
    benchmarks: dict[tuple[date, date, int], BenchmarkCurve | None] | None = None,
    record: bool = True,
) -> RunOutcome:
    """Record the trial of a finished simulation and write its run directory, in that order.

    ``start`` and ``end`` are the window the simulation was asked for. Every append to the
    evidence happens here, so a caller that simulates elsewhere (another process) still
    records in the order it commits.
    """
    root, strategy = resolved.root, resolved.strategy
    spec = strategy.spec
    initial = spec.account.initial_cash
    result = simulated.result
    traded: tuple[Decimal, Decimal] | None = None  # notional bought and sold (equity runs)
    fill_rows: Iterable[Iterable[Any]]
    decision_chunks: Iterable[str]
    if simulated.spool is None:
        assert isinstance(result, BacktestResult)
        fills = result.fills
        fill_count = len(fills)
        fees = sum((fill.fee for fill in fills), Decimal(0))
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
            for f in fills
        )
        decision_chunks = jsonl_chunks(result.decisions)
    else:
        spool = simulated.spool
        fill_count = spool.fill_count
        fees = spool.fees
        traded = (spool.bought, spool.sold)
        fill_rows = spool.fills_csv_rows()
        decision_chunks = spool.decisions_chunks()
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
    with TemporaryDirectory(prefix="signalquarry-backtest-") as scratch:
        simulated = simulate_run(resolved, start=start, end=end, scratch=Path(scratch))
        return commit_run(
            resolved,
            simulated,
            command=command,
            start=start,
            end=end,
            label=label,
            detail=detail,
            benchmarks=benchmarks,
            record=record,
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


def _unavailable(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "reason": reason}


def _diagnostic_regimes(directory: Path, series: RunSeries) -> dict[str, Any]:
    benchmark_returns = None
    levels = None
    if (directory / "benchmark.csv").is_file():
        try:
            sessions, equity = read_curve(directory / "benchmark.csv")
            if tuple(sessions) != series.sessions:
                raise ValueError("benchmark sessions differ")
            initial = Decimal(str(series.document["spec"]["account"]["initial_cash"]))
            values = np.array([float(initial), *(float(item) for item in equity)])
            levels, benchmark_returns = values[1:], values[1:] / values[:-1] - 1.0
        except (OSError, KeyError, ValueError, ArithmeticError):
            raise _RunError("RUN_ARTIFACT_INVALID", "benchmark.csv is unreadable or not aligned") from None
    calendar = regime_table(calendar_regime(series.sessions), series.returns, benchmark_returns)
    block = {
        "status": "ok",
        "note": "These regimes were not declared before the run; they are descriptive hypothesis generation.",
        "calendar": {"status": "ok", "rows": calendar},
        "trend": _unavailable("The run has no benchmark.csv."),
        "volatility": _unavailable("The run has no benchmark.csv."),
    }
    if levels is not None and benchmark_returns is not None:
        for kind, labels in (
            ("trend", trend_regime(levels)),
            ("volatility", volatility_regime(benchmark_returns)),
        ):
            block[kind] = {"status": "ok", "rows": regime_table(labels, series.returns, benchmark_returns)}
    return block


_NOT_REPRODUCED = "The project no longer reproduces that run."


def _reproduced(root: Path, document: dict[str, Any]) -> Resolved | None:
    """The run's configuration rebuilt from the project, when it still hashes to the run's."""
    resolved = resolve("diagnose", document["strategy_id"], root)
    if isinstance(resolved, Envelope):
        return None
    strategy = resolved.strategy
    try:
        strategy = replace(strategy, params=strategy.definition.params.model_validate(document["params"]))
        if (
            strategy.configuration_hash != document["configuration_hash"]
            or resolved.dataset.identity() != document["dataset_identity"]
        ):
            return None
    except (ValidationError, KeyError, ValueError, ArithmeticError):
        return None
    return replace(resolved, strategy=strategy)


def _diagnostic_costs(resolved: Resolved | None, document: dict[str, Any]) -> dict[str, Any]:
    recorded_spec: dict[str, Any] = document.get("spec") or {}
    if recorded_spec.get("kind") == "options_single_leg":
        return _unavailable("Options strategies use a different cost model.")
    reason = _NOT_REPRODUCED
    if resolved is None or is_options(resolved.strategy.spec):
        return _unavailable(reason)
    strategy, params = resolved.strategy, resolved.strategy.params
    try:
        window = document["window"]
        start = date.fromisoformat(window["start"]) if window["start"] else None
        end = date.fromisoformat(window["end"]) if window["end"] else None
        points: list[dict[str, Any]] = []
        for multiplier in (0, 1, 2, 4):
            result = simulate(
                strategy.spec,
                strategy.definition,
                params,
                resolved.dataset,
                start=start,
                end=end,
                cost_multiplier=Decimal(multiplier),
            )
            metrics = summarize(result.sessions, result.equity, strategy.spec.account.initial_cash)
            points.append(
                {
                    "multiplier": multiplier,
                    **{key: metrics[key] for key in ("total_return", "sharpe", "max_drawdown")},
                }
            )
    except (ValidationError, EngineError, KeyError, ValueError, ArithmeticError):
        return _unavailable(reason)
    if points[1]["total_return"] != document["metrics"]["total_return"]:
        return _unavailable("The multiplier-1 simulation does not reproduce the run's recorded total return.")
    return {
        "status": "ok",
        "points": points,
        "break_even": cost_curve_break_even(
            [(point["multiplier"], point["total_return"]) for point in points]
        ),
        "note": "Descriptive: multipliers scale the simulator's execution cost bps; other fees stay as declared.",
    }


def _diagnostic_symbols(symbols: Sequence[object]) -> tuple[str, ...]:
    if isinstance(symbols, str):
        raise ValueError("USAGE_INVALID")
    normalized: list[str] = []
    for symbol in symbols:
        if not isinstance(symbol, str):
            raise ValueError("USAGE_INVALID")
        normalized.append(symbol.upper())
    upper = tuple(normalized)
    if (
        len(upper) > MAX_REFERENCES
        or len(set(upper)) != len(upper)
        or any(not re.fullmatch(SYMBOL_PATTERN, symbol) for symbol in upper)
    ):
        raise ValueError("USAGE_INVALID")
    return upper


def _diagnostic_exposure(
    resolved: Resolved | None, series: RunSeries, symbols: Sequence[str]
) -> tuple[dict[str, Any], list[str]]:
    """Returns-based exposure to buy-and-hold positions in ``symbols``, and any warnings."""
    if resolved is None:
        return _unavailable(_NOT_REPRODUCED), []
    strategy = resolved.strategy
    references: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for symbol in symbols:
        source = resolved.symbol_source(symbol)
        if source is None:
            missing.append(symbol)
            continue
        try:
            references[symbol] = benchmark_curve(
                strategy.spec, source[0], source[1], series.sessions
            ).returns(strategy.spec.account.initial_cash)
        except EngineError:
            missing.append(symbol)
    if missing:
        return _unavailable("Passive return data is unavailable for: " + ", ".join(missing) + "."), [
            "EXPOSURE_DATA_MISSING"
        ]
    regression = exposures(series.returns, references)
    rows = rolling_exposures(series.returns, references) if regression["status"] == "ok" else []
    for row in rows:
        row["end"] = series.sessions[cast(int, row["end"])].isoformat()
    return {
        "status": regression["status"],
        "references_requested": list(symbols),
        "regression": regression,
        "rolling": {"window": 126, "step": 21, "rows": rows},
        "note": "The reference symbols were chosen after the run, the fit is in-sample over the whole window, and a beta is an association with a passive position, not a holding.",
    }, []


def diagnose(
    strategy_id: str,
    *,
    run_id: str | None = None,
    exposures: Sequence[str] = (),
    project: Path | None = None,
) -> Envelope:
    """Write descriptive diagnostics for a recorded run without changing it or its evidence."""
    try:
        symbols = _diagnostic_symbols(exposures)
    except ValueError:
        return Envelope(
            command="diagnose",
            status="usage",
            reason_codes=["USAGE_INVALID"],
            summary="give up to eight different valid reference symbols",
        )
    root = _project_root("diagnose", project)
    if isinstance(root, Envelope):
        return root
    if run_id is None:
        found = iter_runs(root, strategy_id=strategy_id)
        if not found:
            return Envelope(
                command="diagnose",
                status="invalid",
                reason_codes=["REPORT_NO_RUNS"],
                summary=f"no backtest runs for {strategy_id}",
            )
        run_id = found[-1][0].name
    try:
        directory, document = _load(root, run_id)
        if document["strategy_id"] != strategy_id:
            raise _RunError("RUN_ARTIFACT_INVALID", f"{run_id}: run belongs to another strategy")
        series = _series(directory, document)
        regimes = _diagnostic_regimes(directory, series)
    except _RunError as exc:
        return Envelope(command="diagnose", status="invalid", reason_codes=[exc.code], summary=exc.detail)
    reproduced = _reproduced(root, document)
    costs = _diagnostic_costs(reproduced, document)
    configuration = document["configuration_hash"]
    evaluations = [
        item
        for item in iter_runs(root, "evaluation.json")
        if item[1].get("configuration_hash") == configuration
    ]
    folds: dict[str, Any] = (
        {
            "status": "ok",
            "evaluation_id": evaluations[-1][0].name,
            "consistency": fold_consistency(evaluations[-1][1]["folds"]),
        }
        if evaluations
        else _unavailable("No recorded evaluation matches this configuration.")
    )
    sweeps = [
        item
        for item in iter_runs(root, "sweep.json", kind="sweeps")
        if any(point.get("configuration_hash") == configuration for point in item[1].get("points", []))
    ]
    parameters: dict[str, Any] = (
        {
            "status": "ok",
            "sweep_id": sweeps[-1][0].name,
            **sensitivity(sweeps[-1][1]["grid"], sweeps[-1][1]["points"]),
        }
        if sweeps
        else _unavailable("No recorded sweep contains this configuration.")
    )
    sections = {"regimes": regimes, "costs": costs, "folds": folds, "parameters": parameters}
    exposure_data: dict[str, Any] = {}
    warnings: list[str] = []
    if symbols:
        exposure, warnings = _diagnostic_exposure(reproduced, series, symbols)
        sections["exposure"] = exposure
        compact = {"status": exposure["status"]}
        if exposure["status"] == "ok":
            regression = exposure["regression"]
            compact.update(
                r_squared=regression["r_squared"],
                alpha_annual=regression["alpha_annual"],
                betas={row["name"]: row["beta"] for row in regression["references"]},
            )
        exposure_data["exposure"] = compact
    diagnostics_id = unique_run_id(root, run_id, kind="diagnostics")
    artifacts = write_run(
        root,
        diagnostics_id,
        {
            "diagnostics.json": result_document(
                schema="signalquarry.diagnostics/v1",
                run_id=run_id,
                diagnostics_id=diagnostics_id,
                strategy_id=strategy_id,
                configuration_hash=configuration,
                dataset_identity=document["dataset_identity"],
                evidence=document["evidence"],
                **sections,
            )
        },
        kind="diagnostics",
    )
    scored = [
        {"kind": kind, **row}
        for kind in ("calendar", "trend", "volatility")
        for row in regimes[kind].get("rows", [])
        if row["status"] == "ok" and row["sharpe"] is not None
    ]
    consistency: dict[str, Any] = folds.get("consistency", {})
    count = consistency.get("folds", 0)
    ahead = consistency.get("ahead_of_benchmark")
    return Envelope(
        command="diagnose",
        summary=f"descriptive diagnostics for {run_id}",
        evidence=document["evidence"],
        artifacts=artifacts,
        warnings=warnings,
        data={
            "run_id": run_id,
            "diagnostics_id": diagnostics_id,
            "available": [name for name, section in sections.items() if section["status"] == "ok"],
            "break_even": costs.get("break_even"),
            "share_positive": consistency.get("share_positive"),
            "share_ahead_of_benchmark": round(ahead / count, 6) if ahead is not None and count else None,
            "plateau": parameters.get("plateau"),
            "worst_regime": min(scored, key=lambda row: row["sharpe"], default=None),
            **exposure_data,
        },
    )
