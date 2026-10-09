# SPDX-License-Identifier: Apache-2.0
"""One way to execute a backtest and persist it as a run directory.

``sqy backtest`` and ``sqy sweep`` both go through :func:`execute_run`, so every run has the
same artifacts, the same trial accounting and the same descriptive context, and runs can be
compared with each other.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import numpy as np

from signalquarry._internal.engine.backtest import EngineError
from signalquarry._internal.engine.run import is_options, simulate, simulate_equity_ticks
from signalquarry._internal.evidence.run_spool import spool_equity_run
from signalquarry._internal.evidence.runs import (
    csv_chunks,
    jsonl_chunks,
    new_run_id,
    result_document,
    unique_run_id,
    write_run,
)
from signalquarry._internal.validation.benchmark import BenchmarkCurve, benchmark_curve, benchmark_summary
from signalquarry._internal.validation.evaluate import equity_returns
from signalquarry._internal.validation.metrics import drawdown_episodes, summarize, trading_activity
from signalquarry._internal.validation.stats import moments
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
) -> RunOutcome:
    """Simulate ``resolved.strategy``, record its trial and write its run directory.

    Raises :class:`EngineError` before anything is written when the simulation fails.
    ``detail="summary"`` keeps ``result.json`` and the equity curves and leaves out fills and
    decisions. ``benchmarks`` lets a caller that runs many variants share benchmark curves.
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
        files: dict[str, str | bytes | Iterable[str | bytes]] = {
            "result.json": result_document(
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
                spec=spec.outcome_document(),
                **({} if label is None else {"label": label}),
                **context,
            ),
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
        warnings=warnings,
        next_actions=next_actions,
    )
