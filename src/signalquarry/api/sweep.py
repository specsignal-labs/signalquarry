# SPDX-License-Identifier: Apache-2.0
"""``sqy sweep``: backtest a parameter grid. Every point on real data is a recorded trial."""

from __future__ import annotations

import itertools
from concurrent.futures import Future
from contextlib import ExitStack
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import ValidationError

from signalquarry._internal.contracts import progress
from signalquarry._internal.engine.backtest import EngineError
from signalquarry._internal.evidence.runs import (
    csv_text,
    new_run_id,
    result_document,
    unique_run_id,
    write_run,
)
from signalquarry._internal.project.project import LoadedStrategy
from signalquarry._internal.validation.benchmark import BenchmarkCurve
from signalquarry._internal.validation.stats import pbo_cscv
from signalquarry.api.envelope import Envelope
from signalquarry.api.evidence import budget_shortfall, budget_warnings, family_seal, trial_evidence
from signalquarry.api.resolve import Resolved, resolve
from signalquarry.api.runs import (
    Detail,
    Simulated,
    commit_run,
    execute_run,
    simulate_run,
    simulation_pool,
)

MAX_POINTS = 200
SWEEP_SCHEMA = "signalquarry.sweep/v1"


def _grid(assignments: list[str]) -> list[dict[str, Any]] | str:
    axes: list[tuple[str, list[Any]]] = []
    for assignment in assignments:
        name, _, values = assignment.partition("=")
        if not name or not values:
            return f"expected NAME=V1,V2,…, got {assignment!r}"
        axes.append((name.strip(), [yaml.safe_load(v.strip()) for v in values.split(",") if v.strip()]))
    points = [
        dict(zip([a for a, _ in axes], combo, strict=True))
        for combo in itertools.product(*(v for _, v in axes))
    ]
    return points


@dataclass(frozen=True)
class _Job:
    """What a worker process needs to rebuild one grid point from the project and simulate it."""

    strategy_id: str
    root: Path
    point: dict[str, Any]
    configuration_hash: str
    dataset_identity: str
    end: date | None
    scratch: Path


_worker_resolved: dict[tuple[Path, str], tuple[Resolved, str] | None] = {}  # one per worker process


def _simulate_point(job: _Job) -> Simulated | str:
    """Simulate one grid point in a worker process; what changed, when the project is not the parent's.

    It appends nothing to the evidence and writes only into ``job.scratch``: the parent records
    and writes every point, in grid order.
    """
    key = (job.root, job.strategy_id)
    if key not in _worker_resolved:
        resolved = resolve("sweep", job.strategy_id, job.root)
        _worker_resolved[key] = (
            None if isinstance(resolved, Envelope) else (resolved, resolved.dataset.identity())
        )
    found = _worker_resolved[key]
    if found is None:
        return "the strategy no longer resolves"
    resolved, dataset_identity = found
    strategy = resolved.strategy
    try:
        params = strategy.definition.params.model_validate({**strategy.params.model_dump(), **job.point})
    except ValidationError:
        return "its parameters are no longer valid"
    variant = replace(strategy, params=params)
    if variant.configuration_hash != job.configuration_hash:
        return "its configuration is not the one that was checked"
    if dataset_identity != job.dataset_identity:
        return "the dataset is not the one that was checked"
    job.scratch.mkdir()
    return simulate_run(replace(resolved, strategy=variant), end=job.end, scratch=job.scratch)


def sweep(
    strategy_id: str,
    assignments: list[str],
    *,
    summary_only: bool = False,
    jobs: int = 1,
    project: Path | None = None,
) -> Envelope:
    """Backtest every grid point as its own run and record the sweep.

    Each point is written under ``.signalquarry/runs/`` like a backtest (``summary_only``
    leaves out its fills and decisions); the grid, the run ids and the PBO go to
    ``.signalquarry/sweeps/<sweep_id>/``. ``jobs`` above 1 simulates the points in that many
    worker processes; trials and runs are still written by this process in grid order, so
    what is recorded does not depend on ``jobs``.
    """
    envelope = Envelope(command="sweep")
    if jobs < 1:
        envelope.status, envelope.reason_codes = "usage", ["USAGE_INVALID"]
        envelope.summary = "--jobs must be at least 1"
        return envelope
    grid = _grid(assignments)
    if isinstance(grid, str) or not grid:
        envelope.status, envelope.reason_codes = "usage", ["USAGE_INVALID"]
        envelope.summary = grid if isinstance(grid, str) and grid else "no grid"
        return envelope
    if len(grid) > MAX_POINTS:
        envelope.status, envelope.reason_codes = "usage", ["SWEEP_TOO_LARGE"]
        envelope.summary = (
            f"{len(grid)} points; the limit is {MAX_POINTS} (each point on real data is a trial)"
        )
        return envelope
    resolved = resolve("sweep", strategy_id, project)
    if isinstance(resolved, Envelope):
        return resolved
    root, strategy = resolved.root, resolved.strategy
    spec = strategy.spec
    base = strategy.params.model_dump()
    variants: list[tuple[dict[str, Any], LoadedStrategy]] = []
    for point in grid:
        try:
            params = strategy.definition.params.model_validate({**base, **point})
        except ValidationError as exc:
            envelope.status, envelope.reason_codes = "invalid", ["STRATEGY_PARAMS_INVALID"]
            envelope.summary = f"{point}: {exc.errors()[0]['msg']}"
            return envelope
        variants.append((point, replace(strategy, params=params)))
    if resolved.grade != "synthetic":
        shortfall = budget_shortfall(root, spec, {v.configuration_hash for _, v in variants})
        if shortfall is not None:
            envelope.status, envelope.reason_codes = "blocked", ["TRIAL_BUDGET_EXHAUSTED"]
            envelope.summary = shortfall
            return envelope
    seal = family_seal(root, spec.family)
    end = seal - timedelta(days=1) if seal is not None else None
    rows: list[dict[str, Any]] = []
    series: list[np.ndarray] = []
    benchmarks: dict[Any, BenchmarkCurve | None] = {}
    warnings: set[str] = set()
    dataset_identity = ""
    with ExitStack() as stack:
        simulations: list[Future[Simulated | str]] = []
        if jobs > 1 and len(variants) > 1:
            pool, scratch = simulation_pool(stack, min(jobs, len(variants)))
            identity = resolved.dataset.identity()
            simulations = [
                pool.submit(
                    _simulate_point,
                    _Job(
                        strategy_id,
                        root,
                        point,
                        variant.configuration_hash,
                        identity,
                        end,
                        scratch / str(number),
                    ),
                )
                for number, (point, variant) in enumerate(variants)
            ]
        for number, (point, variant) in enumerate(variants):
            progress.emit("sweep", point=point, done=number, total=len(variants))
            detail: Detail = "summary" if summary_only else "full"
            try:
                if simulations:
                    simulated = simulations[number].result()  # an EngineError in the worker is raised here
                    if isinstance(simulated, str):
                        envelope.status, envelope.reason_codes = "blocked", ["PROJECT_CHANGED_DURING_RUN"]
                        envelope.summary = f"{point}: {simulated}; the project changed while the sweep ran"
                        return envelope
                    run = commit_run(
                        replace(resolved, strategy=variant),
                        simulated,
                        command="sweep",
                        end=end,
                        detail=detail,
                        benchmarks=benchmarks,
                    )
                else:
                    run = execute_run(
                        replace(resolved, strategy=variant),
                        command="sweep",
                        end=end,
                        detail=detail,
                        benchmarks=benchmarks,
                    )
            except EngineError as exc:
                envelope.status, envelope.reason_codes, envelope.summary = (
                    "invalid",
                    [str(exc).split(":", 1)[0]],
                    str(exc),
                )
                return envelope
            warnings.update(run.warnings)
            dataset_identity = run.dataset_identity
            block: dict[str, Any] = run.context.get("benchmark") or {}
            relative: dict[str, Any] = block.get("relative", {})
            rows.append(
                {
                    "params": point,
                    "configuration_hash": variant.configuration_hash,
                    "run_id": run.run_id,
                    "total_return": run.metrics.get("total_return"),
                    "sharpe": run.metrics.get("sharpe"),
                    "max_drawdown": run.metrics.get("max_drawdown"),
                    "fills": run.fill_count,
                    **(
                        {"excess_total_return": relative["excess_total_return"]}
                        if "excess_total_return" in relative
                        else {}
                    ),
                }
            )
            series.append(run.returns)
    length = min(len(s) for s in series)
    pbo = pbo_cscv(np.column_stack([s[-length:] for s in series])) if len(series) >= 2 else None
    envelope.evidence = {
        "grade": resolved.evidence_grade,
        "claim_level": "none" if resolved.grade == "synthetic" else "in_sample",
        "trial": None
        if resolved.grade == "synthetic"
        else trial_evidence(root, spec.family, strategy.configuration_hash),
    }
    axes = list(grid[0])
    sweep_id = unique_run_id(
        root, new_run_id(strategy.configuration_hash, datetime.now(UTC)) + "-sweep", kind="sweeps"
    )
    envelope.artifacts = write_run(
        root,
        sweep_id,
        {
            "sweep.json": result_document(
                schema=SWEEP_SCHEMA,
                sweep_id=sweep_id,
                strategy_id=spec.id,
                family=spec.family,
                base_configuration_hash=strategy.configuration_hash,
                dataset_id=resolved.dataset_id,
                dataset_identity=dataset_identity,
                grid={axis: list(dict.fromkeys(point[axis] for point in grid)) for axis in axes},
                points=rows,
                pbo=pbo,
                holdout_clipped=seal is not None,
                summary_only=summary_only,
                evidence=envelope.evidence,
            ),
            "sweep.csv": csv_text(
                [
                    *axes,
                    "configuration_hash",
                    "run_id",
                    "total_return",
                    "sharpe",
                    "max_drawdown",
                    "fills",
                    "excess_total_return",
                ],
                [
                    [
                        *(str(row["params"][axis]) for axis in axes),
                        row["configuration_hash"],
                        row["run_id"],
                        *(
                            "" if row.get(key) is None else str(row[key])
                            for key in (
                                "total_return",
                                "sharpe",
                                "max_drawdown",
                                "fills",
                                "excess_total_return",
                            )
                        ),
                    ]
                    for row in rows
                ],
            ),
        },
        kind="sweeps",
    )
    envelope.data = {"sweep_id": sweep_id, "points": rows, "pbo": pbo, "holdout_clipped": seal is not None}
    envelope.warnings += sorted(warnings)
    if resolved.grade != "synthetic":
        envelope.warnings += budget_warnings(root, spec.evaluation.trial_budget, spec.family)
    envelope.summary = (
        f"{len(rows)} configurations of {strategy_id}"
        + (f"; PBO {pbo['pbo']:.2f} over {pbo['splits']} CSCV splits" if pbo and pbo["splits"] else "")
        + ("" if resolved.grade == "synthetic" else f"; {len(rows)} trials recorded")
    )
    envelope.warnings.append("SWEEP_RESULTS_ARE_IN_SAMPLE")
    envelope.next_actions = [
        {
            "command": "edit strategy.yaml params, then sqy spec freeze",
            "why": "Choose one configuration by reasoning, not by the top row.",
        },
        {
            "command": f"sqy evaluate --strategy {strategy_id}",
            "why": "Only walk-forward gates on the frozen choice support a claim.",
        },
    ]
    return envelope
