# SPDX-License-Identifier: Apache-2.0
"""``sqy sweep``: backtest a parameter grid. Every point on real data is a recorded trial."""

from __future__ import annotations

import itertools
from dataclasses import replace
from datetime import UTC, datetime, timedelta
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
from signalquarry.api.resolve import resolve
from signalquarry.api.runs import execute_run

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


def sweep(
    strategy_id: str,
    assignments: list[str],
    *,
    summary_only: bool = False,
    project: Path | None = None,
) -> Envelope:
    """Backtest every grid point as its own run and record the sweep.

    Each point is written under ``.signalquarry/runs/`` like a backtest (``summary_only``
    leaves out its fills and decisions); the grid, the run ids and the PBO go to
    ``.signalquarry/sweeps/<sweep_id>/``.
    """
    envelope = Envelope(command="sweep")
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
    for number, (point, variant) in enumerate(variants):
        progress.emit("sweep", point=point, done=number, total=len(variants))
        try:
            run = execute_run(
                replace(resolved, strategy=variant),
                command="sweep",
                end=end,
                detail="summary" if summary_only else "full",
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
