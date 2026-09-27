# SPDX-License-Identifier: Apache-2.0
"""``sqy sweep``: backtest a parameter grid. Every point on real data is a recorded trial."""

from __future__ import annotations

import itertools
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import ValidationError

from signalquarry._internal.contracts import progress
from signalquarry._internal.engine.backtest import EngineError
from signalquarry._internal.engine.run import simulate
from signalquarry._internal.project.project import LoadedStrategy
from signalquarry._internal.validation import ledger
from signalquarry._internal.validation.evaluate import equity_returns
from signalquarry._internal.validation.metrics import summarize
from signalquarry._internal.validation.stats import moments, pbo_cscv
from signalquarry.api.envelope import Envelope
from signalquarry.api.evidence import (
    budget_warnings,
    family_seal,
    record_run_trial,
    trial_budget,
    trial_evidence,
)
from signalquarry.api.resolve import resolve

MAX_POINTS = 200


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


def sweep(strategy_id: str, assignments: list[str], *, project: Path | None = None) -> Envelope:
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
        known = {
            e["configuration_hash"]
            for e in ledger.trials(root, spec.family).entries()
            if e.get("kind") == "trial"
        }
        new = {v.configuration_hash for _, v in variants} - known
        used = ledger.trial_summary(root, spec.family)["family_count"]
        budget = trial_budget(root, spec.evaluation.trial_budget, spec.family)
        if used + len(new) > budget:
            envelope.status, envelope.reason_codes = "blocked", ["TRIAL_BUDGET_EXHAUSTED"]
            envelope.summary = (
                f"{len(new)} new trials would exceed family {spec.family}'s budget ({used}/{budget} used)"
            )
            return envelope
    seal = family_seal(root, spec.family)
    end = seal - timedelta(days=1) if seal is not None else None
    rows: list[dict[str, Any]] = []
    series: list[np.ndarray] = []
    chains = resolved.recorded_chains()
    for number, (point, variant) in enumerate(variants):
        progress.emit("sweep", point=point, done=number, total=len(variants))
        try:
            result = simulate(
                spec, variant.definition, variant.params, resolved.dataset, end=end, recorded_chains=chains
            )
        except EngineError as exc:
            envelope.status, envelope.reason_codes, envelope.summary = (
                "invalid",
                [str(exc).split(":", 1)[0]],
                str(exc),
            )
            return envelope
        returns = equity_returns(result, spec.account.initial_cash)
        m = moments(returns)
        record_run_trial(
            replace(resolved, strategy=variant),
            command="sweep",
            returns=returns,
            moments={"n": m.n, "sharpe": m.sharpe, "skew": m.skew, "kurtosis": m.kurtosis},
            window=(result.sessions[0], result.sessions[-1]),
        )
        metrics = summarize(
            result.sessions,
            result.equity,
            spec.account.initial_cash,
            fills=len(result.fills),
            fees=Decimal(0),
        )
        rows.append(
            {
                "params": point,
                "configuration_hash": variant.configuration_hash,
                "total_return": metrics.get("total_return"),
                "sharpe": metrics.get("sharpe"),
                "max_drawdown": metrics.get("max_drawdown"),
                "fills": len(result.fills),
            }
        )
        series.append(returns)
    length = min(len(s) for s in series)
    pbo = pbo_cscv(np.column_stack([s[-length:] for s in series])) if len(series) >= 2 else None
    envelope.data = {"points": rows, "pbo": pbo, "holdout_clipped": seal is not None}
    envelope.evidence = {
        "grade": resolved.evidence_grade,
        "claim_level": "none" if resolved.grade == "synthetic" else "in_sample",
        "trial": None
        if resolved.grade == "synthetic"
        else trial_evidence(root, spec.family, strategy.configuration_hash),
    }
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
