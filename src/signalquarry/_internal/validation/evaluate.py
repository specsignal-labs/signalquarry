# SPDX-License-Identifier: Apache-2.0
"""Walk-forward, stress and holdout evaluation with the default gates (``signalquarry.gates/v1``).

Parameters are frozen, so walk-forward means: one continuous run over the
pre-holdout data, a ``train_months`` burn-in excluded, and consecutive
``test_months`` out-of-sample folds scored separately and together.

| Gate | Passes when | Unlocks |
|---|---|---|
| G1 sample | ≥ 5 years before the holdout and ≥ 30 rebalancing sessions | — |
| G2 walk-forward | ≥ 6 folds, OOS PSR(0) ≥ 0.95, OOS DSR ≥ 0.95 (project-wide trials) | ``walk_forward`` with G1+G3 |
| G3 stress | costs ×2 and a one-session delay each keep Sharpe ≥ 0.5 × base and net return > 0 | |
| G4 holdout | Sharpe > 0 and max drawdown ≤ 1.5 × worst walk-forward fold drawdown | ``holdout_passed`` |
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Protocol

import numpy as np

from signalquarry._internal.contracts.spec import StrategySpecV1
from signalquarry._internal.data.dataset import Dataset
from signalquarry._internal.engine.backtest import BacktestResult
from signalquarry._internal.engine.run import is_options, simulate
from signalquarry._internal.validation.stats import (
    block_bootstrap_sharpe,
    deflated_sharpe,
    min_track_record_length,
    moments,
    probabilistic_sharpe,
)
from signalquarry.sdk.strategy import Params, StrategyDef

GATES_VERSION = "signalquarry.gates/v1"
CLAIMS = ("none", "in_sample", "walk_forward", "holdout_passed", "paper_forward")


def add_months(day: date, months: int) -> date:
    year, month = divmod(day.month - 1 + months, 12)
    year, month = day.year + year, month + 1
    for candidate in range(day.day, 0, -1):
        try:
            return date(year, month, candidate)
        except ValueError:
            continue
    raise ValueError("DATE_INVALID")


class EquityCurve(Protocol):
    @property
    def equity(self) -> Sequence[Decimal]: ...


def equity_returns(result: EquityCurve, initial: Decimal) -> np.ndarray:
    values = np.array([float(initial), *(float(e) for e in result.equity)])
    return values[1:] / values[:-1] - 1.0


def max_drawdown(returns: np.ndarray) -> float:
    if not len(returns):
        return 0.0
    curve = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(np.concatenate(([1.0], curve)))
    return float((1.0 - np.concatenate(([1.0], curve)) / peak).max())


def _window(result: BacktestResult, returns: np.ndarray, start: date, end: date) -> np.ndarray:
    mask = np.array([start <= s <= end for s in result.sessions])
    return returns[mask]


def _annual_interval(interval: tuple[float, float]) -> list[float] | None:
    low, high = interval
    if not (math.isfinite(low) and math.isfinite(high)):
        return None
    return [round(low * math.sqrt(252), 4), round(high * math.sqrt(252), 4)]


def _stats(returns: np.ndarray) -> dict[str, Any]:
    m = moments(returns)
    return {
        "sessions": int(len(returns)),
        "total_return": float(np.prod(1.0 + returns) - 1.0) if len(returns) else 0.0,
        "sharpe_daily": None if not math.isfinite(m.sharpe) else round(m.sharpe, 8),
        "sharpe_annual": None if not math.isfinite(m.sharpe) else round(m.sharpe * math.sqrt(252), 6),
        "max_drawdown": round(max_drawdown(returns), 8),
    }


@dataclass
class Evaluation:
    gates: dict[str, dict[str, Any]] = field(default_factory=dict)
    folds: list[dict[str, Any]] = field(default_factory=list)
    oos: dict[str, Any] = field(default_factory=dict)
    trial_moments: dict[str, float] = field(default_factory=dict)
    oos_returns: tuple[float, ...] = ()
    base: BacktestResult | None = None

    def passed(self, *names: str) -> bool:
        return all(self.gates.get(name, {}).get("ok") is True for name in names)


def evaluate(
    spec: StrategySpecV1,
    definition: StrategyDef,
    params: Params,
    dataset: Dataset,
    *,
    holdout_start: date | None,
    project_trials: int,
    sharpe_variance: float,
    stress: bool = True,
    open_holdout: bool = False,
    recorded_chains: Mapping[tuple[str, date], Mapping[str, tuple[Decimal, Decimal]]] | None = None,
) -> Evaluation:
    evaluation = Evaluation()
    pre_end = (holdout_start - timedelta(days=1)) if holdout_start else dataset.sessions[-1]
    options = is_options(spec)
    base = simulate(spec, definition, params, dataset, end=pre_end, recorded_chains=recorded_chains)
    evaluation.base = base
    returns = equity_returns(base, spec.account.initial_cash)
    first = base.sessions[0]
    oos_start = add_months(first, spec.evaluation.walk_forward.train_months)
    folds: list[dict[str, Any]] = []
    cursor = oos_start
    while True:
        fold_end = add_months(cursor, spec.evaluation.walk_forward.test_months) - timedelta(days=1)
        if fold_end > pre_end:
            break
        folds.append(
            {
                "start": cursor.isoformat(),
                "end": fold_end.isoformat(),
                **_stats(_window(base, returns, cursor, fold_end)),
            }
        )
        cursor = fold_end + timedelta(days=1)
    evaluation.folds = folds
    oos_returns = _window(base, returns, oos_start, pre_end)
    evaluation.oos_returns = tuple(float(r) for r in oos_returns)
    m = moments(oos_returns)
    evaluation.trial_moments = {"n": m.n, "sharpe": m.sharpe, "skew": m.skew, "kurtosis": m.kurtosis}
    psr = probabilistic_sharpe(m)
    dsr = deflated_sharpe(m, max(project_trials, 1), sharpe_variance)
    evaluation.oos = {
        **_stats(oos_returns),
        "psr": None if math.isnan(psr) else round(psr, 6),
        "dsr": None if math.isnan(dsr) else round(dsr, 6),
        "min_track_record_sessions": None
        if not math.isfinite(min_track_record_length(m))
        else round(min_track_record_length(m), 1),
        "project_trials": project_trials,
        # Moving-block bootstrap (20-session blocks, fixed seed): how uncertain the Sharpe is.
        "sharpe_annual_90": _annual_interval(block_bootstrap_sharpe(oos_returns)),
    }

    years = (pre_end - first).days / 365.25
    rebalances = len({fill.session for fill in base.fills})
    evaluation.gates["G1_sample"] = {
        "ok": years >= 5 and rebalances >= 30,
        "years": round(years, 2),
        "rebalancing_sessions": rebalances,
    }
    evaluation.gates["G2_walk_forward"] = {
        "ok": len(folds) >= 6 and psr >= 0.95 and dsr >= 0.95,
        "folds": len(folds),
        "psr": evaluation.oos["psr"],
        "dsr": evaluation.oos["dsr"],
    }
    if stress:
        base_sharpe = m.sharpe if math.isfinite(m.sharpe) else 0.0
        scenarios = {}
        # Options have no execution-delay model: only the cost scenario applies.
        stresses: dict[str, dict[str, Any]] = {"costs_x2": {"cost_multiplier": Decimal(2)}}
        if not options:
            stresses["delay_1"] = {"delay_sessions": 1}
        for name, stress_args in stresses.items():
            run = simulate(
                spec, definition, params, dataset, end=pre_end, recorded_chains=recorded_chains, **stress_args
            )
            window = _window(run, equity_returns(run, spec.account.initial_cash), oos_start, pre_end)
            stats = _stats(window)
            sharpe = stats["sharpe_daily"] or 0.0
            scenarios[name] = {
                **stats,
                "ok": base_sharpe > 0 and sharpe >= 0.5 * base_sharpe and stats["total_return"] > 0,
            }
        evaluation.gates["G3_stress"] = {
            "ok": all(s["ok"] for s in scenarios.values()),
            "scenarios": scenarios,
        }
    if open_holdout and holdout_start is not None and not options:
        full = simulate(spec, definition, params, dataset, recorded_chains=recorded_chains)
        window = _window(
            full, equity_returns(full, spec.account.initial_cash), holdout_start, dataset.sessions[-1]
        )
        stats = _stats(window)
        worst_fold = max((fold["max_drawdown"] for fold in folds), default=0.0)
        evaluation.gates["G4_holdout"] = {
            **stats,
            "ok": (stats["sharpe_daily"] or 0) > 0 and stats["max_drawdown"] <= 1.5 * max(worst_fold, 1e-9),
            "worst_walk_forward_drawdown": worst_fold,
        }
    return evaluation


def claim_level(evaluation: Evaluation, *, grade: str, frozen: bool, conformance_ok: bool) -> str:
    if grade == "synthetic" or not conformance_ok:
        return "none"
    # Plugin gates are add-only: a failing one caps the claim, a passing one unlocks nothing.
    if any(not gate.get("ok") for name, gate in evaluation.gates.items() if name.startswith("plugin:")):
        return "in_sample"
    if not frozen or not evaluation.passed("G1_sample", "G2_walk_forward", "G3_stress"):
        return "in_sample"
    if grade == "low_evidence_options":
        return "walk_forward"  # options: no holdout gate; only a paper forward test (G5) goes higher
    if evaluation.passed("G4_holdout"):
        return "holdout_passed"
    return "walk_forward"
