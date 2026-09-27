# SPDX-License-Identifier: Apache-2.0
"""One entry point for every strategy kind: equity (session engine) or options (low-evidence simulator)."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from decimal import Decimal

from signalquarry._internal.contracts.spec import StrategySpecV1
from signalquarry._internal.data.dataset import Dataset
from signalquarry._internal.engine.backtest import BacktestResult, EngineError, run_backtest
from signalquarry._internal.engine.options_sim import run_options_backtest
from signalquarry.sdk.strategy import Params, StrategyDef


def is_options(spec: StrategySpecV1) -> bool:
    return spec.kind == "options_single_leg"


def simulate(
    spec: StrategySpecV1,
    definition: StrategyDef,
    params: Params,
    dataset: Dataset,
    *,
    start: date | None = None,
    end: date | None = None,
    cost_multiplier: Decimal = Decimal(1),
    delay_sessions: int = 0,
    recorded_chains: Mapping[tuple[str, date], Mapping[str, tuple[Decimal, Decimal]]] | None = None,
) -> BacktestResult:
    if is_options(spec) != (definition.kind == "options"):
        raise EngineError("STRATEGY_KIND_MISMATCH")
    if is_options(spec):
        if delay_sessions:
            raise EngineError("OPTIONS_DELAY_STRESS_UNSUPPORTED")
        return run_options_backtest(
            spec,
            definition,
            params,
            dataset,
            start=start,
            end=end,
            fee_multiplier=cost_multiplier,
            haircut_multiplier=cost_multiplier,
            recorded_chains=recorded_chains,
        )
    execution = spec.execution
    if cost_multiplier != 1 or delay_sessions:
        execution = execution.model_copy(
            update={
                "costs": execution.costs.model_copy(update={"bps": execution.costs.bps * cost_multiplier}),
                "execution_delay_sessions": execution.execution_delay_sessions + delay_sessions,
            }
        )
        spec = spec.model_copy(update={"execution": execution})
    return run_backtest(spec, definition, params, dataset, start=start, end=end)
