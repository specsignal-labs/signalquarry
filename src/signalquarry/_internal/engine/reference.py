# SPDX-License-Identifier: Apache-2.0
"""A buy-and-hold reference curve, simulated by the same engine as the strategy.

The reference holds one symbol at full weight from the first session it can and reinvests
cash whenever a whole order is affordable. It runs under the strategy's own account,
execution and cost settings, so splits, dividends, settlement and fees are treated exactly
as they are for the strategy. It is a yardstick, not a candidate: nothing here records a
trial or carries a claim.
"""

from __future__ import annotations

from datetime import date

from signalquarry._internal.contracts.spec import LimitsSpec, StrategySpecV1
from signalquarry._internal.data.dataset import Dataset
from signalquarry._internal.engine.backtest import BacktestResult, run_backtest
from signalquarry.sdk.context import Ctx
from signalquarry.sdk.decision import Decision
from signalquarry.sdk.strategy import Params, StrategyDef

HOLD_CODE = "REFERENCE_HOLD"


class _HoldParams(Params):
    symbol: str


def _hold(ctx: Ctx, params: Params) -> Decision:
    assert isinstance(params, _HoldParams)
    return Decision.target({params.symbol: 1}, HOLD_CODE)


_DEFINITION = StrategyDef(_hold, _HoldParams, lambda _: 1, __name__, "_hold")


def reference_spec(spec: StrategySpecV1, symbol: str) -> StrategySpecV1:
    """The strategy's account, execution and costs applied to holding ``symbol`` alone."""
    return spec.model_copy(
        update={
            "kind": "equity_daily",
            "data": spec.data.model_copy(update={"symbols": (symbol,)}),
            # Rebalance at every decision so dividends are reinvested once a share is affordable.
            "execution": spec.execution.model_copy(update={"rebalance": "every_decision"}),
            "limits": LimitsSpec(),
            "params": {"symbol": symbol},
            "reason_codes": {HOLD_CODE: "Hold the reference symbol at full weight."},
            "options": None,
            "benchmark": None,
        }
    )


def buy_and_hold(
    spec: StrategySpecV1,
    dataset: Dataset,
    symbol: str,
    *,
    start: date | None = None,
    end: date | None = None,
) -> BacktestResult:
    """Simulate holding ``symbol`` over the same sessions and assumptions as ``spec``."""
    return run_backtest(
        reference_spec(spec, symbol),
        _DEFINITION,
        _HoldParams(symbol=symbol),
        dataset,
        start=start,
        end=end,
    )
