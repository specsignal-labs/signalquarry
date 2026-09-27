# SPDX-License-Identifier: Apache-2.0
"""Strategy authoring API (stable).

A strategy is one pure function::

    from signalquarry.sdk import Ctx, Decision, Field, Params, strategy, ta

    class P(Params):
        symbol: str = "SPY"
        period: int = Field(200, ge=20, le=400)

    @strategy(params=P, lookback=lambda p: p.period)
    def decide(ctx: Ctx, p: P) -> Decision:
        close = ctx.bars(p.symbol).close
        if close[-1] > ta.sma(close, p.period)[-1]:
            return Decision.target({p.symbol: "0.95"}, "PRICE_ABOVE_SMA")
        return Decision.target({}, "PRICE_BELOW_SMA")
"""

from pydantic import Field

from signalquarry.sdk import ta
from signalquarry.sdk.context import Bars, Ctx
from signalquarry.sdk.decision import Decision
from signalquarry.sdk.strategy import Params, StrategyDef, definition_of, strategy

__all__ = ["Bars", "Ctx", "Decision", "Field", "Params", "StrategyDef", "definition_of", "strategy", "ta"]
