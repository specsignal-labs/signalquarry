"""Transfer-drill strategy. It exists to rehearse a family sale; it never trades."""

from decimal import Decimal

from signalquarry.sdk import Ctx, Decision, Field, Params, strategy, ta


class DrillParams(Params):
    symbol: str = "SPY"
    period: int = Field(100, ge=20, le=400)
    weight: Decimal = Field(Decimal("0.5"), ge=0, le=1)


@strategy(params=DrillParams, lookback=lambda p: p.period)
def decide(ctx: Ctx, p: DrillParams) -> Decision:
    close = ctx.bars(p.symbol).close
    if close[-1] > ta.sma(close, p.period)[-1]:
        return Decision.target({p.symbol: p.weight}, "ABOVE_AVERAGE")
    return Decision.target({}, "BELOW_AVERAGE")
