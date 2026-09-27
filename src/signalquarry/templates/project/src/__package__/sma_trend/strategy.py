"""Trend filter: hold the symbol above its simple moving average, otherwise cash."""

from decimal import Decimal

from signalquarry.sdk import Ctx, Decision, Field, Params, strategy, ta


class SmaParams(Params):
    symbol: str = "{{symbol}}"
    period: int = Field(200, ge=20, le=400)
    weight: Decimal = Field(Decimal("0.95"), ge=0, le=1)


@strategy(params=SmaParams, lookback=lambda p: p.period)
def decide(ctx: Ctx, p: SmaParams) -> Decision:
    close = ctx.bars(p.symbol).close
    if close[-1] > ta.sma(close, p.period)[-1]:
        return Decision.target({p.symbol: p.weight}, "PRICE_ABOVE_SMA")
    return Decision.target({}, "PRICE_BELOW_SMA")
