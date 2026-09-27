"""A trend-aware wheel: cash-secured puts, then covered calls after assignment.

Rules (from a published reference wheel): the trend is the last completed close
against its trailing mean; strikes sit a fixed fraction out of the money that depends
on the trend; expirations 7-14 days out; buy the leg back once more than the target
fraction of its premium is captured; at most one new entry per underlying per week.
"""

from decimal import Decimal

from signalquarry.sdk import Field, Params
from signalquarry.sdk.options import (
    OD,
    OptionsCtx,
    OptionsDecision,
    Strike,
    options_strategy,
    sell_call,
    sell_put,
)


class WheelParams(Params):
    underlying: str = "{{underlying}}"
    trend_sessions: int = Field(50, ge=20, le=252)
    min_dte: int = Field(7, ge=1, le=30)
    max_dte: int = Field(14, ge=1, le=45)
    uptrend_put_otm: Decimal = Field(Decimal("0.01"), ge=0, lt=Decimal("0.2"))
    downtrend_put_otm: Decimal = Field(Decimal("0.03"), ge=0, lt=Decimal("0.2"))
    uptrend_call_otm: Decimal = Field(Decimal("0.03"), ge=0, lt=Decimal("0.2"))
    downtrend_call_otm: Decimal = Field(Decimal("0.01"), ge=0, lt=Decimal("0.2"))
    take_profit: Decimal = Field(Decimal("0.15"), gt=0, lt=1)


@options_strategy(params=WheelParams, lookback=lambda p: p.trend_sessions)
def decide(ctx: OptionsCtx, p: WheelParams) -> OptionsDecision:
    leg = ctx.leg(p.underlying)
    if leg is not None:
        if leg.captured_fraction is not None and leg.captured_fraction > p.take_profit:
            return OD.close(p.underlying, "TAKE_PROFIT")
        return OD.hold("HOLD_LEG")
    week = ctx.decision_time.isocalendar()[:2]
    if list(ctx.state.get("last_entry_week", [])) == list(week):
        return OD.hold("WEEKLY_ENTRY_USED")
    closes = ctx.bars(p.underlying).close
    trend_up = closes[-1] > closes[-p.trend_sessions :].mean()
    dte = (p.min_dte, p.max_dte)
    state = {"last_entry_week": list(week)}
    if ctx.wheel(p.underlying).state == "flat":
        otm = p.uptrend_put_otm if trend_up else p.downtrend_put_otm
        code = "SELL_PUT_UPTREND" if trend_up else "SELL_PUT_DOWNTREND"
        return OD.open(sell_put(p.underlying, dte=dte, strike=Strike.otm(otm)), code, state=state)
    otm = p.uptrend_call_otm if trend_up else p.downtrend_call_otm
    code = "SELL_CALL_UPTREND" if trend_up else "SELL_CALL_DOWNTREND"
    return OD.open(sell_call(p.underlying, dte=dte, strike=Strike.otm(otm)), code, state=state)
