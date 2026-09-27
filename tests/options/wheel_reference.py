# SPDX-License-Identifier: Apache-2.0
"""The reference wheel (a published trend-aware wheel rule set) in the options authoring API, for tests."""

from __future__ import annotations

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
    underlying: str = "QQQ"
    trend_sessions: int = Field(50, ge=20, le=252)
    min_dte: int = Field(7, ge=1, le=30)
    max_dte: int = Field(14, ge=1, le=45)
    uptrend_put_otm: Decimal = Decimal("0.01")
    downtrend_put_otm: Decimal = Decimal("0.03")
    uptrend_call_otm: Decimal = Decimal("0.03")
    downtrend_call_otm: Decimal = Decimal("0.01")
    take_profit: Decimal = Field(Decimal("0.15"), gt=0, lt=1)


@options_strategy(params=WheelParams, lookback=lambda p: p.trend_sessions)
def wheel(ctx: OptionsCtx, p: WheelParams) -> OptionsDecision:
    closes = ctx.bars(p.underlying).close
    trend_up = closes[-1] > closes[-p.trend_sessions :].mean()
    state = ctx.wheel(p.underlying).state
    leg = ctx.leg(p.underlying)
    if leg is not None:
        if leg.captured_fraction is not None and leg.captured_fraction > p.take_profit:
            return OD.close(p.underlying, "TAKE_PROFIT")
        return OD.hold("HOLD_LEG")
    dte = (p.min_dte, p.max_dte)
    if state == "flat":
        otm = p.uptrend_put_otm if trend_up else p.downtrend_put_otm
        return OD.open(
            sell_put(p.underlying, dte=dte, strike=Strike.otm(otm)),
            "SELL_PUT_UPTREND" if trend_up else "SELL_PUT_DOWNTREND",
        )
    if state == "long_shares":
        otm = p.uptrend_call_otm if trend_up else p.downtrend_call_otm
        return OD.open(
            sell_call(p.underlying, dte=dte, strike=Strike.otm(otm)),
            "SELL_CALL_UPTREND" if trend_up else "SELL_CALL_DOWNTREND",
        )
    return OD.hold("HOLD_LEG")


CODES = {
    "TAKE_PROFIT": "More than the target fraction of the premium was captured; buy the leg back.",
    "HOLD_LEG": "A leg is open and the profit target is not reached.",
    "SELL_PUT_UPTREND": "Flat and above the trend mean: sell a put 1% out of the money.",
    "SELL_PUT_DOWNTREND": "Flat and below the trend mean: sell a put 3% out of the money.",
    "SELL_CALL_UPTREND": "Holding shares and above the trend mean: sell a call 3% out of the money.",
    "SELL_CALL_DOWNTREND": "Holding shares and below the trend mean: sell a call 1% out of the money.",
}


def wheel_spec(**overrides):
    from signalquarry._internal.contracts.spec import StrategySpecV1

    body = {
        "schema": "signalquarry.strategy/v1",
        "id": "wheel-reference",
        "family": "wheel",
        "version": "1.0.0",
        "kind": "options_single_leg",
        "hypothesis": {
            "statement": "Selling short-dated out-of-the-money puts and calls on QQQ harvests premium.",
            "falsification": "The wheel underperforms holding QQQ after fees and spread costs.",
        },
        "data": {"symbols": ["QQQ"], "feed": "synthetic"},
        "account": {"model": "cash", "initial_cash": "100000"},
        "reason_codes": CODES,
        "options": {"underlyings": ["QQQ"], "per_contract_fee": "0.65"},
    }
    body.update(overrides)
    return StrategySpecV1.model_validate(body)
