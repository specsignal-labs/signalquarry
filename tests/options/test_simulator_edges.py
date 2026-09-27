# SPDX-License-Identifier: Apache-2.0
"""Error and refusal paths of the options simulator and the simulate() dispatcher."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal

import numpy as np
import pytest

from signalquarry._internal.data.dataset import Split, SymbolSeries
from signalquarry._internal.data.synthetic import synthetic_dataset
from signalquarry._internal.engine.backtest import EngineError
from signalquarry._internal.engine.run import simulate
from signalquarry.sdk import Ctx, Decision, Params, definition_of, strategy
from signalquarry.sdk.options import OD, OptionsCtx, Strike, options_strategy, sell_call, sell_put
from tests.helpers import spec as equity_spec
from tests.options.wheel_reference import WheelParams, wheel, wheel_spec

DATA = synthetic_dataset(date(2022, 1, 3), date(2023, 6, 30), symbols=("QQQ",))


def _decisions(result, key="why"):
    return {d.get(key) for d in result.decisions if d.get("outcome") == "refused"}


@options_strategy(params=WheelParams, lookback=lambda p: 5)
def always_call(ctx: OptionsCtx, p: WheelParams):
    return OD.open(sell_call("QQQ", dte=(7, 14), strike=Strike.otm("0.02")), "SELL_CALL_UPTREND")


@options_strategy(params=WheelParams, lookback=lambda p: 5)
def always_close(ctx: OptionsCtx, p: WheelParams):
    return OD.close("QQQ", "TAKE_PROFIT")


@options_strategy(params=WheelParams, lookback=lambda p: 5)
def far_out(ctx: OptionsCtx, p: WheelParams):
    return OD.open(sell_put("QQQ", dte=(100, 110), strike=Strike.otm("0.02")), "SELL_PUT_UPTREND")


@options_strategy(params=WheelParams, lookback=lambda p: 5)
def returns_garbage(ctx: OptionsCtx, p: WheelParams):
    return "nope"


@options_strategy(params=WheelParams, lookback=lambda p: 5)
def other_underlying(ctx: OptionsCtx, p: WheelParams):
    return OD.close("SPY", "TAKE_PROFIT")


def _sim(decide, *, data=DATA, **spec_overrides):
    return simulate(wheel_spec(**spec_overrides), definition_of(decide), WheelParams(), data)


def test_refusals() -> None:
    assert "OPTIONS_OPEN_NOT_ALLOWED_IN_FLAT" in _decisions(_sim(always_call))
    assert "OPTIONS_NO_OPEN_LEG" in _decisions(_sim(always_close))
    strict = _sim(far_out, options={"underlyings": ["QQQ"], "max_relative_spread": "0.0001"})
    assert "OPTIONS_NO_ELIGIBLE_CONTRACT" in _decisions(strict)
    poor = _sim(wheel, account={"model": "cash", "initial_cash": "1000"})
    assert "OPTIONS_INSUFFICIENT_COLLATERAL" in _decisions(poor)


def test_contract_violations() -> None:
    with pytest.raises(EngineError, match="STRATEGY_RETURNED_NON_DECISION"):
        _sim(returns_garbage)
    with pytest.raises(EngineError, match="DECISION_SYMBOL_NOT_DECLARED"):
        _sim(other_underlying)
    with pytest.raises(EngineError, match="DATASET_SYMBOLS_MISSING"):
        _sim(wheel, data=synthetic_dataset(date(2022, 1, 3), date(2022, 6, 30), symbols=("SPY",)))
    with pytest.raises(EngineError, match="BACKTEST_RANGE_EMPTY"):
        simulate(wheel_spec(), definition_of(wheel), WheelParams(), DATA, start=date(2030, 1, 1))

    @options_strategy(params=WheelParams, lookback=lambda p: 0)
    def no_history(ctx: OptionsCtx, p: WheelParams):
        return OD.hold("HOLD_LEG")

    with pytest.raises(EngineError, match="STRATEGY_LOOKBACK_INVALID"):
        _sim(no_history)
    with pytest.raises(EngineError, match="OPTIONS_DELAY_STRESS_UNSUPPORTED"):
        simulate(wheel_spec(), definition_of(wheel), WheelParams(), DATA, delay_sessions=1)


def test_split_while_the_wheel_is_open_stops_the_run() -> None:
    split_day = DATA.sessions[200]
    data = replace(DATA, splits=(Split("QQQ", split_day, Decimal(2)),))
    data.__post_init__()
    with pytest.raises(EngineError, match="CORPORATE_ACTION_UNSUPPORTED"):
        simulate(wheel_spec(), definition_of(wheel), WheelParams(take_profit=Decimal("0.99")), data)


def test_missing_price_at_expiry_stops_the_run() -> None:
    item = DATA.series["QQQ"]
    present = item.present.copy()
    present[120:] = False
    data = replace(DATA, series={"QQQ": SymbolSeries(item.micro, item.volume, present)})
    with pytest.raises(EngineError, match="PRICE_MISSING"):
        simulate(wheel_spec(), definition_of(wheel), WheelParams(take_profit=Decimal("0.99")), data)


class EP(Params):
    pass


@strategy(params=EP, lookback=lambda p: 1)
def hold_aaa(ctx: Ctx, p: EP) -> Decision:
    return Decision.target({"QQQ": "0.5"}, "GO")


def test_equity_stress_paths_through_simulate() -> None:
    spec = equity_spec(("QQQ",))
    base = simulate(spec, definition_of(hold_aaa), EP(), DATA)
    costly = simulate(spec, definition_of(hold_aaa), EP(), DATA, cost_multiplier=Decimal(3))
    delayed = simulate(spec, definition_of(hold_aaa), EP(), DATA, delay_sessions=1)
    assert costly.equity[-1] < base.equity[-1]
    assert delayed.fills[0].session > base.fills[0].session
    with pytest.raises(EngineError, match="STRATEGY_KIND_MISMATCH"):
        simulate(spec, definition_of(wheel), WheelParams(), DATA)
    assert np.isfinite(float(base.equity[-1]))
