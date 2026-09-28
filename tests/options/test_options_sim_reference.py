# SPDX-License-Identifier: Apache-2.0
"""Short deterministic option-wheel cases for mutation testing the simulator."""

from __future__ import annotations

from datetime import date, time
from decimal import Decimal
from typing import Literal

import pytest

from signalquarry._internal.data.synthetic import synthetic_dataset
from signalquarry._internal.engine.backtest import BacktestResult, EngineError
from signalquarry._internal.engine.options_sim import _act, _at, _Book, _expire, _haircut, _leg_view
from signalquarry._internal.engine.run import simulate
from signalquarry._internal.options.chains import ChainModel
from signalquarry._internal.options.contracts import Quote, contract
from signalquarry._internal.options.resolver import QuoteRules
from signalquarry._internal.options.wheel import Leg, Wheel, WheelState
from signalquarry.sdk import definition_of
from signalquarry.sdk.options import OD, OptionsCtx, Strike, options_strategy, sell_call, sell_put
from tests.helpers import dataset
from tests.options.wheel_reference import WheelParams, wheel, wheel_spec

DATA = synthetic_dataset(date(2022, 1, 3), date(2022, 4, 1), symbols=("QQQ",))
DEFINITION = definition_of(wheel)
PARAMS = WheelParams(trend_sessions=20)


def _run(**kwargs):
    return simulate(wheel_spec(), DEFINITION, PARAMS, DATA, **kwargs)


def test_short_wheel_entry_cash_and_terminal_positions() -> None:
    result = _run()
    first = result.fills[0]
    assert (first.session, first.symbol, first.side, first.quantity, first.price, first.fee) == (
        date(2022, 1, 31),
        "QQQ220211P00049000",
        "sell",
        Decimal("1"),
        Decimal("0.0950"),
        Decimal("0.65"),
    )
    assert result.cash[0] == Decimal("100008.850000")  # 100000 + 100 * 0.095 - 0.65
    assert result.equity[0] == Decimal("100003.850000")
    assert {d["action"] for d in result.decisions} >= {"open", "close", "expired", "assigned"}
    assert result.positions == {"QQQ": Decimal("100")}
    assert result.ledger_hash == _run().ledger_hash


def test_recorded_quote_changes_fill_and_reports_provenance() -> None:
    base = _run()
    first = base.fills[0]
    recorded = {("QQQ", first.session): {first.symbol: (Decimal("9.99"), Decimal("10.01"))}}
    replay = _run(recorded_chains=recorded)
    assert replay.fills[0].price >= Decimal("9.9")
    assert replay.cash[0] > base.cash[0]
    assert "OPTIONS_RECORDED_CHAINS_USED:1" in replay.warnings
    assert not any(w.startswith("OPTIONS_RECORDED") for w in base.warnings)


def test_cost_stress_decreases_cash_after_first_entry() -> None:
    base = _run()
    stressed = _run(cost_multiplier=Decimal("3"))
    assert stressed.fills[0].fee == Decimal("1.95")
    assert stressed.cash[0] < base.cash[0]
    assert stressed.equity[-1] < base.equity[-1]


@options_strategy(params=WheelParams, lookback=lambda p: 1)
def _bad_reason(ctx: OptionsCtx, p: WheelParams):
    return OD.hold("NOT_DECLARED")


def test_options_contract_and_range_errors() -> None:
    with pytest.raises(EngineError, match="BACKTEST_RANGE_EMPTY"):
        _run(start=date(2030, 1, 1))
    with pytest.raises(EngineError, match="REASON_CODE_UNDECLARED"):
        simulate(wheel_spec(), definition_of(_bad_reason), PARAMS, DATA)


def test_checkpoint_time_haircut_and_leg_mark_are_exact() -> None:
    at = _at(date(2022, 1, 3), time(9, 35))
    assert at.isoformat() == "2022-01-03T14:35:00+00:00"
    assert _haircut(Decimal("0.015"), Decimal("0.3")) == Decimal("0.0045")
    leg = Leg(
        contract("QQQ", date(2022, 1, 14), "PUT", Decimal("100")), Decimal("0.20"), date(2022, 1, 3), "test"
    )
    wheel = Wheel("QQQ", WheelState.SHORT_PUT, leg)
    marked = _leg_view(wheel, Quote(Decimal("0.08"), Decimal("0.10"), at), date(2022, 1, 7))
    assert marked is not None
    assert (marked.dte, marked.close_cost, marked.captured_fraction) == (
        7,
        Decimal("0.10"),
        Decimal("0.500000"),
    )
    unmarked = _leg_view(wheel, None, date(2022, 1, 7))
    assert unmarked is not None and unmarked.close_cost is None and unmarked.captured_fraction is None
    assert _leg_view(Wheel("QQQ"), None, date(2022, 1, 7)) is None


@pytest.mark.parametrize(
    ("right", "close", "action", "cash", "shares"),
    [
        ("PUT", "99.99", "assigned", "0", 100),
        ("PUT", "100", "expired", "10000", 0),
        ("CALL", "100.01", "assigned", "20000", 0),
        ("CALL", "100", "expired", "10000", 100),
    ],
)
def test_expiry_one_cent_exercise_boundary(
    right: Literal["CALL", "PUT"], close: str, action: str, cash: str, shares: int
) -> None:
    session = date(2022, 1, 14)
    prices = dataset((session,), {"QQQ": {"open": [float(close)], "close": [float(close)]}})
    leg = Leg(contract("QQQ", session, right, Decimal("100")), Decimal("1"), date(2022, 1, 3), "test")
    wheel = Wheel(
        "QQQ",
        WheelState.SHORT_PUT if right == "PUT" else WheelState.COVERED_CALL,
        leg,
        0 if right == "PUT" else 100,
    )
    book = _Book(Decimal("10000"), {"QQQ": wheel})
    result = BacktestResult([], [], [], [], [], {}, [], prices.identity())
    _expire(book, prices, 0, session, Decimal("0.65"), result)
    assert book.cash == Decimal(cash)
    assert book.wheels["QQQ"].shares == shares
    assert book.wheels["QQQ"].leg is None
    assert result.decisions == [
        {
            "session": session.isoformat(),
            "checkpoint": "expiry",
            "action": action,
            "reason_codes": [],
            "contract": leg.contract.symbol,
        }
    ]
    assert [(fill.side, fill.quantity, fill.price, fill.fee) for fill in result.fills] == (
        []
        if action == "expired"
        else [
            (
                "buy" if right == "PUT" else "sell",
                Decimal("100"),
                Decimal("100"),
                Decimal("0"),
            )
        ]
    )


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("call_without_shares", "OPTIONS_OPEN_NOT_ALLOWED_IN_FLAT"),
        ("entry_at_cutoff", "OPTIONS_ENTRY_CUTOFF"),
        ("missing_open_price", "PRICE_MISSING"),
        ("close_without_leg", "OPTIONS_NO_OPEN_LEG"),
        ("missing_close_quote", "QUOTE_MISSING"),
        ("insufficient_collateral", "OPTIONS_INSUFFICIENT_COLLATERAL"),
    ],
)
def test_action_refusal_leaves_book_unchanged(case: str, expected: str) -> None:
    session = date(2022, 1, 3)
    moment = time(15, 55) if case == "entry_at_cutoff" else time(9, 35)
    at = _at(session, moment)
    put = sell_put("QQQ", dte=(7, 14), strike=Strike.otm("0.02"))
    call = sell_call("QQQ", dte=(7, 14), strike=Strike.otm("0.02"))
    decision = (
        OD.close("QQQ", "TAKE_PROFIT")
        if case.startswith("close") or case == "missing_close_quote"
        else OD.open(call if case == "call_without_shares" else put, "SELL_PUT_UPTREND")
    )
    wheel = Wheel("QQQ")
    if case == "missing_close_quote":
        leg = Leg(contract("QQQ", date(2022, 1, 14), "PUT", Decimal("95")), Decimal("1"), session, "test")
        wheel = Wheel("QQQ", WheelState.SHORT_PUT, leg)
    cash = Decimal("1") if case == "insufficient_collateral" else Decimal("10000")
    book = _Book(cash, {"QQQ": wheel})
    prices = {} if case == "missing_open_price" else {"QQQ": Decimal("100")}
    chains = (
        {
            "QQQ": ChainModel(
                "QQQ", spot=Decimal("100"), session=session, at=at, sigma=0.2, tick=Decimal("0.01")
            )
        }
        if case == "insufficient_collateral"
        else {}
    )
    result = BacktestResult([], [], [], [], [], {}, [], "test")
    outcome = _act(
        decision,
        book,
        chains,
        prices,
        session,
        at,
        moment,
        time(15, 55),
        QuoteRules(Decimal("0.01"), Decimal("0.5"), 300),
        Decimal("0.01"),
        Decimal("0.65"),
        Decimal("0.5"),
        result,
    )
    assert outcome == {"outcome": "refused", "why": expected}
    assert book.cash == cash and book.wheels == {"QQQ": wheel}
    assert result.fills == []
