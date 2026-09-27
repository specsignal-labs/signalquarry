# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal

import numpy as np
import pytest

from signalquarry._internal.data.dataset import MICRO
from signalquarry._internal.data.synthetic import synthetic_dataset
from signalquarry._internal.engine.backtest import EngineError, run_backtest
from signalquarry.sdk import Ctx, Decision, Field, Params, definition_of, strategy, ta
from tests.helpers import Dividend, Split, dataset, spec, weekdays


class P(Params):
    weight: Decimal = Decimal("0.5")
    symbol: str = "AAA"


@strategy(params=P, lookback=lambda p: 1)
def always_target(ctx: Ctx, p: P) -> Decision:
    return Decision.target({p.symbol: p.weight}, "GO")


DEFAULT_PARAMS = P()


def _run(strat, data, the_spec, params=DEFAULT_PARAMS, **kwargs):
    return run_backtest(the_spec, definition_of(strat), params, data, **kwargs)


def test_next_open_fill_prior_close_sizing_and_costs() -> None:
    days = weekdays(date(2025, 1, 6), 4)
    data = dataset(days, {"AAA": {"open": [100, 102, 103, 104], "close": [100, 104, 105, 106]}})
    result = _run(always_target, data, spec(("AAA",)))
    fill = result.fills[0]
    assert (fill.session, fill.side, fill.quantity, fill.price, fill.fee) == (
        days[1],
        "buy",
        Decimal(50),
        Decimal(102),
        Decimal("5.10"),
    )
    assert result.equity[0] == Decimal("10094.900000")  # 10000 - 5100 - 5.10 + 50 * 104
    assert len(result.fills) == 1  # unchanged, completed target does not churn


def test_state_is_persisted_on_hold() -> None:
    seen = []

    class Q(Params):
        pass

    @strategy(params=Q, lookback=lambda p: 1)
    def counter(ctx: Ctx, p: Q) -> Decision:
        seen.append(dict(ctx.state))
        return Decision.hold("WAIT", state={"calls": ctx.state.get("calls", 0) + 1})

    days = weekdays(date(2025, 1, 6), 4)
    _run(counter, dataset(days, {"AAA": {"open": [1, 1, 1, 1]}}), spec(("AAA",)), Q())
    assert seen == [{}, {"calls": 1}, {"calls": 2}]


def test_cash_limit_is_reported_and_never_overspends() -> None:
    days = weekdays(date(2025, 1, 6), 4)
    data = dataset(days, {"AAA": {"open": [100, 110, 100, 100], "close": [100, 100, 100, 100]}})
    result = _run(always_target, data, spec(("AAA",)), P(weight=Decimal("1")))
    assert any(w.startswith("INSUFFICIENT_SETTLED_CASH") for w in result.warnings)
    assert result.fills[0].quantity == Decimal(90)  # sized at 100 from prior close; gap-up open 110 caps it
    assert all(cash >= 0 for cash in result.cash)


def test_incomplete_target_is_retried_on_the_next_decision() -> None:
    days = weekdays(date(2025, 1, 6), 4)
    data = dataset(days, {"AAA": {"open": [100, 100, 100, 100], "present": [True, False, True, True]}})
    result = _run(always_target, data, spec(("AAA",)))
    assert any(w.startswith("PRICE_MISSING") for w in result.warnings)
    # Day 2 has no bar in its one-session window (INSUFFICIENT_HISTORY); the unchanged, incomplete target is retried next.
    assert [d["reason_codes"] for d in result.decisions][1] == ["INSUFFICIENT_HISTORY"]
    assert [f.session for f in result.fills] == [days[3]]


def test_rotation_cash_vs_margin_account() -> None:
    days = weekdays(date(2025, 1, 6), 5)
    prices = {"AAA": {"open": [100] * 5}, "BBB": {"open": [50] * 5}}

    class R(Params):
        pass

    @strategy(params=R, lookback=lambda p: 1)
    def rotate(ctx: Ctx, p: R) -> Decision:
        return Decision.target({"AAA": "0.9"} if ctx.decision_session <= days[1] else {"BBB": "0.9"}, "GO")

    cash = _run(rotate, dataset(days, prices), spec(("AAA", "BBB")), R())
    margin = _run(rotate, dataset(days, prices), spec(("AAA", "BBB"), account={"model": "margin"}), R())
    cash_buys = [(f.session, f.symbol) for f in cash.fills if f.side == "buy" and f.symbol == "BBB"]
    margin_buys = [(f.session, f.symbol) for f in margin.fills if f.side == "buy" and f.symbol == "BBB"]
    margin_qty = {f.session: f.quantity for f in margin.fills if f.symbol == "BBB"}
    cash_qty = {f.session: f.quantity for f in cash.fills if f.symbol == "BBB"}
    assert margin_buys == [(days[2], "BBB")]  # margin: sale proceeds fund the buy in the same session
    assert cash_qty[days[2]] < margin_qty[days[2]]  # cash: only already-settled cash that day
    assert cash_buys[1][0] == days[3]  # completes once proceeds settle (T+1)
    assert all(e >= 0 for e in cash.cash)


def test_split_keeps_value_and_bars_are_point_in_time() -> None:
    days = weekdays(date(2025, 1, 6), 5)
    data = dataset(
        days,
        {"AAA": {"open": [100, 100, 50, 50, 50], "close": [100, 100, 50, 50, 50]}},
        splits=[Split("AAA", days[2], Decimal(2))],
    )
    seen = []

    class S(Params):
        pass

    @strategy(params=S, lookback=lambda p: 2)
    def watch(ctx: Ctx, p: S) -> Decision:
        seen.append(ctx.bars("AAA").close.tolist())
        return Decision.target({"AAA": "0.5"}, "GO")

    result = _run(watch, data, spec(("AAA",)), S())
    assert seen[0] == [100.0, 100.0] and seen[1] == [
        50.0,
        50.0,
    ]  # history split-adjusted once the split is known
    # The first decision lands on the split day, sized on the post-split basis (mark 100 / 2 = 50).
    assert result.fills[0].session == days[2] and result.fills[0].quantity == Decimal(100)
    assert result.positions["AAA"] == Decimal(100)


def test_dividend_is_paid_on_pay_date_only() -> None:
    days = weekdays(date(2025, 1, 6), 6)
    data = dataset(
        days, {"AAA": {"open": [100] * 6}}, dividends=[Dividend("AAA", days[2], days[4], Decimal("1.00"))]
    )
    result = _run(always_target, data, spec(("AAA",)))
    shares = result.fills[0].quantity
    cash_by_day = dict(zip(result.sessions, result.cash, strict=True))
    assert cash_by_day[days[4]] - cash_by_day[days[3]] == shares * Decimal("1.00")
    assert result.equity[1] == result.equity[2]  # receivable counted in equity before payment


def test_settlement_switches_to_t_plus_one() -> None:
    before = dataset(weekdays(date(2024, 5, 20), 6), {"AAA": {"open": [1] * 6}})
    assert before.settlement_index(0) == 2
    after = dataset(weekdays(date(2024, 5, 28), 4), {"AAA": {"open": [1] * 4}})
    assert after.settlement_index(0) == 1


def test_undeclared_reason_code_stops_the_run() -> None:
    class U(Params):
        pass

    @strategy(params=U, lookback=lambda p: 1)
    def rogue(ctx: Ctx, p: U) -> Decision:
        return Decision.hold("UNDECLARED_THING")

    days = weekdays(date(2025, 1, 6), 3)
    with pytest.raises(EngineError, match="REASON_CODE_UNDECLARED"):
        _run(rogue, dataset(days, {"AAA": {"open": [1, 1, 1]}}), spec(("AAA",)), U())


class SmaP(Params):
    symbol: str = "SYNA"
    period: int = Field(50, ge=5, le=400)


@strategy(params=SmaP, lookback=lambda p: p.period)
def sma_trend(ctx: Ctx, p: SmaP) -> Decision:
    close = ctx.bars(p.symbol).close
    if close[-1] > ta.sma(close, p.period)[-1]:
        return Decision.target({p.symbol: "0.95"}, "GO")
    return Decision.target({}, "WAIT")


def test_future_data_cannot_change_past_decisions() -> None:
    data = synthetic_dataset(date(2020, 1, 1), date(2021, 12, 31), symbols=("SYNA",))
    base = run_backtest(spec(("SYNA",)), definition_of(sma_trend), SmaP(), data)
    cut = 300
    series = data.series["SYNA"]
    perturbed_micro = {name: values.copy() for name, values in series.micro.items()}
    for values in perturbed_micro.values():
        values[cut:] = (values[cut:] * np.linspace(0.5, 2.0, len(values) - cut)).astype(np.int64)
    mutated = replace(data, series={"SYNA": replace(series, micro=perturbed_micro)})
    mutated.__post_init__()
    other = run_backtest(spec(("SYNA",)), definition_of(sma_trend), SmaP(), mutated)
    # Decisions up to and including session `cut` use bars through `cut - 1` only.
    assert base.decisions[: cut - 1] == other.decisions[: cut - 1]
    assert base.decisions != other.decisions


def test_runs_are_deterministic_and_equity_identity_holds() -> None:
    data = synthetic_dataset(date(2020, 1, 1), date(2021, 12, 31), symbols=("SYNA",))
    first = run_backtest(spec(("SYNA",)), definition_of(sma_trend), SmaP(), data)
    second = run_backtest(spec(("SYNA",)), definition_of(sma_trend), SmaP(), data)
    assert first.ledger_hash == second.ledger_hash
    last_close = Decimal(int(data.series["SYNA"].micro["close"][-1])) / MICRO
    positions_value = first.positions.get("SYNA", Decimal(0)) * last_close
    assert first.equity[-1] - first.cash[-1] == pytest.approx(positions_value, abs=Decimal("0.01"))
