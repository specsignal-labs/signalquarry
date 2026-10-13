# SPDX-License-Identifier: Apache-2.0
"""Optional engine marks preserve the ledger and include its actual cash bridge."""

from datetime import date
from decimal import Decimal as D

import pytest

from signalquarry._internal.data.synthetic import synthetic_dataset
from signalquarry._internal.engine import backtest as engine
from signalquarry._internal.engine.run import simulate_equity_ticks
from signalquarry._internal.evidence.run_spool import spool_equity_run
from signalquarry._internal.validation.attribution import attribution
from signalquarry.sdk import Decision, Params, definition_of, strategy
from tests.helpers import Dividend, Split, dataset, spec, weekdays
from tests.unit.test_engine import SmaP, sma_trend


class P(Params):
    pass


def _collect(stream):
    ticks = []
    while True:
        try:
            ticks.append(next(stream))
        except StopIteration as completed:
            return ticks, completed.value


def _compare(base, marked):
    assert len(base) == len(marked)
    for off, on in zip(base, marked, strict=True):
        assert off.positions is None and off.positions_cash is None
        assert off.valuation_cash is None
        assert (off.session, off.equity, off.cash, off.fills, off.decision) == (
            on.session,
            on.equity,
            on.cash,
            on.fills,
            on.decision,
        )
        assert on.positions is not None and on.positions_cash is not None
        value = sum((quantity * mark for quantity, mark in on.positions.values()), D(0))
        assert value + on.cash + sum(on.positions_cash, D(0)) == on.equity
        assert value + on.valuation_cash == on.equity


def test_engine_positions_demo_parity_and_spool(tmp_path, monkeypatch):
    data = synthetic_dataset(date(2022, 1, 3), date(2023, 12, 29), symbols=("SYNA",))
    base_spec, definition, params = spec(("SYNA",)), definition_of(sma_trend), SmaP()
    calls = []
    capture = engine._position_marks

    def count(*args):
        calls.append(1)
        return capture(*args)

    monkeypatch.setattr(engine, "_position_marks", count)
    base, off_end = _collect(simulate_equity_ticks(base_spec, definition, params, data))
    assert calls == []
    marked, on_end = _collect(simulate_equity_ticks(base_spec, definition, params, data, positions=True))
    assert len(calls) == len(marked)
    _compare(base, marked)
    assert on_end == off_end
    original = engine.run_backtest(base_spec, definition, params, data)
    spooled = spool_equity_run(
        simulate_equity_ticks(base_spec, definition, params, data, positions=True), tmp_path
    )
    assert spooled.result.ledger_hash == original.ledger_hash
    assert spooled.result.equity == original.equity and spooled.result.cash == original.cash
    assert spooled.fill_count == len(original.fills) and spooled.result.positions == original.positions
    assert spooled.positions_path is not None
    assert "".join(spooled.positions_chunks()).startswith("session,symbol,quantity,mark,value\n")
    assert len(spooled.positions_cash) == len(marked)


def test_engine_positions_split_dividend_missing_bar_and_settlement(tmp_path, monkeypatch):
    days = weekdays(date(2025, 1, 6), 7)
    data = dataset(
        days,
        {
            "AAA": {"open": [10, 10, 10, 5, 6, 6, 6], "present": [True, True, True, False, True, True, True]},
            "BBB": {"open": [20] * 7},
        },
        splits=(Split("AAA", days[3], D(2)),),
        dividends=(Dividend("AAA", days[3], days[5], D(1)),),
    )

    @strategy(params=P, lookback=lambda p: 1)
    def buy_then_sell(ctx, p):
        if ctx.decision_session == days[1]:
            return Decision.target({"AAA": ".4", "BBB": ".3"}, "GO")
        if ctx.decision_session == days[5]:
            return Decision.target({}, "GO")
        return Decision.hold("WAIT")

    base_spec = spec(("AAA", "BBB"))
    definition = definition_of(buy_then_sell)
    marks = []
    actual = engine._close_mark

    def counting(*args):
        marks.append(args[2:])
        return actual(*args)

    monkeypatch.setattr(engine, "_close_mark", counting)
    off, off_end = _collect(engine.iter_equity_backtest(base_spec, definition, P(), data))
    count_off = len(marks)
    marks.clear()
    on, on_end = _collect(engine.iter_equity_backtest(base_spec, definition, P(), data, positions=True))
    assert len(marks) == count_off  # collection never recomputes closing marks
    _compare(off, on)
    assert off_end == on_end
    before = on[1].positions["AAA"]
    split = on[2].positions["AAA"]
    assert split == (before[0] * 2, before[1] / 2)  # absent split-day bar uses the rebased last close
    assert on[2].positions_cash[1] == split[0]
    assert on[4].positions == {} and on[4].positions_cash[0] > 0  # unsettled sale proceeds
    assert on[5].positions_cash[0] == on[5].positions_cash[1] == 0
    initial = base_spec.account.initial_cash
    result = attribution(
        [t.session for t in on],
        [t.equity for t in on],
        initial,
        [
            (t.session, symbol, quantity, mark, quantity * mark)
            for t in on
            for symbol, (quantity, mark) in t.positions.items()
        ],
        [(f.session, f.symbol, f.side, f.quantity, f.price, f.fee) for t in on for f in t.fills],
    )
    parts = result["reconciliation"]
    assert D(parts["remainder"]) == split[0]  # entitlement, unaffected by payment/settlement timing
    assert D(parts["total"]) == D(parts["symbols"]) + D(parts["fees"]) + D(parts["remainder"])
    plain_dir, marked_dir = tmp_path / "plain", tmp_path / "marked"
    plain_dir.mkdir()
    marked_dir.mkdir()
    plain = spool_equity_run(engine.iter_equity_backtest(base_spec, definition, P(), data), plain_dir)
    marked = spool_equity_run(
        engine.iter_equity_backtest(base_spec, definition, P(), data, positions=True), marked_dir
    )
    assert plain.result.ledger_hash == marked.result.ledger_hash
    assert plain.decisions_path.read_bytes() == marked.decisions_path.read_bytes()
    assert plain.fills_path.read_bytes() == marked.fills_path.read_bytes()
    assert plain.positions_path is None and plain.positions_cash is None


def test_engine_positions_cash_rounding_and_zero_quantity_filter():
    days = weekdays(date(2025, 1, 6), 4)
    data = dataset(days, {"AAA": {"open": [10, 10, 10, 10], "close": [10, 10.123456, 10, 10]}})

    @strategy(params=P, lookback=lambda p: 1)
    def fractional(ctx, p):
        return Decision.target({"AAA": ".333333"} if ctx.decision_session == days[1] else {}, "GO")

    base_spec = spec(("AAA",), execution={"sizing": "fractional"})
    off, _ = _collect(engine.iter_equity_backtest(base_spec, definition_of(fractional), P(), data))
    on, _ = _collect(
        engine.iter_equity_backtest(base_spec, definition_of(fractional), P(), data, positions=True)
    )
    _compare(off, on)
    assert on[0].positions_cash[2] != 0
    assert on[-1].positions == {}


def test_engine_positions_empty_marks_and_zero_fallback():
    days = weekdays(date(2025, 1, 6), 2)
    data = dataset(days, {"AAA": {"open": [10, 10], "present": [False, False]}})
    views = engine._Views(data, ("AAA",))
    assert engine._position_marks(data, views, {"AAA": D(1)}, 1) == {"AAA": (D(1), D(0))}
    assert engine._position_marks(data, views, {"AAA": D(0)}, 1) == {}


def test_engine_positions_failed_spool_is_removed(tmp_path):
    def broken():
        yield engine.BacktestTick(date(2025, 1, 6), {}, (), D(10), D(10), {}, (D(0), D(0), D(0)))
        raise engine.EngineError("test interruption")

    with pytest.raises(engine.EngineError, match="interruption"):
        spool_equity_run(broken(), tmp_path)
    assert list(tmp_path.iterdir()) == []
