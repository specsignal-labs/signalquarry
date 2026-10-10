# SPDX-License-Identifier: Apache-2.0
"""Every way a strategy or input can break the engine contract stops the run with a registered code."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from signalquarry._internal.contracts.reason_codes import REASON_CODES
from signalquarry._internal.data.dataset import with_pending_session
from signalquarry._internal.engine.backtest import EngineError, plan_pre_open, run_backtest
from signalquarry.sdk import Decision, Params, definition_of, strategy
from tests.helpers import dataset, spec, weekdays

DAYS = weekdays(date(2025, 1, 6), 6)
DATA = dataset(DAYS, {"AAA": {"open": [10, 11, 12, 13, 14, 15]}})


class P(Params):
    lookback: int = 1


def _def(body, lookback=lambda p: p.lookback):
    return definition_of(strategy(params=P, lookback=lookback)(body))


def _code(action) -> str:
    with pytest.raises(EngineError) as info:
        action()
    code = str(info.value).split(":", 1)[0]
    assert code in REASON_CODES, code
    return code


@pytest.mark.parametrize(
    ("body", "code"),
    [
        (lambda ctx, p: "not a decision", "STRATEGY_RETURNED_NON_DECISION"),
        (lambda ctx, p: Decision.target({"AAA": "0.5"}, "UNDECLARED"), "REASON_CODE_UNDECLARED"),
        (lambda ctx, p: Decision.target({"BBB": "0.5"}, "GO"), "DECISION_SYMBOL_NOT_DECLARED"),
        (lambda ctx, p: Decision.target({"AAA": "0.9"}, "GO"), "DECISION_WEIGHT_ABOVE_LIMIT"),
        (lambda ctx, p: Decision.hold("GO", state={"x": object()}), "STRATEGY_STATE_NOT_SERIALIZABLE"),
        (lambda ctx, p: Decision.hold("GO", state={"x": "y" * 20000}), "STRATEGY_STATE_TOO_LARGE"),
    ],
)
def test_contract_violations(body, code) -> None:
    the_spec = spec(("AAA",), limits={"max_weight_per_symbol": "0.8"})
    assert _code(lambda: run_backtest(the_spec, _def(body), P(), DATA)) == code
    pending = with_pending_session(DATA, date(2025, 1, 14))
    assert (
        _code(
            lambda: plan_pre_open(
                the_spec,
                _def(body),
                P(),
                pending,
                quantity={},
                cash=Decimal(1000),
                state={},
                last_target=None,
                target_complete=True,
            )
        )
        == code
    )


def test_an_exception_in_the_strategy_is_the_strategys_error_with_its_session() -> None:
    def broken(ctx, p):
        return Decision.target({"AAA": ctx.bars("ZZZ").close[-1]}, "GO")

    with pytest.raises(EngineError) as info:
        run_backtest(spec(("AAA",)), _def(broken), P(), DATA)
    assert str(info.value) == "STRATEGY_RAISED:2025-01-07: KeyError: 'SYMBOL_NOT_DECLARED:ZZZ'"
    assert isinstance(info.value.__cause__, KeyError) and "STRATEGY_RAISED" in REASON_CODES

    def divides(ctx, p):
        return Decision.target({"AAA": str(1 / 0)}, "GO")

    assert _code(lambda: run_backtest(spec(("AAA",)), _def(divides), P(), DATA)) == "STRATEGY_RAISED"


def test_bars_are_built_only_for_the_symbols_a_strategy_reads() -> None:
    from signalquarry._internal.engine import backtest as engine

    data = dataset(DAYS, {name: {"open": [10, 11, 12, 13, 14, 15]} for name in ("AAA", "BBB", "CCC")})
    seen: list[tuple[tuple[str, ...], int, float]] = []
    built: list[str] = []
    real = engine._Views.bars

    def counting(self, symbol, start, stop):
        built.append(symbol)
        return real(self, symbol, start, stop)

    def reads_one(ctx, p):
        bars = ctx.bars("bbb")  # lower case, as the SDK allows
        seen.append((ctx.symbols, len(bars), float(bars.close[-1])))
        assert ctx.bars("BBB") is bars  # built once per decision
        with pytest.raises(KeyError, match="SYMBOL_NOT_DECLARED:ZZZ"):
            ctx.bars("ZZZ")
        return Decision.target({"BBB": "0.5"}, "GO")

    engine._Views.bars = counting
    try:
        lazy = run_backtest(spec(("AAA", "BBB", "CCC")), _def(reads_one), P(), data)
    finally:
        engine._Views.bars = real
    assert set(built) == {"BBB"} and len(built) == len(seen) == len(DAYS) - 1
    assert seen[0][0] == ("AAA", "BBB", "CCC") and seen[0][1] == 1

    # Reading every symbol gives the ledger an eager engine gave: the views are the same.
    def reads_all(ctx, p):
        assert [ctx.bars(name).symbol for name in ctx.symbols] == ["AAA", "BBB", "CCC"]
        assert len(ctx._bars) == 3 and list(ctx._bars) == ["AAA", "BBB", "CCC"]
        return Decision.target({"BBB": "0.5"}, "GO")

    assert (
        run_backtest(spec(("AAA", "BBB", "CCC")), _def(reads_all), P(), data).ledger_hash == lazy.ledger_hash
    )


def test_input_errors() -> None:
    go = _def(lambda ctx, p: Decision.target({"AAA": "0.5"}, "GO"))
    assert _code(lambda: run_backtest(spec(("ZZZ",)), go, P(), DATA)) == "DATASET_SYMBOLS_MISSING"
    assert (
        _code(lambda: run_backtest(spec(("AAA",)), _def(lambda c, p: None, lambda p: 0), P(), DATA))
        == "STRATEGY_LOOKBACK_INVALID"
    )
    with pytest.raises(EngineError) as empty:
        run_backtest(spec(("AAA",)), go, P(), DATA, start=date(2030, 1, 1))
    assert str(empty.value) == (
        "BACKTEST_RANGE_EMPTY:asked for 2030-01-01 to the last; the data covers 2025-01-06 to 2025-01-13"
    )
    with pytest.raises(EngineError) as early:
        run_backtest(spec(("AAA",)), go, P(), DATA, end=date(2020, 1, 1))
    assert str(early.value).startswith("BACKTEST_RANGE_EMPTY:asked for the first session to 2020-01-01;")
    from signalquarry._internal.engine.backtest import _range_empty

    assert _range_empty((), None, None).endswith("the dataset has no sessions")
    pending = with_pending_session(DATA, date(2025, 1, 14))
    kwargs = dict(quantity={}, cash=Decimal(1000), state={}, last_target=None, target_complete=True)
    assert (
        _code(lambda: plan_pre_open(spec(("ZZZ",)), go, P(), pending, **kwargs)) == "DATASET_SYMBOLS_MISSING"
    )
    assert (
        _code(
            lambda: plan_pre_open(
                spec(("AAA",)), _def(lambda c, p: None, lambda p: 0), P(), pending, **kwargs
            )
        )
        == "STRATEGY_LOOKBACK_INVALID"
    )
    delayed = spec(("AAA",), execution={"execution_delay_sessions": 1})
    assert (
        _code(lambda: plan_pre_open(delayed, go, P(), pending, **kwargs))
        == "PAPER_EXECUTION_DELAY_UNSUPPORTED"
    )


def test_stale_observations_make_the_decision_unavailable() -> None:
    data = dataset(DAYS, {"AAA": {"open": [10] * 6, "present": [True, True, True, True, False, False]}})
    result = run_backtest(
        spec(("AAA",), data={"max_staleness_sessions": 0}),
        _def(lambda ctx, p: Decision.hold("GO"), lambda p: 3),
        P(),
        data,
    )
    assert "STALE_OBSERVATIONS" in {code for d in result.decisions for code in d["reason_codes"]}
