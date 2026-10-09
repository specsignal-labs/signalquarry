# SPDX-License-Identifier: Apache-2.0
"""Hand-worked cases for the buy-and-hold reference curve."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from signalquarry._internal.contracts.spec import LimitsSpec
from signalquarry._internal.engine.backtest import EngineError
from signalquarry._internal.engine.reference import HOLD_CODE, buy_and_hold, reference_spec
from tests.helpers import Dividend, dataset, spec, weekdays

SESSIONS = weekdays(date(2025, 1, 6), 4)  # Monday to Thursday
PRICES = {
    "AAA": {"open": [50.0, 50.0, 50.0, 50.0]},
    "BBB": {"open": [10.0, 10.0, 12.0, 12.0], "close": [10.0, 11.0, 12.0, 13.0]},
}
FREE = {"execution": {"costs": {"bps": "0"}}, "account": {"initial_cash": "1000"}}


def test_buys_at_the_first_open_and_holds() -> None:
    # The first session only supplies history. On the second, 1000 buys 100 shares at 10;
    # the curve then follows the close: 11, 12, 13.
    result = buy_and_hold(spec(("AAA", "BBB"), **FREE), dataset(SESSIONS, PRICES), "BBB")
    assert result.sessions == list(SESSIONS[1:])
    assert result.equity == [Decimal("1100"), Decimal("1200"), Decimal("1300")]
    assert [(f.session, f.symbol, f.side, f.quantity, f.price) for f in result.fills] == [
        (SESSIONS[1], "BBB", "buy", Decimal("100"), Decimal("10"))
    ]
    assert result.positions == {"BBB": Decimal("100")}
    assert {code for d in result.decisions for code in d["reason_codes"]} == {HOLD_CODE}


def test_reinvests_a_dividend_once_whole_shares_are_affordable() -> None:
    # 100 shares earn 1.00 each on the third session, paid that day and settled the next.
    # The fourth session sizes 1300 of equity at the prior close of 12: 108 shares, so 8 more
    # at the open of 12 for 96, leaving 4 in cash. 108 * 13 + 4 = 1408.
    paid = dataset(SESSIONS, PRICES, dividends=[Dividend("BBB", SESSIONS[2], SESSIONS[2], Decimal("1"))])
    result = buy_and_hold(spec(("BBB",), **FREE), paid, "BBB")
    assert result.equity == [Decimal("1100"), Decimal("1300"), Decimal("1408")]
    assert [(f.session, f.quantity, f.price) for f in result.fills] == [
        (SESSIONS[1], Decimal("100"), Decimal("10")),
        (SESSIONS[3], Decimal("8"), Decimal("12")),
    ]
    assert result.cash[-1] == Decimal("4")


def test_pays_the_strategy_costs() -> None:
    # 10 bps on 10,000: 1,000 shares would cost 10,010, so 999 are bought for 9,990 + 9.99.
    result = buy_and_hold(spec(("BBB",)), dataset(SESSIONS, PRICES), "BBB")
    (fill,) = result.fills
    assert (fill.quantity, fill.fee) == (Decimal("999"), Decimal("9.99"))
    assert result.equity[0] == Decimal("999") * 11 + Decimal("0.01")


def test_reference_spec_keeps_account_and_costs_and_drops_strategy_choices() -> None:
    base = spec(
        ("AAA", "BBB"),
        params={"anything": 1},
        limits={"max_weight_per_symbol": "0.5"},
        execution={"costs": {"bps": "7"}, "rebalance": "on_change", "execution_delay_sessions": 1},
        benchmark="BBB",
    )
    reference = reference_spec(base, "BBB")
    assert reference.data.symbols == ("BBB",)
    assert reference.data.max_staleness_sessions == base.data.max_staleness_sessions
    assert reference.account == base.account
    assert reference.execution.costs == base.execution.costs
    assert reference.execution.execution_delay_sessions == 1
    assert reference.execution.rebalance == "every_decision"
    assert reference.limits == LimitsSpec()
    assert reference.params == {"symbol": "BBB"}
    assert list(reference.reason_codes) == [HOLD_CODE]
    assert reference.benchmark is None
    assert reference.options is None
    assert base.execution.rebalance == "on_change"  # the strategy's own spec is untouched


def test_window_and_missing_symbol() -> None:
    data = dataset(SESSIONS, PRICES)
    clipped = buy_and_hold(spec(("BBB",), **FREE), data, "BBB", start=SESSIONS[2], end=SESSIONS[2])
    assert clipped.sessions == [SESSIONS[2]]
    # Sized at the prior close of 11 (90 shares), but 1000 buys only 83 at the open of 12;
    # marked at the close of 12 with 4 left in cash, the account is still worth 1000.
    assert clipped.equity == [Decimal("1000")]
    assert clipped.fills[0].quantity == Decimal("83")
    with pytest.raises(EngineError, match="DATASET_SYMBOLS_MISSING:ZZZ"):
        buy_and_hold(spec(("BBB",), **FREE), data, "ZZZ")


def test_is_deterministic() -> None:
    data = dataset(SESSIONS, PRICES)
    first = buy_and_hold(spec(("BBB",)), data, "BBB")
    second = buy_and_hold(spec(("BBB",)), data, "BBB")
    assert first.ledger_hash == second.ledger_hash
    assert first.ledger_hash.startswith("sha256:")
