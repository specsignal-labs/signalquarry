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


def test_input_errors() -> None:
    go = _def(lambda ctx, p: Decision.target({"AAA": "0.5"}, "GO"))
    assert _code(lambda: run_backtest(spec(("ZZZ",)), go, P(), DATA)) == "DATASET_SYMBOLS_MISSING"
    assert (
        _code(lambda: run_backtest(spec(("AAA",)), _def(lambda c, p: None, lambda p: 0), P(), DATA))
        == "STRATEGY_LOOKBACK_INVALID"
    )
    assert (
        _code(lambda: run_backtest(spec(("AAA",)), go, P(), DATA, start=date(2030, 1, 1)))
        == "BACKTEST_RANGE_EMPTY"
    )
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
