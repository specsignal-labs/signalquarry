# SPDX-License-Identifier: Apache-2.0
"""Hand-worked reference cases for order planning and cash accounting."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from signalquarry._internal.data.dataset import with_pending_session
from signalquarry._internal.engine.backtest import (
    EngineError,
    _draw_pending,
    affordable_quantity,
    plan_orders,
    plan_pre_open,
)
from signalquarry.sdk import Ctx, Decision, Params, definition_of, strategy
from tests.helpers import dataset, spec, weekdays

SESSION = date(2025, 1, 7)


def _orders(
    weights: dict[str, Decimal],
    quantity: dict[str, Decimal],
    marks: dict[str, Decimal | None],
    *,
    equity: Decimal = Decimal("100"),
    fractional: bool = False,
    minimum: Decimal = Decimal("0"),
    opens: dict[str, Decimal | None] | None = None,
):
    return plan_orders(
        weights,
        quantity,
        equity,
        marks,
        fractional=fractional,
        min_order_notional=minimum,
        session=SESSION,
        opens=opens,
    )


def test_plan_orders_sizes_at_marks_and_sells_before_buys() -> None:
    orders, complete, warnings = _orders(
        {"BBB": Decimal("0.5"), "AAA": Decimal("0.2"), "CCC": Decimal("0.3")},
        {"AAA": Decimal("5"), "BBB": Decimal("1"), "CCC": Decimal("3")},
        {"AAA": Decimal("10"), "BBB": Decimal("20"), "CCC": Decimal("10")},
    )
    assert [(o.symbol, o.delta, o.mark) for o in orders] == [
        ("AAA", Decimal("-3"), Decimal("10")),
        ("BBB", Decimal("1"), Decimal("20")),
    ]
    assert complete is True
    assert warnings == []


def test_plan_orders_fractional_quantum_and_exact_minimum() -> None:
    orders, complete, warnings = _orders(
        {"AAA": Decimal("0.005"), "BBB": Decimal("0.01")},
        {},
        {"AAA": Decimal("2"), "BBB": Decimal("2")},
        fractional=True,
        minimum=Decimal("0.01"),
    )
    assert [(o.symbol, o.delta) for o in orders] == [
        ("AAA", Decimal("0.250000")),
        ("BBB", Decimal("0.500000")),
    ]
    assert complete is True and warnings == []


def test_plan_orders_filters_zero_and_below_minimum() -> None:
    orders, complete, warnings = _orders(
        {"AAA": Decimal("0.1"), "BBB": Decimal("0.1")},
        {"AAA": Decimal("1")},
        {"AAA": Decimal("10"), "BBB": Decimal("10")},
        minimum=Decimal("11"),
    )
    assert orders == []
    assert complete is True and warnings == []


def test_plan_orders_accepts_exact_minimum_and_positive_sub_dollar_mark() -> None:
    orders, complete, warnings = _orders(
        {"AAA": Decimal("0.01")},
        {},
        {"AAA": Decimal("0.5")},
        fractional=True,
        minimum=Decimal("1"),
    )
    assert [(o.symbol, o.delta, o.mark) for o in orders] == [("AAA", Decimal("2.000000"), Decimal("0.5"))]
    assert complete is True and warnings == []


@pytest.mark.parametrize(
    ("marks", "opens", "weights", "quantity", "expected_warning"),
    [
        ({"AAA": None}, None, {"AAA": Decimal("1")}, {}, True),
        ({"AAA": Decimal("0")}, None, {}, {"AAA": Decimal("1")}, True),
        ({"AAA": Decimal("10")}, {"AAA": None}, {"AAA": Decimal("1")}, {}, True),
        ({"AAA": None}, None, {}, {}, False),
    ],
)
def test_plan_orders_missing_prices(
    marks: dict[str, Decimal | None],
    opens: dict[str, Decimal | None] | None,
    weights: dict[str, Decimal],
    quantity: dict[str, Decimal],
    expected_warning: bool,
) -> None:
    orders, complete, warnings = _orders(
        weights | {"BBB": Decimal("0.5")},
        quantity,
        marks | {"BBB": Decimal("10")},
        opens=(opens | {"BBB": Decimal("10")}) if opens is not None else None,
    )
    assert [(o.symbol, o.delta) for o in orders] == [("BBB", Decimal("5"))]
    assert complete is (not expected_warning)
    assert warnings == ([f"PRICE_MISSING:{SESSION.isoformat()}:AAA"] if expected_warning else [])


@pytest.mark.parametrize(
    ("available", "price", "bps", "per_share", "fractional", "expected"),
    [
        ("1", "0.5", "0", "0", False, "2"),
        ("100", "30", "0", "0", False, "3"),
        ("10", "4", "0", "0", True, "2.500000"),
        ("10", "4", "0", "1", False, "2"),
        ("10", "4", "0.01", "0", False, "2"),
        ("2.006", "1", "0.003", "0", False, "1"),
        ("3.009", "1", "0.003", "0", False, "2"),
        ("0", "4", "0", "0", False, "0"),
    ],
)
def test_affordable_quantity_reference_cases(
    available: str, price: str, bps: str, per_share: str, fractional: bool, expected: str
) -> None:
    assert affordable_quantity(
        Decimal(available), Decimal(price), Decimal(bps), Decimal(per_share), fractional
    ) == Decimal(expected)


def test_draw_pending_uses_earliest_settlement_first_and_never_borrows() -> None:
    pending = [(3, Decimal("5")), (1, Decimal("7")), (2, Decimal("4"))]
    assert _draw_pending(pending, Decimal("9")) == [(2, Decimal("2")), (3, Decimal("5"))]
    assert pending == [(3, Decimal("5")), (1, Decimal("7")), (2, Decimal("4"))]
    with pytest.raises(EngineError, match="MARGIN_MODEL_WOULD_BORROW"):
        _draw_pending(pending, Decimal("17"))
    assert _draw_pending([(1, Decimal("2"))], Decimal("1.5")) == [(1, Decimal("0.5"))]


class P(Params):
    pass


@strategy(params=P, lookback=lambda p: 1)
def _half_target(ctx: Ctx, p: P) -> Decision:
    return Decision.target({"AAA": "0.5"}, "GO", state={"visits": ctx.state.get("visits", 0) + 1})


@strategy(params=P, lookback=lambda p: 1)
def _hold_target(ctx: Ctx, p: P) -> Decision:
    return Decision.hold("WAIT")


def test_pre_open_retries_incomplete_target_and_preserves_state() -> None:
    days = weekdays(date(2025, 1, 6), 2)
    prices = dataset(days, {"AAA": {"open": [10, 10]}})
    pending = with_pending_session(prices, weekdays(date(2025, 1, 8), 1)[0])
    kwargs = dict(
        quantity={},
        cash=Decimal("100"),
        state={"visits": 2},
        last_target={"AAA": Decimal("0.5")},
    )
    definition = definition_of(_half_target)
    strategy_spec = spec(("AAA",))
    unchanged = plan_pre_open(strategy_spec, definition, P(), pending, target_complete=True, **kwargs)
    assert unchanged.orders == []
    assert unchanged.record == {
        "session": pending.sessions[-1].isoformat(),
        "action": "target",
        "weights": {"AAA": Decimal("0.5")},
        "reason_codes": ["GO"],
    }
    assert unchanged.state == {"visits": 3}
    assert unchanged.last_target == {"AAA": Decimal("0.5")}
    assert unchanged.complete is True
    retried = plan_pre_open(strategy_spec, definition, P(), pending, target_complete=False, **kwargs)
    assert [(o.symbol, o.delta, o.mark) for o in retried.orders] == [("AAA", Decimal("5"), Decimal("10"))]
    assert retried.record["queued_for"] == pending.sessions[-1].isoformat()
    assert retried.complete is True
    assert retried.warnings == []
    assert retried.equity == Decimal("100")
    assert retried.marks == {"AAA": Decimal("10")}


def test_pre_open_hold_keeps_state_and_last_target() -> None:
    days = weekdays(date(2025, 1, 6), 2)
    pending = with_pending_session(dataset(days, {"AAA": {"open": [10, 10]}}), date(2025, 1, 8))
    last_target = {"AAA": Decimal("0.5")}
    plan = plan_pre_open(
        spec(("AAA",)),
        definition_of(_hold_target),
        P(),
        pending,
        quantity={},
        cash=Decimal("100"),
        state={"visits": 2},
        last_target=last_target,
        target_complete=False,
    )
    assert plan.state == {"visits": 2}
    assert plan.last_target == last_target
    assert plan.complete is False
    assert plan.orders == []
    assert plan.record["action"] == "hold"


def test_pre_open_fractional_every_decision_and_held_symbol_mark() -> None:
    days = weekdays(date(2025, 1, 6), 2)
    prices = dataset(days, {"AAA": {"open": [30, 30]}, "BBB": {"open": [20, 20]}})
    pending = with_pending_session(prices, date(2025, 1, 8))
    plan = plan_pre_open(
        spec(("AAA",), execution={"sizing": "fractional", "rebalance": "every_decision"}),
        definition_of(_half_target),
        P(),
        pending,
        quantity={"BBB": Decimal("2")},
        cash=Decimal("100"),
        state={},
        last_target={"AAA": Decimal("0.5")},
        target_complete=True,
    )
    assert plan.equity == Decimal("140")
    assert plan.marks == {"AAA": Decimal("30"), "BBB": Decimal("20")}
    assert [(o.symbol, o.delta) for o in plan.orders] == [
        ("BBB", Decimal("-2")),
        ("AAA", Decimal("2.333333")),
    ]
    assert plan.complete is True
    assert plan.warnings == []
    assert plan.record["queued_for"] == "2025-01-08"
    assert plan.decision.action == "target"
