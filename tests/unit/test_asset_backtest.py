# SPDX-License-Identifier: Apache-2.0
"""Synthetic lifecycle backtest vectors; no provider data or broker orders."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from signalquarry._internal.data.asset_dataset import AssetDatasetError, AssetDatasetV1, AssetSeries
from signalquarry._internal.data.dataset import FIELDS, MICRO
from signalquarry._internal.data.identity import AliasBook, AliasObservation, AssetKey, IdentityError
from signalquarry._internal.data.lifecycle import LifecycleError, LifecycleEvent
from signalquarry._internal.engine import asset_backtest
from signalquarry._internal.engine.asset_backtest import (
    AssetPlannedOrder,
    _asset_mark,
    _decide_asset,
    _marks_for_positions,
    plan_asset_orders,
    run_asset_backtest,
)
from signalquarry._internal.engine.backtest import EngineError, plan_orders
from signalquarry._internal.engine.lifecycle import LifecycleState
from signalquarry.sdk import Ctx, Decision, Params, definition_of, strategy
from tests.helpers import spec, weekdays

OLD = AssetKey("synthetic", "asset-old")
NEW = AssetKey("synthetic", "asset-new")


class Weight(Params):
    value: Decimal = Decimal("0.5")


@strategy(params=Weight, lookback=lambda _: 1)
def target_current_symbol(ctx: Ctx, params: Weight) -> Decision:
    return Decision.target({ctx.symbols[0]: params.value}, "GO")


def alias(
    record_id: str,
    asset: AssetKey,
    symbol: str,
    start: date,
    end: date | None,
    observed_at: datetime,
) -> AliasObservation:
    return AliasObservation(
        record_id,
        asset,
        symbol,
        start,
        end,
        start,
        observed_at,
        ("a" * 64,),
        1,
    )


def action(
    kind: str,
    sessions: tuple[date, ...],
    *,
    source: AssetKey = OLD,
    observed_at: datetime | None = None,
    target: AssetKey | None = None,
    target_symbol: str | None = None,
    ratio: Decimal | None = None,
    cash: Decimal | None = None,
    pay_date: date | None = None,
    event_id: str = "synthetic-action",
) -> LifecycleEvent:
    terms: dict[str, object] = {}
    if kind == "rename":
        terms["target_symbol"] = target_symbol
    if kind in ("split", "stock_merger", "mixed_merger"):
        terms |= {"share_ratio": ratio or Decimal(1), "fraction_policy": "retain"}
    if kind == "split":
        terms["target_symbol"] = target_symbol
    if kind in ("stock_merger", "mixed_merger"):
        terms |= {"target": target, "target_symbol": target_symbol}
    if kind in ("cash_merger", "mixed_merger", "worthless_removal"):
        terms["cash_per_old_share"] = cash if cash is not None else Decimal(0)
    if kind in ("cash_merger", "mixed_merger"):
        terms["cash_pay_date"] = pay_date or sessions[2]
    return LifecycleEvent(
        event_id=event_id,
        kind=kind,
        source=source,
        source_symbol="SYNA",
        effective_date=sessions[2],
        process_date=sessions[1],
        observed_at=observed_at or datetime.combine(sessions[2], time(12), UTC),
        page_hashes=("b" * 64,),
        normalization_version=1,
        sequence=0,
        fee_per_old_share=Decimal(0),
        **terms,
    )


def make_dataset(
    prices: dict[AssetKey, tuple[list[float], list[bool]]],
    aliases: tuple[AliasObservation, ...],
    events: tuple[LifecycleEvent, ...] = (),
    *,
    count: int = 6,
    open_prices: dict[AssetKey, list[float]] | None = None,
    volumes: dict[AssetKey, list[float]] | None = None,
) -> AssetDatasetV1:
    sessions = weekdays(date(2024, 1, 2), count)
    series: dict[AssetKey, AssetSeries] = {}
    for asset, (closes, present_values) in prices.items():
        values = np.rint(np.asarray(closes, dtype=np.float64) * MICRO).astype(np.int64)
        micro = {name: values.copy() for name in FIELDS}
        if open_prices is not None and asset in open_prices:
            micro["open"] = np.rint(np.asarray(open_prices[asset], dtype=np.float64) * MICRO).astype(np.int64)
        series[asset] = AssetSeries(
            micro,
            np.asarray(volumes[asset], dtype=np.float64)
            if volumes is not None and asset in volumes
            else np.full(count, 1_000_000, dtype=np.float64),
            np.asarray(present_values, dtype=bool),
        )
    cutoffs = tuple(datetime.combine(session, time(13), UTC) for session in sessions)
    return AssetDatasetV1(
        sessions=sessions,
        series=series,
        aliases=AliasBook(aliases),
        events=events,
        decision_cutoffs=cutoffs,
        observation_cutoff=cutoffs[-1] + timedelta(hours=2),
    )


def make_aliases(
    sessions: tuple[date, ...], *, change_symbol: str | None = None, successor: AssetKey | None = None
) -> tuple[AliasObservation, ...]:
    seen = datetime.combine(sessions[0], time(10), UTC)
    if successor is None:
        if change_symbol is None:
            return (alias("old", OLD, "SYNA", sessions[0], None, seen),)
        return (
            alias("old", OLD, "SYNA", sessions[0], sessions[2], seen),
            alias("new", OLD, change_symbol, sessions[2], None, seen),
        )
    return (
        alias("old", OLD, "SYNA", sessions[0], sessions[2], seen),
        alias("successor", successor, "SYNA", sessions[2], None, seen),
    )


def _run(data: AssetDatasetV1, symbols: tuple[str, ...], *, delay: int = 0):
    strategy_spec = spec(
        symbols,
        execution={"costs": {"bps": "0", "per_share": "0"}, "execution_delay_sessions": delay},
    )
    return run_asset_backtest(strategy_spec, definition_of(target_current_symbol), Weight(), data)


def _definition(decide: Callable[[Ctx, Weight], Any], *, lookback: int = 1):
    @strategy(params=Weight, lookback=lambda _: lookback)
    def custom(ctx: Ctx, params: Weight) -> Any:
        return decide(ctx, params)

    return definition_of(custom)


def _raise(error: Exception):
    def raising(*args: Any, **kwargs: Any) -> Any:
        raise error

    return raising


def _direct_decision(
    data: AssetDatasetV1,
    symbols: tuple[str, ...],
    *,
    index: int,
    lookback: int = 1,
    max_staleness: int = 2,
    max_weight: Decimal = Decimal(1),
    state: Any = None,
    decide: Callable[[Ctx, Weight], Any] | None = None,
):
    strategy_spec = spec(
        symbols,
        data={"max_staleness_sessions": max_staleness},
        limits={"max_weight_per_symbol": str(max_weight)},
    )
    session = data.sessions[index]
    cutoff = data.decision_cutoffs[index]
    aliases = data.aliases_as_of(session, cutoff)
    by_asset = {asset: symbol for symbol, asset in aliases.items()}
    return _decide_asset(
        strategy_spec,
        _definition(decide or (lambda _ctx, _params: Decision.hold("WAIT")), lookback=lookback),
        Weight(),
        data,
        data.events_as_of(cutoff),
        aliases,
        state if state is not None else LifecycleState({}, by_asset),
        {asset: Decimal(100) for asset in by_asset},
        Decimal(10_000),
        {},
        index,
        lookback,
        set(strategy_spec.reason_codes),
    )


def test_asset_orders_reuse_shared_planner_sizing_and_sells_first() -> None:
    marks = {OLD: Decimal(100), NEW: Decimal(50)}
    quantity = {OLD: Decimal(20), NEW: Decimal(0)}
    weights = {NEW: Decimal("0.5")}
    orders, complete, warnings = plan_asset_orders(
        weights,
        quantity,
        Decimal(10_000),
        marks,
        fractional=False,
        min_order_notional=Decimal(0),
        session=date(2024, 1, 2),
        aliases={OLD: "SYNA", NEW: "SYNB"},
        opens={OLD: Decimal(100), NEW: Decimal(50)},
    )
    tokens = {asset: f"A{index:08X}" for index, asset in enumerate(sorted((OLD, NEW)))}
    shared, shared_complete, _ = plan_orders(
        {tokens[NEW]: Decimal("0.5")},
        {tokens[OLD]: Decimal(20), tokens[NEW]: Decimal(0)},
        Decimal(10_000),
        {tokens[OLD]: Decimal(100), tokens[NEW]: Decimal(50)},
        fractional=False,
        min_order_notional=Decimal(0),
        session=date(2024, 1, 2),
        opens={tokens[OLD]: Decimal(100), tokens[NEW]: Decimal(50)},
    )
    assert complete is shared_complete and warnings == []
    assert [(order.asset, order.delta, order.mark) for order in orders] == [
        ({token: asset for asset, token in tokens.items()}[order.symbol], order.delta, order.mark)
        for order in shared
    ]


def test_rename_keeps_asset_keyed_holding_and_resolves_new_fill_alias() -> None:
    sessions = weekdays(date(2024, 1, 2), 6)
    event = action("rename", sessions, target_symbol="SYNB")
    data = make_dataset(
        {OLD: ([100, 100, 100, 100, 100, 100], [True] * 6)},
        make_aliases(sessions, change_symbol="SYNB"),
        (event,),
    )
    result = _run(data, ("SYNA", "SYNB"))
    assert result.fills[0].asset == OLD and result.fills[0].symbol == "SYNA"
    assert result.positions == {OLD: Decimal(50)}
    assert not [fill for fill in result.fills if fill.session >= sessions[2]]
    assert result.equity[-1] == Decimal("10000.000000")


def test_ticker_reuse_stock_merger_moves_holding_to_successor_asset() -> None:
    sessions = weekdays(date(2024, 1, 2), 6)
    event = action("stock_merger", sessions, target=NEW, target_symbol="SYNA", ratio=Decimal(1))
    data = make_dataset(
        {
            OLD: ([100, 100, 0, 0, 0, 0], [True, True, False, False, False, False]),
            NEW: ([100, 100, 100, 100, 100, 100], [True] * 6),
        },
        make_aliases(sessions, successor=NEW),
        (event,),
    )
    result = _run(data, ("SYNA",))
    assert OLD not in result.positions
    assert result.positions == {NEW: Decimal(50)}
    assert result.applied_events[0]["event_id"] == event.event_id
    assert result.fills[0].asset == OLD


def test_split_with_new_symbol_changes_share_basis_without_changing_value() -> None:
    sessions = weekdays(date(2024, 1, 2), 6)
    event = action("split", sessions, target_symbol="SYNB", ratio=Decimal(2))
    data = make_dataset(
        {OLD: ([100, 100, 50, 50, 50, 50], [True] * 6)},
        make_aliases(sessions, change_symbol="SYNB"),
        (event,),
    )
    seen = []

    @strategy(params=Weight, lookback=lambda _: 1)
    def record_close(ctx: Ctx, params: Weight) -> Decision:
        seen.append((ctx.symbols[0], float(ctx.bars(ctx.symbols[0]).close[-1])))
        return Decision.target({ctx.symbols[0]: params.value}, "GO")

    result = run_asset_backtest(
        spec(("SYNA", "SYNB"), execution={"costs": {"bps": "0"}}), definition_of(record_close), Weight(), data
    )
    assert ("SYNA", 100.0) in seen
    assert ("SYNB", 50.0) in seen
    assert result.positions == {OLD: Decimal(100)}
    assert result.equity[0] == result.equity[1]


def test_worthless_removal_requires_explicit_zero_and_removes_position() -> None:
    sessions = weekdays(date(2024, 1, 2), 6)
    event = action("worthless_removal", sessions, cash=Decimal(0))
    data = make_dataset(
        {OLD: ([100, 100, 0, 0, 0, 0], [True, True, False, False, False, False])},
        (alias("old", OLD, "SYNA", sessions[0], sessions[2], datetime.combine(sessions[0], time(10), UTC)),),
        (event,),
    )
    result = _run(data, ("SYNA",))
    assert result.positions == {}
    assert result.equity[0] == Decimal("10000.000000")
    assert result.equity[1] == Decimal("5000.000000")


def test_cash_merger_receivable_is_in_equity_then_paid_once() -> None:
    sessions = weekdays(date(2024, 1, 2), 6)
    event = action("cash_merger", sessions, cash=Decimal(100), pay_date=sessions[4])
    data = make_dataset(
        {OLD: ([100, 100, 0, 0, 0, 0], [True, True, False, False, False, False])},
        (alias("old", OLD, "SYNA", sessions[0], sessions[2], datetime.combine(sessions[0], time(18), UTC)),),
        (event,),
    )
    result = _run(data, ("SYNA",))
    assert result.positions == {}
    assert result.equity[1] == Decimal("10000.000000")
    assert result.cash[1] == Decimal("5000.000000")
    assert result.cash[3] == Decimal("10000.000000")
    assert result.equity[-1] == Decimal("10000.000000")


def test_mixed_merger_moves_stock_and_pays_cash_consideration() -> None:
    sessions = weekdays(date(2024, 1, 2), 6)
    event = action(
        "mixed_merger",
        sessions,
        target=NEW,
        target_symbol="SYNB",
        ratio=Decimal("0.5"),
        cash=Decimal(10),
        pay_date=sessions[4],
    )
    aliases = (
        alias("old", OLD, "SYNA", sessions[0], sessions[2], datetime.combine(sessions[0], time(10), UTC)),
        alias("successor", NEW, "SYNB", sessions[2], None, datetime.combine(sessions[0], time(10), UTC)),
    )
    data = make_dataset(
        {
            OLD: ([100, 100, 0, 0, 0, 0], [True, True, False, False, False, False]),
            NEW: ([100, 100, 100, 100, 100, 100], [True] * 6),
        },
        aliases,
        (event,),
    )

    @strategy(params=Weight, lookback=lambda _: 1)
    def enter_then_hold(ctx: Ctx, params: Weight) -> Decision:
        if ctx.decision_session < sessions[2]:
            return Decision.target({ctx.symbols[0]: params.value}, "GO")
        return Decision.hold("WAIT")

    strategy_spec = spec(
        ("SYNA", "SYNB"),
        execution={"costs": {"bps": "0", "per_share": "0"}},
    )
    result = run_asset_backtest(strategy_spec, definition_of(enter_then_hold), Weight(), data)
    assert result.positions == {NEW: Decimal(25)}
    assert result.equity[1] == Decimal("8000.000000")
    assert result.cash[1] == Decimal("5000.000000")
    assert result.cash[3] == Decimal("5500.000000")


def test_stock_merger_without_successor_preopen_mark_blocks_valuation() -> None:
    sessions = weekdays(date(2024, 1, 2), 6)
    event = action("stock_merger", sessions, target=NEW, target_symbol="SYNA", ratio=Decimal(1))
    data = make_dataset(
        {
            OLD: ([100, 100, 0, 0, 0, 0], [True, True, False, False, False, False]),
            NEW: ([0, 0, 100, 100, 100, 100], [False, False, True, True, True, True]),
        },
        make_aliases(sessions, successor=NEW),
        (event,),
    )
    with pytest.raises(EngineError, match="missing or invalid mark"):
        _run(data, ("SYNA",))


def test_late_observed_event_and_late_revision_fail_closed() -> None:
    sessions = weekdays(date(2024, 1, 2), 6)
    late = action(
        "rename",
        sessions,
        target_symbol="SYNB",
        observed_at=datetime.combine(sessions[2], time(14), UTC),
    )
    stable_aliases = make_aliases(sessions)
    data = make_dataset({OLD: ([100] * 6, [True] * 6)}, stable_aliases, (late,))
    with pytest.raises(EngineError, match="first observed after its effective session"):
        _run(data, ("SYNA", "SYNB"))

    first = action(
        "rename",
        sessions,
        target_symbol="SYNB",
        observed_at=datetime.combine(sessions[2], time(12), UTC),
    )
    revised = dataclasses.replace(
        first,
        source_symbol="SYNB",
        effective_date=sessions[4],
        process_date=sessions[3],
        observed_at=datetime.combine(sessions[3], time(14), UTC),
        page_hashes=("c" * 64,),
        target_symbol="SYNX",
    )
    revised_aliases = (
        alias("old", OLD, "SYNA", sessions[0], sessions[2], datetime.combine(sessions[0], time(10), UTC)),
        alias("new", OLD, "SYNB", sessions[2], sessions[4], datetime.combine(sessions[0], time(10), UTC)),
        alias("revision", OLD, "SYNX", sessions[4], None, datetime.combine(sessions[3], time(14), UTC)),
    )
    corrected = make_dataset({OLD: ([100] * 6, [True] * 6)}, revised_aliases, (first, revised))
    with pytest.raises(EngineError, match="applied event was revised"):
        _run(corrected, ("SYNA", "SYNB", "SYNX"))


def test_queued_order_for_retired_asset_is_blocked_and_provider_data_is_rejected() -> None:
    sessions = weekdays(date(2024, 1, 2), 6)
    event = action("stock_merger", sessions, target=NEW, target_symbol="SYNA", ratio=Decimal(1))
    data = make_dataset(
        {
            OLD: ([100, 100, 0, 0, 0, 0], [True, True, False, False, False, False]),
            NEW: ([100] * 6, [True] * 6),
        },
        make_aliases(sessions, successor=NEW),
        (event,),
    )
    with pytest.raises(EngineError, match="queued target references a retired asset"):
        _run(data, ("SYNA",), delay=1)
    with pytest.raises(EngineError, match="synthetic-only"):
        _run(dataclasses.replace(data, source="provider"), ("SYNA",))


def test_asset_order_adapter_reports_missing_marks_and_bounds_token_space(monkeypatch) -> None:
    orders, complete, warnings = plan_asset_orders(
        {OLD: Decimal("0.5")},
        {},
        Decimal(10_000),
        {OLD: None},
        fractional=False,
        min_order_notional=Decimal(0),
        session=date(2024, 1, 2),
        aliases={},
    )
    assert orders == [] and not complete
    assert warnings == ["PRICE_MISSING:2024-01-02:synthetic:asset-old"]

    monkeypatch.setattr(asset_backtest, "MAX_PLANNER_ASSETS", 1)
    with pytest.raises(EngineError, match="too many assets"):
        plan_asset_orders(
            {OLD: Decimal("0.5"), NEW: Decimal("0.5")},
            {},
            Decimal(10_000),
            {OLD: Decimal(100), NEW: Decimal(100)},
            fractional=False,
            min_order_notional=Decimal(0),
            session=date(2024, 1, 2),
            aliases={},
        )


@pytest.mark.parametrize("provider_side", ["source", "target"])
def test_asset_backtest_rejects_provider_lifecycle_assets(provider_side: str) -> None:
    provider_asset = SimpleNamespace(provider="alpaca")
    synthetic_asset = SimpleNamespace(provider="synthetic")
    event = SimpleNamespace(
        source=provider_asset if provider_side == "source" else synthetic_asset,
        target=provider_asset if provider_side == "target" else None,
    )
    untrusted = SimpleNamespace(source="synthetic", series={}, events=(event,))
    with pytest.raises(EngineError, match="synthetic-only"):
        run_asset_backtest(spec(("SYNA",)), definition_of(target_current_symbol), Weight(), untrusted)


def test_invalid_lookback_and_empty_asset_backtest_ranges_fail_closed() -> None:
    sessions = weekdays(date(2024, 1, 2), 6)
    data = make_dataset(
        {OLD: ([100] * 6, [True] * 6)},
        (alias("old", OLD, "SYNA", sessions[0], None, datetime.combine(sessions[0], time(10), UTC)),),
    )
    with pytest.raises(EngineError, match="STRATEGY_LOOKBACK_INVALID"):
        run_asset_backtest(
            spec(("SYNA",)),
            _definition(lambda _ctx, _params: Decision.hold("WAIT"), lookback=0),
            Weight(),
            data,
        )
    with pytest.raises(EngineError, match="BACKTEST_RANGE_EMPTY"):
        run_asset_backtest(
            spec(("SYNA",)),
            definition_of(target_current_symbol),
            Weight(),
            data,
            start=sessions[4],
            end=sessions[1],
        )
    with pytest.raises(EngineError, match="BACKTEST_RANGE_EMPTY"):
        run_asset_backtest(
            spec(("SYNA",)),
            definition_of(target_current_symbol),
            Weight(),
            data,
            start=sessions[0],
            end=sessions[0],
        )


@pytest.mark.parametrize(
    ("target", "error"),
    [
        ("aliases_as_of", IdentityError("TEST", "alias revision invalid")),
        ("events_as_of", AssetDatasetError("event revision invalid")),
        ("settle_lifecycle_receivables", LifecycleError("settlement invalid")),
        ("apply_lifecycle_event", LifecycleError("event application invalid")),
        ("value_lifecycle_state", LifecycleError("valuation invalid")),
    ],
)
def test_asset_backtest_wraps_metadata_and_lifecycle_errors(
    monkeypatch, target: str, error: Exception
) -> None:
    sessions = weekdays(date(2024, 1, 2), 6)
    event = action("rename", sessions, target_symbol="SYNB")
    data = make_dataset(
        {OLD: ([100] * 6, [True] * 6)},
        make_aliases(sessions, change_symbol="SYNB"),
        (event,),
    )
    owner = AssetDatasetV1 if target in {"aliases_as_of", "events_as_of"} else asset_backtest
    monkeypatch.setattr(owner, target, _raise(error))
    with pytest.raises(EngineError, match=str(error)):
        _run(data, ("SYNA", "SYNB"))


def test_asset_backtest_wraps_invalid_intermediate_lifecycle_state(monkeypatch) -> None:
    sessions = weekdays(date(2024, 1, 2), 6)
    data = make_dataset(
        {OLD: ([100] * 6, [True] * 6)},
        make_aliases(sessions),
    )
    original = asset_backtest.LifecycleState
    calls = 0

    def fail_on_alias_sync(*args: Any, **kwargs: Any) -> LifecycleState:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise LifecycleError("intermediate state invalid")
        return original(*args, **kwargs)

    monkeypatch.setattr(asset_backtest, "LifecycleState", fail_on_alias_sync)
    with pytest.raises(EngineError, match="intermediate state invalid"):
        _run(data, ("SYNA",))


def test_event_effective_before_requested_window_is_not_replayed() -> None:
    sessions = weekdays(date(2024, 1, 2), 6)
    event = action("rename", sessions, target_symbol="SYNB")
    data = make_dataset(
        {OLD: ([100] * 6, [True] * 6)},
        make_aliases(sessions, change_symbol="SYNB"),
        (event,),
    )
    result = run_asset_backtest(
        spec(("SYNA", "SYNB")),
        definition_of(target_current_symbol),
        Weight(),
        data,
        start=sessions[3],
    )
    assert result.sessions[0] == sessions[3]
    assert result.applied_events == []


def test_held_asset_alias_must_match_lifecycle_transition() -> None:
    sessions = weekdays(date(2024, 1, 2), 6)
    event = action("rename", sessions, target_symbol="SYNB")
    data = make_dataset(
        {OLD: ([100] * 6, [True] * 6)},
        make_aliases(sessions),
        (event,),
    )
    with pytest.raises(EngineError, match="held asset alias differs"):
        _run(data, ("SYNA", "SYNB"))


def test_decision_cannot_target_a_declared_symbol_without_an_asset() -> None:
    sessions = weekdays(date(2024, 1, 2), 6)
    data = make_dataset(
        {OLD: ([100] * 6, [True] * 6)},
        make_aliases(sessions),
    )
    definition = _definition(lambda _ctx, _params: Decision.target({"SYNB": Decimal("0.5")}, "GO"))
    with pytest.raises(EngineError, match="decision symbol has no asset"):
        run_asset_backtest(spec(("SYNA", "SYNB")), definition, Weight(), data)


@pytest.mark.parametrize("order_asset", [OLD, NEW])
def test_planned_order_without_current_alias_fails_or_skips_missing_open(
    monkeypatch, order_asset: AssetKey
) -> None:
    sessions = weekdays(date(2024, 1, 2), 6)
    present = [True, False, False, False, False, False]
    data = make_dataset(
        {OLD: ([100] * 6, present)},
        make_aliases(sessions),
    )
    monkeypatch.setattr(
        asset_backtest,
        "plan_asset_orders",
        lambda *args, **kwargs: ([AssetPlannedOrder(order_asset, Decimal(10), Decimal(100))], True, []),
    )
    if order_asset == NEW:
        with pytest.raises(EngineError, match="order asset lacks a current alias"):
            _run(data, ("SYNA",))
    else:
        result = _run(data, ("SYNA",))
        assert result.fills == [] and result.positions == {}


@pytest.mark.parametrize(("volume", "expected_quantity"), [(10.0, Decimal(1)), (0.0, None)])
def test_volume_cap_records_partial_fill_and_skips_zero_cap(
    volume: float, expected_quantity: Decimal | None
) -> None:
    sessions = weekdays(date(2024, 1, 2), 6)
    data = make_dataset(
        {OLD: ([100] * 6, [True] * 6)},
        make_aliases(sessions),
        volumes={OLD: [volume] * 6},
    )
    strategy_spec = spec(
        ("SYNA",),
        execution={
            "costs": {"bps": "0", "per_share": "0"},
            "fill": {"model": "volume_cap", "max_volume_fraction": "0.1"},
        },
    )
    result = run_asset_backtest(strategy_spec, definition_of(target_current_symbol), Weight(), data)
    if expected_quantity is None:
        assert result.fills == []
        assert any(warning.startswith("VOLUME_CAPPED:") for warning in result.warnings)
    else:
        assert result.fills[0].quantity == expected_quantity
        assert any(warning.startswith("VOLUME_CAPPED:") for warning in result.warnings)


def test_open_gap_limits_affordability_and_minimum_notional_can_skip_a_fill(monkeypatch) -> None:
    sessions = weekdays(date(2024, 1, 2), 6)
    data = make_dataset(
        {OLD: ([100] * 6, [True] * 6)},
        make_aliases(sessions),
        open_prices={OLD: [100, 110, 100, 100, 100, 100]},
    )
    full_weight = _definition(lambda _ctx, _params: Decision.target({"SYNA": Decimal(1)}, "GO"))
    strategy_spec = spec(("SYNA",), execution={"costs": {"bps": "0", "per_share": "0"}})
    capped = run_asset_backtest(strategy_spec, full_weight, Weight(), data)
    assert capped.fills[0].quantity == Decimal(90)
    assert any(warning.startswith("INSUFFICIENT_SETTLED_CASH:") for warning in capped.warnings)

    monkeypatch.setattr(
        asset_backtest,
        "plan_asset_orders",
        lambda *args, **kwargs: ([AssetPlannedOrder(OLD, Decimal(1), Decimal(100))], True, []),
    )
    too_small = run_asset_backtest(
        spec(
            ("SYNA",),
            execution={
                "costs": {"bps": "0", "per_share": "0"},
                "min_order_notional": "200",
            },
        ),
        definition_of(target_current_symbol),
        Weight(),
        data,
    )
    assert too_small.fills == []


def test_full_sell_removes_position_and_margin_buy_draws_from_pending_sale() -> None:
    sessions = weekdays(date(2024, 1, 2), 6)
    data = make_dataset(
        {OLD: ([100] * 6, [True] * 6)},
        make_aliases(sessions),
    )

    def exit_after_first_trade(ctx: Ctx, _params: Weight) -> Decision:
        weight = Decimal("0.5") if ctx.decision_session < sessions[2] else Decimal(0)
        return Decision.target({"SYNA": weight}, "GO")

    sold = run_asset_backtest(
        spec(("SYNA",), execution={"costs": {"bps": "0", "per_share": "0"}}),
        _definition(exit_after_first_trade),
        Weight(),
        data,
    )
    assert [fill.side for fill in sold.fills] == ["buy", "sell"]
    assert sold.positions == {}

    aliases = (
        alias("old", OLD, "SYNA", sessions[0], None, datetime.combine(sessions[0], time(10), UTC)),
        alias("new", NEW, "SYNB", sessions[0], None, datetime.combine(sessions[0], time(10), UTC)),
    )
    two_assets = make_dataset(
        {OLD: ([100] * 6, [True] * 6), NEW: ([100] * 6, [True] * 6)},
        aliases,
    )

    def rotate(ctx: Ctx, _params: Weight) -> Decision:
        symbol = "SYNA" if ctx.decision_session < sessions[2] else "SYNB"
        return Decision.target({symbol: Decimal(1)}, "GO")

    margin = run_asset_backtest(
        spec(
            ("SYNA", "SYNB"),
            account={"model": "margin"},
            execution={"costs": {"bps": "0", "per_share": "0"}},
        ),
        _definition(rotate),
        Weight(),
        two_assets,
    )
    assert [fill.side for fill in margin.fills] == ["buy", "sell", "buy"]
    assert margin.positions == {NEW: Decimal(100)}


def test_decision_history_gates_cover_missing_stale_and_unmapped_bars() -> None:
    sessions = weekdays(date(2024, 1, 2), 6)
    aliases = make_aliases(sessions)
    enough_data = make_dataset({OLD: ([100] * 6, [True] * 6)}, aliases)
    short_history = _direct_decision(enough_data, ("SYNA",), index=1, lookback=2)
    assert short_history.action == "unavailable"
    assert short_history.reason_codes == ("INSUFFICIENT_HISTORY",)

    no_observations = make_dataset({OLD: ([100] * 6, [False] * 6)}, aliases)
    empty = _direct_decision(no_observations, ("SYNA",), index=1)
    assert empty.reason_codes == ("INSUFFICIENT_HISTORY",)

    outside_lookback = make_dataset({OLD: ([100] * 6, [True, False, False, False, False, False])}, aliases)
    aged_out = _direct_decision(outside_lookback, ("SYNA",), index=2)
    assert aged_out.reason_codes == ("INSUFFICIENT_HISTORY",)

    stale_within_lookback = make_dataset(
        {OLD: ([100] * 6, [False, True, False, False, False, False])}, aliases
    )
    stale = _direct_decision(stale_within_lookback, ("SYNA",), index=3, lookback=3, max_staleness=0)
    assert stale.reason_codes == ("STALE_OBSERVATIONS",)

    no_aliases = make_dataset({OLD: ([100] * 6, [True] * 6)}, ())
    unmapped = _direct_decision(no_aliases, ("SYNA",), index=1)
    assert unmapped.reason_codes == ("INSUFFICIENT_HISTORY",)


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (LifecycleState({OLD: Decimal(0)}, {OLD: "SYNA"}), Decision.hold("WAIT")),
        (SimpleNamespace(positions={OLD: Decimal(1)}, symbols={}), "held asset lacks a current alias"),
    ],
)
def test_decision_handles_zero_holdings_and_refuses_missing_held_alias(state: Any, expected: Any) -> None:
    sessions = weekdays(date(2024, 1, 2), 6)
    data = make_dataset(
        {OLD: ([100] * 6, [True] * 6)},
        make_aliases(sessions),
    )
    if isinstance(expected, str):
        with pytest.raises(EngineError, match=expected):
            _direct_decision(data, ("SYNA",), index=1, state=state)
    else:
        result = _direct_decision(data, ("SYNA",), index=1, state=state)
        assert result.action == expected.action


@pytest.mark.parametrize(
    ("decide", "message"),
    [
        (lambda _ctx, _params: "not a decision", "STRATEGY_RETURNED_NON_DECISION"),
        (lambda _ctx, _params: Decision.hold("UNDECLARED"), "REASON_CODE_UNDECLARED"),
        (
            lambda _ctx, _params: Decision.target({"OTHER": Decimal("0.5")}, "GO"),
            "DECISION_SYMBOL_NOT_DECLARED",
        ),
        (
            lambda _ctx, _params: Decision.target({"SYNA": Decimal("0.75")}, "GO"),
            "DECISION_WEIGHT_ABOVE_LIMIT",
        ),
    ],
)
def test_asset_strategy_contract_errors_are_enforced(decide, message: str) -> None:
    sessions = weekdays(date(2024, 1, 2), 6)
    data = make_dataset(
        {OLD: ([100] * 6, [True] * 6)},
        make_aliases(sessions),
    )
    with pytest.raises(EngineError, match=message):
        _direct_decision(
            data,
            ("SYNA",),
            index=1,
            decide=decide,
            max_weight=Decimal("0.5") if message == "DECISION_WEIGHT_ABOVE_LIMIT" else Decimal(1),
        )


def test_asset_bars_keep_gaps_as_nan_and_missing_marks_fail_closed() -> None:
    sessions = weekdays(date(2024, 1, 2), 6)
    aliases = make_aliases(sessions)
    gapped = make_dataset(
        {OLD: ([100] * 6, [True, True, False, True, True, True])},
        aliases,
    )

    def inspect_gap(ctx: Ctx, _params: Weight) -> Decision:
        bars = ctx.bars("SYNA")
        assert np.isnan(bars.close[2]) and np.isnan(bars.volume[2])
        return Decision.hold("WAIT")

    _direct_decision(gapped, ("SYNA",), index=4, lookback=4, decide=inspect_gap)
    stale_marks = make_dataset(
        {OLD: ([100] * 6, [True, False, False, False, False, False])},
        aliases,
    )
    assert _asset_mark(stale_marks, (), OLD, 2, at_close=False, max_staleness=0) is None
    assert (
        _asset_mark(
            make_dataset({OLD: ([100] * 6, [False] * 6)}, aliases),
            (),
            OLD,
            2,
            at_close=False,
            max_staleness=0,
        )
        is None
    )
    zero_state = LifecycleState({OLD: Decimal(0)}, {OLD: "SYNA"})
    assert _marks_for_positions(gapped, (), zero_state, 2, at_close=False, max_staleness=0) == {}
