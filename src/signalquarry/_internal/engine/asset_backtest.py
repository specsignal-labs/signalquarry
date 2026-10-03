# SPDX-License-Identifier: Apache-2.0
"""Synthetic-only backtest path with asset-keyed holdings and dated symbols."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_DOWN, ROUND_HALF_EVEN, Decimal
from typing import Any

import numpy as np

from signalquarry._internal.canonical import canonical_hash
from signalquarry._internal.contracts.reason_codes import REASON_CODES
from signalquarry._internal.contracts.spec import StrategySpecV1
from signalquarry._internal.data.asset_dataset import (
    AssetDatasetError,
    AssetDatasetV1,
    _asset_document,
    _event_document,
)
from signalquarry._internal.data.dataset import T_PLUS_ONE_FROM
from signalquarry._internal.data.identity import AssetKey, IdentityError
from signalquarry._internal.data.lifecycle import LifecycleError, LifecycleEvent
from signalquarry._internal.engine.backtest import (
    CENT,
    MICRO_QTY,
    EngineError,
    _draw_pending,
    _fee,
    _quantize_cash,
    _state_or_raise,
    affordable_quantity,
    plan_orders,
)
from signalquarry._internal.engine.lifecycle import (
    LifecycleState,
    apply_lifecycle_event,
    settle_lifecycle_receivables,
    value_lifecycle_state,
)
from signalquarry.sdk.context import Bars, Ctx, freeze_state
from signalquarry.sdk.decision import Decision
from signalquarry.sdk.strategy import Params, StrategyDef

ZERO = Decimal(0)
WEIGHT_QUANTUM = Decimal("0.000001")
MAX_PLANNER_ASSETS = 0xFFFFFFFF


@dataclass(frozen=True)
class AssetPlannedOrder:
    asset: AssetKey
    delta: Decimal
    mark: Decimal


@dataclass(frozen=True)
class AssetPreOpenPlan:
    """Asset-keyed account transition and order plan for one session."""

    session: date
    state: LifecycleState
    settled_cash: Decimal
    pending_cash: tuple[tuple[int, Decimal], ...]
    applied_events: Mapping[str, LifecycleEvent]
    applied_event_records: tuple[dict[str, str], ...]
    known_events: tuple[LifecycleEvent, ...]
    strategy_state: dict[str, Any]
    last_target: dict[AssetKey, Decimal] | None
    target_complete: bool
    queued_targets: dict[int, dict[AssetKey, Decimal]]
    equity: Decimal
    decision: Decision
    decision_record: dict[str, Any]
    orders: tuple[AssetPlannedOrder, ...]
    warnings: tuple[str, ...]


def plan_asset_orders(
    weights: Mapping[AssetKey, Decimal],
    quantity: Mapping[AssetKey, Decimal],
    equity: Decimal,
    marks: Mapping[AssetKey, Decimal | None],
    *,
    fractional: bool,
    min_order_notional: Decimal,
    session: date,
    aliases: Mapping[AssetKey, str],
    opens: Mapping[AssetKey, Decimal | None] | None = None,
) -> tuple[list[AssetPlannedOrder], bool, list[str]]:
    """Adapt asset keys to the shared planner while retaining its sizing rules."""
    assets = sorted(set(weights) | set(quantity) | set(marks) | (set(opens) if opens is not None else set()))
    if len(assets) > MAX_PLANNER_ASSETS:
        raise EngineError("CORPORATE_ACTION_UNSUPPORTED:too many assets for deterministic planner keys")
    tokens = {asset: f"A{index:08X}" for index, asset in enumerate(assets)}
    reversed_tokens = {token: asset for asset, token in tokens.items()}
    planned, complete, warnings = plan_orders(
        {tokens[asset]: weight for asset, weight in weights.items()},
        {tokens[asset]: amount for asset, amount in quantity.items()},
        equity,
        {tokens[asset]: mark for asset, mark in marks.items()},
        fractional=fractional,
        min_order_notional=min_order_notional,
        session=session,
        opens={tokens[asset]: price for asset, price in opens.items()} if opens is not None else None,
    )
    orders = [AssetPlannedOrder(reversed_tokens[order.symbol], order.delta, order.mark) for order in planned]
    display_warnings: list[str] = []
    for warning in warnings:
        token = warning.rsplit(":", 1)[-1]
        asset = reversed_tokens[token]
        display = aliases.get(asset, f"{asset.provider}:{asset.asset_id}")
        display_warnings.append(f"{warning[: -len(token)]}{display}")
    return orders, complete, display_warnings


@dataclass(frozen=True)
class AssetFill:
    session: date
    asset: AssetKey
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    fee: Decimal
    settle_session: date | None


@dataclass
class AssetBacktestResult:
    sessions: list[date]
    equity: list[Decimal]
    cash: list[Decimal]
    decisions: list[dict[str, Any]]
    fills: list[AssetFill]
    positions: dict[AssetKey, Decimal]
    warnings: list[str]
    dataset_identity: str
    applied_events: list[dict[str, str]]
    ledger_hash: str = ""

    def ledger_document(self) -> dict[str, Any]:
        return {
            "dataset_identity": self.dataset_identity,
            "equity": [
                [session.isoformat(), equity, cash]
                for session, equity, cash in zip(self.sessions, self.equity, self.cash, strict=True)
            ],
            "decisions": self.decisions,
            "fills": [
                [
                    fill.session.isoformat(),
                    _asset_document(fill.asset),
                    fill.symbol,
                    fill.side,
                    fill.quantity,
                    fill.price,
                    fill.fee,
                    fill.settle_session.isoformat() if fill.settle_session else None,
                ]
                for fill in self.fills
            ],
            "positions": [
                {"asset": _asset_document(asset), "quantity": quantity}
                for asset, quantity in sorted(self.positions.items())
            ],
            "applied_events": self.applied_events,
        }


@dataclass
class _AssetAccount:
    settled: Decimal
    state: LifecycleState
    pending: list[tuple[int, Decimal]] = field(default_factory=list)

    def pending_total(self) -> Decimal:
        return sum((amount for _, amount in self.pending), ZERO)

    def receivable_total(self) -> Decimal:
        return sum((item.amount for item in self.state.receivables), ZERO)

    def total_cash(self) -> Decimal:
        return self.settled + self.pending_total() + self.receivable_total()


def plan_asset_pre_open(
    spec: StrategySpecV1,
    definition: StrategyDef,
    params: Params,
    dataset: AssetDatasetV1,
    *,
    index: int,
    first_index: int,
    lookback: int,
    state: LifecycleState,
    settled_cash: Decimal,
    pending_cash: list[tuple[int, Decimal]],
    applied_events: Mapping[str, LifecycleEvent],
    strategy_state: dict[str, Any],
    last_target: Mapping[AssetKey, Decimal] | None,
    target_complete: bool,
    queued_targets: Mapping[int, Mapping[AssetKey, Decimal]],
    allowed_codes: set[str],
    simulated_open_price: Callable[[AssetKey], Decimal | None] | None = None,
) -> AssetPreOpenPlan:
    """Apply known lifecycle events, decide, and size asset-keyed orders.

    This pure planning path is shared by synthetic asset-keyed backtests and
    pre-open parity checks. ``simulated_open_price`` is simulator-only; omit it
    when producing a true pre-open intent. Provider data and paper activation
    remain outside this checkpoint.
    """
    if not (dataset.source == "synthetic" or dataset.source.startswith("synthetic:")) or any(
        asset.provider != "synthetic" for asset in dataset.series
    ):
        raise EngineError("CORPORATE_ACTION_UNSUPPORTED:asset-keyed planning is synthetic-only")
    if any(
        event.source.provider != "synthetic"
        or (event.target is not None and event.target.provider != "synthetic")
        for event in dataset.events
    ):
        raise EngineError("CORPORATE_ACTION_UNSUPPORTED:asset-keyed planning is synthetic-only")
    if index < 0 or index >= len(dataset.sessions) or first_index < 0 or first_index > index:
        raise EngineError("BACKTEST_RANGE_EMPTY")
    if lookback < 1:
        raise EngineError("STRATEGY_LOOKBACK_INVALID")

    session = dataset.sessions[index]
    cutoff = dataset.decision_cutoffs[index]
    try:
        aliases = dataset.aliases_as_of(session, cutoff)
        known_events = dataset.events_as_of(cutoff)
    except (AssetDatasetError, IdentityError, LifecycleError) as exc:
        raise EngineError(str(exc)) from exc
    aliases_by_asset = {asset: symbol for symbol, asset in aliases.items()}

    settled = settled_cash + sum((amount for due, amount in pending_cash if due <= index), ZERO)
    pending = [(due, amount) for due, amount in pending_cash if due > index]
    try:
        state, settled = settle_lifecycle_receivables(state, settled, session)
    except LifecycleError as exc:
        raise EngineError(str(exc)) from exc

    due_events: list[LifecycleEvent] = []
    applied = dict(applied_events)
    for event in known_events:
        effective_index = next(
            (i for i, item in enumerate(dataset.sessions) if item >= event.effective_date),
            len(dataset.sessions),
        )
        previous = applied.get(event.event_id)
        if previous is not None:
            if previous != event:
                raise EngineError("CORPORATE_ACTION_UNSUPPORTED:applied event was revised")
            continue
        if effective_index < index:
            if effective_index >= first_index:
                raise EngineError(
                    "CORPORATE_ACTION_UNSUPPORTED:event was first observed after its effective session"
                )
            continue
        if effective_index == index:
            due_events.append(event)

    protected = {asset for asset, quantity in state.positions.items() if quantity}
    protected.update(event.source for event in due_events)
    protected.update(event.target for event in due_events if event.target is not None)
    symbols_before_events: dict[AssetKey, str] = {}
    for asset, symbol in aliases_by_asset.items():
        if asset in protected:
            previous_symbol = state.symbols.get(asset)
            if previous_symbol is not None:
                symbols_before_events[asset] = previous_symbol
        else:
            symbols_before_events[asset] = symbol
    for asset in protected:
        if asset not in symbols_before_events and asset in state.symbols:
            symbols_before_events[asset] = state.symbols[asset]
    try:
        state = LifecycleState(
            state.positions,
            symbols_before_events,
            state.receivables,
            state.applied_event_ids,
            state.last_event_key,
        )
    except LifecycleError as exc:
        raise EngineError(str(exc)) from exc

    applied_records: list[dict[str, str]] = []
    try:
        for event in sorted(due_events, key=lambda item: (item.effective_date, item.sequence, item.event_id)):
            state = apply_lifecycle_event(state, event)
            applied[event.event_id] = event
            applied_records.append(
                {"event_id": event.event_id, "version_identity": canonical_hash(_event_document(event))}
            )
        state, settled = settle_lifecycle_receivables(state, settled, session)
    except LifecycleError as exc:
        raise EngineError(str(exc)) from exc

    held = {asset for asset, quantity in state.positions.items() if quantity}
    for asset in held:
        if aliases_by_asset.get(asset) != state.symbols.get(asset):
            raise EngineError("CORPORATE_ACTION_UNSUPPORTED:held asset alias differs from point-in-time map")
    current_symbols = dict(aliases_by_asset)
    for asset in held:
        current_symbols[asset] = state.symbols[asset]
    state = LifecycleState(
        state.positions,
        current_symbols,
        state.receivables,
        state.applied_event_ids,
        state.last_event_key,
    )

    previous_marks = _marks_for_positions(
        dataset,
        known_events,
        state,
        index,
        at_close=False,
        max_staleness=spec.data.max_staleness_sessions,
    )
    cash = (
        settled
        + sum((amount for _, amount in pending), ZERO)
        + sum((item.amount for item in state.receivables), ZERO)
    )
    invested = sum(
        (quantity * previous_marks[asset] for asset, quantity in state.positions.items() if quantity), ZERO
    )
    equity = cash + invested
    aliases_for_spec = {symbol: asset for symbol, asset in aliases.items() if symbol in spec.data.symbols}
    decision = _decide_asset(
        spec,
        definition,
        params,
        dataset,
        known_events,
        aliases_for_spec,
        state,
        previous_marks,
        cash,
        strategy_state,
        index,
        lookback,
        allowed_codes,
    )
    record: dict[str, Any] = {
        "session": session.isoformat(),
        "action": decision.action,
        "weights": dict(decision.weights),
        "reason_codes": list(decision.reason_codes),
        "aliases": {symbol: _asset_document(asset) for symbol, asset in sorted(aliases_for_spec.items())},
    }
    next_strategy_state = strategy_state
    if decision.action != "unavailable":
        next_strategy_state = _state_or_raise(decision.state, strategy_state)

    next_last_target = dict(last_target) if last_target is not None else None
    next_complete = target_complete
    queue = {due: dict(target) for due, target in queued_targets.items()}
    if decision.action == "target":
        missing_aliases = [symbol for symbol in decision.weights if symbol not in aliases_for_spec]
        if missing_aliases:
            raise EngineError(
                f"CORPORATE_ACTION_UNSUPPORTED:decision symbol has no asset at cutoff: {','.join(missing_aliases)}"
            )
        target = {aliases_for_spec[symbol]: weight for symbol, weight in decision.weights.items()}
        changed = last_target is None or target != last_target
        if changed or not target_complete or spec.execution.rebalance == "every_decision":
            execute_index = index + spec.execution.execution_delay_sessions
            queue[execute_index] = target
            record["queued_for"] = dataset.sessions[min(execute_index, len(dataset.sessions) - 1)].isoformat()
            record["asset_targets"] = [
                {"asset": _asset_document(asset), "weight": weight}
                for asset, weight in sorted(target.items())
            ]
        next_last_target = target

    order_target = queue.pop(index, None)
    planned_orders: list[AssetPlannedOrder] = []
    warnings: list[str] = []
    if order_target is not None:
        retired = [asset for asset in order_target if asset not in aliases_by_asset]
        if retired:
            raise EngineError("CORPORATE_ACTION_UNSUPPORTED:queued target references a retired asset")
        order_assets = set(state.positions) | set(order_target)
        marks = {
            asset: _asset_mark(
                dataset,
                known_events,
                asset,
                index,
                at_close=False,
                max_staleness=spec.data.max_staleness_sessions,
            )
            for asset in order_assets
        }
        opens = (
            {asset: simulated_open_price(asset) for asset in order_assets}
            if simulated_open_price is not None
            else None
        )
        planned, next_complete, plan_warnings = plan_asset_orders(
            order_target,
            state.positions,
            equity,
            marks,
            fractional=spec.execution.sizing == "fractional",
            min_order_notional=spec.execution.min_order_notional,
            session=session,
            aliases=state.symbols,
            opens=opens,
        )
        planned_orders = planned
        warnings.extend(plan_warnings)

    return AssetPreOpenPlan(
        session=session,
        state=state,
        settled_cash=settled,
        pending_cash=tuple(pending),
        applied_events=applied,
        applied_event_records=tuple(applied_records),
        known_events=known_events,
        strategy_state=next_strategy_state,
        last_target=next_last_target,
        target_complete=next_complete,
        queued_targets=queue,
        equity=equity,
        decision=decision,
        decision_record=record,
        orders=tuple(planned_orders),
        warnings=tuple(warnings),
    )


def run_asset_backtest(
    spec: StrategySpecV1,
    definition: StrategyDef,
    params: Params,
    dataset: AssetDatasetV1,
    *,
    start: date | None = None,
    end: date | None = None,
) -> AssetBacktestResult:
    """Run the strategy against synthetic-only, point-in-time asset identities.

    Position keys and target weights are ``AssetKey`` values for the whole
    simulation. Display symbols enter only when building strategy context or
    recording fills. Lifecycle terms known at each decision cutoff are applied
    before planning. Provider data and paper execution are deliberately outside
    this path.
    """
    if not (dataset.source == "synthetic" or dataset.source.startswith("synthetic:")) or any(
        asset.provider != "synthetic" for asset in dataset.series
    ):
        raise EngineError("CORPORATE_ACTION_UNSUPPORTED:asset-keyed backtest is synthetic-only")
    if any(
        event.source.provider != "synthetic"
        or (event.target is not None and event.target.provider != "synthetic")
        for event in dataset.events
    ):
        raise EngineError("CORPORATE_ACTION_UNSUPPORTED:asset-keyed backtest is synthetic-only")

    lookback = int(definition.lookback(params))
    if lookback < 1:
        raise EngineError("STRATEGY_LOOKBACK_INVALID")
    sessions = dataset.sessions
    first = next(
        (index for index, session in enumerate(sessions) if start is None or session >= start), len(sessions)
    )
    last = max((index for index, session in enumerate(sessions) if end is None or session <= end), default=-1)
    if first > last:
        raise EngineError("BACKTEST_RANGE_EMPTY")
    first = max(first, 1)
    if first > last:
        raise EngineError("BACKTEST_RANGE_EMPTY")

    try:
        initial_aliases = dataset.aliases_as_of(sessions[first - 1], dataset.decision_cutoffs[first - 1])
    except (AssetDatasetError, IdentityError) as exc:
        raise EngineError(str(exc)) from exc
    initial_symbols = {asset: symbol for symbol, asset in initial_aliases.items()}
    account = _AssetAccount(spec.account.initial_cash, LifecycleState({}, initial_symbols))
    strategy_state: dict[str, Any] = {}
    last_target: dict[AssetKey, Decimal] | None = None
    target_complete = True
    queued: dict[int, dict[AssetKey, Decimal]] = {}
    applied: dict[str, LifecycleEvent] = {}
    result = AssetBacktestResult([], [], [], [], [], {}, [], dataset.identity(), [])
    allowed_codes = set(spec.reason_codes) | set(REASON_CODES)
    execution = spec.execution
    bps = execution.costs.bps / Decimal(10000)
    per_share = execution.costs.per_share
    fractional = execution.sizing == "fractional"

    for index in range(first, last + 1):
        plan = plan_asset_pre_open(
            spec,
            definition,
            params,
            dataset,
            index=index,
            first_index=first,
            lookback=lookback,
            state=account.state,
            settled_cash=account.settled,
            pending_cash=account.pending,
            applied_events=applied,
            strategy_state=strategy_state,
            last_target=last_target,
            target_complete=target_complete,
            queued_targets=queued,
            allowed_codes=allowed_codes,
            simulated_open_price=lambda asset, session_index=index: dataset.price(
                asset, "open", session_index
            ),
        )
        session = plan.session
        account = _AssetAccount(plan.settled_cash, plan.state, list(plan.pending_cash))
        applied = dict(plan.applied_events)
        result.applied_events.extend(plan.applied_event_records)
        result.warnings.extend(plan.warnings)
        result.decisions.append(plan.decision_record)
        strategy_state = plan.strategy_state
        last_target = plan.last_target
        target_complete = plan.target_complete
        queued = plan.queued_targets
        known_events = plan.known_events

        settle_index = _settlement_index(sessions, index)
        settle_session = sessions[settle_index] if settle_index < len(sessions) else None
        for order in plan.orders:
            symbol = account.state.symbols.get(order.asset)
            if symbol is None:
                raise EngineError("CORPORATE_ACTION_UNSUPPORTED:order asset lacks a current alias")
            open_price = dataset.price(order.asset, "open", index)
            if open_price is None:
                continue
            amount = abs(order.delta)
            if execution.fill is not None:
                volume = Decimal(str(float(dataset.series[order.asset].volume[index])))
                quantum = MICRO_QTY if fractional else Decimal(1)
                cap = (volume * execution.fill.max_volume_fraction).quantize(quantum, rounding=ROUND_DOWN)
                if amount > cap:
                    result.warnings.append(f"VOLUME_CAPPED:{session.isoformat()}:{symbol}")
                    amount, target_complete = cap, False
                if amount <= 0:
                    continue
            positions = dict(account.state.positions)
            if order.delta < 0:
                notional = amount * open_price
                fee = (
                    notional * bps + amount * per_share + execution.costs.sell_fees(notional, amount)
                ).quantize(CENT, rounding=ROUND_HALF_EVEN)
                account.pending.append((settle_index, _quantize_cash(notional - fee)))
                positions[order.asset] = positions.get(order.asset, ZERO) - amount
                side = "sell"
            else:
                available = (
                    account.settled
                    if spec.account.model == "cash"
                    else account.settled + account.pending_total()
                )
                affordable = affordable_quantity(available, open_price, bps, per_share, fractional)
                if affordable < amount:
                    result.warnings.append(f"INSUFFICIENT_SETTLED_CASH:{session.isoformat()}:{symbol}")
                    amount, target_complete = affordable, False
                if amount <= 0 or amount * open_price < execution.min_order_notional:
                    continue
                notional = amount * open_price
                fee = _fee(notional, amount, bps, per_share)
                cost = _quantize_cash(notional + fee)
                if spec.account.model == "cash" or cost <= account.settled:
                    account.settled -= cost
                else:
                    remaining = cost - account.settled
                    account.settled = ZERO
                    account.pending = _draw_pending(account.pending, remaining)
                positions[order.asset] = positions.get(order.asset, ZERO) + amount
                side = "buy"
            if positions[order.asset] == 0:
                positions.pop(order.asset)
            account.state = LifecycleState(
                positions,
                account.state.symbols,
                account.state.receivables,
                account.state.applied_event_ids,
                account.state.last_event_key,
            )
            result.fills.append(
                AssetFill(
                    session,
                    order.asset,
                    symbol,
                    side,
                    amount,
                    open_price,
                    fee,
                    settle_session if side == "sell" else None,
                )
            )

        # 5. Close valuation requires a valid mark for every remaining holding.
        try:
            close_marks = _marks_for_positions(
                dataset,
                known_events,
                account.state,
                index,
                at_close=True,
                max_staleness=spec.data.max_staleness_sessions,
            )
            close_equity = value_lifecycle_state(
                account.state, close_marks, account.settled + account.pending_total(), session
            )
        except LifecycleError as exc:
            raise EngineError(str(exc)) from exc
        result.sessions.append(session)
        result.equity.append(_quantize_cash(close_equity))
        result.cash.append(_quantize_cash(account.settled))

    result.positions = {
        asset: quantity for asset, quantity in sorted(account.state.positions.items()) if quantity
    }
    result.ledger_hash = canonical_hash(result.ledger_document())
    return result


def _decide_asset(
    spec: StrategySpecV1,
    definition: StrategyDef,
    params: Params,
    dataset: AssetDatasetV1,
    events: tuple[LifecycleEvent, ...],
    aliases: dict[str, AssetKey],
    state: LifecycleState,
    marks: dict[AssetKey, Decimal],
    cash: Decimal,
    strategy_state: dict[str, Any],
    index: int,
    lookback: int,
    allowed_codes: set[str],
) -> Decision:
    if index < lookback:
        return Decision.unavailable("INSUFFICIENT_HISTORY")
    start = index - lookback
    bars: dict[str, Bars] = {}
    for symbol in spec.data.symbols:
        asset = aliases.get(symbol)
        if asset is None:
            continue
        present = dataset.series[asset].present
        observed = np.flatnonzero(present[:index])
        if observed.size == 0:
            return Decision.unavailable("INSUFFICIENT_HISTORY")
        last = int(observed[-1])
        if last < start:
            return Decision.unavailable("INSUFFICIENT_HISTORY")
        if index - 1 - last > spec.data.max_staleness_sessions:
            return Decision.unavailable("STALE_OBSERVATIONS")
        bars[symbol] = _bars(dataset, events, asset, symbol, start, index, dataset.sessions[index])
    if not bars:
        return Decision.unavailable("INSUFFICIENT_HISTORY")

    holdings: dict[str, Decimal] = {}
    for asset, amount in state.positions.items():
        if amount:
            symbol = state.symbols.get(asset)
            if symbol is None:
                raise EngineError("CORPORATE_ACTION_UNSUPPORTED:held asset lacks a current alias")
            holdings[symbol] = amount
    invested = sum((amount * marks[asset] for asset, amount in state.positions.items() if amount), ZERO)
    equity = cash + invested
    weights = {
        symbol: (
            amount
            * marks[next(asset for asset, current in state.symbols.items() if current == symbol)]
            / equity
        ).quantize(WEIGHT_QUANTUM)
        if equity
        else ZERO
        for symbol, amount in holdings.items()
    }
    ctx = Ctx(
        decision_session=dataset.sessions[index],
        _bars=freeze_state(bars),
        positions=freeze_state(holdings),
        weights=freeze_state(weights),
        cash=_quantize_cash(cash),
        equity=_quantize_cash(equity),
        state=freeze_state(strategy_state),
    )
    decision = definition.decide(ctx, params)
    if not isinstance(decision, Decision):
        raise EngineError(f"STRATEGY_RETURNED_NON_DECISION:{type(decision).__name__}")
    undeclared = [code for code in decision.reason_codes if code not in allowed_codes]
    if undeclared:
        raise EngineError(f"REASON_CODE_UNDECLARED:{','.join(undeclared)}")
    unknown = [symbol for symbol in decision.weights if symbol not in spec.data.symbols]
    if unknown:
        raise EngineError(f"DECISION_SYMBOL_NOT_DECLARED:{','.join(unknown)}")
    over = [
        symbol for symbol, weight in decision.weights.items() if weight > spec.limits.max_weight_per_symbol
    ]
    if over:
        raise EngineError(f"DECISION_WEIGHT_ABOVE_LIMIT:{','.join(over)}")
    return decision


def _bars(
    dataset: AssetDatasetV1,
    events: tuple[LifecycleEvent, ...],
    asset: AssetKey,
    symbol: str,
    start: int,
    stop: int,
    basis_session: date,
) -> Bars:
    item = dataset.series[asset]
    basis = _split_factor(events, asset, basis_session)
    arrays: dict[str, np.ndarray] = {}
    for name in ("open", "high", "low", "close"):
        values = np.full(stop - start, np.nan, dtype=np.float64)
        for offset, index in enumerate(range(start, stop)):
            price = item.price(name, index)
            if price is not None:
                values[offset] = float(price * _split_factor(events, asset, dataset.sessions[index]) / basis)
        values.setflags(write=False)
        arrays[name] = values
    volume = np.full(stop - start, np.nan, dtype=np.float64)
    for offset, index in enumerate(range(start, stop)):
        if item.present[index]:
            volume[offset] = float(item.volume[index]) * float(
                basis / _split_factor(events, asset, dataset.sessions[index])
            )
    volume.setflags(write=False)
    dates = np.asarray(dataset.sessions[start:stop], dtype="datetime64[D]")
    dates.setflags(write=False)
    return Bars(symbol, dates, arrays["open"], arrays["high"], arrays["low"], arrays["close"], volume)


def _split_factor(events: tuple[LifecycleEvent, ...], asset: AssetKey, session: date) -> Decimal:
    factor = Decimal(1)
    for event in events:
        if event.kind == "split" and event.source == asset and event.effective_date <= session:
            assert event.share_ratio is not None
            factor *= event.share_ratio
    return factor


def _asset_mark(
    dataset: AssetDatasetV1,
    events: tuple[LifecycleEvent, ...],
    asset: AssetKey,
    index: int,
    *,
    at_close: bool,
    max_staleness: int,
) -> Decimal | None:
    item = dataset.series[asset]
    stop = index + 1 if at_close else index
    observed = np.flatnonzero(item.present[:stop])
    if observed.size == 0:
        return None
    last = int(observed[-1])
    if stop - 1 - last > max_staleness:
        return None
    raw = item.price("close", last)
    assert raw is not None
    basis = _split_factor(events, asset, dataset.sessions[index])
    historical = _split_factor(events, asset, dataset.sessions[last])
    return raw * historical / basis


def _marks_for_positions(
    dataset: AssetDatasetV1,
    events: tuple[LifecycleEvent, ...],
    state: LifecycleState,
    index: int,
    *,
    at_close: bool,
    max_staleness: int,
) -> dict[AssetKey, Decimal]:
    marks: dict[AssetKey, Decimal] = {}
    for asset, quantity in state.positions.items():
        if not quantity:
            continue
        mark = _asset_mark(dataset, events, asset, index, at_close=at_close, max_staleness=max_staleness)
        if mark is None or mark <= 0:
            raise EngineError(
                f"CORPORATE_ACTION_UNSUPPORTED:missing or invalid mark for {asset.provider}:{asset.asset_id}"
            )
        marks[asset] = mark
    return marks


def _settlement_index(sessions: tuple[date, ...], trade_index: int) -> int:
    lag = 2 if sessions[trade_index] < T_PLUS_ONE_FROM else 1
    return trade_index + lag
