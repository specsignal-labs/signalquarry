# SPDX-License-Identifier: Apache-2.0
"""The session engine. Backtests drive it with a historical clock and a simulated broker.

Per session ``D`` (fixed order):

1. Settle sale proceeds and pay dividends whose settlement/payment falls on ``D``.
2. Apply corporate actions with ex-date ``D``: splits (quantities × ratio), then
   dividend entitlements for shares held after ``D-1``'s close.
3. PRE_OPEN: build the context from completed bars through ``D-1``
   (point-in-time split-adjusted, exactly ``lookback`` sessions) and call ``decide``.
4. Queue the decision for execution at ``D + execution_delay_sessions``.
5. OPEN: execute queued decisions at the raw open. Size from prior-close marks,
   sells before buys, symbols sorted; cash accounts spend settled cash only.
6. CLOSE: mark positions at the raw close.

Strategy state from ``target`` and ``hold`` is persisted identically; an
incomplete rebalance (partial fill, missing price, cash limit) is retried on the
next unchanged target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_DOWN, ROUND_HALF_EVEN, Decimal
from typing import Any

import numpy as np

from signalquarry._internal.canonical import canonical_hash, canonical_json, to_canonical
from signalquarry._internal.contracts.reason_codes import REASON_CODES
from signalquarry._internal.contracts.spec import StrategySpecV1
from signalquarry._internal.data.dataset import FIELDS, MICRO, Dataset
from signalquarry.sdk.context import Bars, Ctx, freeze_state
from signalquarry.sdk.decision import Decision
from signalquarry.sdk.strategy import Params, StrategyDef

CENT = Decimal("0.01")
MICRO_QTY = Decimal("0.000001")
CASH_QUANTUM = Decimal("0.000001")
MAX_STATE_BYTES = 16 * 1024
ZERO = Decimal(0)


class EngineError(RuntimeError):
    """A strategy or input broke the contract; the run stops."""


@dataclass
class Fill:
    session: date
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    fee: Decimal
    settle_session: date | None


@dataclass
class BacktestResult:
    sessions: list[date]
    equity: list[Decimal]
    cash: list[Decimal]
    decisions: list[dict[str, Any]]
    fills: list[Fill]
    positions: dict[str, Decimal]
    warnings: list[str]
    dataset_identity: str
    ledger_hash: str = ""

    def ledger_document(self) -> dict[str, Any]:
        return {
            "dataset_identity": self.dataset_identity,
            "equity": [
                [s.isoformat(), e, c] for s, e, c in zip(self.sessions, self.equity, self.cash, strict=True)
            ],
            "decisions": self.decisions,
            "fills": [
                [
                    f.session.isoformat(),
                    f.symbol,
                    f.side,
                    f.quantity,
                    f.price,
                    f.fee,
                    f.settle_session.isoformat() if f.settle_session else None,
                ]
                for f in self.fills
            ],
            "positions": dict(sorted(self.positions.items())),
        }


@dataclass
class _Account:
    settled: Decimal
    pending: list[tuple[int, Decimal]] = field(default_factory=list)  # (settle index, amount)
    receivable: list[tuple[int, Decimal]] = field(default_factory=list)  # (pay index, amount)
    quantity: dict[str, Decimal] = field(default_factory=dict)

    def pending_total(self) -> Decimal:
        return sum((amount for _, amount in self.pending), ZERO)

    def receivable_total(self) -> Decimal:
        return sum((amount for _, amount in self.receivable), ZERO)


def _quantize_cash(value: Decimal) -> Decimal:
    return value.quantize(CASH_QUANTUM, rounding=ROUND_HALF_EVEN)


def _micro_to_float(values: np.ndarray, present: np.ndarray) -> np.ndarray:
    out = values.astype(np.float64) / MICRO
    out[~present] = np.nan
    return out


class _Views:
    """Precomputed float arrays for O(lookback) context construction."""

    def __init__(self, dataset: Dataset, symbols: tuple[str, ...]) -> None:
        self.dataset = dataset
        self.adjusted: dict[str, dict[str, np.ndarray]] = {}
        self.cumulative: dict[str, np.ndarray] = {}
        self.volume: dict[str, np.ndarray] = {}
        self.present: dict[str, np.ndarray] = {}
        self.last_before: dict[str, np.ndarray] = {}  # [stop] -> last present index < stop, or -1
        self.sessions = np.array(dataset.sessions, dtype="datetime64[D]")
        for symbol in symbols:
            item = dataset.series[symbol]
            cumulative = dataset.cumulative_split(symbol)
            self.cumulative[symbol] = cumulative
            # raw × F(t): adjusted to the first session's share basis; divide by F(cutoff) at use.
            self.adjusted[symbol] = {
                name: _micro_to_float(item.micro[name], item.present) * cumulative for name in FIELDS
            }
            self.volume[symbol] = np.where(item.present, item.volume / cumulative, np.nan)
            self.present[symbol] = item.present
            running = np.maximum.accumulate(np.where(item.present, np.arange(len(item.present)), -1))
            self.last_before[symbol] = np.concatenate(([-1], running)) if len(running) else np.array([-1])

    def bars(self, symbol: str, start: int, stop: int) -> Bars:
        cutoff = self.cumulative[symbol][stop - 1]

        def view(array: np.ndarray) -> np.ndarray:
            window = array[start:stop] / cutoff
            window.setflags(write=False)
            return window

        sessions = self.sessions[start:stop].copy()
        sessions.setflags(write=False)
        volume = self.volume[symbol][start:stop] * cutoff
        volume.setflags(write=False)
        opens, highs, lows, closes = (view(self.adjusted[symbol][name]) for name in FIELDS)
        return Bars(symbol, sessions, opens, highs, lows, closes, volume)

    def last_present(self, symbol: str, stop: int) -> int | None:
        last = int(self.last_before[symbol][stop])
        return last if last >= 0 else None


def _close_on_basis(dataset: Dataset, views: _Views, symbol: str, stop: int, basis: int) -> Decimal | None:
    """Last close at an index below ``stop``, expressed in session ``basis``'s share basis."""
    last = views.last_present(symbol, stop)
    if last is None:
        return None
    raw = dataset.price(symbol, "close", last)
    assert raw is not None
    factor = Decimal(repr(float(views.cumulative[symbol][basis] / views.cumulative[symbol][last])))
    return raw / factor


def _mark_price(dataset: Dataset, views: _Views, symbol: str, index: int) -> Decimal | None:
    """Last completed close before session ``index``, in that session's share basis."""
    return _close_on_basis(dataset, views, symbol, index, index)


def _close_mark(dataset: Dataset, views: _Views, symbol: str, index: int) -> Decimal:
    """Close of session ``index`` (or the last close before it)."""
    price = _close_on_basis(dataset, views, symbol, index + 1, index)
    return price if price is not None else ZERO


def _state_or_raise(state: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
    if state is None:
        return current
    try:
        canonical = to_canonical(state)
    except (TypeError, ValueError) as exc:
        raise EngineError(f"STRATEGY_STATE_NOT_SERIALIZABLE:{exc}") from exc
    if len(canonical_json(canonical).encode("utf-8")) > MAX_STATE_BYTES:
        raise EngineError("STRATEGY_STATE_TOO_LARGE")
    return canonical


def run_backtest(
    spec: StrategySpecV1,
    definition: StrategyDef,
    params: Params,
    dataset: Dataset,
    *,
    start: date | None = None,
    end: date | None = None,
) -> BacktestResult:
    symbols = spec.data.symbols
    missing = [symbol for symbol in symbols if symbol not in dataset.series]
    if missing:
        raise EngineError(f"DATASET_SYMBOLS_MISSING:{','.join(missing)}")
    lookback = int(definition.lookback(params))
    if lookback < 1:
        raise EngineError("STRATEGY_LOOKBACK_INVALID")
    allowed_codes = set(spec.reason_codes) | set(REASON_CODES)
    views = _Views(dataset, symbols)
    sessions = dataset.sessions
    first = next((i for i, s in enumerate(sessions) if start is None or s >= start), len(sessions))
    last = max((i for i, s in enumerate(sessions) if end is None or s <= end), default=-1)
    if first > last:
        raise EngineError("BACKTEST_RANGE_EMPTY")
    first = max(first, 1)

    execution = spec.execution
    bps = execution.costs.bps / Decimal(10000)
    per_share = execution.costs.per_share
    fractional = execution.sizing == "fractional"
    account = _Account(settled=spec.account.initial_cash)
    strategy_state: dict[str, Any] = {}
    last_target: dict[str, Decimal] | None = None
    target_complete = True
    queue: dict[int, Decision] = {}
    result = BacktestResult([], [], [], [], [], {}, [], dataset.identity())
    dividends_by_ex: dict[date, list] = {}
    for dividend in dataset.dividends:
        dividends_by_ex.setdefault(dividend.ex_date, []).append(dividend)
    splits_by_ex: dict[date, list] = {}
    for split in dataset.splits:
        splits_by_ex.setdefault(split.ex_date, []).append(split)

    def positions_value(index: int, *, at_close: bool) -> Decimal:
        total = ZERO
        for symbol, quantity in account.quantity.items():
            if quantity:
                price = (
                    _close_mark(dataset, views, symbol, index)
                    if at_close
                    else (_mark_price(dataset, views, symbol, index) or ZERO)
                )
                total += quantity * price
        return total

    for i in range(first, last + 1):
        session = sessions[i]
        # 1. settlement and dividend payments due today
        account.settled += sum((a for idx, a in account.pending if idx <= i), ZERO)
        account.pending = [(idx, a) for idx, a in account.pending if idx > i]
        account.settled += sum((a for idx, a in account.receivable if idx <= i), ZERO)
        account.receivable = [(idx, a) for idx, a in account.receivable if idx > i]
        # 2. corporate actions effective today
        for split in splits_by_ex.get(session, ()):
            if split.symbol in account.quantity:
                account.quantity[split.symbol] *= split.ratio
        for dividend in dividends_by_ex.get(session, ()):
            held = account.quantity.get(dividend.symbol, ZERO)
            if held > 0:
                pay_index = next(
                    (k for k in range(i, len(sessions)) if sessions[k] >= dividend.pay_date),
                    len(sessions) - 1,
                )
                # Dividend amounts are per share on the ex-date basis, so a same-day split is applied first.
                account.receivable.append((pay_index, _quantize_cash(held * dividend.amount)))

        # 3. decide on completed bars through D-1
        decision = _decide(
            spec,
            definition,
            params,
            views,
            dataset,
            account.quantity,
            account.settled + account.pending_total() + account.receivable_total(),
            strategy_state,
            i,
            lookback,
            allowed_codes,
        )
        record: dict[str, Any] = {
            "session": session.isoformat(),
            "action": decision.action,
            "weights": dict(decision.weights),
            "reason_codes": list(decision.reason_codes),
        }
        if decision.action != "unavailable":
            strategy_state = _state_or_raise(decision.state, strategy_state)
        if decision.action == "target":
            changed = last_target is None or decision.weights != last_target
            if changed or not target_complete or execution.rebalance == "every_decision":
                queue[i + execution.execution_delay_sessions] = decision
                record["queued_for"] = sessions[
                    min(i + execution.execution_delay_sessions, len(sessions) - 1)
                ].isoformat()
            last_target = dict(decision.weights)
        result.decisions.append(record)

        # 5. execute at today's open
        pending_decision = queue.pop(i, None)
        if pending_decision is not None:
            target_complete = _execute(
                pending_decision,
                dataset,
                views,
                account,
                i,
                bps,
                per_share,
                fractional,
                spec,
                result,
                positions_value,
            )

        # 6. mark at the close
        value = positions_value(i, at_close=True)
        cash_total = account.settled + account.pending_total() + account.receivable_total()
        result.sessions.append(session)
        result.equity.append(_quantize_cash(cash_total + value))
        result.cash.append(_quantize_cash(account.settled))

    result.positions = {symbol: quantity for symbol, quantity in sorted(account.quantity.items()) if quantity}
    result.ledger_hash = canonical_hash(result.ledger_document())
    return result


def _decide(
    spec: StrategySpecV1,
    definition: StrategyDef,
    params: Params,
    views: _Views,
    dataset: Dataset,
    quantity: dict[str, Decimal],
    cash: Decimal,
    state: dict[str, Any],
    index: int,
    lookback: int,
    allowed_codes: set[str],
) -> Decision:
    if index < lookback:
        return Decision.unavailable("INSUFFICIENT_HISTORY")
    start = index - lookback
    bars: dict[str, Bars] = {}
    for symbol in spec.data.symbols:
        last = views.last_present(symbol, index)
        if last is None or last < start:
            return Decision.unavailable("INSUFFICIENT_HISTORY")
        if index - 1 - last > spec.data.max_staleness_sessions:
            return Decision.unavailable("STALE_OBSERVATIONS")
        bars[symbol] = views.bars(symbol, start, index)
    marks = {symbol: _mark_price(dataset, views, symbol, index) or ZERO for symbol in quantity}
    holdings = {symbol: amount for symbol, amount in quantity.items() if amount}
    invested = sum((holdings[s] * marks[s] for s in holdings), ZERO)
    equity = cash + invested
    weights = {
        s: (holdings[s] * marks[s] / equity).quantize(Decimal("0.000001")) if equity else ZERO
        for s in holdings
    }
    ctx = Ctx(
        decision_session=dataset.sessions[index],
        _bars=bars,
        positions=freeze_state(holdings),
        weights=freeze_state(weights),
        cash=_quantize_cash(cash),
        equity=_quantize_cash(equity),
        state=freeze_state(state),
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


@dataclass(frozen=True)
class PlannedOrder:
    symbol: str
    delta: Decimal  # signed share quantity: negative sells, positive buys
    mark: Decimal  # prior-close mark used for sizing, in the execution session's share basis


def plan_orders(
    weights: dict[str, Decimal],
    quantity: dict[str, Decimal],
    equity: Decimal,
    marks: dict[str, Decimal | None],
    *,
    fractional: bool,
    min_order_notional: Decimal,
    session: date,
    opens: dict[str, Decimal | None] | None = None,
) -> tuple[list[PlannedOrder], bool, list[str]]:
    """Size a target at prior-close marks. Shared by the simulator and the paper runner.

    Returns sells first then buys (each sorted by symbol), whether the plan covers the
    whole target, and warnings. ``opens`` is known only to the simulator; a missing
    open there makes the symbol unexecutable exactly like a missing mark.
    """
    complete, warnings, orders = True, [], []
    quantum = MICRO_QTY if fractional else Decimal(1)
    for symbol in sorted(set(quantity) | set(weights)):
        current = quantity.get(symbol, ZERO)
        weight = weights.get(symbol, ZERO)
        mark = marks.get(symbol)
        open_missing = opens is not None and opens.get(symbol) is None
        if open_missing or mark is None or mark <= 0:
            if weight != 0 or current != 0:
                warnings.append(f"PRICE_MISSING:{session.isoformat()}:{symbol}")
                complete = False
            continue
        desired = (equity * weight / mark).quantize(quantum, rounding=ROUND_DOWN) if weight > 0 else ZERO
        delta = desired - current
        if delta == 0 or abs(delta) * mark < min_order_notional:
            continue
        orders.append(PlannedOrder(symbol, delta, mark))
    return [o for o in orders if o.delta < 0] + [o for o in orders if o.delta > 0], complete, warnings


def affordable_quantity(
    available: Decimal, price: Decimal, bps: Decimal, per_share: Decimal, fractional: bool
) -> Decimal:
    """Largest quantity whose cost including fees (rounded to the cent) fits in ``available``."""
    quantum = MICRO_QTY if fractional else Decimal(1)
    unit_cost = price * (1 + bps) + per_share
    quantity = (available / unit_cost).quantize(quantum, rounding=ROUND_DOWN) if unit_cost > 0 else ZERO
    while (
        quantity > 0
        and _quantize_cash(quantity * price + _fee(quantity * price, quantity, bps, per_share)) > available
    ):
        quantity -= quantum  # fee rounding to the cent can exceed the estimate by a cent
    return max(quantity, ZERO)


def _execute(
    decision: Decision,
    dataset: Dataset,
    views: _Views,
    account: _Account,
    index: int,
    bps: Decimal,
    per_share: Decimal,
    fractional: bool,
    spec: StrategySpecV1,
    result: BacktestResult,
    positions_value: Any,
) -> bool:
    session = dataset.sessions[index]
    cash_total = account.settled + account.pending_total() + account.receivable_total()
    equity = cash_total + positions_value(index, at_close=False)
    symbols = sorted(set(account.quantity) | set(decision.weights))
    planned, complete, warnings = plan_orders(
        dict(decision.weights),
        account.quantity,
        equity,
        {symbol: _mark_price(dataset, views, symbol, index) for symbol in symbols},
        fractional=fractional,
        min_order_notional=spec.execution.min_order_notional,
        session=session,
        opens={symbol: dataset.price(symbol, "open", index) for symbol in symbols},
    )
    result.warnings.extend(warnings)
    orders = [(o.symbol, o.delta, dataset.price(o.symbol, "open", index)) for o in planned]
    settle_index = dataset.settlement_index(index)
    settle_session = dataset.sessions[settle_index] if settle_index < len(dataset.sessions) else None
    fill = spec.execution.fill
    quantum = MICRO_QTY if fractional else Decimal(1)
    for symbol, delta, price in orders:
        assert price is not None
        quantity = abs(delta)
        if fill is not None:
            volume = Decimal(str(float(dataset.series[symbol].volume[index])))
            cap = (volume * fill.max_volume_fraction).quantize(quantum, rounding=ROUND_DOWN)
            if quantity > cap:
                result.warnings.append(f"VOLUME_CAPPED:{session.isoformat()}:{symbol}")
                quantity, complete = cap, False
                if quantity <= 0:
                    continue
        if delta < 0:
            notional = quantity * price
            fee = (
                notional * bps + quantity * per_share + spec.execution.costs.sell_fees(notional, quantity)
            ).quantize(CENT, rounding=ROUND_HALF_EVEN)
            account.pending.append((settle_index, _quantize_cash(notional - fee)))
            account.quantity[symbol] = account.quantity.get(symbol, ZERO) - quantity
            result.fills.append(Fill(session, symbol, "sell", quantity, price, fee, settle_session))
            continue
        available = (
            account.settled if spec.account.model == "cash" else account.settled + account.pending_total()
        )
        affordable = affordable_quantity(available, price, bps, per_share, fractional)
        if affordable < quantity:
            result.warnings.append(f"INSUFFICIENT_SETTLED_CASH:{session.isoformat()}:{symbol}")
            quantity = affordable
            complete = False
        if quantity <= 0 or quantity * price < spec.execution.min_order_notional:
            continue
        notional = quantity * price
        fee = _fee(notional, quantity, bps, per_share)
        cost = _quantize_cash(notional + fee)
        if spec.account.model == "cash" or cost <= account.settled:
            account.settled -= cost
        else:  # margin model spends unsettled proceeds after settled cash; never borrows
            remaining = cost - account.settled
            account.settled = ZERO
            account.pending = _draw_pending(account.pending, remaining)
        account.quantity[symbol] = account.quantity.get(symbol, ZERO) + quantity
        result.fills.append(Fill(session, symbol, "buy", quantity, price, fee, None))
    return complete


def _fee(notional: Decimal, quantity: Decimal, bps: Decimal, per_share: Decimal) -> Decimal:
    return (notional * bps + quantity * per_share).quantize(CENT, rounding=ROUND_HALF_EVEN)


def _draw_pending(pending: list[tuple[int, Decimal]], amount: Decimal) -> list[tuple[int, Decimal]]:
    remaining, out = amount, []
    for index, value in sorted(pending):
        take = min(value, remaining)
        remaining -= take
        if value - take > 0:
            out.append((index, value - take))
    if remaining > 0:
        raise EngineError("MARGIN_MODEL_WOULD_BORROW")
    return out


@dataclass(frozen=True)
class PreOpenPlan:
    """What the paper runner submits for one session: the same decide and sizing as the simulator."""

    session: date
    decision: Decision
    record: dict[str, Any]
    state: dict[str, Any]
    last_target: dict[str, Decimal] | None
    orders: list[PlannedOrder]
    complete: bool
    warnings: list[str]
    equity: Decimal
    marks: dict[str, Decimal | None]


def plan_pre_open(
    spec: StrategySpecV1,
    definition: StrategyDef,
    params: Params,
    dataset: Dataset,
    *,
    quantity: dict[str, Decimal],
    cash: Decimal,
    state: dict[str, Any],
    last_target: dict[str, Decimal] | None,
    target_complete: bool,
) -> PreOpenPlan:
    """Decide and size the last session of ``dataset`` before its open.

    ``dataset`` ends with the session being traded; its bars there are absent. The
    context is built from bars through the previous session, exactly as in a backtest.
    ``cash`` is the account's total cash (settled and unsettled).
    """
    if spec.execution.execution_delay_sessions:
        raise EngineError("PAPER_EXECUTION_DELAY_UNSUPPORTED")
    symbols = spec.data.symbols
    missing = [symbol for symbol in symbols if symbol not in dataset.series]
    if missing:
        raise EngineError(f"DATASET_SYMBOLS_MISSING:{','.join(missing)}")
    lookback = int(definition.lookback(params))
    if lookback < 1:
        raise EngineError("STRATEGY_LOOKBACK_INVALID")
    views = _Views(dataset, tuple(sorted(set(symbols) | {s for s in quantity if s in dataset.series})))
    index = len(dataset.sessions) - 1
    session = dataset.sessions[index]
    decision = _decide(
        spec,
        definition,
        params,
        views,
        dataset,
        quantity,
        cash,
        state,
        index,
        lookback,
        set(spec.reason_codes) | set(REASON_CODES),
    )
    record: dict[str, Any] = {
        "session": session.isoformat(),
        "action": decision.action,
        "weights": dict(decision.weights),
        "reason_codes": list(decision.reason_codes),
    }
    new_state = state if decision.action == "unavailable" else _state_or_raise(decision.state, state)
    new_target = last_target
    held = sorted(symbol for symbol, amount in quantity.items() if amount)
    marks = {
        symbol: _mark_price(dataset, views, symbol, index) if symbol in views.present else None
        for symbol in sorted(set(held) | set(symbols))
    }
    equity = cash + sum((quantity[s] * (marks.get(s) or ZERO) for s in held), ZERO)
    orders: list[PlannedOrder] = []
    complete, warnings = target_complete, []
    if decision.action == "target":
        changed = last_target is None or decision.weights != last_target
        if changed or not target_complete or spec.execution.rebalance == "every_decision":
            record["queued_for"] = session.isoformat()
            orders, complete, warnings = plan_orders(
                dict(decision.weights),
                quantity,
                equity,
                marks,
                fractional=spec.execution.sizing == "fractional",
                min_order_notional=spec.execution.min_order_notional,
                session=session,
            )
        new_target = dict(decision.weights)
    return PreOpenPlan(
        session, decision, record, new_state, new_target, orders, complete, warnings, equity, marks
    )
