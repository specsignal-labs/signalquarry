# SPDX-License-Identifier: Apache-2.0
"""Low-evidence simulator for single-leg options strategies (the wheel family).

Per session ``D`` (after ``lookback`` completed sessions):

1. OPEN checkpoint (09:35 New York): spot = D's open; the strategy decides on bars
   through D-1, its wheel and its open leg (marked from a synthetic chain priced at
   the open). ``open`` resolves a selector with the shared resolver and sells at the
   bid floored to the tick, less the spread haircut; ``close`` buys back at the ask
   raised to the tick, plus the haircut. Opens need cash collateral for puts.
2. CLOSE checkpoint (15:55, or 12:55 on an NYSE 1pm early close — see
   ``signalquarry._internal.calendar.nyse``): the same with spot = D's close; opens
   after the entry cutoff are refused.
3. EXPIRY at D's close: a leg expiring on or before D is assigned when in the money by
   at least $0.01 (put: buy 100 shares at the strike; call: sell them), otherwise it
   expires. Early assignment is not modelled.
4. MARK: equity = cash + shares × close − the leg's ask × 100.

Everything about option prices is modelled, so results are graded
``low_evidence_options`` and can never support more than a ``walk_forward`` claim
until a paper forward test passes G5.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Any
from zoneinfo import ZoneInfo

from signalquarry._internal.canonical import canonical_hash
from signalquarry._internal.contracts.reason_codes import REASON_CODES
from signalquarry._internal.contracts.spec import StrategySpecV1
from signalquarry._internal.data.dataset import Dataset
from signalquarry._internal.engine.backtest import (
    BacktestResult,
    EngineError,
    Fill,
    _state_or_raise,
    _Views,
)
from signalquarry._internal.options.chains import (
    ChainModel,
    RecordedChain,
    realized_sigma,
    session_checkpoints,
)
from signalquarry._internal.options.contracts import OptionsError, Quote, quote_violations
from signalquarry._internal.options.resolver import (
    EntryPlan,
    QuoteRules,
    Selection,
    ceil_to_tick,
    resolve,
    target_strike,
)
from signalquarry._internal.options.wheel import Leg, Wheel, WheelState, transition
from signalquarry.sdk.context import freeze_state
from signalquarry.sdk.options import LegView, OptionsCtx, OptionsDecision, WheelView
from signalquarry.sdk.strategy import Params, StrategyDef

NEW_YORK = ZoneInfo("America/New_York")
CASH = Decimal("0.01")


@dataclass
class _Book:
    cash: Decimal
    wheels: dict[str, Wheel]


def _at(session: date, moment: time) -> datetime:
    return datetime.combine(session, moment, tzinfo=NEW_YORK).astimezone(UTC)


def _haircut(quote_half_spread: Decimal, spec_haircut: Decimal) -> Decimal:
    return (quote_half_spread * spec_haircut).quantize(Decimal("0.0001"))


def _leg_view(wheel: Wheel, quote: Quote | None, today: date) -> LegView | None:
    leg = wheel.leg
    if leg is None:
        return None
    cost = quote.ask if quote is not None else None
    captured = (
        (leg.entry_credit - cost) / leg.entry_credit if cost is not None and leg.entry_credit > 0 else None
    )
    return LegView(
        symbol=leg.contract.symbol,
        right=leg.contract.right,
        strike=leg.contract.strike,
        expiration=leg.contract.expiration,
        dte=leg.contract.dte(today),
        entry_credit=leg.entry_credit,
        close_cost=cost,
        captured_fraction=captured.quantize(Decimal("0.000001")) if captured is not None else None,
    )


def run_options_backtest(
    spec: StrategySpecV1,
    definition: StrategyDef,
    params: Params,
    dataset: Dataset,
    *,
    start: date | None = None,
    end: date | None = None,
    fee_multiplier: Decimal = Decimal(1),
    haircut_multiplier: Decimal = Decimal(1),
    recorded_chains: Mapping[tuple[str, date], Mapping[str, tuple[Decimal, Decimal]]] | None = None,
) -> BacktestResult:
    options = spec.options
    if options is None or definition.kind != "options":
        raise EngineError("STRATEGY_KIND_MISMATCH")
    missing = [s for s in spec.data.symbols if s not in dataset.series]
    if missing:
        raise EngineError(f"DATASET_SYMBOLS_MISSING:{','.join(missing)}")
    lookback = int(definition.lookback(params))
    if lookback < 1:
        raise EngineError("STRATEGY_LOOKBACK_INVALID")
    allowed = set(spec.reason_codes) | set(REASON_CODES)
    views = _Views(dataset, spec.data.symbols)
    sessions = dataset.sessions
    first = next((i for i, s in enumerate(sessions) if start is None or s >= start), len(sessions))
    last = max((i for i, s in enumerate(sessions) if end is None or s <= end), default=-1)
    if first > last:
        raise EngineError("BACKTEST_RANGE_EMPTY")
    first = max(first, lookback)
    fee = options.per_contract_fee * fee_multiplier
    haircut = options.spread_haircut * haircut_multiplier
    rules = QuoteRules(options.min_bid, options.max_relative_spread, options.max_quote_age_seconds)
    cutoff = time.fromisoformat(options.entry_cutoff)
    book = _Book(spec.account.initial_cash, {u: Wheel(u) for u in options.underlyings})
    state: dict[str, Any] = {}
    result = BacktestResult([], [], [], [], [], {}, [], dataset.identity())
    split_dates = {(s.symbol, s.ex_date) for s in dataset.splits}
    recorded_sessions: set[date] = set()

    for i in range(first, last + 1):
        session = sessions[i]
        for underlying in options.underlyings:
            wheel = book.wheels[underlying]
            if (underlying, session) in split_dates and (wheel.shares or wheel.leg):
                raise EngineError(f"CORPORATE_ACTION_UNSUPPORTED:{underlying} split while the wheel is open")
        chains: dict[str, ChainModel | RecordedChain] = {}  # the last checkpoint's chains mark the close
        for checkpoint, moment in session_checkpoints(session):
            field = "open" if checkpoint == "open" else "close"
            spots: dict[str, Decimal] = {}
            chains = {}
            at = _at(session, moment)
            for underlying in options.underlyings:
                price = dataset.price(underlying, field, i)
                if price is None:
                    continue
                spots[underlying] = price
                closes = views.bars(underlying, max(0, i - 25), i).close
                model = ChainModel(
                    underlying,
                    spot=price,
                    session=session,
                    at=at,
                    sigma=realized_sigma(closes),
                    tick=options.tick,
                )
                recorded = (recorded_chains or {}).get((underlying, session))
                if recorded:
                    chains[underlying] = RecordedChain(recorded, model)
                    recorded_sessions.add(session)
                else:
                    chains[underlying] = model
            decision = options_decide(
                spec,
                definition,
                params,
                views,
                index=i,
                lookback=lookback,
                at=at,
                today=session,
                wheels=book.wheels,
                leg_quotes={
                    u: chains[u].quote(w.leg.contract) if w.leg is not None and u in chains else None
                    for u, w in book.wheels.items()
                },
                spots=spots,
                cash=book.cash,
                state=state,
                allowed=allowed,
            )
            record: dict[str, Any] = {
                "session": session.isoformat(),
                "checkpoint": checkpoint,
                "action": decision.action,
                "reason_codes": list(decision.reason_codes),
            }
            if decision.action != "unavailable":
                state = _state_or_raise(decision.state, state)
            if decision.action in ("open", "close"):
                outcome = _act(
                    decision,
                    book,
                    chains,
                    spots,
                    session,
                    at,
                    moment,
                    cutoff,
                    rules,
                    options.tick,
                    fee,
                    haircut,
                    result,
                )
                record.update(outcome)
            result.decisions.append(record)
        _expire(book, dataset, i, session, fee, result)
        equity = book.cash
        for underlying, wheel in book.wheels.items():
            close = dataset.price(underlying, "close", i) or Decimal(0)
            equity += wheel.shares * close
            if wheel.leg is not None:
                chain = chains.get(underlying)
                quote = chain.quote(wheel.leg.contract) if chain is not None else None
                equity -= (quote.ask if quote else Decimal(0)) * 100
        result.sessions.append(session)
        result.equity.append(equity.quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN))
        result.cash.append(book.cash.quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN))
    for underlying, wheel in book.wheels.items():
        if wheel.shares:
            result.positions[underlying] = Decimal(wheel.shares)
        if wheel.leg is not None:
            result.positions[wheel.leg.contract.symbol] = Decimal(-1)
    result.ledger_hash = canonical_hash(result.ledger_document())
    if recorded_sessions:
        result.warnings.append(f"OPTIONS_RECORDED_CHAINS_USED:{len(recorded_sessions)}")
    return result


def options_decide(
    spec: StrategySpecV1,
    definition: StrategyDef,
    params: Params,
    views: _Views,
    *,
    index: int,
    lookback: int,
    at: datetime,
    today: date,
    wheels: dict[str, Wheel],
    leg_quotes: dict[str, Quote | None],
    spots: dict[str, Decimal],
    cash: Decimal,
    state: dict[str, Any],
    allowed: set[str],
) -> OptionsDecision:
    """Build the options context and call the strategy. Shared by the simulator and the paper runner."""
    from signalquarry.sdk.options import OD

    start = index - lookback
    bars = {}
    for symbol in spec.data.symbols:
        last = views.last_present(symbol, index)
        if last is None or last < start:
            return OD.unavailable("INSUFFICIENT_HISTORY")
        bars[symbol] = views.bars(symbol, start, index)
    wheel_views = {
        u: WheelView(u, w.state.value, w.shares, w.share_cost_basis)  # type: ignore[arg-type]
        for u, w in wheels.items()
    }
    legs = {
        u: view for u, w in wheels.items() if (view := _leg_view(w, leg_quotes.get(u), today)) is not None
    }
    equity = cash + sum(
        (
            w.shares * spots.get(u, Decimal(0))
            - ((legs[u].close_cost or Decimal(0)) * 100 if u in legs else Decimal(0))
            for u, w in wheels.items()
        ),
        Decimal(0),
    )
    ctx = OptionsCtx(
        decision_time=at,
        _bars=bars,
        _wheels=wheel_views,
        _legs=legs,
        spots=dict(spots),
        cash=cash.quantize(CASH),
        equity=equity.quantize(CASH),
        state=freeze_state(state),
    )
    decision = definition.decide(ctx, params)  # type: ignore[arg-type]
    if not isinstance(decision, OptionsDecision):
        raise EngineError(f"STRATEGY_RETURNED_NON_DECISION:{type(decision).__name__}")
    undeclared = [code for code in decision.reason_codes if code not in allowed]
    if undeclared:
        raise EngineError(f"REASON_CODE_UNDECLARED:{','.join(undeclared)}")
    if decision.underlying is not None and decision.underlying not in wheels:
        raise EngineError(f"DECISION_SYMBOL_NOT_DECLARED:{decision.underlying}")
    return decision


def _act(
    decision: OptionsDecision,
    book: _Book,
    chains: dict[str, ChainModel | RecordedChain],
    spots: dict[str, Decimal],
    session: date,
    at: datetime,
    moment: time,
    cutoff: time,
    rules: QuoteRules,
    tick: Decimal,
    fee: Decimal,
    haircut: Decimal,
    result: BacktestResult,
) -> dict[str, Any]:
    underlying = decision.underlying
    assert underlying is not None
    wheel = book.wheels[underlying]
    chain = chains.get(underlying)
    if decision.action == "open":
        selector = decision.selector
        assert selector is not None
        expected = {WheelState.FLAT: "PUT", WheelState.LONG_SHARES: "CALL"}.get(wheel.state)
        if expected != selector.right:
            return {"outcome": "refused", "why": f"OPTIONS_OPEN_NOT_ALLOWED_IN_{wheel.state.value.upper()}"}
        if moment >= cutoff:
            return {"outcome": "refused", "why": "OPTIONS_ENTRY_CUTOFF"}
        if underlying not in spots or chain is None:
            return {"outcome": "refused", "why": "PRICE_MISSING"}
        selection = Selection(selector.right, selector.min_dte, selector.max_dte, selector.strike.value)
        plan = resolve(
            selection,
            spot=spots[underlying],
            today=session,
            now=at,
            candidates=chain.candidates(
                selector.right,
                selector.min_dte,
                selector.max_dte,
                target_strike(selection, spots[underlying]),
            ),
            rules=rules,
            tick=tick,
        )
        if isinstance(plan, tuple):
            return {"outcome": "refused", "why": plan[0]}
        if plan.contract.right == "PUT" and book.cash < plan.collateral:
            return {"outcome": "refused", "why": "OPTIONS_INSUFFICIENT_COLLATERAL"}
        price = _entry_fill(plan, haircut, tick)
        book.cash += price * 100 - fee
        event = "put_opened" if plan.contract.right == "PUT" else "call_opened"
        book.wheels[underlying] = transition(
            wheel, event, leg=Leg(plan.contract, price, session, f"sim-{session.isoformat()}")
        )
        result.fills.append(Fill(session, plan.contract.symbol, "sell", Decimal(1), price, fee, None))
        return {"outcome": "filled", "contract": plan.contract.symbol, "price": price}
    if wheel.leg is None:
        return {"outcome": "refused", "why": "OPTIONS_NO_OPEN_LEG"}
    quote = chain.quote(wheel.leg.contract) if chain is not None else None
    if quote is None:
        return {"outcome": "refused", "why": "QUOTE_MISSING"}
    # The same quote rules as the paper runner (minimum bid does not apply to buying back).
    unusable = quote_violations(
        quote,
        now=at,
        max_age_seconds=rules.max_age_seconds,
        min_bid=Decimal(0),
        max_relative_spread=rules.max_relative_spread,
    )
    if unusable:
        return {"outcome": "refused", "why": unusable[0]}
    half = (quote.ask - quote.bid) / 2
    price = ceil_to_tick(quote.ask, tick) + _haircut(half, haircut)
    book.cash -= price * 100 + fee
    event = "put_closed" if wheel.leg.contract.right == "PUT" else "call_closed"
    symbol = wheel.leg.contract.symbol
    book.wheels[underlying] = transition(wheel, event)
    result.fills.append(Fill(session, symbol, "buy", Decimal(1), price, fee, None))
    return {"outcome": "filled", "contract": symbol, "price": price}


def _entry_fill(plan: EntryPlan, haircut: Decimal, tick: Decimal) -> Decimal:
    half = (plan.quote.ask - plan.quote.bid) / 2
    return max(tick, plan.limit_price - _haircut(half, haircut))


def _expire(
    book: _Book, dataset: Dataset, index: int, session: date, fee: Decimal, result: BacktestResult
) -> None:
    for underlying, wheel in list(book.wheels.items()):
        leg = wheel.leg
        if leg is None or leg.contract.expiration > session:
            continue
        close = dataset.price(underlying, "close", index)
        if close is None:
            raise EngineError(f"PRICE_MISSING:{session.isoformat()}:{underlying} at expiry")
        strike = leg.contract.strike
        in_the_money = (
            (strike - close) >= Decimal("0.01")
            if leg.contract.right == "PUT"
            else (close - strike) >= Decimal("0.01")
        )
        try:
            if not in_the_money:
                book.wheels[underlying] = transition(
                    wheel, "put_expired" if leg.contract.right == "PUT" else "call_expired"
                )
                result.decisions.append(
                    {
                        "session": session.isoformat(),
                        "checkpoint": "expiry",
                        "action": "expired",
                        "reason_codes": [],
                        "contract": leg.contract.symbol,
                    }
                )
                continue
            if leg.contract.right == "PUT":
                book.cash -= strike * 100
                book.wheels[underlying] = transition(wheel, "put_assigned")
                result.fills.append(Fill(session, underlying, "buy", Decimal(100), strike, Decimal(0), None))
            else:
                book.cash += strike * 100
                book.wheels[underlying] = transition(wheel, "call_assigned")
                result.fills.append(Fill(session, underlying, "sell", Decimal(100), strike, Decimal(0), None))
        except OptionsError as exc:
            raise EngineError(f"{exc.code}:{exc}") from exc
        result.decisions.append(
            {
                "session": session.isoformat(),
                "checkpoint": "expiry",
                "action": "assigned",
                "reason_codes": [],
                "contract": leg.contract.symbol,
            }
        )
    _ = fee, replace  # fees are charged on fills; assignment has none in this model
