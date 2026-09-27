# SPDX-License-Identifier: Apache-2.0
"""Paper runner for single-leg options strategies (the wheel family).

``poll`` runs during market hours every ``poll_seconds`` (under the ``RunLease``):

1. Verify the journal, the submission switch, the account, the arm token and the clock.
2. Record the session's opening equity once (drawdown guard, forward records).
3. Reconcile own orders: record fills (wheel transitions) and terminal states; cancel
   entries left unfilled after ``cancel_after_seconds`` (confirming the cancel).
4. Apply account activities — assignment (OPASN) and expiration (OPEXP) — to the wheel
   state machine. Exercise (OPEXC) of a short leg is unexpected and halts.
5. Compare broker positions with the wheels. Differences that a pending lifecycle event
   explains (an expired leg, until ``lifecycle_deadline``) are retried; anything else halts.
6. Decide once per underlying (no pending order), with the same context builder and
   resolver as the simulator; entry plans must satisfy the plan invariants. Limit orders
   are journaled before submission with deterministic client order ids.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol
from zoneinfo import ZoneInfo

from signalquarry._internal.canonical import canonical_hash
from signalquarry._internal.contracts.reason_codes import REASON_CODES
from signalquarry._internal.data.dataset import Dataset
from signalquarry._internal.engine.backtest import _state_or_raise, _Views
from signalquarry._internal.engine.options_sim import options_decide
from signalquarry._internal.options.contracts import Quote, parse_occ, quote_violations
from signalquarry._internal.options.resolver import (
    Candidate,
    QuoteRules,
    Selection,
    ceil_to_tick,
    plan_invariants,
    resolve,
    target_strike,
)
from signalquarry._internal.options.wheel import Leg, Wheel, transition
from signalquarry._internal.paper import arm
from signalquarry._internal.paper.journal import Journal
from signalquarry._internal.paper.lease import RunLease
from signalquarry._internal.paper.models import (
    TERMINAL_ORDER_STATUSES,
    Activity,
    BrokerAccount,
    BrokerClock,
    BrokerOrder,
    BrokerPosition,
    OrderRequest,
    PaperError,
)
from signalquarry._internal.paper.runner import Deployment, Outcome

NEW_YORK = ZoneInfo("America/New_York")


def lifecycle_deadline(expiration: date) -> datetime:
    """Assignment/expiration activities may arrive until the close of the first weekday after expiry."""
    following = expiration + timedelta(days=1)
    while following.weekday() >= 5:
        following += timedelta(days=1)
    return datetime.combine(following, time(16, 0), tzinfo=NEW_YORK)


class OptionsVenue(Protocol):
    paper_only: bool
    name: str

    def account(self) -> BrokerAccount: ...
    def clock(self) -> BrokerClock: ...
    def positions(self) -> list[BrokerPosition]: ...
    def open_orders(self) -> list[BrokerOrder]: ...
    def order_by_client_id(self, client_order_id: str) -> BrokerOrder | None: ...
    def submit(self, request: OrderRequest) -> BrokerOrder: ...
    def cancel(self, order_id: str) -> None: ...
    def activities(self, after: str | None) -> list[Activity]: ...
    def stock_quote(self, symbol: str) -> Quote | None: ...
    def option_quote(self, symbol: str) -> Quote | None: ...
    def option_candidates(
        self, underlying: str, right: str, min_dte: int, max_dte: int, target: Decimal
    ) -> list[Candidate]: ...


@dataclass(frozen=True)
class _Intent:
    entry: dict[str, Any]

    @property
    def cid(self) -> str:
        return self.entry["client_order_id"]


class OptionsKernel:
    if TYPE_CHECKING:  # attached by _attach() below, shared with the equity kernel's commands

        def arm(self, *, reason: str = "", days: int | None = None) -> Outcome: ...
        def status(self) -> Outcome: ...
        def halt(self, reason: str) -> Outcome: ...
        def run_once(self) -> Outcome: ...

    def __init__(
        self,
        deployment: Deployment,
        venue: OptionsVenue,
        load_dataset: Callable[[date], Dataset],
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if getattr(venue, "paper_only", False) is not True:
            raise PaperError("BROKER_NOT_PAPER_ONLY", "invalid", getattr(venue, "name", "?"))
        if deployment.spec.kind != "options_single_leg" or deployment.spec.options is None:
            raise PaperError(
                "PAPER_KIND_UNSUPPORTED", "invalid", "the options runner trades options_single_leg strategies"
            )
        self.deployment = deployment
        self.venue = venue
        self.load_dataset = load_dataset
        self.now = now
        self.options = deployment.spec.options
        self.rules = QuoteRules(
            self.options.min_bid, self.options.max_relative_spread, self.options.max_quote_age_seconds
        )

    # -- journal-derived state -------------------------------------------------------
    def wheels(self, journal: Journal) -> dict[str, Wheel]:
        wheels = {u: Wheel(u) for u in self.options.underlyings}
        for entry in journal.of_kind("wheel_event"):
            leg = None
            if entry.get("leg"):
                data = entry["leg"]
                leg = Leg(
                    parse_occ(data["symbol"]),
                    Decimal(data["entry_credit"]),
                    date.fromisoformat(data["opened"]),
                    data["client_order_id"],
                )
            wheels[entry["underlying"]] = transition(wheels[entry["underlying"]], entry["event"], leg=leg)
        return wheels

    @staticmethod
    def _state(journal: Journal) -> dict[str, Any]:
        last = journal.last("strategy_state")
        return dict(last["state"]) if last else {}

    @staticmethod
    def _pending(journal: Journal) -> list[_Intent]:
        finals = {e["client_order_id"] for e in journal.of_kind("order_final")}
        return [_Intent(e) for e in journal.of_kind("order_intent") if e["client_order_id"] not in finals]

    # -- steps -----------------------------------------------------------------------
    def _record_fill(
        self, journal: Journal, intent: _Intent, order: BrokerOrder, wheels: dict[str, Wheel], now: datetime
    ) -> None:
        journal.append(
            "order_final",
            {
                "client_order_id": intent.cid,
                "order_id": order.order_id,
                "session": intent.entry["session"],
                "symbol": order.symbol,
                "side": order.side,
                "quantity": intent.entry["quantity"],
                "status": order.status,
                "filled_quantity": order.filled_quantity,
                "filled_average_price": order.filled_average_price,
            },
            now=now,
        )
        if order.status != "filled" or order.filled_average_price is None:
            return
        underlying = intent.entry["underlying"]
        contract = parse_occ(order.symbol)
        if intent.entry["action"] == "open":
            event = "put_opened" if contract.right == "PUT" else "call_opened"
            leg = {
                "symbol": order.symbol,
                "entry_credit": order.filled_average_price,
                "opened": intent.entry["session"],
                "client_order_id": intent.cid,
            }
        else:
            event = "put_closed" if contract.right == "PUT" else "call_closed"
            leg = None
        wheels[underlying] = transition(
            wheels[underlying],
            event,
            leg=Leg(
                contract,
                Decimal(order.filled_average_price),
                date.fromisoformat(intent.entry["session"]),
                intent.cid,
            )
            if leg
            else None,
        )
        journal.append(
            "wheel_event",
            {"underlying": underlying, "event": event, "leg": leg, "source": f"order:{order.order_id}"},
            now=now,
        )

    def _reconcile_orders(self, journal: Journal, wheels: dict[str, Wheel], now: datetime) -> list[str]:
        warnings = []
        own = {i.cid for i in (_Intent(e) for e in journal.of_kind("order_intent"))}
        foreign = [o for o in self.venue.open_orders() if o.client_order_id not in own]
        if foreign:
            raise PaperError("PAPER_UNMANAGED_ORDERS", "blocked", ",".join(sorted(o.symbol for o in foreign)))
        for intent in self._pending(journal):
            order = self.venue.order_by_client_id(intent.cid)
            if order is None:
                if now - datetime.fromisoformat(intent.entry["at"].replace("Z", "+00:00")) > timedelta(
                    minutes=10
                ):
                    journal.append(
                        "order_final",
                        {
                            "client_order_id": intent.cid,
                            "order_id": None,
                            "session": intent.entry["session"],
                            "symbol": intent.entry["symbol"],
                            "side": intent.entry["side"],
                            "quantity": intent.entry["quantity"],
                            "status": "not_found",
                            "filled_quantity": "0",
                            "filled_average_price": None,
                        },
                        now=now,
                    )
                    warnings.append(f"PAPER_ORDER_NOT_FOUND:{intent.cid}")
                continue
            if order.status in TERMINAL_ORDER_STATUSES:
                self._record_fill(journal, intent, order, wheels, now)
                continue
            submitted = datetime.fromisoformat(intent.entry["at"].replace("Z", "+00:00"))
            if (now - submitted).total_seconds() > self.options.cancel_after_seconds:
                self.venue.cancel(order.order_id)
                again = self.venue.order_by_client_id(intent.cid)
                if again is None or again.status not in TERMINAL_ORDER_STATUSES:
                    raise PaperError("PAPER_CANCEL_UNCONFIRMED", "busy", intent.cid)
                self._record_fill(journal, intent, again, wheels, now)
                warnings.append(f"PAPER_STALE_ORDER_CANCELED:{intent.cid}")
        return warnings

    def _apply_activities(self, journal: Journal, wheels: dict[str, Wheel], now: datetime) -> None:
        seen = {
            e["source"].split(":", 1)[1]
            for e in journal.of_kind("wheel_event")
            if e["source"].startswith("activity:")
        }
        seen |= {e["activity_id"] for e in journal.of_kind("activity_ignored")}
        legs = {w.leg.contract.symbol: u for u, w in wheels.items() if w.leg is not None}
        for activity in self.venue.activities(None):
            if activity.activity_id in seen:
                continue
            underlying = legs.get(activity.symbol)
            if underlying is None:
                journal.append(
                    "activity_ignored",
                    {
                        "activity_id": activity.activity_id,
                        "activity_kind": activity.kind,
                        "symbol": activity.symbol,
                    },
                    now=now,
                )
                continue
            if activity.kind == "OPEXC":
                raise PaperError("PAPER_OPTIONS_UNEXPECTED_EXERCISE", "blocked", activity.symbol)
            right = wheels[underlying].leg.contract.right  # type: ignore[union-attr]
            event = ("put_" if right == "PUT" else "call_") + (
                "assigned" if activity.kind == "OPASN" else "expired"
            )
            wheels[underlying] = transition(wheels[underlying], event)
            journal.append(
                "wheel_event",
                {
                    "underlying": underlying,
                    "event": event,
                    "leg": None,
                    "source": f"activity:{activity.activity_id}",
                },
                now=now,
            )
            legs.pop(activity.symbol, None)

    def _check_positions(self, journal: Journal, wheels: dict[str, Wheel], now: datetime) -> None:
        actual = {p.symbol: p.quantity for p in self.venue.positions() if p.quantity}
        expected: dict[str, Decimal] = {}
        for underlying, wheel in wheels.items():
            if wheel.shares:
                expected[underlying] = Decimal(wheel.shares)
            if wheel.leg is not None:
                expected[wheel.leg.contract.symbol] = Decimal(-1)
        if actual == expected:
            return
        today = now.astimezone(NEW_YORK).date()
        lifecycle_pending = any(
            w.leg is not None
            and w.leg.contract.expiration <= today
            and now < lifecycle_deadline(w.leg.contract.expiration)
            for w in wheels.values()
        )
        if lifecycle_pending:
            raise PaperError(
                "PAPER_OPTIONS_LIFECYCLE_PENDING",
                "busy",
                "an expiring leg has no assignment or expiration activity yet",
            )
        unmanaged = sorted(set(actual) - set(expected))
        detail = {
            s: [str(expected.get(s, 0)), str(actual.get(s, 0))]
            for s in sorted(set(actual) | set(expected))
            if expected.get(s) != actual.get(s)
        }
        journal.append("halted", {"reason_codes": ["PAPER_POSITION_DRIFT"], "detail": str(detail)}, now=now)
        raise PaperError(
            "PAPER_UNMANAGED_POSITION"
            if unmanaged and not (set(expected) - set(actual))
            else "PAPER_POSITION_DRIFT",
            "blocked",
            str(detail),
        )

    def _client_id(self, journal: Journal, session: str, underlying: str, action: str, symbol: str) -> str:
        retry = sum(
            1
            for e in journal.of_kind("order_intent")
            if (e["session"], e["underlying"], e["action"], e["symbol"])
            == (session, underlying, action, symbol)
        )
        digest = canonical_hash(
            {
                "alias": self.deployment.config.alias,
                "configuration_hash": self.deployment.configuration_hash,
                "session": session,
                "underlying": underlying,
                "action": action,
                "symbol": symbol,
            }
        )
        return f"{self.deployment.prefix}{digest[7:17]}-{retry}"

    # -- the poll --------------------------------------------------------------------
    def poll(self, *, submit: bool = True) -> Outcome:
        deployment = self.deployment
        lease = RunLease(deployment.lease_path) if submit else None
        if lease is not None:
            lease.__enter__()
        try:
            return self._poll(submit)
        finally:
            if lease is not None:
                lease.__exit__(None, None, None)

    def _poll(self, submit: bool) -> Outcome:
        deployment = self.deployment
        journal = Journal.open(deployment.journal_path)
        now = self.now()
        if submit and deployment.config.submission != "enabled":
            raise PaperError(
                "PAPER_SUBMISSION_DISABLED", "disabled", "set submission: enabled in the paper config"
            )
        account = self.venue.account()
        if account.trading_blocked or account.status.upper() != "ACTIVE":
            raise PaperError("PAPER_ACCOUNT_BLOCKED", "blocked", account.status)
        if submit:
            arm.check(
                arm.read(deployment.arm_path),
                journal,
                alias=deployment.config.alias,
                configuration_hash=deployment.configuration_hash,
                freeze_hash=deployment.freeze_hash,
                account_id=account.account_id,
                now=now,
            )
        clock = self.venue.clock()
        skew = abs((clock.timestamp - now).total_seconds())
        if skew > deployment.config.guards.max_clock_skew_seconds:
            raise PaperError("PAPER_CLOCK_SKEW", "blocked", f"{skew:.1f}s")
        if not clock.is_open:
            return Outcome(
                "market closed; nothing to do",
                {"next_open": clock.next_open.isoformat()},
                reason_codes=["PAPER_MARKET_CLOSED"],
            )
        local = now.astimezone(NEW_YORK)
        session = local.date()
        wheels = self.wheels(journal)
        warnings: list[str] = []
        if submit:
            started = next(
                (e for e in journal.of_kind("session_started") if e["session"] == session.isoformat()), None
            )
            if started is None:
                previous = journal.last("session_started")
                if previous is not None and Decimal(previous["equity"]) > 0:
                    drop = (Decimal(previous["equity"]) - account.equity) / Decimal(previous["equity"])
                    if drop > deployment.config.guards.max_daily_drawdown:
                        journal.append(
                            "halted",
                            {"reason_codes": ["PAPER_DRAWDOWN_HALT"], "detail": f"{drop:.4f}"},
                            now=now,
                        )
                        raise PaperError("PAPER_DRAWDOWN_HALT", "blocked", f"equity fell {drop:.4f}")
                journal.append(
                    "session_started",
                    {"session": session, "equity": account.equity, "cash": account.cash, "runner": "options"},
                    now=now,
                )
            warnings += self._reconcile_orders(journal, wheels, now)
            self._apply_activities(journal, wheels, now)
        self._check_positions(journal, wheels, now)
        dataset = self.load_dataset(session)
        if dataset.sessions[-1] != session:
            raise PaperError("PAPER_DATA_STALE", "busy", f"dataset does not end at {session}")
        views = _Views(dataset, deployment.spec.data.symbols)
        index = len(dataset.sessions) - 1
        pending = {i.entry["underlying"] for i in self._pending(journal)}
        state = self._state(journal)
        spots: dict[str, Decimal] = {}
        for underlying in self.options.underlyings:
            quote = self.venue.stock_quote(underlying)
            reasons = quote_violations(
                quote,
                now=now,
                max_age_seconds=self.options.max_quote_age_seconds,
                min_bid=Decimal("0.01"),
                max_relative_spread=Decimal("0.05"),
            )
            if not reasons and quote is not None:
                spots[underlying] = quote.mid
            else:
                warnings.append(f"QUOTE_UNUSABLE:{underlying}:{','.join(reasons)}")
        leg_quotes = {
            u: self.venue.option_quote(w.leg.contract.symbol) if w.leg is not None else None
            for u, w in wheels.items()
        }
        decision = options_decide(
            deployment.spec,
            deployment.definition,
            deployment.params,
            views,
            index=index,
            lookback=int(deployment.definition.lookback(deployment.params)),
            at=now,
            today=session,
            wheels=wheels,
            leg_quotes=leg_quotes,
            spots=spots,
            cash=account.cash,
            state=state,
            allowed=set(deployment.spec.reason_codes) | set(REASON_CODES),
        )
        record = {
            "session": session.isoformat(),
            "at": local.strftime("%H:%M"),
            "action": decision.action,
            "reason_codes": list(decision.reason_codes),
        }
        if decision.action != "unavailable":
            new_state = _state_or_raise(decision.state, state)
            if submit and new_state != state:
                journal.append("strategy_state", {"state": new_state}, now=now)
        if decision.action not in ("open", "close"):
            return Outcome(
                f"{decision.action} ({', '.join(decision.reason_codes)})",
                {"decision": record},
                warnings=warnings,
            )
        underlying = decision.underlying
        assert underlying is not None
        if underlying in pending:
            return Outcome(
                f"waiting for the pending order on {underlying}",
                {"decision": record},
                warnings=warnings,
                reason_codes=["PAPER_ORDER_PENDING"],
            )
        order = self._plan(decision, wheels[underlying], spots, leg_quotes, now, session, account)
        if isinstance(order, str):
            return Outcome(
                f"{decision.action} refused: {order}",
                {"decision": record, "refused": order},
                warnings=warnings,
                reason_codes=[order],
            )
        request, entry = order
        record["order"] = {k: str(v) for k, v in entry.items()}
        if not submit:
            return Outcome(
                f"would {decision.action} {request.symbol} at {request.limit_price}",
                {"decision": record},
                warnings=warnings,
            )
        cid = self._client_id(journal, session.isoformat(), underlying, decision.action, request.symbol)
        request = OrderRequest(cid, request.symbol, request.side, Decimal(1), "day", request.limit_price)
        journal.append(
            "order_intent",
            {
                "session": session,
                "underlying": underlying,
                "action": decision.action,
                "client_order_id": cid,
                "symbol": request.symbol,
                "side": request.side,
                "quantity": Decimal(1),
                "limit_price": request.limit_price,
                "decision": record,
            },
            now=now,
        )
        try:
            placed = self.venue.submit(request)
        except PaperError as exc:
            if exc.code == "BROKER_ORDER_REJECTED":
                journal.append(
                    "order_final",
                    {
                        "client_order_id": cid,
                        "order_id": None,
                        "session": session,
                        "symbol": request.symbol,
                        "side": request.side,
                        "quantity": Decimal(1),
                        "status": "rejected",
                        "filled_quantity": Decimal(0),
                        "filled_average_price": None,
                    },
                    now=now,
                )
                return Outcome(
                    "order rejected",
                    {"decision": record},
                    warnings=[*warnings, f"BROKER_ORDER_REJECTED:{request.symbol}"],
                )
            found = self.venue.order_by_client_id(cid)
            if found is None:
                raise PaperError(exc.code, "busy", f"{exc.detail}; the next poll resolves it") from exc
            placed = found
        journal.append(
            "order_submitted",
            {"client_order_id": cid, "order_id": placed.order_id, "status": placed.status},
            now=now,
        )
        if placed.status in TERMINAL_ORDER_STATUSES:
            self._record_fill(journal, _Intent(journal.of_kind("order_intent")[-1]), placed, wheels, now)
        return Outcome(
            f"{decision.action} {request.symbol} limit {request.limit_price}: {placed.status}",
            {"decision": record, "journal_head": journal.head},
            warnings=warnings,
        )

    def _plan(
        self,
        decision: Any,
        wheel: Wheel,
        spots: dict[str, Decimal],
        leg_quotes: dict[str, Quote | None],
        now: datetime,
        session: date,
        account: BrokerAccount,
    ) -> tuple[OrderRequest, dict[str, Any]] | str:
        underlying = decision.underlying
        if decision.action == "close":
            if wheel.leg is None:
                return "OPTIONS_NO_OPEN_LEG"
            quote = leg_quotes.get(underlying)
            reasons = quote_violations(
                quote,
                now=now,
                max_age_seconds=self.options.max_quote_age_seconds,
                min_bid=Decimal(0),
                max_relative_spread=self.options.max_relative_spread,
            )
            if reasons or quote is None:
                return reasons[0] if reasons else "QUOTE_MISSING"
            limit = ceil_to_tick(quote.ask, self.options.tick)
            return OrderRequest("pending", wheel.leg.contract.symbol, "buy", Decimal(1), "day", limit), {
                "limit": limit,
                "ask": quote.ask,
            }
        selector = decision.selector
        expected = {"flat": "PUT", "long_shares": "CALL"}.get(wheel.state.value)
        if expected != selector.right:
            return f"OPTIONS_OPEN_NOT_ALLOWED_IN_{wheel.state.value.upper()}"
        if now.astimezone(NEW_YORK).time() >= time.fromisoformat(self.options.entry_cutoff):
            return "OPTIONS_ENTRY_CUTOFF"
        if underlying not in spots:
            return "PRICE_MISSING"
        selection = Selection(selector.right, selector.min_dte, selector.max_dte, selector.strike.value)
        spot = spots[underlying]
        candidates = self.venue.option_candidates(
            underlying, selector.right, selector.min_dte, selector.max_dte, target_strike(selection, spot)
        )
        plan = resolve(
            selection,
            spot=spot,
            today=session,
            now=now,
            candidates=candidates,
            rules=self.rules,
            tick=self.options.tick,
        )
        if isinstance(plan, tuple):
            return plan[0]
        problems = plan_invariants(plan, selection, today=session, tick=self.options.tick)
        if problems:
            raise PaperError("PAPER_OPTIONS_PLAN_INVALID", "blocked", ",".join(problems))
        if selector.right == "PUT" and account.cash < plan.collateral:
            return "OPTIONS_INSUFFICIENT_COLLATERAL"
        return OrderRequest("pending", plan.contract.symbol, "sell", Decimal(1), "day", plan.limit_price), {
            "limit": plan.limit_price,
            "bid": plan.quote.bid,
            "strike": plan.contract.strike,
            "expiration": plan.contract.expiration,
            "collateral": plan.collateral,
        }


def _arm(self: OptionsKernel, *, reason: str = "", days: int | None = None) -> Outcome:
    deployment = self.deployment
    if deployment.freeze_hash is None:
        raise PaperError("FREEZE_REQUIRED", "blocked", "run `sqy spec freeze` for the strategy first")
    with RunLease(deployment.lease_path):
        journal = Journal.open(deployment.journal_path)
        account = self.venue.account()
        now = self.now()
        clock = self.venue.clock()
        if abs((clock.timestamp - now).total_seconds()) > deployment.config.guards.max_clock_skew_seconds:
            raise PaperError("PAPER_CLOCK_SKEW", "blocked", "clock")
        # Bring the journal up to date first (fills, expirations, assignments since the last
        # poll), so arming the morning after an expiry sees the leg as settled.
        wheels = self.wheels(journal)
        self._reconcile_orders(journal, wheels, now)
        self._apply_activities(journal, wheels, now)
        if any(o.client_order_id.startswith(deployment.prefix) for o in self.venue.open_orders()):
            raise PaperError("PAPER_OPEN_ORDERS", "blocked", "own orders are still open")
        self._check_positions(journal, self.wheels(journal), now)
        token = arm.issue(
            alias=deployment.config.alias,
            strategy_id=deployment.spec.id,
            configuration_hash=deployment.configuration_hash,
            freeze_hash=deployment.freeze_hash,
            account_id=account.account_id,
            broker=self.venue.name,
            journal_head=journal.head,
            now=now,
            days=days or deployment.config.arm_days,
        )
        arm.write(deployment.arm_path, token)
        journal.append(
            "armed",
            {
                "token_hash": token["token_hash"],
                "expires_at": token["expires_at"],
                "baseline_session": now.astimezone(NEW_YORK).date(),
                "baseline_positions": {},
                "reason": reason,
            },
            now=now,
        )
        return Outcome(
            f"{deployment.config.alias} armed until {token['expires_at']}",
            {"expires_at": token["expires_at"], "journal_head": journal.head},
        )


def _status(self: OptionsKernel) -> Outcome:
    journal = Journal.open(self.deployment.journal_path)
    state = journal.last("armed", "disarmed", "halted")
    wheels = self.wheels(journal)
    return Outcome(
        f"{self.deployment.config.alias}: {state['kind'] if state else 'never armed'}; "
        + ", ".join(f"{u} {w.state.value}" for u, w in wheels.items()),
        {
            "alias": self.deployment.config.alias,
            "state": state["kind"] if state else "unarmed",
            "wheels": {
                u: {
                    "state": w.state.value,
                    "shares": w.shares,
                    "leg": w.leg.contract.symbol if w.leg else None,
                }
                for u, w in wheels.items()
            },
            "pending_orders": [i.cid for i in self._pending(journal)],
            "journal_entries": len(journal.entries),
            "journal_head": journal.head,
        },
    )


def _halt(self: OptionsKernel, reason: str) -> Outcome:
    with RunLease(self.deployment.lease_path):
        journal = Journal.open(self.deployment.journal_path)
        canceled = []
        for order in self.venue.open_orders():
            if order.client_order_id.startswith(self.deployment.prefix):
                self.venue.cancel(order.order_id)
                canceled.append(order.order_id)
        journal.append("halted", {"reason_codes": ["PAPER_HALTED"], "detail": reason}, now=self.now())
        return Outcome(
            f"{self.deployment.config.alias} halted; {len(canceled)} order(s) canceled",
            {"canceled_orders": canceled},
        )


def _preflight(self: OptionsKernel) -> Outcome:
    checks = []

    def check(name: str, action: Callable[[], str]) -> None:
        try:
            checks.append({"name": name, "ok": True, "detail": action()})
        except PaperError as exc:
            checks.append({"name": name, "ok": False, "code": exc.code, "detail": exc.detail})

    journal = Journal.open(self.deployment.journal_path)
    account = self.venue.account()
    check("account", lambda: f"equity {account.equity}")
    check(
        "armed",
        lambda: (
            "armed until "
            + arm.check(
                arm.read(self.deployment.arm_path),
                journal,
                alias=self.deployment.config.alias,
                configuration_hash=self.deployment.configuration_hash,
                freeze_hash=self.deployment.freeze_hash,
                account_id=account.account_id,
                now=self.now(),
            )["expires_at"]
        ),
    )

    def submission() -> str:
        if self.deployment.config.submission != "enabled":
            raise PaperError("PAPER_SUBMISSION_DISABLED", "disabled", "submission: disabled")
        return "enabled"

    def positions() -> str:
        self._check_positions(journal, self.wheels(journal), self.now())
        return "positions match the wheels"

    check("submission", submission)
    check("positions", positions)
    failed = [c["code"] for c in checks if not c["ok"]]
    return Outcome(
        "ready to run" if not failed else f"{len(failed)} check(s) not ready",
        {"checks": checks, "ready": not failed},
        reason_codes=failed,
    )


def _unsupported(self: OptionsKernel, *args: Any, **kwargs: Any) -> Outcome:
    raise PaperError(
        "PAPER_KIND_UNSUPPORTED",
        "invalid",
        "not available for options deployments (poll reconciles on every run)",
    )


def _drift(self: OptionsKernel) -> Outcome:
    """G5 for options, from the journal alone (read-only).

    Option quotes are not journaled, so decisions cannot be replayed as for equities.
    A session is clean when it was not halted and every order it placed reached a final
    state; plan invariants were enforced before each order was journaled. G5 passes with
    at least ``G5_MIN_SESSIONS`` clean sessions and at least one expiration or assignment
    applied from broker activities (the lifecycle has been exercised for real).
    """
    from signalquarry._internal.paper.runner import G5_MIN_SESSIONS

    journal = Journal.open(self.deployment.journal_path)
    halted_sessions = {e["at"][:10] for e in journal.of_kind("halted")}
    finals = {e["client_order_id"]: e for e in journal.of_kind("order_final")}
    intents: dict[str, list[dict[str, Any]]] = {}
    for intent in journal.of_kind("order_intent"):
        intents.setdefault(str(intent["session"]), []).append(intent)
    fee = self.options.per_contract_fee
    sessions: list[dict[str, Any]] = []
    model_costs = Decimal(0)
    for started in journal.of_kind("session_started"):
        placed = intents.get(started["session"], [])
        orders_final = all(i["client_order_id"] in finals for i in placed)
        filled = [
            finals[i["client_order_id"]]
            for i in placed
            if i["client_order_id"] in finals
            and Decimal(str(finals[i["client_order_id"]]["filled_quantity"])) > 0
        ]
        model_costs += fee * sum(Decimal(str(f["filled_quantity"])) for f in filled)
        halted = started["session"] in halted_sessions
        sessions.append(
            {
                "session": started["session"],
                "orders": len(placed),
                "fills": len(filled),
                "orders_final": orders_final,
                "halted": halted,
                "clean": orders_final and not halted,
            }
        )
    lifecycle = [
        e["event"]
        for e in journal.of_kind("wheel_event")
        if str(e.get("source", "")).startswith("activity:") and e["event"].endswith(("_expired", "_assigned"))
    ]
    clean = sum(1 for row in sessions if row["clean"])
    g5 = {
        "ok": clean >= G5_MIN_SESSIONS and bool(lifecycle),
        "clean_sessions": clean,
        "required_sessions": G5_MIN_SESSIONS,
        "lifecycle_events": len(lifecycle),
        "decision_replay": "not_available_for_options",
    }
    completed = journal.last("session_completed") or journal.last("session_started")
    broker_equity = Decimal(str(completed["equity"])) if completed else None
    shadow = {
        "model_costs": model_costs.quantize(Decimal("0.01")),
        "broker_equity": broker_equity,
        "shadow_equity": (broker_equity - model_costs).quantize(Decimal("0.01"))
        if broker_equity is not None
        else None,
    }
    return Outcome(
        f"{len(sessions)} session(s): {clean} clean, {len(lifecycle)} expiration/assignment event(s); "
        f"G5 {'passed' if g5['ok'] else 'not passed'}",
        {"sessions": sessions, "g5": g5, "shadow": shadow, "fills_measured": 0},
    )


def _attach() -> None:
    from signalquarry._internal.paper.runner import PaperKernel

    OptionsKernel.arm = _arm  # type: ignore[attr-defined]
    OptionsKernel.status = _status  # type: ignore[attr-defined]
    OptionsKernel.halt = _halt  # type: ignore[attr-defined]
    OptionsKernel.backup = PaperKernel.backup  # type: ignore[attr-defined]
    OptionsKernel.verify_continuity = PaperKernel.verify_continuity  # type: ignore[attr-defined]
    OptionsKernel.snapshot = PaperKernel.snapshot  # type: ignore[attr-defined]
    OptionsKernel.broker = property(lambda self: self.venue)  # type: ignore[attr-defined]  # snapshot reads self.broker
    OptionsKernel.run_once = lambda self: self.poll()  # type: ignore[attr-defined]
    OptionsKernel.dry_run = lambda self: self.poll(submit=False)  # type: ignore[attr-defined]
    OptionsKernel.preflight = _preflight  # type: ignore[attr-defined]
    OptionsKernel.drift = _drift  # type: ignore[attr-defined]
    for name in ("reconcile",):
        setattr(OptionsKernel, name, _unsupported)


_attach()

__all__ = ["OptionsKernel", "OptionsVenue", "lifecycle_deadline"]
