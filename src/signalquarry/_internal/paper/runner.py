# SPDX-License-Identifier: Apache-2.0
"""The paper kernel: reconcile before decide, one pre-open submission per session, fail closed.

``run_once`` for session ``D`` (under the ``RunLease``):

1. Verify the journal chain, the submission switch, the account and the arm token.
2. Check the broker clock (skew guard) and that ``D`` is in its pre-open window.
3. Load the dataset ending with ``D`` (bars through ``D-1``).
4. Reconcile: finalize own orders from earlier runs, refuse unmanaged orders or
   positions, and halt on position drift that splits cannot explain.
5. Decide and size with the engine's own ``plan_pre_open`` (the backtest code path),
   then clip buys to settled cash at a buffered prior-close price.
6. Journal every order intent **before** submitting it. Client order ids are
   deterministic (``sq-<alias6>-<intent10>-<retry>``), so a crash or an ambiguous
   broker error is resolved by lookup, never by a blind resubmit.

A completed session is never re-run; an interrupted one resumes from its journaled plan.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

from signalquarry._internal.canonical import canonical_hash, to_canonical
from signalquarry._internal.contracts.paper import PaperDeploymentV1
from signalquarry._internal.contracts.spec import StrategySpecV1
from signalquarry._internal.data.dataset import Dataset
from signalquarry._internal.engine.backtest import PreOpenPlan, affordable_quantity, plan_pre_open
from signalquarry._internal.options.contracts import OptionsError, parse_occ
from signalquarry._internal.paper import arm
from signalquarry._internal.paper.isolate import plan_in_child
from signalquarry._internal.paper.journal import Journal
from signalquarry._internal.paper.lease import RunLease
from signalquarry._internal.paper.models import (
    TERMINAL_ORDER_STATUSES,
    BrokerAccount,
    BrokerClock,
    BrokerOrder,
    OrderRequest,
    PaperBroker,
    PaperError,
)
from signalquarry.sdk.strategy import Params, StrategyDef

EXCHANGE = ZoneInfo("America/New_York")
G5_MIN_SESSIONS = 20  # a framework minimum; projects cannot loosen it
ZERO = Decimal(0)
DatasetLoader = Callable[[date], Dataset]


@dataclass(frozen=True)
class Deployment:
    config: PaperDeploymentV1
    spec: StrategySpecV1
    definition: StrategyDef
    params: Params
    configuration_hash: str
    freeze_hash: str | None
    state_dir: Path

    @property
    def prefix(self) -> str:
        return f"sq-{self.config.alias[:6]}-"

    @property
    def journal_path(self) -> Path:
        return self.state_dir / "journal.jsonl"

    @property
    def arm_path(self) -> Path:
        return self.state_dir / "arm.json"

    @property
    def lease_path(self) -> Path:
        return self.state_dir / "lease"


@dataclass
class Outcome:
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)


def client_order_id(deployment: Deployment, session: date, symbol: str, side: str, quantity: Decimal) -> str:
    intent = canonical_hash(
        {
            "alias": deployment.config.alias,
            "configuration_hash": deployment.configuration_hash,
            "session": session,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
        }
    )
    return f"{deployment.prefix}{intent[7:17]}-0"


def _decimals(values: dict[str, Any]) -> dict[str, Decimal]:
    return {key: Decimal(str(value)) for key, value in values.items()}


class PaperKernel:
    def __init__(
        self,
        deployment: Deployment,
        broker: PaperBroker,
        load_dataset: DatasetLoader,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if getattr(broker, "paper_only", False) is not True:
            raise PaperError("BROKER_NOT_PAPER_ONLY", "invalid", getattr(broker, "name", "?"))
        if deployment.spec.kind != "equity_daily":
            raise PaperError(
                "PAPER_KIND_UNSUPPORTED", "invalid", "this runner trades daily equity strategies"
            )
        execution = deployment.spec.execution
        if execution.sizing != "whole_shares":
            raise PaperError(
                "PAPER_FRACTIONAL_UNSUPPORTED", "invalid", "market-on-open orders need whole shares"
            )
        if execution.execution_delay_sessions:
            raise PaperError("PAPER_EXECUTION_DELAY_UNSUPPORTED", "invalid")
        self.deployment = deployment
        self.broker = broker
        self.load_dataset = load_dataset
        self.now = now

    # -- shared checks --------------------------------------------------------
    def _check_account(self, account: BrokerAccount) -> None:
        expected = self.deployment.config.expected_account_id_sha256
        if expected is not None and arm.sha256_hex(account.account_id) != expected:
            raise PaperError(
                "PAPER_ACCOUNT_MISMATCH", "blocked", "account id hash differs from the deployment"
            )
        if account.trading_blocked or account.status.upper() != "ACTIVE":
            raise PaperError("PAPER_ACCOUNT_BLOCKED", "blocked", account.status)

    def _check_skew(self, clock: BrokerClock, now: datetime) -> None:
        skew = abs((clock.timestamp - now).total_seconds())
        if skew > self.deployment.config.guards.max_clock_skew_seconds:
            raise PaperError("PAPER_CLOCK_SKEW", "blocked", f"{skew:.1f}s")

    @staticmethod
    def _phase(clock: BrokerClock, now: datetime) -> tuple[Literal["pre_open", "open", "closed"], date]:
        today = now.astimezone(EXCHANGE).date()
        if clock.is_open:
            return "open", today
        if clock.next_open.astimezone(EXCHANGE).date() == today:
            return "pre_open", today
        return "closed", clock.next_open.astimezone(EXCHANGE).date()

    def _in_window(self, now: datetime) -> bool:
        window = self.deployment.config.window
        local = now.astimezone(EXCHANGE).time()
        return window.start <= local <= window.end

    def _load(self, session: date) -> Dataset:
        try:
            dataset = self.load_dataset(session)
        except PaperError:
            raise
        except Exception as exc:
            raise PaperError("PAPER_DATA_UNAVAILABLE", "unavailable", str(exc)) from exc
        if len(dataset.sessions) < 2 or dataset.sessions[-1] != session:
            raise PaperError("PAPER_DATA_STALE", "busy", f"dataset does not end at {session.isoformat()}")
        if not any(
            dataset.series[s].present[-2] for s in self.deployment.spec.data.symbols if s in dataset.series
        ):
            raise PaperError("PAPER_DATA_STALE", "busy", f"no bars for {dataset.sessions[-2].isoformat()}")
        return dataset

    # -- journal-derived state --------------------------------------------------
    @staticmethod
    def _completed(journal: Journal, session: date) -> dict[str, Any] | None:
        return next(
            (e for e in journal.of_kind("session_completed") if e["session"] == session.isoformat()), None
        )

    @staticmethod
    def _started(journal: Journal, session: date) -> dict[str, Any] | None:
        return next(
            (e for e in journal.of_kind("session_started") if e["session"] == session.isoformat()), None
        )

    @staticmethod
    def _memory(journal: Journal) -> tuple[dict[str, Any], dict[str, Decimal] | None, bool]:
        """Strategy state, last target and whether that target was fully executed."""
        started = journal.last("session_started")
        if started is None:
            return {}, None, True
        completed = next(
            (e for e in journal.of_kind("session_completed") if e["session"] == started["session"]), None
        )
        complete = bool(completed and completed["complete"])
        finals = {e["client_order_id"]: e for e in journal.of_kind("order_final")}
        for order in started["orders"]:
            final = finals.get(order["client_order_id"])
            if final is None or Decimal(final["filled_quantity"]) != Decimal(order["quantity"]):
                complete = False
        target = started["last_target"]
        return started["state"], _decimals(target) if target is not None else None, complete

    @staticmethod
    def _split_factor(dataset: Dataset, symbol: str, after: date, through: date) -> Decimal:
        factor = Decimal(1)
        for split in dataset.splits:
            if split.symbol == symbol and after < split.ex_date <= through:
                factor *= split.ratio
        return factor

    def expected_positions(
        self, journal: Journal, dataset: Dataset, session: date, *, through: date
    ) -> dict[str, Decimal]:
        armed = journal.last("armed")
        if armed is None:
            return {}
        base_session = date.fromisoformat(armed["baseline_session"])
        expected = {
            symbol: Decimal(q) * self._split_factor(dataset, symbol, base_session, through)
            for symbol, q in armed["baseline_positions"].items()
        }
        for entry in journal.entries[armed["seq"] :]:
            if entry["kind"] != "order_final" or Decimal(entry["filled_quantity"]) == 0:
                continue
            filled_on = date.fromisoformat(entry["session"])
            sign = 1 if entry["side"] == "buy" else -1
            quantity = Decimal(entry["filled_quantity"]) * self._split_factor(
                dataset, entry["symbol"], filled_on, through
            )
            expected[entry["symbol"]] = expected.get(entry["symbol"], ZERO) + sign * quantity
        return {symbol: quantity for symbol, quantity in sorted(expected.items()) if quantity}

    def _unsettled_proceeds(self, journal: Journal, dataset: Dataset) -> Decimal:
        index = {s: i for i, s in enumerate(dataset.sessions)}
        today = len(dataset.sessions) - 1
        total = ZERO
        for entry in journal.of_kind("order_final"):
            if entry["side"] != "sell" or Decimal(entry["filled_quantity"]) == 0:
                continue
            traded = index.get(date.fromisoformat(entry["session"]))
            if traded is not None and dataset.settlement_index(traded) > today:
                total += Decimal(entry["filled_quantity"]) * Decimal(entry["filled_average_price"])
        return total

    # -- reconcile ----------------------------------------------------------------
    def _final(
        self, journal: Journal, intent: dict[str, Any], order: BrokerOrder | None, status: str = "not_found"
    ) -> None:
        journal.append(
            "order_final",
            {
                "client_order_id": intent["client_order_id"],
                "order_id": order.order_id if order else None,
                "session": intent["session"],
                "symbol": intent["symbol"],
                "side": intent["side"],
                "quantity": intent["quantity"],
                "status": order.status if order else status,
                "filled_quantity": order.filled_quantity if order else ZERO,
                "filled_average_price": order.filled_average_price if order else None,
            },
            now=self.now(),
        )

    def reconcile_orders(self, journal: Journal, session: date) -> list[str]:
        """Finalize own orders from the journal; returns warnings. Refuses unmanaged open orders."""
        warnings: list[str] = []
        intents = {e["client_order_id"]: e for e in journal.of_kind("order_intent")}
        finals = {e["client_order_id"] for e in journal.of_kind("order_final")}
        submitted = {e["client_order_id"] for e in journal.of_kind("order_submitted")}
        foreign = [o for o in self.broker.open_orders() if o.client_order_id not in intents]
        if foreign:
            raise PaperError("PAPER_UNMANAGED_ORDERS", "blocked", ",".join(sorted(o.symbol for o in foreign)))
        for cid, intent in intents.items():
            if cid in finals:
                continue
            earlier = date.fromisoformat(intent["session"]) < session
            order = self.broker.order_by_client_id(cid)
            if order is None:
                if earlier:
                    self._final(journal, intent, None)
                    warnings.append(f"PAPER_ORDER_NOT_FOUND:{cid}")
                continue
            if cid not in submitted:
                journal.append(
                    "order_submitted",
                    {
                        "client_order_id": cid,
                        "order_id": order.order_id,
                        "status": order.status,
                        "recovered": True,
                    },
                    now=self.now(),
                )
            if order.status in TERMINAL_ORDER_STATUSES:
                self._final(journal, intent, order)
            elif earlier:
                self.broker.cancel(order.order_id)
                again = self.broker.order_by_client_id(cid)
                if again is None or again.status not in TERMINAL_ORDER_STATUSES:
                    raise PaperError("PAPER_CANCEL_UNCONFIRMED", "busy", cid)
                self._final(journal, intent, again)
                warnings.append(f"PAPER_STALE_ORDER_CANCELED:{cid}")
        return warnings

    def _halt(self, journal: Journal, code: str, detail: str) -> PaperError:
        journal.append("halted", {"reason_codes": [code], "detail": detail}, now=self.now())
        for order in self.broker.open_orders():
            if order.client_order_id.startswith(self.deployment.prefix):
                self.broker.cancel(order.order_id)
        return PaperError(code, "blocked", detail)

    def reconcile_positions(
        self, journal: Journal, dataset: Dataset, session: date, broker_positions: dict[str, Decimal]
    ) -> None:
        managed = set(self.deployment.spec.data.symbols)
        expected = self.expected_positions(journal, dataset, session, through=session)
        unmanaged = sorted(s for s in broker_positions if s not in managed and s not in expected)
        if unmanaged:
            raise PaperError("PAPER_UNMANAGED_POSITION", "blocked", ",".join(unmanaged))
        if broker_positions == expected:
            return
        if broker_positions == self.expected_positions(
            journal, dataset, session, through=dataset.sessions[-2]
        ):
            raise PaperError("PAPER_CORPORATE_ACTION_PENDING", "busy", "broker has not applied today's split")
        drift = {
            s: [str(expected.get(s, ZERO)), str(broker_positions.get(s, ZERO))]
            for s in sorted(set(expected) | set(broker_positions))
            if expected.get(s, ZERO) != broker_positions.get(s, ZERO)
        }
        raise self._halt(journal, "PAPER_POSITION_DRIFT", canonical_hash(drift) + " " + str(drift))

    # -- commands -----------------------------------------------------------------
    def _positions(self) -> dict[str, Decimal]:
        return {p.symbol: p.quantity for p in self.broker.positions() if p.quantity}

    def _size(
        self, plan: PreOpenPlan, account: BrokerAccount, settled: Decimal
    ) -> tuple[list[dict[str, Any]], bool, list[str]]:
        spec = self.deployment.spec
        buffer = 1 + self.deployment.config.guards.buy_price_buffer_bps / Decimal(10000)
        available = settled if spec.account.model == "cash" else account.cash
        complete, warnings, orders = plan.complete, list(plan.warnings), []
        for order in plan.orders:
            if order.delta < 0:
                orders.append({"symbol": order.symbol, "side": "sell", "quantity": -order.delta})
                if spec.account.model == "margin":
                    available += -order.delta * order.mark
                continue
            price = order.mark * buffer
            quantity = min(order.delta, affordable_quantity(available, price, ZERO, ZERO, False))
            if quantity < order.delta:
                warnings.append(f"INSUFFICIENT_SETTLED_CASH:{plan.session.isoformat()}:{order.symbol}")
                complete = False
            if quantity <= 0 or quantity * order.mark < spec.execution.min_order_notional:
                continue
            available -= quantity * price
            orders.append({"symbol": order.symbol, "side": "buy", "quantity": quantity})
        for item in orders:
            item["client_order_id"] = client_order_id(
                self.deployment, plan.session, item["symbol"], item["side"], item["quantity"]
            )
        return orders, complete, warnings

    def _plan(
        self, journal: Journal, dataset: Dataset, account: BrokerAccount, positions: dict[str, Decimal]
    ) -> tuple[PreOpenPlan, list[dict[str, Any]], bool, list[str], Decimal]:
        state, last_target, target_complete = self._memory(journal)
        planner = plan_in_child if self.deployment.config.isolation == "process" else plan_pre_open
        plan = planner(
            self.deployment.spec,
            self.deployment.definition,
            self.deployment.params,
            dataset,
            quantity=positions,
            cash=account.cash,
            state=state,
            last_target=last_target,
            target_complete=target_complete,
        )
        settled = account.cash - self._unsettled_proceeds(journal, dataset)
        orders, complete, warnings = self._size(plan, account, settled)
        return plan, orders, complete, warnings, settled

    def run_once(self) -> Outcome:
        deployment = self.deployment
        with RunLease(deployment.lease_path):
            journal = Journal.open(deployment.journal_path)
            if deployment.config.submission != "enabled":
                raise PaperError(
                    "PAPER_SUBMISSION_DISABLED", "disabled", "set submission: enabled in the paper config"
                )
            account = self.broker.account()
            self._check_account(account)
            now = self.now()
            token = arm.check(
                arm.read(deployment.arm_path),
                journal,
                alias=deployment.config.alias,
                configuration_hash=deployment.configuration_hash,
                freeze_hash=deployment.freeze_hash,
                account_id=account.account_id,
                now=now,
            )
            clock = self.broker.clock()
            self._check_skew(clock, now)
            phase, session = self._phase(clock, now)
            if phase == "closed":
                return Outcome(
                    f"no session today; next session {session.isoformat()}",
                    {"next_session": session.isoformat()},
                    reason_codes=["PAPER_NO_SESSION_TODAY"],
                )
            done = self._completed(journal, session)
            if done is not None:
                return Outcome(
                    f"session {session.isoformat()} already completed",
                    {"session": session.isoformat(), "journal_head": journal.head, "orders": done["orders"]},
                    reason_codes=["PAPER_SESSION_ALREADY_COMPLETED"],
                )
            if phase != "pre_open" or not self._in_window(now):
                raise PaperError(
                    "PAPER_OUTSIDE_WINDOW", "busy", now.astimezone(EXCHANGE).strftime("%H:%M %Z")
                )
            dataset = self._load(session)
            warnings = self.reconcile_orders(journal, session)
            positions = self._positions()
            self.reconcile_positions(journal, dataset, session, positions)
            started = self._started(journal, session)
            if started is None:
                previous = journal.last("session_completed")
                if previous is not None and Decimal(previous["equity"]) > 0:
                    drop = (Decimal(previous["equity"]) - account.equity) / Decimal(previous["equity"])
                    if drop > deployment.config.guards.max_daily_drawdown:
                        raise self._halt(
                            journal,
                            "PAPER_DRAWDOWN_HALT",
                            f"equity fell {drop:.4f} since {previous['session']}",
                        )
                try:
                    plan, orders, complete, size_warnings, settled = self._plan(
                        journal, dataset, account, positions
                    )
                except Exception as exc:  # strategy or contract failure: stop this deployment
                    raise self._halt(
                        journal, "PAPER_STRATEGY_FAILED", f"{type(exc).__name__}: {exc}"
                    ) from exc
                warnings += size_warnings
                started = journal.append(
                    "session_started",
                    {
                        "session": session,
                        "arm_token_hash": token["token_hash"],
                        "configuration_hash": deployment.configuration_hash,
                        "dataset_identity": dataset.identity(),
                        "decision": plan.record,
                        "state": plan.state,
                        "last_target": plan.last_target,
                        "equity": account.equity,
                        "cash": account.cash,
                        "settled_cash": settled,
                        "positions": positions,
                        "orders": orders,
                        "complete": complete,
                        "warnings": warnings,
                    },
                    now=self.now(),
                )
            complete = bool(started["complete"])
            intents = {e["client_order_id"] for e in journal.of_kind("order_intent")}
            submitted = {e["client_order_id"] for e in journal.of_kind("order_submitted")}
            finals = {e["client_order_id"] for e in journal.of_kind("order_final")}
            for item in started["orders"]:
                cid = item["client_order_id"]
                if cid in submitted or cid in finals:
                    continue
                if cid not in intents:
                    journal.append("order_intent", {"session": session, **item}, now=self.now())
                request = OrderRequest(cid, item["symbol"], item["side"], Decimal(item["quantity"]))
                try:
                    order = self.broker.submit(request)
                except PaperError as exc:
                    if exc.code == "BROKER_ORDER_REJECTED":
                        self._final(journal, {"session": session.isoformat(), **item}, None, "rejected")
                        warnings.append(f"BROKER_ORDER_REJECTED:{item['symbol']}")
                        complete = False
                        continue
                    found = self.broker.order_by_client_id(cid)
                    if found is None:
                        raise PaperError(
                            exc.code, "busy", f"{exc.detail}; rerun to resume {session.isoformat()}"
                        ) from exc
                    order = found
                journal.append(
                    "order_submitted",
                    {"client_order_id": cid, "order_id": order.order_id, "status": order.status},
                    now=self.now(),
                )
            orders = [item["client_order_id"] for item in started["orders"]]
            journal.append(
                "session_completed",
                {"session": session, "equity": account.equity, "complete": complete, "orders": orders},
                now=self.now(),
            )
            return Outcome(
                f"{session.isoformat()}: {started['decision']['action']} "
                f"({', '.join(started['decision']['reason_codes'])}); {len(orders)} order(s) submitted",
                {
                    "session": session.isoformat(),
                    "decision": started["decision"],
                    "orders": started["orders"],
                    "journal_head": journal.head,
                },
                warnings=warnings,
            )

    def arm(self, *, reason: str = "", days: int | None = None) -> Outcome:
        """Record a human's arm decision. The CLI obtains the confirmation; this binds the token."""
        deployment = self.deployment
        if deployment.freeze_hash is None:
            raise PaperError("FREEZE_REQUIRED", "blocked", "run `sqy spec freeze` for the strategy first")
        with RunLease(deployment.lease_path):
            journal = Journal.open(deployment.journal_path)
            account = self.broker.account()
            self._check_account(account)
            now = self.now()
            clock = self.broker.clock()
            self._check_skew(clock, now)
            today = now.astimezone(EXCHANGE).date()
            warnings = self.reconcile_orders(journal, today)
            if any(o.client_order_id.startswith(deployment.prefix) for o in self.broker.open_orders()):
                raise PaperError(
                    "PAPER_OPEN_ORDERS", "blocked", "own orders are still open; retry after they finish"
                )
            positions = self._positions()
            unmanaged = sorted(s for s in positions if s not in deployment.spec.data.symbols)
            if unmanaged:
                raise PaperError("PAPER_UNMANAGED_POSITION", "blocked", ",".join(unmanaged))
            token = arm.issue(
                alias=deployment.config.alias,
                strategy_id=deployment.spec.id,
                configuration_hash=deployment.configuration_hash,
                freeze_hash=deployment.freeze_hash,
                account_id=account.account_id,
                broker=self.broker.name,
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
                    "account_id_sha256": token["account_id_sha256"],
                    "configuration_hash": deployment.configuration_hash,
                    "freeze_hash": deployment.freeze_hash,
                    "baseline_session": today,
                    "baseline_positions": positions,
                    "reason": reason,
                },
                now=now,
            )
            return Outcome(
                f"{deployment.config.alias} armed until {token['expires_at']}",
                {
                    "alias": deployment.config.alias,
                    "expires_at": token["expires_at"],
                    "token_hash": token["token_hash"],
                    "baseline_positions": {k: str(v) for k, v in positions.items()},
                    "journal_head": journal.head,
                },
                warnings=warnings,
            )

    def halt(self, reason: str) -> Outcome:
        with RunLease(self.deployment.lease_path):
            journal = Journal.open(self.deployment.journal_path)
            before = {
                o.order_id
                for o in self.broker.open_orders()
                if o.client_order_id.startswith(self.deployment.prefix)
            }
            self._halt(journal, "PAPER_HALTED", reason)
            return Outcome(
                f"{self.deployment.config.alias} halted; {len(before)} open order(s) canceled; re-arm to resume",
                {"canceled_orders": sorted(before), "journal_head": journal.head},
            )

    def reconcile(self) -> Outcome:
        with RunLease(self.deployment.lease_path):
            journal = Journal.open(self.deployment.journal_path)
            now = self.now()
            clock = self.broker.clock()
            self._check_skew(clock, now)
            phase, session = self._phase(clock, now)
            today = now.astimezone(EXCHANGE).date() if phase != "closed" else session
            warnings = self.reconcile_orders(journal, today)
            positions = self._positions()
            if journal.last("armed") is not None:
                dataset = self._load(session if phase != "open" else today)
                self.reconcile_positions(journal, dataset, dataset.sessions[-1], positions)
            account = self.broker.account()
            entry = journal.append(
                "reconciled",
                {"positions": positions, "cash": account.cash, "equity": account.equity},
                now=now,
            )
            return Outcome(
                f"reconciled; positions match the journal ({len(positions)} held)",
                {"positions": {k: str(v) for k, v in positions.items()}, "journal_head": entry["hash"]},
                warnings=warnings,
            )

    def status(self) -> Outcome:
        journal = Journal.open(self.deployment.journal_path)
        token = arm.read(self.deployment.arm_path)
        state = journal.last("armed", "disarmed", "halted")
        completed = journal.last("session_completed")
        finals = {e["client_order_id"] for e in journal.of_kind("order_final")}
        pending = [
            e["client_order_id"]
            for e in journal.of_kind("order_intent")
            if e["client_order_id"] not in finals
        ]
        sessions = journal.of_kind("session_completed")
        return Outcome(
            f"{self.deployment.config.alias}: {state['kind'] if state else 'never armed'}; "
            f"{len(sessions)} session(s); head {str(journal.head)[:19]}",
            {
                "alias": self.deployment.config.alias,
                "state": state["kind"] if state else "unarmed",
                "halt_reason_codes": state.get("reason_codes")
                if state and state["kind"] == "halted"
                else None,
                "arm_expires_at": token.get("expires_at") if token else None,
                "submission": self.deployment.config.submission,
                "sessions_completed": len(sessions),
                "last_session": completed["session"] if completed else None,
                "pending_orders": pending,
                "journal_entries": len(journal.entries),
                "journal_head": journal.head,
            },
        )

    def dry_run(self) -> Outcome:
        """Decide and size the next session against the live paper account. Writes nothing."""
        journal = Journal.open(self.deployment.journal_path)
        account = self.broker.account()
        self._check_account(account)
        now = self.now()
        clock = self.broker.clock()
        self._check_skew(clock, now)
        phase, session = self._phase(clock, now)
        if phase == "open":
            session = clock.next_open.astimezone(EXCHANGE).date()
        dataset = self._load(session)
        positions = self._positions()
        unmanaged = sorted(s for s in positions if s not in self.deployment.spec.data.symbols)
        if unmanaged:
            raise PaperError("PAPER_UNMANAGED_POSITION", "blocked", ",".join(unmanaged))
        plan, orders, complete, warnings, settled = self._plan(journal, dataset, account, positions)
        finals = {e["client_order_id"] for e in journal.of_kind("order_final")}
        pending = [e for e in journal.of_kind("order_intent") if e["client_order_id"] not in finals]
        if pending:
            warnings.append(f"PAPER_PENDING_ORDERS:{len(pending)}")
        return Outcome(
            f"{session.isoformat()}: {plan.decision.action} ({', '.join(plan.decision.reason_codes)}); "
            f"{len(orders)} order(s) would be submitted",
            {
                "session": session.isoformat(),
                "decision": plan.record,
                "orders": [{**o, "quantity": str(o["quantity"])} for o in orders],
                "equity": str(account.equity),
                "settled_cash": str(settled),
                "complete": complete,
                "positions": {k: str(v) for k, v in positions.items()},
            },
            warnings=warnings,
        )

    def preflight(self) -> Outcome:
        """Read-only readiness checks; each check reports independently."""
        checks: list[dict[str, Any]] = []

        def check(name: str, action: Callable[[], str]) -> None:
            try:
                checks.append({"name": name, "ok": True, "detail": action()})
            except PaperError as exc:
                checks.append({"name": name, "ok": False, "code": exc.code, "detail": exc.detail})

        state: dict[str, Any] = {}

        def journal_ok() -> str:
            state["journal"] = Journal.open(self.deployment.journal_path)
            return f"{len(state['journal'].entries)} entries, chain intact"

        def account_ok() -> str:
            state["account"] = account = self.broker.account()
            self._check_account(account)
            return f"account active; equity {account.equity}"

        def clock_ok() -> str:
            now = self.now()
            clock = self.broker.clock()
            self._check_skew(clock, now)
            phase, session = self._phase(clock, now)
            state["session"] = session if phase != "open" else clock.next_open.astimezone(EXCHANGE).date()
            return f"{phase}; next session {state['session'].isoformat()}"

        def armed_ok() -> str:
            if "journal" not in state or "account" not in state:
                raise PaperError("PAPER_NOT_ARMED", "disabled", "journal or account unavailable")
            token = arm.check(
                arm.read(self.deployment.arm_path),
                state["journal"],
                alias=self.deployment.config.alias,
                configuration_hash=self.deployment.configuration_hash,
                freeze_hash=self.deployment.freeze_hash,
                account_id=state["account"].account_id,
                now=self.now(),
            )
            return f"armed until {token['expires_at']}"

        def submission_ok() -> str:
            if self.deployment.config.submission != "enabled":
                raise PaperError("PAPER_SUBMISSION_DISABLED", "disabled", "submission: disabled")
            return "enabled"

        def orders_ok() -> str:
            foreign = [
                o
                for o in self.broker.open_orders()
                if not o.client_order_id.startswith(self.deployment.prefix)
            ]
            if foreign:
                raise PaperError(
                    "PAPER_UNMANAGED_ORDERS", "blocked", ",".join(sorted(o.symbol for o in foreign))
                )
            return "no unmanaged open orders"

        def positions_ok() -> str:
            positions = self._positions()
            unmanaged = sorted(s for s in positions if s not in self.deployment.spec.data.symbols)
            if unmanaged:
                raise PaperError("PAPER_UNMANAGED_POSITION", "blocked", ",".join(unmanaged))
            return f"{len(positions)} managed position(s)"

        def data_ok() -> str:
            if "session" not in state:
                raise PaperError("PAPER_DATA_UNAVAILABLE", "unavailable", "session unknown")
            dataset = self._load(state["session"])
            return f"bars through {dataset.sessions[-2].isoformat()}"

        for name, action in (
            ("journal", journal_ok),
            ("account", account_ok),
            ("clock", clock_ok),
            ("armed", armed_ok),
            ("submission", submission_ok),
            ("orders", orders_ok),
            ("positions", positions_ok),
            ("data", data_ok),
        ):
            check(name, action)
        failed = [c["code"] for c in checks if not c["ok"]]
        return Outcome(
            "ready to run" if not failed else f"{len(failed)} check(s) not ready",
            {"checks": checks, "ready": not failed},
            reason_codes=failed,
        )

    def drift(self) -> Outcome:
        """Replay every journaled session against the engine; measure fills; apply model costs and dividends.

        Read-only. G5 passes with at least ``G5_MIN_SESSIONS`` clean sessions (decision parity, orders
        final, no halt) and mean adverse slippage within the deployment's tolerance.
        """
        journal = Journal.open(self.deployment.journal_path)
        spec = self.deployment.spec
        bps = spec.execution.costs.bps / Decimal(10000)
        per_share = spec.execution.costs.per_share
        halted_sessions = {e["at"][:10] for e in journal.of_kind("halted")}
        finals = {e["client_order_id"]: e for e in journal.of_kind("order_final")}
        sessions: list[dict[str, Any]] = []
        slippages: list[float] = []
        model_costs = ZERO
        dividends = ZERO
        previous_session: date | None = None
        for started in journal.of_kind("session_started"):
            session = date.fromisoformat(started["session"])
            row: dict[str, Any] = {"session": started["session"]}
            prior = Journal(journal.path, [e for e in journal.entries if e["seq"] < started["seq"]])
            try:
                dataset = self._load(session)
            except PaperError as exc:
                row.update(parity="unverifiable", reason=exc.code)
                sessions.append(row)
                previous_session = session
                continue
            if dataset.identity() != started["dataset_identity"]:
                row.update(parity="unverifiable", reason="PROVIDER_REVISION")
            else:
                state, last_target, complete = self._memory(prior)
                positions = _decimals(started["positions"])
                plan = plan_pre_open(
                    spec,
                    self.deployment.definition,
                    self.deployment.params,
                    dataset,
                    quantity=positions,
                    cash=Decimal(started["cash"]),
                    state=state,
                    last_target=last_target,
                    target_complete=complete,
                )
                account = BrokerAccount(
                    "replay",
                    "ACTIVE",
                    Decimal(started["cash"]),
                    Decimal(started["equity"]),
                    Decimal(started["cash"]),
                    False,
                )
                orders, _, _ = self._size(plan, account, Decimal(started["settled_cash"]))
                same_decision = to_canonical(plan.record) == started["decision"]
                same_orders = [(o["symbol"], o["side"], str(o["quantity"])) for o in orders] == [
                    (o["symbol"], o["side"], str(o["quantity"])) for o in started["orders"]
                ]
                row["parity"] = "ok" if same_decision and same_orders else "mismatch"
                if not same_decision:
                    row["expected_decision"] = to_canonical(plan.record)
                # Dividends Alpaca paper does not credit: ex-dates since the previous session on held shares.
                for dividend in dataset.dividends:
                    if (
                        previous_session is None or previous_session < dividend.ex_date
                    ) and dividend.ex_date <= session:
                        held = positions.get(dividend.symbol, ZERO)
                        dividends += held * dividend.amount
            orders_final = all(o["client_order_id"] in finals for o in started["orders"])
            fills = []
            for order in started["orders"]:
                final = finals.get(order["client_order_id"])
                if (
                    final is None
                    or Decimal(final["filled_quantity"]) == 0
                    or final["filled_average_price"] is None
                ):
                    continue
                price = Decimal(final["filled_average_price"])
                quantity = Decimal(final["filled_quantity"])
                model_costs += quantity * price * bps + quantity * per_share
                if order["side"] == "sell":
                    model_costs += spec.execution.costs.sell_fees(quantity * price, quantity)
                fills.append((order["symbol"], order["side"], price))
            if fills:
                after = self._session_open_prices(session)
                for symbol, side, price in fills:
                    opened = after.get(symbol)
                    if opened:
                        adverse = (
                            (float(price) / float(opened) - 1.0) * 10000.0 * (1 if side == "buy" else -1)
                        )
                        slippages.append(adverse)
            row["orders_final"] = orders_final
            row["halted"] = started["session"] in halted_sessions
            row["clean"] = row.get("parity") == "ok" and orders_final and not row["halted"]
            sessions.append(row)
            previous_session = session
        clean = sum(1 for row in sessions if row.get("clean"))
        mismatches = [row["session"] for row in sessions if row.get("parity") == "mismatch"]
        mean_slippage = sum(slippages) / len(slippages) if slippages else 0.0
        tolerance = float(self.deployment.config.guards.max_mean_slippage_bps)
        g5 = {
            "ok": clean >= G5_MIN_SESSIONS and not mismatches and mean_slippage <= tolerance,
            "clean_sessions": clean,
            "required_sessions": G5_MIN_SESSIONS,
            "parity_mismatches": len(mismatches),
            "mean_adverse_slippage_bps": round(mean_slippage, 3),
            "tolerance_bps": tolerance,
        }
        completed = journal.last("session_completed")
        broker_equity = Decimal(completed["equity"]) if completed else None
        shadow = {
            "model_costs": model_costs.quantize(Decimal("0.01")),
            "uncredited_dividends": dividends.quantize(Decimal("0.01")),
            "broker_equity": broker_equity,
            "shadow_equity": (broker_equity - model_costs + dividends).quantize(Decimal("0.01"))
            if broker_equity is not None
            else None,
        }
        codes = [] if not mismatches else ["PAPER_PARITY_MISMATCH"]
        return Outcome(
            f"{len(sessions)} session(s): {clean} clean, {len(mismatches)} parity mismatch(es); "
            f"mean adverse slippage {mean_slippage:.1f} bps; G5 {'passed' if g5['ok'] else 'not passed'}",
            {"sessions": sessions, "g5": g5, "shadow": shadow, "fills_measured": len(slippages)},
            reason_codes=codes,
        )

    def _session_open_prices(self, session: date) -> dict[str, Decimal]:
        """Raw opens of ``session`` from the dataset of the following session (empty if not yet available)."""
        try:
            probe = self.load_dataset(session)
            calendar_next = self._next_session(probe, session)
            if calendar_next is None:
                return {}
            dataset = self.load_dataset(calendar_next)
        except Exception:  # noqa: BLE001 - measurement is best-effort; missing data just skips the fill
            return {}
        if session not in dataset.sessions:
            return {}
        index = dataset.index_of(session)
        return {
            symbol: price
            for symbol in dataset.series
            if (price := dataset.price(symbol, "open", index)) is not None
        }

    def _next_session(self, dataset: Dataset, session: date) -> date | None:
        finder = getattr(self.broker, "next_session_after", None)
        found = finder(session) if callable(finder) else None
        return found if isinstance(found, date) else None

    def backup(self, out_dir: Path, *, now: datetime | None = None) -> Outcome:
        """Write a tar.gz of the journal and arm token with a manifest of their hashes."""
        import hashlib
        import io
        import tarfile

        journal = Journal.open(self.deployment.journal_path)
        stamp = (now or self.now()).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
        files = [p for p in (self.deployment.journal_path, self.deployment.arm_path) if p.is_file()]
        manifest = {
            "schema": "signalquarry.paper-backup/v1",
            "alias": self.deployment.config.alias,
            "created_at": stamp,
            "journal_head": journal.head,
            "journal_entries": len(journal.entries),
            "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"{self.deployment.config.alias}-{stamp}.tar.gz"
        with tarfile.open(target, "w:gz") as archive:
            for path in files:
                archive.add(path, arcname=path.name)
            data = json.dumps(manifest, indent=2, sort_keys=True).encode()
            info = tarfile.TarInfo("backup.json")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        return Outcome(
            f"backup of {self.deployment.config.alias} at {journal.head and journal.head[:19]}… ({len(journal.entries)} entries)",
            {"path": str(target), **manifest},
        )

    def verify_continuity(self, backup: Path) -> Outcome:
        """The live journal must extend the backed-up journal: same bytes, then only appends."""
        import tarfile

        try:
            with tarfile.open(backup, "r:gz") as archive:
                manifest = json.loads(archive.extractfile("backup.json").read())  # type: ignore[union-attr]
                saved = archive.extractfile("journal.jsonl")
                old = saved.read() if saved is not None else b""
        except (OSError, KeyError, tarfile.TarError, ValueError) as exc:
            raise PaperError("PAPER_BACKUP_INVALID", "invalid", str(exc)) from exc
        if manifest.get("alias") != self.deployment.config.alias:
            raise PaperError("PAPER_BACKUP_INVALID", "invalid", "backup belongs to another deployment")
        current = self.deployment.journal_path.read_bytes() if self.deployment.journal_path.is_file() else b""
        journal = Journal.open(self.deployment.journal_path)
        if not current.startswith(old):
            raise PaperError(
                "PAPER_CONTINUITY_BROKEN",
                "blocked",
                f"journal no longer extends the backup from {manifest.get('created_at')}",
            )
        return Outcome(
            f"journal extends the {manifest.get('created_at')} backup by {len(journal.entries) - manifest['journal_entries']} entries",
            {
                "backup_head": manifest["journal_head"],
                "current_head": journal.head,
                "appended": len(journal.entries) - manifest["journal_entries"],
            },
        )

    def snapshot(
        self, *, recent_fills: int = 20, previous_snapshot_hash: str | None = None
    ) -> dict[str, Any]:
        """A sanitized public-performance snapshot (format ``signalquarry-public-performance/v1``).

        Equity history is the journal's pre-open equity per session (the prior close);
        positions come from the broker; fills from the journal, without order identifiers.
        Figures that need cash-flow data are marked unavailable rather than estimated.
        """
        journal = Journal.open(self.deployment.journal_path)
        account = self.broker.account()
        now = self.now().astimezone(UTC)
        history: list[dict[str, Any]] = []
        peak = None
        for entry in journal.of_kind("session_started"):
            equity = Decimal(entry["equity"])
            peak = equity if peak is None else max(peak, equity)
            history.append(
                {
                    "timestamp": entry["session"],
                    "equity": str(equity),
                    "drawdown": str(
                        ((peak - equity) / peak).quantize(Decimal("0.000001")) if peak else Decimal(0)
                    ),
                }
            )
        first = Decimal(history[0]["equity"]) if history else None
        previous = Decimal(history[-1]["equity"]) if history else None
        armed = journal.last("armed")
        baseline = None
        if armed is not None:
            baseline_started = next(
                (e for e in journal.of_kind("session_started") if e["seq"] > armed["seq"]), None
            )
            baseline = Decimal(baseline_started["equity"]) if baseline_started else None

        def pnl(label: str, attribution: str, base: Decimal | None) -> dict[str, Any]:
            if base is None:
                return {
                    "status": "unavailable",
                    "amount": None,
                    "return_value": None,
                    "label": label,
                    "attribution": attribution,
                }
            amount = account.equity - base
            return {
                "status": "available",
                "amount": str(amount.quantize(Decimal("0.01"))),
                "return_value": str((amount / base).quantize(Decimal("0.000001"))) if base else None,
                "label": label,
                "attribution": attribution,
            }

        finals = [e for e in journal.of_kind("order_final") if Decimal(e["filled_quantity"]) > 0]
        managed = set(self.deployment.spec.data.symbols)
        options = getattr(self.deployment.spec, "options", None)
        underlyings = set(options.underlyings) if options is not None else set()

        def is_managed(symbol: str) -> bool:
            if symbol in managed:
                return True
            try:  # an option leg on a managed underlying (OCC symbol)
                return bool(underlyings) and parse_occ(symbol).underlying in underlyings
            except (ValueError, OptionsError):
                return False

        document: dict[str, Any] = {
            "schema_version": "signalquarry-public-performance/v1",
            "deployment_alias": self.deployment.config.alias,
            "evidence_mode": "paper",
            "captured_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "broker_observed_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "account_history_epoch": armed["baseline_session"] if armed else None,
            "account": {
                "equity": str(account.equity),
                "cash": str(account.cash),
                "buying_power": str(account.buying_power),
                "status": account.status,
            },
            "pnl": {
                "broker_reference": pnl(
                    "Equity versus the equity at the first armed session",
                    "account equity minus that baseline; not adjusted for external cash flows",
                    baseline,
                ),
                "day": pnl(
                    "Movement since the last recorded pre-open equity",
                    "current equity minus the latest journaled equity",
                    previous,
                ),
                "net_dollar_pnl": pnl(
                    "Net dollar P&L since the first journaled session",
                    "current equity minus the first journaled equity; not adjusted for cash flows",
                    first,
                ),
                "time_weighted_return": {
                    "status": "unavailable",
                    "amount": None,
                    "return_value": None,
                    "label": "Time-weighted return",
                    "attribution": "requires dated external cash flows, which this feed does not collect",
                },
            },
            "equity_history": history,
            "drawdown_sampling": "pre_open_daily",
            "positions": [
                {
                    "symbol": p.symbol,
                    "quantity": str(p.quantity),
                    "market_value": None,
                    "unrealized_pnl": None,
                }
                for p in self.broker.positions()
                if is_managed(p.symbol)
            ],
            "recent_fills": [
                {
                    "filled_at": e["session"],
                    "action": e["side"],
                    "instrument": e["symbol"],
                    "quantity": str(e["filled_quantity"]),
                    "average_fill_price": str(e["filled_average_price"]),
                }
                for e in finals[-recent_fills:]
            ],
            "external_cash_flows": [],
            "availability": {
                "equity_history": "available" if history else "unavailable",
                "positions": "available",
                "recent_fills": "available",
                "external_cash_flows": "unavailable",
                "realized_pnl": "unavailable",
                "source_feed": f"signalquarry-journal/{self.broker.name}",
            },
            "limitations": [
                "Paper trading simulates money and does not reproduce all live execution effects.",
                "Equity history is sampled from pre-open equity (the prior close), one point per session.",
                "External cash flows are not collected, so P&L figures are not cash-flow adjusted.",
            ],
        }
        document["previous_snapshot_hash"] = previous_snapshot_hash
        document["snapshot_hash"] = canonical_hash(document)
        return document
