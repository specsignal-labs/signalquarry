# SPDX-License-Identifier: Apache-2.0
"""Kernel commands and guard branches not covered by the fault tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import time
from decimal import Decimal
from pathlib import Path

import pytest

from signalquarry._internal.data.dataset import truncated
from signalquarry._internal.paper.arm import sha256_hex
from signalquarry._internal.paper.journal import Journal
from signalquarry._internal.paper.models import PaperError
from signalquarry._internal.paper.runner import PaperKernel
from signalquarry.sdk import Ctx, Decision, strategy
from tests.paper.harness import Momentum, hold_syna, paper_spec, rig


def _code(action) -> tuple[str, str]:
    with pytest.raises(PaperError) as info:
        action()
    return info.value.code, info.value.status


def _armed(tmp_path: Path, **kwargs):
    paper = rig(tmp_path, decide=hold_syna, **kwargs)
    sessions = list(paper.dataset.sessions[60:70])
    paper.arm(sessions[0])
    return paper, sessions


def test_reconcile_halt_and_status_commands(tmp_path: Path) -> None:
    paper, sessions = _armed(tmp_path)
    paper.session(sessions[0])
    paper.at(sessions[1], time(9, 1))
    reconciled = paper.kernel.reconcile()
    assert reconciled.data["positions"] == {"SYNA": str(paper.broker.positions_["SYNA"])}
    paper.at(sessions[1])
    paper.kernel.run_once()  # opg orders are open now
    halted = paper.kernel.halt("manual stop for review")
    assert isinstance(halted.data["canceled_orders"], list)
    status = paper.kernel.status()
    assert status.data["state"] == "halted" and status.data["halt_reason_codes"] == ["PAPER_HALTED"]
    assert status.data["sessions_completed"] == 2
    assert _code(paper.kernel.run_once) == ("PAPER_HALTED", "disabled")


def test_preflight_ready_when_armed_and_enabled(tmp_path: Path) -> None:
    paper, sessions = _armed(tmp_path)
    paper.at(sessions[0])
    outcome = paper.kernel.preflight()
    assert outcome.data["ready"] is True and outcome.reason_codes == []


def test_preflight_reports_unmanaged_orders_and_positions(tmp_path: Path) -> None:
    from signalquarry._internal.paper.brokers.fake import FakeBroker
    from signalquarry._internal.paper.models import OrderRequest

    paper, sessions = _armed(tmp_path)
    paper.at(sessions[0])
    paper.broker.positions_["ZZZ"] = Decimal(1)
    FakeBroker.submit(paper.broker, OrderRequest("manual", "SYNA", "buy", Decimal(1)))
    codes = paper.kernel.preflight().reason_codes
    assert {"PAPER_UNMANAGED_ORDERS", "PAPER_UNMANAGED_POSITION"} <= set(codes)
    assert _code(paper.kernel.dry_run)[0] == "PAPER_UNMANAGED_POSITION"


def test_pending_split_is_retried_not_halted(tmp_path: Path) -> None:
    @strategy(params=Momentum, lookback=lambda p: p.lookback)
    def hold_synb(ctx: Ctx, p: Momentum) -> Decision:
        return Decision.target({"SYNB": p.weight}, "GO")

    paper = rig(tmp_path, decide=hold_synb)
    split = next(s for s in paper.dataset.splits)
    index = paper.dataset.index_of(split.ex_date)
    paper.arm(paper.dataset.sessions[index - 2])
    paper.session(paper.dataset.sessions[index - 2])
    paper.session(paper.dataset.sessions[index - 1])
    held = paper.broker.positions_["SYNB"]
    paper.at(split.ex_date)  # the fake broker applies the split here...
    paper.broker.positions_["SYNB"] = held  # ...but a real broker may lag: undo it
    assert _code(paper.kernel.run_once) == ("PAPER_CORPORATE_ACTION_PENDING", "busy")
    paper.broker.positions_["SYNB"] = held * split.ratio  # applied later in the window
    paper.at(split.ex_date, time(9, 15))
    assert paper.kernel.run_once().data["session"] == split.ex_date.isoformat()


def test_data_problems_are_retryable(tmp_path: Path) -> None:
    paper, sessions = _armed(tmp_path)
    paper.at(sessions[0])
    stale = PaperKernel(
        paper.deployment,
        paper.broker,
        lambda s: truncated(paper.dataset, paper.dataset.index_of(s)),
        now=lambda: paper.broker.now,
    )
    assert _code(stale.run_once) == ("PAPER_DATA_STALE", "busy")

    def broken(session):
        raise RuntimeError("provider exploded")

    failing = PaperKernel(paper.deployment, paper.broker, broken, now=lambda: paper.broker.now)
    assert _code(failing.run_once) == ("PAPER_DATA_UNAVAILABLE", "unavailable")


def test_strategy_failure_halts(tmp_path: Path) -> None:
    @strategy(params=Momentum, lookback=lambda p: p.lookback)
    def explode(ctx: Ctx, p: Momentum) -> Decision:
        raise ZeroDivisionError("bug in strategy")

    paper = rig(tmp_path, decide=explode)
    session = paper.dataset.sessions[60]
    paper.arm(session)
    paper.at(session)
    assert _code(paper.kernel.run_once) == ("PAPER_STRATEGY_FAILED", "blocked")
    assert Journal.open(paper.deployment.journal_path).last("halted")["reason_codes"] == [
        "PAPER_STRATEGY_FAILED"
    ]


def test_margin_sizing_and_cash_clipping(tmp_path: Path) -> None:
    @strategy(params=Momentum, lookback=lambda p: p.lookback)
    def all_in(ctx: Ctx, p: Momentum) -> Decision:
        return Decision.target({"SYNA": "1"}, "GO")

    clipped = rig(tmp_path / "cash", decide=all_in, config={"guards": {"buy_price_buffer_bps": "200"}})
    session = clipped.dataset.sessions[60]
    clipped.arm(session)
    clipped.at(session)
    outcome = clipped.kernel.run_once()
    assert any(w.startswith("INSUFFICIENT_SETTLED_CASH") for w in outcome.warnings)

    margin = rig(tmp_path / "margin", decide=hold_syna, the_spec=paper_spec(account={"model": "margin"}))
    margin.arm(session)
    assert margin.session(session).data["orders"]


def test_expected_account_is_enforced_from_arming_on(tmp_path: Path) -> None:
    paper = rig(tmp_path, config={"expected_account_id_sha256": sha256_hex("someone-else")})
    paper.at(paper.dataset.sessions[60], time(9, 5))
    assert _code(paper.kernel.arm) == ("PAPER_ACCOUNT_MISMATCH", "blocked")
    assert _code(paper.kernel.run_once) == ("PAPER_ACCOUNT_MISMATCH", "blocked")


def test_blocked_account(tmp_path: Path) -> None:
    from signalquarry._internal.paper.brokers.fake import FakeBroker
    from signalquarry._internal.paper.models import BrokerAccount

    class Blocked(FakeBroker):
        def account(self) -> BrokerAccount:
            return replace(super().account(), trading_blocked=True)

    paper = rig(tmp_path, broker_class=Blocked)
    paper.at(paper.dataset.sessions[60])
    assert _code(paper.kernel.run_once) == ("PAPER_ACCOUNT_BLOCKED", "blocked")


def test_intent_that_never_reached_the_broker_is_finalized_next_session(tmp_path: Path) -> None:
    paper, sessions = _armed(tmp_path)
    paper.broker.faults = ["error_before_accept"]
    paper.at(sessions[0])
    assert _code(paper.kernel.run_once)[0] == "BROKER_UNAVAILABLE"
    paper.at(sessions[1])
    outcome = paper.kernel.run_once()
    assert any(w.startswith("PAPER_ORDER_NOT_FOUND") for w in outcome.warnings)


def test_dry_run_during_market_hours_plans_the_next_session(tmp_path: Path) -> None:
    paper, sessions = _armed(tmp_path)
    paper.at(sessions[0], time(11, 0))
    assert paper.kernel.dry_run().data["session"] == sessions[1].isoformat()


def test_kernel_refuses_execution_delay(tmp_path: Path) -> None:
    paper = rig(tmp_path)
    delayed = replace(
        paper.deployment,
        spec=paper.deployment.spec.model_copy(
            update={
                "execution": paper.deployment.spec.execution.model_copy(
                    update={"execution_delay_sessions": 1}
                )
            }
        ),
    )
    with pytest.raises(PaperError) as info:
        PaperKernel(delayed, paper.broker, paper.loader)
    assert info.value.code == "PAPER_EXECUTION_DELAY_UNSUPPORTED"


def test_missing_previous_session_bars_are_stale(tmp_path: Path) -> None:
    from signalquarry._internal.data.dataset import SymbolSeries, with_pending_session

    paper, sessions = _armed(tmp_path)
    paper.at(sessions[0])

    def gap(session):
        data = with_pending_session(truncated(paper.dataset, paper.dataset.index_of(session)), session)
        for symbol, item in list(data.series.items()):
            present = item.present.copy()
            present[-2] = False
            data.series[symbol] = SymbolSeries(item.micro, item.volume, present)
        return data

    kernel = PaperKernel(paper.deployment, paper.broker, gap, now=lambda: paper.broker.now)
    assert _code(kernel.run_once) == ("PAPER_DATA_STALE", "busy")


def test_arm_refuses_open_orders_and_unmanaged_positions(tmp_path: Path) -> None:
    paper, sessions = _armed(tmp_path)
    paper.at(sessions[0])
    paper.kernel.run_once()  # leaves own opg orders open until the open
    paper.at(sessions[0], time(9, 20))
    assert _code(paper.kernel.arm) == ("PAPER_OPEN_ORDERS", "blocked")
    paper.broker.open_session(sessions[0])
    paper.broker.positions_["ZZZ"] = Decimal(2)
    paper.at(sessions[1], time(9, 5))
    assert _code(paper.kernel.arm) == ("PAPER_UNMANAGED_POSITION", "blocked")


def test_preflight_when_the_account_is_unreachable(tmp_path: Path) -> None:
    from signalquarry._internal.paper.brokers.fake import FakeBroker

    class Down(FakeBroker):
        def account(self):
            raise PaperError("BROKER_UNAVAILABLE", "unavailable", "down")

        def clock(self):
            raise PaperError("BROKER_UNAVAILABLE", "unavailable", "down")

    paper = rig(tmp_path, broker_class=Down)
    checks = {c["name"]: c for c in paper.kernel.preflight().data["checks"]}
    assert checks["account"]["code"] == "BROKER_UNAVAILABLE"
    assert checks["armed"]["code"] == "PAPER_NOT_ARMED"
    assert checks["data"]["code"] == "PAPER_DATA_UNAVAILABLE"


def test_reconcile_before_any_arm_checks_orders_only(tmp_path: Path) -> None:
    paper = rig(tmp_path)
    paper.at(paper.dataset.sessions[60])
    assert paper.kernel.reconcile().data["positions"] == {}
