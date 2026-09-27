# SPDX-License-Identifier: Apache-2.0
"""Kernel safety: idempotency, N1/N2 recovery, guards, halts and arm binding."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, time, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from signalquarry._internal.paper.brokers.fake import FakeBroker
from signalquarry._internal.paper.journal import Journal
from signalquarry._internal.paper.lease import RunLease
from signalquarry._internal.paper.models import PaperError
from signalquarry._internal.paper.runner import PaperKernel
from tests.paper.harness import Rig, hold_syna, rig


def _trading_rig(tmp_path: Path, **kwargs) -> tuple[Rig, list[date]]:
    """A rig armed on a session where the strategy wants to trade, plus the following sessions."""
    paper = rig(tmp_path, **kwargs)
    sessions = list(paper.dataset.sessions[60:])
    paper.arm(sessions[0])
    for index, session in enumerate(sessions):
        paper.at(session)
        if paper.kernel.dry_run().data["orders"]:
            return paper, sessions[index:]
        paper.kernel.run_once()
        paper.broker.open_session(session)
    raise AssertionError("no trading session found")


def _code(action) -> tuple[str, str]:
    with pytest.raises(PaperError) as info:
        action()
    return info.value.code, info.value.status


def test_run_once_is_idempotent_per_session(tmp_path: Path) -> None:
    paper, sessions = _trading_rig(tmp_path)
    paper.at(sessions[0])
    first = paper.kernel.run_once()
    submissions = paper.broker.submissions
    second = paper.kernel.run_once()
    assert first.data["orders"] and second.reason_codes == ["PAPER_SESSION_ALREADY_COMPLETED"]
    assert paper.broker.submissions == submissions


def test_n1_ambiguous_submit_is_resolved_by_client_order_id(tmp_path: Path) -> None:
    paper, sessions = _trading_rig(tmp_path)
    paper.broker.faults = ["error_after_accept"]
    paper.at(sessions[0])
    outcome = paper.kernel.run_once()
    ids = [o["client_order_id"] for o in outcome.data["orders"]]
    assert len({o.client_order_id for o in paper.broker.orders.values()}) == len(paper.broker.orders)
    assert sorted(o.client_order_id for o in paper.broker.orders.values()) == sorted(ids)


def test_submit_failure_before_accept_resumes_the_journaled_plan(tmp_path: Path) -> None:
    paper, sessions = _trading_rig(tmp_path)
    paper.broker.faults = ["error_before_accept"]
    paper.at(sessions[0])
    assert _code(paper.kernel.run_once) == ("BROKER_UNAVAILABLE", "busy")
    journal = Journal.open(paper.deployment.journal_path)
    assert journal.last("session_started")["session"] == sessions[0].isoformat()
    assert (
        journal.last("session_completed") is None
        or journal.last("session_completed")["session"] != sessions[0].isoformat()
    )
    paper.at(sessions[0], time(9, 15))
    outcome = paper.kernel.run_once()
    assert sorted(o.client_order_id for o in paper.broker.orders.values()) == sorted(
        o["client_order_id"] for o in outcome.data["orders"]
    )
    assert len(Journal.open(paper.deployment.journal_path).of_kind("session_started")) == len(
        {e["session"] for e in Journal.open(paper.deployment.journal_path).of_kind("session_started")}
    )


class CrashAfterSubmit(FakeBroker):
    crash = True

    def submit(self, request):
        order = super().submit(request)
        if self.crash:
            self.crash = False
            raise KeyboardInterrupt("process killed after the broker accepted the order")
        return order


def test_n2_orphan_filled_order_is_recovered_next_session(tmp_path: Path) -> None:
    paper, sessions = _trading_rig(tmp_path, broker_class=CrashAfterSubmit)
    paper.broker.crash = True
    paper.at(sessions[0])
    with pytest.raises(KeyboardInterrupt):
        paper.kernel.run_once()
    paper.broker.open_session(sessions[0])  # the orphan fills at the open
    paper.at(sessions[1])
    paper.kernel.run_once()  # reconciles the orphan instead of reporting drift
    journal = Journal.open(paper.deployment.journal_path)
    recovered = [e for e in journal.of_kind("order_submitted") if e.get("recovered")]
    assert recovered and journal.last("halted") is None
    finals = [e for e in journal.of_kind("order_final") if e["session"] == sessions[0].isoformat()]
    assert finals and all(e["status"] == "filled" for e in finals)


def test_rejected_order_marks_the_target_incomplete_and_retries(tmp_path: Path) -> None:
    paper, sessions = _trading_rig(tmp_path, decide=hold_syna)
    paper.broker.faults = ["reject"] * 5
    paper.session(sessions[0])
    journal = Journal.open(paper.deployment.journal_path)
    assert journal.last("session_completed")["complete"] is False
    assert {e["status"] for e in journal.of_kind("order_final")} >= {"rejected"}
    paper.broker.faults = []
    outcome = paper.session(sessions[1])
    assert outcome.data["orders"], "the unchanged target is retried because it was incomplete"


def test_lease_contention_is_busy(tmp_path: Path) -> None:
    paper, sessions = _trading_rig(tmp_path)
    paper.at(sessions[0])
    with RunLease(paper.deployment.lease_path):
        assert _code(paper.kernel.run_once) == ("RUN_LEASE_BUSY", "busy")


def test_clock_skew_blocks(tmp_path: Path) -> None:
    paper, sessions = _trading_rig(tmp_path)
    paper.broker.skew = timedelta(seconds=90)
    paper.at(sessions[0])
    assert _code(paper.kernel.run_once) == ("PAPER_CLOCK_SKEW", "blocked")


def test_window_weekend_and_open_market(tmp_path: Path) -> None:
    paper, sessions = _trading_rig(tmp_path)
    paper.at(sessions[0], time(8, 30))
    assert _code(paper.kernel.run_once) == ("PAPER_OUTSIDE_WINDOW", "busy")
    paper.at(sessions[0], time(10, 0))
    assert _code(paper.kernel.run_once) == ("PAPER_OUTSIDE_WINDOW", "busy")
    saturday = next(
        sessions[0] + timedelta(days=k)
        for k in range(1, 8)
        if (sessions[0] + timedelta(days=k)).weekday() == 5
    )
    paper.broker.now = paper.broker.now.replace(year=saturday.year, month=saturday.month, day=saturday.day)
    assert paper.kernel.run_once().reason_codes == ["PAPER_NO_SESSION_TODAY"]


def test_unconfirmed_cancel_of_a_stale_order_is_busy(tmp_path: Path) -> None:
    class NoCancel(FakeBroker):
        def cancel(self, order_id: str) -> None:
            return None

    paper, sessions = _trading_rig(tmp_path, broker_class=NoCancel)
    paper.at(sessions[0])
    paper.kernel.run_once()  # orders never reach an open: the next session finds them still open
    paper.at(sessions[1])
    assert _code(paper.kernel.run_once) == ("PAPER_CANCEL_UNCONFIRMED", "busy")


def test_stale_open_order_is_canceled_and_the_target_retried(tmp_path: Path) -> None:
    paper, sessions = _trading_rig(tmp_path, decide=hold_syna)
    paper.at(sessions[0])
    paper.kernel.run_once()
    paper.at(sessions[1])
    outcome = paper.kernel.run_once()
    assert any(w.startswith("PAPER_STALE_ORDER_CANCELED") for w in outcome.warnings)
    assert outcome.data["orders"]


def test_position_drift_halts_until_a_human_re_arms(tmp_path: Path) -> None:
    paper, sessions = _trading_rig(tmp_path)
    paper.session(sessions[0])
    symbol = next(iter(paper.broker.positions_))
    paper.broker.positions_[symbol] += Decimal(3)
    paper.at(sessions[1])
    assert _code(paper.kernel.run_once) == ("PAPER_POSITION_DRIFT", "blocked")
    assert _code(paper.kernel.run_once) == ("PAPER_HALTED", "disabled")
    assert paper.kernel.status().data["state"] == "halted"
    paper.at(sessions[1], time(9, 12))
    paper.kernel.arm()
    paper.at(sessions[1], time(9, 14))
    assert paper.kernel.run_once().data["session"] == sessions[1].isoformat()


def test_unmanaged_positions_and_orders_block(tmp_path: Path) -> None:
    paper, sessions = _trading_rig(tmp_path)
    paper.broker.positions_["ZZZ"] = Decimal(1)
    paper.at(sessions[0])
    assert _code(paper.kernel.run_once) == ("PAPER_UNMANAGED_POSITION", "blocked")
    del paper.broker.positions_["ZZZ"]
    from signalquarry._internal.paper.models import OrderRequest

    FakeBroker.submit(paper.broker, OrderRequest("manual-1", "SYNA", "buy", Decimal(1)))
    assert _code(paper.kernel.run_once) == ("PAPER_UNMANAGED_ORDERS", "blocked")


def test_drawdown_guard_halts(tmp_path: Path) -> None:
    paper, sessions = _trading_rig(tmp_path, config={"guards": {"max_daily_drawdown": "0.01"}})
    paper.session(sessions[0])
    paper.broker.cash -= Decimal(500)  # an unexplained 5% equity loss
    paper.at(sessions[1])
    assert _code(paper.kernel.run_once) == ("PAPER_DRAWDOWN_HALT", "blocked")


def test_submission_disabled_and_not_armed(tmp_path: Path) -> None:
    disabled = rig(tmp_path / "a", submission="disabled")
    disabled.at(disabled.dataset.sessions[60])
    assert _code(disabled.kernel.run_once) == ("PAPER_SUBMISSION_DISABLED", "disabled")
    unarmed = rig(tmp_path / "b")
    unarmed.at(unarmed.dataset.sessions[60])
    assert _code(unarmed.kernel.run_once) == ("PAPER_NOT_ARMED", "disabled")
    assert unarmed.kernel.dry_run().data["session"] == unarmed.dataset.sessions[60].isoformat()


def test_arm_token_binds_configuration_account_and_journal(tmp_path: Path) -> None:
    paper, sessions = _trading_rig(tmp_path)
    token_path = paper.deployment.arm_path
    token = json.loads(token_path.read_text())
    token_path.write_text(json.dumps({**token, "expires_at": "2099-01-01T00:00:00Z"}))
    paper.at(sessions[0])
    assert _code(paper.kernel.run_once) == ("PAPER_NOT_ARMED", "disabled")
    token_path.write_text(json.dumps(token))
    changed = PaperKernel(
        replace(paper.deployment, configuration_hash="sha256:" + "d" * 64),
        paper.broker,
        paper.loader,
        now=lambda: paper.broker.now,
    )
    assert _code(changed.run_once) == ("PAPER_ARM_STALE", "disabled")
    paper.broker.account_id = "another-account"
    assert _code(paper.kernel.run_once) == ("PAPER_ACCOUNT_MISMATCH", "blocked")
    paper.broker.account_id = "fake-paper-account"
    paper.broker.now += timedelta(days=45)
    assert _code(paper.kernel.run_once)[0] == "PAPER_ARM_EXPIRED"


def test_arm_requires_a_freeze(tmp_path: Path) -> None:
    paper = rig(tmp_path, freeze_hash=None)
    paper.at(paper.dataset.sessions[60], time(9, 5))
    assert _code(paper.kernel.arm) == ("FREEZE_REQUIRED", "blocked")


def test_tampered_journal_is_refused(tmp_path: Path) -> None:
    paper, sessions = _trading_rig(tmp_path)
    lines = paper.deployment.journal_path.read_text().splitlines()
    entry = json.loads(lines[0])
    entry["baseline_positions"] = {"SYNA": "5"}
    paper.deployment.journal_path.write_text("\n".join([json.dumps(entry), *lines[1:]]) + "\n")
    paper.at(sessions[0])
    assert _code(paper.kernel.run_once) == ("PAPER_JOURNAL_CORRUPT", "blocked")


def test_kernel_refuses_non_paper_brokers_and_fractional_sizing(tmp_path: Path) -> None:
    paper = rig(tmp_path)

    class Live:
        paper_only = False
        name = "live"

    with pytest.raises(PaperError) as info:
        PaperKernel(paper.deployment, Live(), paper.loader)  # type: ignore[arg-type]
    assert info.value.code == "BROKER_NOT_PAPER_ONLY"
    fractional = replace(
        paper.deployment,
        spec=paper.deployment.spec.model_copy(
            update={"execution": paper.deployment.spec.execution.model_copy(update={"sizing": "fractional"})}
        ),
    )
    with pytest.raises(PaperError) as info:
        PaperKernel(fractional, paper.broker, paper.loader)
    assert info.value.code == "PAPER_FRACTIONAL_UNSUPPORTED"


def test_preflight_reports_each_check(tmp_path: Path) -> None:
    paper = rig(tmp_path, submission="disabled")
    paper.at(paper.dataset.sessions[60])
    outcome = paper.kernel.preflight()
    checks = {c["name"]: c["ok"] for c in outcome.data["checks"]}
    assert checks == {
        "journal": True,
        "account": True,
        "clock": True,
        "armed": False,
        "submission": False,
        "orders": True,
        "positions": True,
        "data": True,
    }
    assert outcome.reason_codes == ["PAPER_NOT_ARMED", "PAPER_SUBMISSION_DISABLED"]
