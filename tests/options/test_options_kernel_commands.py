# SPDX-License-Identifier: Apache-2.0
"""Options kernel commands and guards."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, time, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from signalquarry._internal.data.synthetic import synthetic_dataset
from signalquarry._internal.paper.brokers.fake_options import FakeOptionsVenue, at
from signalquarry._internal.paper.journal import Journal
from signalquarry._internal.paper.models import OrderRequest, PaperError
from signalquarry._internal.paper.options_runner import OptionsKernel
from signalquarry._internal.paper.schedule import write_schedule
from tests.options.paper_harness import options_rig
from tests.options.wheel_reference import WheelParams, wheel_spec

DATA = synthetic_dataset(date(2022, 1, 3), date(2022, 12, 30), symbols=("QQQ",))
DAY = DATA.sessions[60]


def _code(action) -> tuple[str, str]:
    with pytest.raises(PaperError) as info:
        action()
    return info.value.code, info.value.status


def _armed(tmp_path: Path, **kwargs):
    rig = options_rig(tmp_path, dataset=DATA, **kwargs)
    rig.venue.now = at(DAY, time(9, 31))
    rig.kernel.arm(reason="test arm for options")
    return rig


def test_arm_status_halt_preflight(tmp_path: Path) -> None:
    rig = _armed(tmp_path)
    rig.venue.now = at(DAY, time(9, 32))
    assert rig.kernel.preflight().data["ready"] is True
    rig.poll(DAY, time(9, 35))
    status = rig.kernel.status()
    assert status.data["state"] == "armed" and status.data["wheels"]["QQQ"]["state"] == "short_put"
    rig.venue.faults = ["rest"]
    rig.venue.now = at(DAY, time(10, 0))
    FakeOptionsVenue.submit(
        rig.venue,
        OrderRequest(
            f"{rig.deployment.prefix}manual-0", "QQQ221230P00300000", "sell", Decimal(1), "day", Decimal("99")
        ),
    )
    halted = rig.kernel.halt("manual stop")
    assert halted.data["canceled_orders"]
    assert _code(lambda: rig.poll(DAY, time(10, 5)))[0] == "PAPER_HALTED"
    assert rig.kernel.preflight().data["ready"] is False
    for command in (rig.kernel.reconcile,):
        assert _code(command)[0] == "PAPER_KIND_UNSUPPORTED"


def test_arm_refusals(tmp_path: Path) -> None:
    rig = options_rig(tmp_path / "a", dataset=DATA)
    unfrozen = OptionsKernel(
        replace(rig.deployment, freeze_hash=None), rig.venue, rig.loader, now=lambda: rig.venue.now
    )
    assert _code(unfrozen.arm)[0] == "FREEZE_REQUIRED"
    rig.venue.now = at(DAY, time(9, 31))
    rig.venue.faults = ["rest"]
    FakeOptionsVenue.submit(
        rig.venue,
        OrderRequest(
            f"{rig.deployment.prefix}x-0", "QQQ221230P00300000", "sell", Decimal(1), "day", Decimal("99")
        ),
    )
    # An own-prefix order the journal never recorded is refused as unmanaged.
    assert _code(rig.kernel.arm)[0] == "PAPER_UNMANAGED_ORDERS"


def test_guards(tmp_path: Path) -> None:
    rig = _armed(tmp_path)
    rig.poll(DAY, time(9, 35))
    rig.venue.end_of_day(DAY)
    rig.venue.cash -= Decimal(50000)  # an unexplained loss overnight
    assert _code(lambda: rig.poll(DATA.sessions[61], time(9, 35)))[0] == "PAPER_DRAWDOWN_HALT"

    skewed = _armed(tmp_path / "s")
    kernel = OptionsKernel(
        skewed.deployment, skewed.venue, skewed.loader, now=lambda: skewed.venue.now + timedelta(minutes=5)
    )
    skewed.venue.now = at(DAY, time(9, 35))
    assert _code(kernel.poll)[0] == "PAPER_CLOCK_SKEW"

    disabled = options_rig(tmp_path / "d", dataset=DATA, config={"submission": "disabled"})
    disabled.venue.now = at(DAY, time(9, 35))
    assert _code(disabled.kernel.poll)[0] == "PAPER_SUBMISSION_DISABLED"


def test_refusals_and_foreign_orders(tmp_path: Path) -> None:
    rig = _armed(tmp_path, cash=Decimal(1000))
    outcome = rig.poll(DAY, time(9, 35))
    assert outcome.reason_codes == ["OPTIONS_INSUFFICIENT_COLLATERAL"]
    late = _armed(
        tmp_path / "late",
        spec=wheel_spec(
            options={"underlyings": ["QQQ"], "entry_cutoff": "09:40", "spread_haircut": "0.000001"}
        ),
    )
    assert late.poll(DAY, time(10, 0)).reason_codes == ["OPTIONS_ENTRY_CUTOFF"]
    rig.venue.faults = ["rest"]
    FakeOptionsVenue.submit(
        rig.venue,
        OrderRequest("someone-else", "QQQ221230P00300000", "sell", Decimal(1), "day", Decimal("99")),
    )
    assert _code(lambda: rig.poll(DAY, time(9, 50)))[0] == "PAPER_UNMANAGED_ORDERS"


def test_lost_intent_is_finalized_after_ten_minutes(tmp_path: Path) -> None:
    rig = _armed(tmp_path)
    rig.venue.faults = ["error_before_accept"]
    assert _code(lambda: rig.poll(DAY, time(9, 35)))[0] == "BROKER_UNAVAILABLE"
    outcome = rig.poll(DAY, time(9, 50))
    assert any(w.startswith("PAPER_ORDER_NOT_FOUND") for w in outcome.warnings)
    finals = Journal.open(rig.deployment.journal_path).of_kind("order_final")
    assert finals[0]["status"] == "not_found"


def test_constructor_refuses_non_paper_and_equity(tmp_path: Path) -> None:
    rig = options_rig(tmp_path, dataset=DATA)

    class Live:
        paper_only = False
        name = "live"

    assert _code(lambda: OptionsKernel(rig.deployment, Live(), rig.loader))[0] == "BROKER_NOT_PAPER_ONLY"  # type: ignore[arg-type]
    equity = replace(rig.deployment, spec=wheel_spec(kind="equity_daily", options=None))
    assert _code(lambda: OptionsKernel(equity, rig.venue, rig.loader))[0] == "PAPER_KIND_UNSUPPORTED"
    assert rig.deployment.config.alias == "wheel" and WheelParams().take_profit > 0


@pytest.mark.parametrize("target", ["systemd", "launchd"])
def test_polling_schedules(tmp_path: Path, target: str) -> None:
    rig = options_rig(tmp_path, dataset=DATA)
    written = write_schedule(tmp_path, rig.deployment.config, target, poll=True)
    text = "".join(p.read_text() for p in written)
    assert (
        ("10..15:*:00" in text)
        if target == "systemd"
        else ("<key>StartInterval</key><integer>60</integer>" in text)
    )
    with pytest.raises(ValueError):
        write_schedule(tmp_path, rig.deployment.config, "github-actions", poll=True)


from signalquarry.sdk import definition_of  # noqa: E402
from signalquarry.sdk.options import OD, OptionsCtx, Strike, options_strategy, sell_call  # noqa: E402


@options_strategy(params=WheelParams, lookback=lambda p: 5)
def close_without_leg(ctx: OptionsCtx, p: WheelParams):
    return OD.close("QQQ", "TAKE_PROFIT", state={"polls": ctx.state.get("polls", 0) + 1})


@options_strategy(params=WheelParams, lookback=lambda p: 5)
def call_when_flat(ctx: OptionsCtx, p: WheelParams):
    return OD.open(sell_call("QQQ", dte=(7, 14), strike=Strike.otm("0.02")), "SELL_CALL_UPTREND")


def _with(tmp_path: Path, decide):
    rig = _armed(tmp_path)
    kernel = OptionsKernel(
        replace(rig.deployment, definition=definition_of(decide)),
        rig.venue,
        rig.loader,
        now=lambda: rig.venue.now,
    )
    return rig, kernel


def test_decision_refusals_and_state(tmp_path: Path) -> None:
    rig, kernel = _with(tmp_path / "a", close_without_leg)
    rig.venue.now = at(DAY, time(9, 35))
    assert kernel.poll().reason_codes == ["OPTIONS_NO_OPEN_LEG"]
    assert Journal.open(rig.deployment.journal_path).last("strategy_state")["state"] == {"polls": 1}
    rig, kernel = _with(tmp_path / "b", call_when_flat)
    rig.venue.now = at(DAY, time(9, 35))
    assert kernel.poll().reason_codes == ["OPTIONS_OPEN_NOT_ALLOWED_IN_FLAT"]


def test_quotes_data_and_account_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _armed(tmp_path)
    monkeypatch.setattr(rig.venue, "stock_quote", lambda symbol: None)
    outcome = rig.poll(DAY, time(9, 35))
    assert outcome.reason_codes == ["PRICE_MISSING"] and outcome.warnings[0].startswith("QUOTE_UNUSABLE")
    monkeypatch.undo()
    stale = OptionsKernel(
        rig.deployment, rig.venue, lambda s: rig.loader(DATA.sessions[59]), now=lambda: rig.venue.now
    )
    assert _code(stale.poll)[0] == "PAPER_DATA_STALE"
    import signalquarry._internal.paper.options_runner as runner

    monkeypatch.setattr(runner, "plan_invariants", lambda *a, **k: ["PLAN_STRIKE_WRONG_SIDE"])
    assert _code(lambda: rig.poll(DAY, time(9, 40)))[0] == "PAPER_OPTIONS_PLAN_INVALID"
    monkeypatch.undo()
    from signalquarry._internal.paper.models import BrokerAccount

    monkeypatch.setattr(
        rig.venue,
        "account",
        lambda: BrokerAccount("x", "ACCOUNT_CLOSED", Decimal(0), Decimal(0), Decimal(0), True),
    )
    assert _code(lambda: rig.poll(DAY, time(9, 45)))[0] == "PAPER_ACCOUNT_BLOCKED"


def test_resting_order_filled_later_and_foreign_activity(tmp_path: Path) -> None:
    from signalquarry._internal.paper.models import Activity

    rig = _armed(tmp_path)
    rig.venue.faults = ["rest"]
    rig.poll(DAY, time(9, 35))
    order = next(iter(rig.venue.orders.values()))
    filled = replace(order, status="filled", filled_quantity=Decimal(1), filled_average_price=Decimal("0.50"))
    rig.venue.orders[order.order_id] = filled
    rig.venue._fill(filled)
    rig.venue.activities_.append(Activity("zz", "OPEXP", "SPY221230P00300000", Decimal(1), DAY))
    rig.poll(DAY, time(9, 36))
    journal = Journal.open(rig.deployment.journal_path)
    assert [e["event"] for e in journal.of_kind("wheel_event")][0] == "put_opened"
    assert journal.of_kind("activity_ignored")[0]["activity_id"] == "zz"


def test_arm_skew_and_preflight_disabled(tmp_path: Path) -> None:
    rig = options_rig(tmp_path, dataset=DATA, config={"submission": "disabled"})
    rig.venue.now = at(DAY, time(9, 31))
    kernel = OptionsKernel(
        rig.deployment, rig.venue, rig.loader, now=lambda: rig.venue.now + timedelta(minutes=5)
    )
    assert _code(kernel.arm)[0] == "PAPER_CLOCK_SKEW"
    assert "PAPER_SUBMISSION_DISABLED" in rig.kernel.preflight().reason_codes


def test_arm_waits_for_a_resting_own_order(tmp_path: Path) -> None:
    rig = _armed(tmp_path)
    rig.venue.faults = ["rest"]
    rig.poll(DAY, time(9, 35))
    assert rig.venue.open_orders()
    rig.venue.now = at(DAY, time(9, 36))  # inside the cancel window: the order is still ours and live
    assert _code(rig.kernel.arm)[0] == "PAPER_OPEN_ORDERS"
