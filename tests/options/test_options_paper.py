# SPDX-License-Identifier: Apache-2.0
"""Options paper runner: parity with the simulator, lifecycle, guards and faults."""

from __future__ import annotations

from datetime import date, time
from decimal import Decimal
from pathlib import Path

import pytest

from signalquarry._internal.data.synthetic import synthetic_dataset
from signalquarry._internal.engine.run import simulate
from signalquarry._internal.paper.brokers.fake_options import at
from signalquarry._internal.paper.journal import Journal
from signalquarry._internal.paper.models import PaperError
from signalquarry.sdk import definition_of
from tests.options.paper_harness import options_rig
from tests.options.wheel_reference import WheelParams, wheel, wheel_spec

DATA = synthetic_dataset(date(2022, 1, 3), date(2022, 12, 30), symbols=("QQQ",))


def _events(journal: Journal) -> list[tuple[str, str]]:
    return [(e["event"], (e.get("leg") or {}).get("symbol", "")) for e in journal.of_kind("wheel_event")]


def _sim_events(result) -> list[tuple[str, str]]:
    out = []
    for d in result.decisions:
        if d.get("outcome") == "filled":
            kind = "opened" if d["action"] == "open" else "closed"
            right = "put" if d["contract"][9] == "P" else "call"
            out.append((f"{right}_{kind}", d["contract"] if kind == "opened" else ""))
        elif d["action"] in ("assigned", "expired"):
            right = "put" if d["contract"][9] == "P" else "call"
            out.append((f"{right}_{d['action']}", ""))
    return out


def test_paper_matches_the_simulator(tmp_path: Path) -> None:
    spec = wheel_spec(options={"underlyings": ["QQQ"], "spread_haircut": "0.000001"})
    params = WheelParams()
    lookback = params.trend_sessions
    sessions = DATA.sessions[lookback : lookback + 120]
    sim = simulate(spec, definition_of(wheel), params, DATA, start=sessions[0], end=sessions[-1])
    rig = options_rig(tmp_path, dataset=DATA, spec=spec, params=params)
    for number, session in enumerate(sessions):
        if number % 50 == 0:
            rig.arm(session)
        rig.day(session)
    paper = _events(Journal.open(rig.deployment.journal_path))
    assert len(paper) >= 10
    assert paper == _sim_events(sim)[: len(paper)] and len(paper) >= len(_sim_events(sim)) - 1


def test_early_assignment_and_unexpected_positions(tmp_path: Path) -> None:
    rig = options_rig(tmp_path, dataset=DATA)
    days = DATA.sessions[60:70]
    rig.arm(days[0])
    rig.day(days[0])
    leg = next(s for s in rig.venue.positions_ if len(s) > 6)
    rig.venue.now = rig.venue.now.replace(hour=15)
    rig.venue.assign_early(leg)
    rig.poll(days[1], time(9, 35))
    journal = Journal.open(rig.deployment.journal_path)
    assert {e["event"] for e in journal.of_kind("wheel_event")} & {"put_assigned", "call_assigned"}
    rig.venue.positions_["SPY"] = Decimal(5)
    with pytest.raises(PaperError) as info:
        rig.poll(days[1], time(10, 0))
    assert info.value.code in ("PAPER_UNMANAGED_POSITION", "PAPER_POSITION_DRIFT")


def test_resting_orders_are_canceled_after_the_timeout(tmp_path: Path) -> None:
    rig = options_rig(tmp_path, dataset=DATA)
    day = DATA.sessions[60]
    rig.arm(day)
    rig.venue.faults = ["rest"]
    rig.poll(day, time(9, 35))
    assert rig.venue.open_orders()
    outcome = rig.poll(day, time(9, 36))
    assert outcome.reason_codes == ["PAPER_ORDER_PENDING"]
    outcome = rig.poll(day, time(9, 40))
    assert any(w.startswith("PAPER_STALE_ORDER_CANCELED") for w in outcome.warnings)
    assert not rig.venue.open_orders()


def test_unconfirmed_cancel_is_busy(tmp_path: Path) -> None:
    rig = options_rig(tmp_path, dataset=DATA)
    day = DATA.sessions[60]
    rig.arm(day)
    rig.venue.faults = ["rest"]
    rig.venue.refuse_cancel = True
    rig.poll(day, time(9, 35))
    with pytest.raises(PaperError) as info:
        rig.poll(day, time(9, 45))
    assert info.value.code == "PAPER_CANCEL_UNCONFIRMED"


def test_lifecycle_pending_then_applied(tmp_path: Path) -> None:
    rig = options_rig(tmp_path, dataset=DATA, params=WheelParams(take_profit=Decimal("0.99")))
    days = DATA.sessions[60:80]
    rig.arm(days[0])
    rig.day(days[0])
    leg = next(s for s in rig.venue.positions_ if len(s) > 6)
    expiry = next(d for d in days if d >= date(2000 + int(leg[3:5]), int(leg[5:7]), int(leg[7:9])))
    for session in days[1:]:
        if session > expiry:
            break
        rig.poll(session, time(9, 35))
        rig.poll(session, time(15, 55))
    rig.venue.positions_.pop(leg, None)  # the broker removed the leg but has not reported the activity yet
    following = days[days.index(expiry) + 1]
    with pytest.raises(PaperError) as info:
        rig.poll(following, time(9, 35))
    assert info.value.code == "PAPER_OPTIONS_LIFECYCLE_PENDING"


def test_unexpected_exercise_halts_and_rejections_are_recorded(tmp_path: Path) -> None:
    from signalquarry._internal.paper.models import Activity

    rig = options_rig(tmp_path, dataset=DATA)
    day = DATA.sessions[60]
    rig.arm(day)
    rig.venue.faults = ["reject"]
    outcome = rig.poll(day, time(9, 35))
    assert any(w.startswith("BROKER_ORDER_REJECTED") for w in outcome.warnings)
    rig.poll(day, time(9, 40))
    leg = next(s for s in rig.venue.positions_ if len(s) > 6)
    rig.venue.activities_.append(Activity("x1", "OPEXC", leg, Decimal(-1), day))
    with pytest.raises(PaperError) as info:
        rig.poll(day, time(9, 45))
    assert info.value.code == "PAPER_OPTIONS_UNEXPECTED_EXERCISE"


def test_submit_errors_resolve_by_client_id_and_market_closed(tmp_path: Path) -> None:
    rig = options_rig(tmp_path, dataset=DATA)
    day = DATA.sessions[60]
    rig.arm(day)
    rig.venue.faults = ["error_after_accept"]
    rig.poll(day, time(9, 35))
    assert len(rig.venue.orders) == 1
    assert rig.poll(day, time(17, 0)).reason_codes == ["PAPER_MARKET_CLOSED"]


def test_dry_poll_writes_nothing(tmp_path: Path) -> None:
    rig = options_rig(tmp_path, dataset=DATA)
    day = DATA.sessions[60]
    rig.venue.now = rig.venue.now.replace()
    from signalquarry._internal.paper.brokers.fake_options import at

    rig.venue.now = at(day, time(9, 35))
    outcome = rig.kernel.poll(submit=False)
    assert "would open" in outcome.summary
    assert not rig.deployment.journal_path.exists() or not Journal.open(rig.deployment.journal_path).entries


def test_options_commands_through_the_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import json
    import uuid

    import yaml

    from signalquarry.cli.main import main

    def sqy(*argv):
        code = main(["--json", *argv])
        return code, json.loads(capsys.readouterr().out)

    package = "wheelcli_" + uuid.uuid4().hex[:8]
    project = tmp_path / "lab"
    sqy("init", str(project), "--demo", "--package", package)
    target = project / "src" / package / "wheel_reference"
    target.mkdir()
    (target / "__init__.py").write_text("")
    (target / "strategy.py").write_text((Path(__file__).parent / "wheel_reference.py").read_text())
    (target / "strategy.yaml").write_text(
        yaml.safe_dump(
            wheel_spec().model_dump(mode="json", by_alias=True, exclude_none=True), sort_keys=False
        )
    )
    config = project / "signalquarry.toml"
    config.write_text(
        config.read_text().replace(
            f'modules = ["{package}.sma_trend.strategy"]',
            f'modules = ["{package}.sma_trend.strategy", "{package}.wheel_reference.strategy"]',
        )
    )
    (project / "paper" / "wheel.paper.yaml").write_text(
        "schema: signalquarry.paper/v1\nalias: wheel\nstrategy: wheel-reference\nbroker: simulated\n"
    )
    code, payload = sqy("paper", "dry-run", "--alias", "wheel", "--project", str(project))
    assert code == 0 and "would open" in payload["summary"], payload
    code, payload = sqy("paper", "status", "--alias", "wheel", "--project", str(project))
    assert code == 0 and payload["data"]["wheels"]["QQQ"]["state"] == "flat"
    code, payload = sqy("paper", "run-once", "--alias", "wheel", "--project", str(project))
    assert payload["reason_codes"] == ["PAPER_BROKER_SIMULATED"]
    code, payload = sqy(
        "paper", "schedule", "--alias", "wheel", "--target", "cron", "--project", str(project)
    )
    assert code == 0 and "* 10-15 * * 1-5" in (project / payload["data"]["files"][0]).read_text()
    code, payload = sqy(
        "paper", "schedule", "--alias", "wheel", "--target", "github-actions", "--project", str(project)
    )
    assert code == 64
    code, payload = sqy("paper", "drift", "--alias", "wheel", "--project", str(project))
    assert code == 0 and payload["data"]["g5"]["ok"] is False  # options G5 comes from the journal


def test_same_day_reentry_after_a_take_profit_close(tmp_path: Path) -> None:
    from signalquarry._internal.options.contracts import Quote

    rig = options_rig(tmp_path, dataset=DATA)
    day = DATA.sessions[60]
    rig.arm(day)
    rig.poll(day, time(9, 35))
    first = next(s for s in rig.venue.positions_ if len(s) > 6)
    entry = Journal.open(rig.deployment.journal_path).of_kind("wheel_event")[-1]["leg"]["entry_credit"]
    assert Decimal(entry) >= Decimal("0.12")  # 0.08 to buy back captures more than the target
    model = rig.venue.option_quote

    def collapsed(symbol: str):
        if symbol != first or first not in rig.venue.positions_:
            return model(symbol)
        return Quote(Decimal("0.07"), Decimal("0.08"), rig.venue.now)

    rig.venue.option_quote = collapsed  # the premium decays: the take-profit fires
    for minute in range(10):
        rig.poll(day, time(11, minute))
    events = [
        (e["event"], (e.get("leg") or {}).get("symbol"))
        for e in Journal.open(rig.deployment.journal_path).of_kind("wheel_event")
    ]
    names = [name for name, _ in events]
    assert names[:3] == ["put_opened", "put_closed", "put_opened"], events
    assert [q for s, q in rig.venue.positions_.items() if len(s) > 6] == [Decimal(-1)]  # one short leg again


def test_arming_the_morning_after_an_expiry_applies_the_activity(tmp_path: Path) -> None:
    rig = options_rig(tmp_path, dataset=DATA, params=WheelParams(take_profit=Decimal("0.99")))
    days = DATA.sessions[60:80]
    rig.arm(days[0])
    leg = None
    for number, session in enumerate(days):
        rig.day(session)
        leg = leg or next((s for s in rig.venue.positions_ if len(s) > 6), None)
        if leg and leg not in rig.venue.positions_:
            rig.venue.now = at(days[number + 1], time(9, 31))  # before any poll that morning
            rig.kernel.arm(reason="re-arm")
            break
    events = [e["event"] for e in Journal.open(rig.deployment.journal_path).of_kind("wheel_event")]
    assert events[-1] in ("put_expired", "put_assigned"), events


def test_options_snapshot_includes_the_leg_and_shares(tmp_path: Path) -> None:
    rig = options_rig(tmp_path, dataset=DATA)
    days = DATA.sessions[60:66]
    rig.arm(days[0])
    for session in days:
        rig.day(session)
    rig.venue.positions_["SPY"] = Decimal(5)  # not managed by this deployment
    document = rig.kernel.snapshot()
    symbols = {p["symbol"] for p in document["positions"]}
    assert symbols and "SPY" not in symbols
    assert any(len(s) > 6 for s in symbols) or "QQQ" in symbols
    assert document["deployment_alias"] == "wheel" and document["snapshot_hash"].startswith("sha256:")
    assert len(document["equity_history"]) == len(days)


def test_options_g5_needs_clean_sessions_and_a_lifecycle_event(tmp_path: Path) -> None:
    rig = options_rig(tmp_path, dataset=DATA, params=WheelParams(take_profit=Decimal("0.99")))
    days = DATA.sessions[60 : 60 + 24]
    for number, session in enumerate(days):
        if number % 20 == 0:
            rig.arm(session)
        rig.day(session)
        if number == 5:
            early = rig.kernel.drift()
            assert early.data["g5"]["ok"] is False and early.data["g5"]["clean_sessions"] == 6
    result = rig.kernel.drift()
    g5 = result.data["g5"]
    assert g5["clean_sessions"] == len(days) and g5["lifecycle_events"] >= 1, g5
    assert g5["ok"] is True and g5["decision_replay"] == "not_available_for_options"
    assert all(row["orders_final"] for row in result.data["sessions"])
