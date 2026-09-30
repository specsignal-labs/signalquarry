# SPDX-License-Identifier: Apache-2.0
"""Independent synthetic reference values for public paper snapshots."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from signalquarry._internal.canonical import canonical_hash
from signalquarry._internal.data.dataset import Split
from signalquarry._internal.engine.backtest import PlannedOrder, PreOpenPlan
from signalquarry._internal.paper.journal import Journal
from signalquarry._internal.paper.models import BrokerAccount, PaperError
from signalquarry._internal.paper.runner import client_order_id
from signalquarry.sdk import Decision
from tests.paper.harness import hold_syna, paper_spec, rig


def _five_sessions(tmp_path: Path, **kwargs):
    paper = rig(tmp_path, decide=hold_syna, **kwargs)
    days = paper.dataset.sessions[60:65]
    paper.arm(days[0])
    for session in days:
        paper.session(session)
    return paper


def test_snapshot_reference_account_drawdown_and_pnl(tmp_path: Path) -> None:
    paper = _five_sessions(tmp_path)
    snapshot = paper.kernel.snapshot(recent_fills=1, previous_snapshot_hash="prior")
    assert snapshot["account_history_epoch"] == "2023-03-27"
    assert snapshot["account"] == {
        "equity": "10044.650000",
        "cash": "5554.270000",
        "buying_power": "5554.270000",
        "status": "ACTIVE",
    }
    assert snapshot["equity_history"] == [
        {"timestamp": "2023-03-27", "equity": "10000", "drawdown": "0.000000"},
        {"timestamp": "2023-03-28", "equity": "9916.81", "drawdown": "0.008319"},
        {"timestamp": "2023-03-29", "equity": "9911.64", "drawdown": "0.008836"},
        {"timestamp": "2023-03-30", "equity": "9984.96", "drawdown": "0.001504"},
        {"timestamp": "2023-03-31", "equity": "10044.65", "drawdown": "0.000000"},
    ]
    assert {k: (v["amount"], v["return_value"]) for k, v in snapshot["pnl"].items()} == {
        "broker_reference": ("44.65", "0.004465"),
        "day": ("0.00", "0.000000"),
        "net_dollar_pnl": ("44.65", "0.004465"),
        "time_weighted_return": (None, None),
    }
    assert snapshot["positions"] == [
        {"symbol": "SYNA", "quantity": "47", "market_value": None, "unrealized_pnl": None}
    ]
    assert snapshot["recent_fills"] == [
        {
            "filled_at": "2023-03-27",
            "action": "buy",
            "instrument": "SYNA",
            "quantity": "47",
            "average_fill_price": "94.59",
        }
    ]
    assert snapshot["previous_snapshot_hash"] == "prior"
    body = {key: value for key, value in snapshot.items() if key != "snapshot_hash"}
    assert snapshot["snapshot_hash"] == canonical_hash(body)


def test_snapshot_zero_recent_fills_is_empty(tmp_path: Path) -> None:
    paper = _five_sessions(tmp_path)
    assert paper.kernel.snapshot(recent_fills=0)["recent_fills"] == []


def test_drift_reference_costs_and_minimum_session_gate(tmp_path: Path) -> None:
    paper = _five_sessions(
        tmp_path, the_spec=paper_spec(execution={"costs": {"bps": "10", "per_share": "0.01"}})
    )
    outcome = paper.kernel.drift()
    assert outcome.reason_codes == []
    assert outcome.data["sessions"] == [
        {
            "session": day.isoformat(),
            "parity": "ok",
            "orders_final": True,
            "halted": False,
            "clean": True,
        }
        for day in paper.dataset.sessions[60:65]
    ]
    assert outcome.data["g5"] == {
        "ok": False,
        "clean_sessions": 5,
        "required_sessions": 20,
        "parity_mismatches": 0,
        "mean_adverse_slippage_bps": 0.0,
        "tolerance_bps": 25.0,
    }
    assert outcome.data["fills_measured"] == 1
    assert outcome.data["shadow"] == {
        "model_costs": Decimal("4.92"),
        "uncredited_dividends": Decimal("0.00"),
        "broker_equity": Decimal("10044.65"),
        "shadow_equity": Decimal("10039.73"),
    }


def test_client_order_id_binds_every_intent_field(tmp_path: Path) -> None:
    deployment = rig(tmp_path, decide=hold_syna).deployment
    session = date(2023, 3, 27)
    base = client_order_id(deployment, session, "SYNA", "buy", Decimal("47"))
    intent = {
        "alias": "demo",
        "configuration_hash": deployment.configuration_hash,
        "session": session,
        "symbol": "SYNA",
        "side": "buy",
        "quantity": Decimal("47"),
    }
    assert base == f"sq-demo-{canonical_hash(intent)[7:17]}-0"
    variants = [
        client_order_id(deployment, date(2023, 3, 28), "SYNA", "buy", Decimal("47")),
        client_order_id(deployment, session, "SYNB", "buy", Decimal("47")),
        client_order_id(deployment, session, "SYNA", "sell", Decimal("47")),
        client_order_id(deployment, session, "SYNA", "buy", Decimal("48")),
        client_order_id(
            replace(deployment, configuration_hash="sha256:" + "d" * 64),
            session,
            "SYNA",
            "buy",
            Decimal("47"),
        ),
        client_order_id(
            replace(deployment, config=deployment.config.model_copy(update={"alias": "other"})),
            session,
            "SYNA",
            "buy",
            Decimal("47"),
        ),
    ]
    assert len(set([base, *variants])) == 7


def _plan(*orders: PlannedOrder) -> PreOpenPlan:
    return PreOpenPlan(
        session=date(2023, 3, 27),
        decision=Decision.target({}, "GO"),
        record={},
        state={},
        last_target=None,
        orders=list(orders),
        complete=True,
        warnings=[],
        equity=Decimal("100"),
        marks={},
    )


def _account(cash: str) -> BrokerAccount:
    return BrokerAccount("synthetic", "ACTIVE", Decimal(cash), Decimal(cash), Decimal(cash), False)


def test_cash_sizing_uses_settled_budget_and_carries_buffered_cost_between_buys(
    tmp_path: Path,
) -> None:
    paper = rig(tmp_path, config={"guards": {"buy_price_buffer_bps": "1000"}})
    plan = _plan(
        PlannedOrder("SYNA", Decimal("6"), Decimal("10")),
        PlannedOrder("SYNB", Decimal("6"), Decimal("10")),
    )
    orders, complete, warnings = paper.kernel._size(plan, _account("100"), Decimal("100"))
    assert [(o["symbol"], o["side"], o["quantity"]) for o in orders] == [
        ("SYNA", "buy", Decimal("6")),
        ("SYNB", "buy", Decimal("3")),
    ]
    assert complete is False
    assert warnings == ["INSUFFICIENT_SETTLED_CASH:2023-03-27:SYNB"]
    assert [o["client_order_id"] for o in orders] == [
        client_order_id(paper.deployment, plan.session, o["symbol"], o["side"], o["quantity"]) for o in orders
    ]

    orders, complete, warnings = paper.kernel._size(
        _plan(PlannedOrder("SYNA", Decimal("5"), Decimal("10"))),
        _account("100"),
        Decimal("20"),
    )
    assert [(o["symbol"], o["quantity"]) for o in orders] == [("SYNA", Decimal("1"))]
    assert complete is False and len(warnings) == 1


def test_margin_sale_proceeds_can_fund_a_later_buy(tmp_path: Path) -> None:
    paper = rig(
        tmp_path,
        the_spec=paper_spec(account={"model": "margin"}),
        config={"guards": {"buy_price_buffer_bps": "0"}},
    )
    plan = _plan(
        PlannedOrder("SYNB", Decimal("-4"), Decimal("10")),
        PlannedOrder("SYNA", Decimal("5"), Decimal("10")),
    )
    orders, complete, warnings = paper.kernel._size(plan, _account("20"), Decimal("0"))
    assert [(o["symbol"], o["side"], o["quantity"]) for o in orders] == [
        ("SYNB", "sell", Decimal("4")),
        ("SYNA", "buy", Decimal("5")),
    ]
    assert complete is True and warnings == []


def test_below_minimum_order_does_not_hide_later_valid_buy(tmp_path: Path) -> None:
    paper = rig(
        tmp_path,
        the_spec=paper_spec(execution={"min_order_notional": "50"}),
        config={"guards": {"buy_price_buffer_bps": "0"}},
    )
    plan = _plan(
        PlannedOrder("SYNA", Decimal("1"), Decimal("10")),
        PlannedOrder("SYNB", Decimal("6"), Decimal("10")),
    )
    orders, complete, warnings = paper.kernel._size(plan, _account("100"), Decimal("100"))
    assert [(o["symbol"], o["quantity"]) for o in orders] == [("SYNB", Decimal("6"))]
    assert complete is True and warnings == []


def test_buffer_denominator_respects_exact_cent_boundary(tmp_path: Path) -> None:
    paper = rig(tmp_path, config={"guards": {"buy_price_buffer_bps": "1000"}})
    plan = _plan(PlannedOrder("SYNA", Decimal("1"), Decimal("10000")))
    orders, complete, warnings = paper.kernel._size(plan, _account("10999.95"), Decimal("10999.95"))
    assert orders == [] and complete is False
    assert warnings == ["INSUFFICIENT_SETTLED_CASH:2023-03-27:SYNA"]


def test_zero_affordability_never_emits_a_zero_share_order(tmp_path: Path) -> None:
    paper = rig(
        tmp_path,
        the_spec=paper_spec(execution={"min_order_notional": "0"}),
        config={"guards": {"buy_price_buffer_bps": "0"}},
    )
    plan = _plan(PlannedOrder("SYNA", Decimal("1"), Decimal("10")))
    orders, complete, warnings = paper.kernel._size(plan, _account("0"), Decimal("0"))
    assert orders == [] and complete is False
    assert warnings == ["INSUFFICIENT_SETTLED_CASH:2023-03-27:SYNA"]


def test_exact_minimum_notional_is_eligible(tmp_path: Path) -> None:
    paper = rig(
        tmp_path,
        the_spec=paper_spec(execution={"min_order_notional": "50"}),
        config={"guards": {"buy_price_buffer_bps": "0"}},
    )
    plan = _plan(PlannedOrder("SYNA", Decimal("5"), Decimal("10")))
    orders, complete, warnings = paper.kernel._size(plan, _account("50"), Decimal("50"))
    assert [(o["symbol"], o["quantity"]) for o in orders] == [("SYNA", Decimal("5"))]
    assert complete is True and warnings == []


def _baseline_journal(paper, session: date, positions: dict[str, str]) -> Journal:
    journal = Journal.open(paper.deployment.journal_path)
    journal.append(
        "armed",
        {"baseline_session": session, "baseline_positions": positions},
        now=paper.broker.now,
    )
    return journal


def test_reconcile_positions_reports_sorted_unmanaged_symbols_without_halting(tmp_path: Path) -> None:
    paper = rig(tmp_path)
    session = paper.dataset.sessions[60]
    journal = _baseline_journal(paper, paper.dataset.sessions[59], {})
    with pytest.raises(PaperError) as caught:
        paper.kernel.reconcile_positions(
            journal,
            paper.loader(session),
            session,
            {"ZZZ": Decimal("1"), "AAA": Decimal("2")},
        )
    assert (caught.value.code, caught.value.status, caught.value.detail) == (
        "PAPER_UNMANAGED_POSITION",
        "blocked",
        "AAA,ZZZ",
    )
    assert journal.last("halted") is None


def test_reconcile_positions_hashes_missing_and_extra_managed_positions(tmp_path: Path) -> None:
    paper = rig(tmp_path)
    session = paper.dataset.sessions[60]
    journal = _baseline_journal(paper, paper.dataset.sessions[59], {"SYNA": "47"})
    with pytest.raises(PaperError) as caught:
        paper.kernel.reconcile_positions(
            journal,
            paper.loader(session),
            session,
            {"SYNB": Decimal("2")},
        )
    drift = {"SYNA": ["47", "0"], "SYNB": ["0", "2"]}
    detail = canonical_hash(drift) + " " + str(drift)
    assert (caught.value.code, caught.value.status, caught.value.detail) == (
        "PAPER_POSITION_DRIFT",
        "blocked",
        detail,
    )
    assert journal.last("halted")["detail"] == detail


def test_reconcile_positions_excludes_unchanged_symbols_from_drift(tmp_path: Path) -> None:
    paper = rig(tmp_path)
    session = paper.dataset.sessions[60]
    journal = _baseline_journal(paper, paper.dataset.sessions[59], {"SYNA": "47", "SYNB": "5"})
    with pytest.raises(PaperError) as caught:
        paper.kernel.reconcile_positions(
            journal,
            paper.loader(session),
            session,
            {"SYNA": Decimal("47"), "SYNB": Decimal("2")},
        )
    drift = {"SYNB": ["5", "2"]}
    assert caught.value.detail == canonical_hash(drift) + " " + str(drift)
    assert journal.last("halted")["reason_codes"] == ["PAPER_POSITION_DRIFT"]


def test_only_todays_split_may_be_pending(tmp_path: Path) -> None:
    paper = rig(tmp_path)
    previous, session = paper.dataset.sessions[59:61]
    journal = _baseline_journal(paper, paper.dataset.sessions[58], {"SYNA": "3"})
    data = replace(
        paper.loader(session),
        splits=(
            Split("SYNA", previous, Decimal("2")),
            Split("SYNA", session, Decimal("3")),
        ),
    )
    assert paper.kernel.expected_positions(journal, data, session, through=session) == {"SYNA": Decimal("18")}
    with pytest.raises(PaperError) as pending:
        paper.kernel.reconcile_positions(journal, data, session, {"SYNA": Decimal("6")})
    assert (pending.value.code, pending.value.status, pending.value.detail) == (
        "PAPER_CORPORATE_ACTION_PENDING",
        "busy",
        "broker has not applied today's split",
    )
    assert journal.last("halted") is None
    with pytest.raises(PaperError) as stale:
        paper.kernel.reconcile_positions(journal, data, session, {"SYNA": Decimal("3")})
    assert (stale.value.code, stale.value.status) == ("PAPER_POSITION_DRIFT", "blocked")
    assert journal.last("halted")["reason_codes"] == ["PAPER_POSITION_DRIFT"]
