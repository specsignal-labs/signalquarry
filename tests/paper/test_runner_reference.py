# SPDX-License-Identifier: Apache-2.0
"""Independent synthetic reference values for public paper snapshots."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path

from signalquarry._internal.canonical import canonical_hash
from signalquarry._internal.paper.runner import client_order_id
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
