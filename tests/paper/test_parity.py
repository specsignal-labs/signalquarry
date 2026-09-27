# SPDX-License-Identifier: Apache-2.0
"""Backtest ↔ paper parity: the same step, two clocks, identical orders, positions and cash."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from signalquarry._internal.canonical import to_canonical
from signalquarry._internal.engine.backtest import run_backtest
from signalquarry._internal.paper.journal import Journal
from signalquarry.sdk import definition_of
from tests.paper.harness import Momentum, paper_spec, parity_dataset, rig, rotate


def test_sixty_plus_sessions_match_the_backtest(tmp_path: Path) -> None:
    data = parity_dataset()
    split = next(s.ex_date for s in data.splits)
    first = data.index_of(split) - 45
    last = data.index_of(split) + 30
    sessions = data.sessions[first : last + 1]
    spec = paper_spec()

    backtest = run_backtest(
        spec, definition_of(rotate), Momentum(), data, start=sessions[0], end=sessions[-1]
    )

    paper = rig(tmp_path, dataset=data, the_spec=spec)
    for number, session in enumerate(sessions):
        if number % 20 == 0:  # tokens expire after arm_days; a human re-arms (baseline carries over)
            paper.arm(session)
        paper.session(session)

    journal = Journal.open(paper.deployment.journal_path)
    session_of = {e["client_order_id"]: e["session"] for e in journal.of_kind("order_intent")}
    simulated = [(f.session.isoformat(), f.symbol, f.side, f.quantity, f.price) for f in backtest.fills]
    brokered = [
        (session_of[o.client_order_id], o.symbol, o.side, o.filled_quantity, o.filled_average_price)
        for o in paper.broker.orders.values()
        if o.status == "filled"
    ]
    assert len(simulated) >= 6, "the window must exercise several rotations"
    assert sorted(brokered) == sorted(simulated)
    assert paper.broker.positions_ == backtest.positions
    final_cash = backtest.equity[-1] - sum(
        q * data.price(s, "close", last) for s, q in backtest.positions.items()
    )
    assert paper.broker.cash == final_cash
    decisions = [e["decision"] for e in journal.of_kind("session_started")]
    assert decisions == to_canonical(backtest.decisions)
    assert any(s.symbol == "SYNB" and s.ex_date in sessions for s in data.splits)
    assert Decimal(0) not in backtest.positions.values()
