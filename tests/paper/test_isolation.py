# SPDX-License-Identifier: Apache-2.0
"""decide runs in a child process with no credentials; results match in-process planning."""

from __future__ import annotations

from datetime import time
from decimal import Decimal
from pathlib import Path

import pytest

from signalquarry._internal.data.dataset import truncated, with_pending_session
from signalquarry._internal.engine.backtest import EngineError, plan_pre_open
from signalquarry._internal.paper.isolate import plan_in_child
from signalquarry._internal.paper.journal import Journal
from signalquarry._internal.paper.models import PaperError
from signalquarry.sdk import Ctx, Decision, definition_of, strategy
from tests.paper.harness import Momentum, paper_spec, parity_dataset, rig, rotate


@strategy(params=Momentum, lookback=lambda p: p.lookback)
def reads_environment(ctx: Ctx, p: Momentum) -> Decision:
    import os  # a policy violation `sqy check` would reject; used here to probe the child

    leaked = [k for k in os.environ if k.startswith(("APCA_", "SIGNALQUARRY_PAPER"))]
    return Decision.target(
        {}, "WAIT", state={"leaked": leaked, "offline": os.environ.get("SIGNALQUARRY_OFFLINE")}
    )


@strategy(params=Momentum, lookback=lambda p: p.lookback)
def spins(ctx: Ctx, p: Momentum) -> Decision:
    while True:
        pass


@strategy(params=Momentum, lookback=lambda p: p.lookback)
def undeclared(ctx: Ctx, p: Momentum) -> Decision:
    return Decision.target({}, "NOT_DECLARED")


def _inputs():
    data = parity_dataset()
    session = data.sessions[80]
    pending = with_pending_session(truncated(data, data.index_of(session)), session)
    kwargs = dict(quantity={}, cash=Decimal(10000), state={}, last_target=None, target_complete=True)
    return pending, kwargs


def test_child_plan_matches_in_process_plan() -> None:
    pending, kwargs = _inputs()
    spec = paper_spec()
    inside = plan_pre_open(spec, definition_of(rotate), Momentum(), pending, **kwargs)
    child = plan_in_child(spec, definition_of(rotate), Momentum(), pending, **kwargs)
    assert child.record == inside.record and child.orders == inside.orders and child.state == inside.state


def test_child_has_no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")
    monkeypatch.setenv("SIGNALQUARRY_PAPER_SECRET_KEY", "secret")
    pending, kwargs = _inputs()
    plan = plan_in_child(paper_spec(), definition_of(reads_environment), Momentum(), pending, **kwargs)
    assert plan.state == {"leaked": [], "offline": "1"}


def test_contract_errors_and_timeouts_cross_the_boundary() -> None:
    pending, kwargs = _inputs()
    with pytest.raises(EngineError, match="REASON_CODE_UNDECLARED"):
        plan_in_child(paper_spec(), definition_of(undeclared), Momentum(), pending, **kwargs)
    with pytest.raises(PaperError) as info:
        plan_in_child(paper_spec(), definition_of(spins), Momentum(), pending, timeout=3, **kwargs)
    assert info.value.code == "PAPER_STRATEGY_TIMEOUT"


def test_run_once_with_process_isolation(tmp_path: Path) -> None:
    paper = rig(tmp_path, config={"isolation": "process"})
    sessions = paper.dataset.sessions[60:63]
    paper.arm(sessions[0])
    for session in sessions:
        paper.session(session)
    assert len(Journal.open(paper.deployment.journal_path).of_kind("session_completed")) == 3
    paper.at(paper.dataset.sessions[63], time(9, 10))
    assert paper.kernel.dry_run().data["session"] == paper.dataset.sessions[63].isoformat()
