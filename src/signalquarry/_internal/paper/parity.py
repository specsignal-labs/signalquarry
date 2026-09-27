# SPDX-License-Identifier: Apache-2.0
"""``sqy check --parity``: the backtest and the paper kernel must act identically.

The last sessions of the synthetic check window run twice, through the backtest engine
and through the real paper kernel against an in-memory fake venue (re-armed as a human
would). Costs, dividends and option spread haircuts are removed from both sides,
because a paper venue reproduces none of them; what is compared is what the strategy
and the kernel decide: orders, fills, positions, cash and journalled decisions.
"""

from __future__ import annotations

import tempfile
from dataclasses import replace
from datetime import time
from decimal import Decimal
from pathlib import Path

from signalquarry._internal.canonical import to_canonical
from signalquarry._internal.contracts.paper import PaperDeploymentV1
from signalquarry._internal.contracts.spec import StrategySpecV1
from signalquarry._internal.data.dataset import Dataset, truncated, with_pending_session
from signalquarry._internal.data.synthetic import synthetic_dataset
from signalquarry._internal.engine.run import is_options, simulate
from signalquarry._internal.options.chains import session_checkpoints
from signalquarry._internal.paper.brokers.fake import FakeBroker
from signalquarry._internal.paper.brokers.fake_options import FakeOptionsVenue, at
from signalquarry._internal.paper.journal import Journal
from signalquarry._internal.paper.models import PaperError
from signalquarry._internal.paper.options_runner import OptionsKernel
from signalquarry._internal.paper.runner import Deployment, PaperKernel
from signalquarry._internal.project.project import LoadedStrategy
from signalquarry._internal.validation.conformance import CHECK_END, CHECK_START, CheckResult

SESSIONS = 60
REARM_EVERY = 20
_PLACEHOLDER_FREEZE = "sha256:" + "0" * 64


def _frictionless(spec: StrategySpecV1) -> StrategySpecV1:
    body = spec.model_dump(mode="json")
    body["execution"]["costs"] = {"bps": "0", "per_share": "0"}  # sell-side fees default to zero
    body["execution"]["fill"] = None  # a paper venue fills whole orders; capacity is a backtest model
    if body.get("options"):
        body["options"]["per_contract_fee"] = "0"
        body["options"]["spread_haircut"] = "0.000001"
    return StrategySpecV1.model_validate(body)


def _deployment(strategy: LoadedStrategy, spec: StrategySpecV1, state_dir: Path) -> Deployment:
    return Deployment(
        config=PaperDeploymentV1.model_validate(
            {
                "schema": "signalquarry.paper/v1",
                "alias": "parity",
                "strategy": spec.id,
                "submission": "enabled",
                "isolation": "none",
            }
        ),
        spec=spec,
        definition=strategy.definition,
        params=strategy.params,
        configuration_hash=strategy.configuration_hash,
        freeze_hash=_PLACEHOLDER_FREEZE,
        state_dir=state_dir,
    )


def _loader(data: Dataset):
    return lambda session: with_pending_session(truncated(data, data.index_of(session)), session)


def _close(data: Dataset, symbol: str, index: int) -> Decimal:
    price = data.price(symbol, "close", index)
    if price is None:
        raise PaperError("PRICE_MISSING", "invalid", f"{symbol} close")
    return price


def _equity(
    strategy: LoadedStrategy, spec: StrategySpecV1, data: Dataset, state_dir: Path
) -> tuple[list[str], str]:
    sessions = data.sessions[-SESSIONS:]
    backtest = simulate(spec, strategy.definition, strategy.params, data, start=sessions[0], end=sessions[-1])
    broker = FakeBroker(data, spec.account.initial_cash)
    deployment = _deployment(strategy, spec, state_dir)
    kernel = PaperKernel(deployment, broker, _loader(data), now=lambda: broker.now)
    for number, session in enumerate(sessions):
        if number % REARM_EVERY == 0:
            broker.pre_open(session, time(9, 5))
            kernel.arm(reason="parity")
        broker.pre_open(session, time(9, 10))
        kernel.run_once()
        broker.open_session(session)
    journal = Journal.open(deployment.journal_path)
    session_of = {e["client_order_id"]: e["session"] for e in journal.of_kind("order_intent")}
    simulated = sorted((f.session.isoformat(), f.symbol, f.side, f.quantity, f.price) for f in backtest.fills)
    brokered = sorted(
        (session_of.get(o.client_order_id, "?"), o.symbol, o.side, o.filled_quantity, o.filled_average_price)
        for o in broker.orders.values()
        if o.status == "filled"
    )
    problems = []
    if brokered != simulated:
        only_sim = [f for f in simulated if f not in brokered][:3]
        only_paper = [f for f in brokered if f not in simulated][:3]
        problems.append(f"fills differ: backtest-only {only_sim}, paper-only {only_paper}")
    positions = {s: q for s, q in backtest.positions.items() if q}
    if broker.positions_ != positions:
        problems.append(f"positions differ: backtest {positions}, paper {broker.positions_}")
    last = data.index_of(sessions[-1])
    cash = backtest.equity[-1] - sum(q * _close(data, s, last) for s, q in positions.items())
    if broker.cash != cash:
        problems.append(f"cash differs: backtest {cash}, paper {broker.cash}")
    decisions = [e["decision"] for e in journal.of_kind("session_started")]
    if decisions != to_canonical(backtest.decisions):
        problems.append("journalled decisions differ from the backtest's")
    return problems, f"{len(simulated)} fills, {len(decisions)} decisions"


def _options(
    strategy: LoadedStrategy, spec: StrategySpecV1, data: Dataset, state_dir: Path
) -> tuple[list[str], str]:
    sessions = data.sessions[-SESSIONS:]
    result = simulate(spec, strategy.definition, strategy.params, data, start=sessions[0], end=sessions[-1])
    simulated = []
    for d in result.decisions:
        if d.get("outcome") == "filled":
            kind = "opened" if d["action"] == "open" else "closed"
            right = "put" if d["contract"][9] == "P" else "call"
            simulated.append(f"{right}_{kind}")
        elif d["action"] in ("assigned", "expired"):
            right = "put" if d["contract"][9] == "P" else "call"
            simulated.append(f"{right}_{d['action']}")
    venue = FakeOptionsVenue(data, spec.account.initial_cash)
    deployment = _deployment(strategy, spec, state_dir)
    kernel = OptionsKernel(deployment, venue, _loader(data), now=lambda: venue.now)
    for number, session in enumerate(sessions):
        if number % REARM_EVERY == 0:
            venue.now = at(session, time(9, 31))
            kernel.arm(reason="parity")
        for _, moment in session_checkpoints(session):
            venue.now = at(session, moment)
            kernel.poll()
        venue.end_of_day(session)
    paper = [e["event"] for e in Journal.open(deployment.journal_path).of_kind("wheel_event")]
    # The last simulated expiry or assignment may land after the final paper poll.
    if paper != simulated[: len(paper)] or len(paper) < len(simulated) - 1:
        return [f"wheel events differ: backtest {simulated[:8]}…, paper {paper[:8]}…"], ""
    return [], f"{len(paper)} wheel events"


def parity(strategy: LoadedStrategy) -> CheckResult:
    spec = _frictionless(strategy.spec)
    data = synthetic_dataset(CHECK_START, CHECK_END, symbols=spec.data.symbols)
    data = replace(data, dividends=())
    runner = _options if is_options(spec) else _equity
    try:
        with tempfile.TemporaryDirectory(prefix="sqy-parity-") as scratch:
            problems, counts = runner(strategy, spec, data, Path(scratch) / "paper" / "parity")
    except PaperError as exc:
        return CheckResult("parity", False, f"paper kernel stopped: {exc.code} {exc.detail}")
    if problems:
        return CheckResult("parity", False, "; ".join(problems)[:2000])
    return CheckResult(
        "parity",
        True,
        f"{SESSIONS} sessions, {counts}: backtest and paper kernel agree (costs and dividends excluded)",
    )
