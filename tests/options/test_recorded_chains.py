# SPDX-License-Identifier: Apache-2.0
"""Recorded option chains replace modelled quotes on recorded sessions, and only there."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

from signalquarry._internal.data.synthetic import synthetic_dataset
from signalquarry._internal.engine.run import simulate
from signalquarry._internal.options.chains import ChainModel, RecordedChain
from signalquarry._internal.options.contracts import parse_occ
from signalquarry.sdk import definition_of
from tests.options.wheel_reference import WheelParams, wheel, wheel_spec

DATA = synthetic_dataset(date(2022, 1, 3), date(2022, 12, 30), symbols=("QQQ",))


def test_recorded_chain_prefers_recordings_and_falls_back() -> None:
    session = date(2022, 3, 28)
    model = ChainModel(
        "QQQ",
        spot=Decimal(100),
        session=session,
        at=datetime(2022, 3, 28, 13, 35, tzinfo=UTC),
        sigma=0.3,
        tick=Decimal("0.01"),
    )
    chain = RecordedChain(
        {"QQQ220408P00095000": (Decimal("1.10"), Decimal("1.20")), "BAD": (Decimal(1), Decimal(1))}, model
    )
    recorded = chain.quote(parse_occ("QQQ220408P00095000"))
    assert (recorded.bid, recorded.ask, recorded.timestamp) == (Decimal("1.10"), Decimal("1.20"), model.at)
    chain.quote(parse_occ("QQQ220408P00090000"))  # not recorded: modelled
    assert (chain.recorded, chain.modelled) == (1, 1)
    puts = chain.candidates("PUT", 7, 14, Decimal(95))
    assert [c.contract.symbol for c in puts] == ["QQQ220408P00095000"]
    calls = chain.candidates("CALL", 7, 14, Decimal(105))  # none recorded: the model's candidates
    assert calls and all(c.contract.right == "CALL" for c in calls)


def test_simulator_fills_at_recorded_quotes() -> None:
    spec, definition, params = wheel_spec(), definition_of(wheel), WheelParams()
    base = simulate(spec, definition, params, DATA)
    opened = next(d for d in base.decisions if d.get("action") == "open" and d.get("outcome") == "filled")
    session = date.fromisoformat(opened["session"])
    symbol = opened["contract"]
    recorded = {("QQQ", session): {symbol: (Decimal("9.99"), Decimal("10.01"))}}
    run = simulate(spec, definition, params, DATA, recorded_chains=recorded)
    fill = next(f for f in run.fills if f.symbol == symbol and f.session == session)
    assert fill.price >= Decimal("9.9")  # the recorded bid, not the model's price
    assert "OPTIONS_RECORDED_CHAINS_USED:1" in run.warnings
    assert all(not w.startswith("OPTIONS_RECORDED") for w in base.warnings)
