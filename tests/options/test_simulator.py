# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import uuid
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from signalquarry._internal.data.synthetic import synthetic_dataset
from signalquarry._internal.engine.backtest import EngineError
from signalquarry._internal.engine.run import simulate
from signalquarry._internal.options.chains import black_scholes, fridays
from signalquarry._internal.validation.conformance import perturb_after
from signalquarry.sdk import definition_of
from signalquarry.sdk.options import OD, OptionsCtx, options_strategy
from tests.options.wheel_reference import CODES, WheelParams, wheel, wheel_spec

DATA = synthetic_dataset(date(2021, 1, 4), date(2024, 12, 31), symbols=("QQQ",))


def _run(**kwargs):
    return simulate(wheel_spec(), definition_of(wheel), WheelParams(), DATA, **kwargs)


def test_black_scholes_parity_and_expirations() -> None:
    call = black_scholes(100, 100, 0.25, 0.2, "CALL")
    put = black_scholes(100, 100, 0.25, 0.2, "PUT")
    assert abs((call - put) - (100 - 100 * np.exp(-0.04 * 0.25))) < 1e-9
    assert black_scholes(100, 90, 0, 0.2, "PUT") == 0 and black_scholes(100, 110, 0, 0.2, "PUT") == 10
    assert all(day.weekday() == 4 for day in fridays(date(2026, 9, 28), 30))


def test_the_wheel_runs_through_its_whole_cycle() -> None:
    result = _run()
    outcomes = [d.get("outcome") for d in result.decisions]
    actions = {d["action"] for d in result.decisions}
    contracts = {d.get("contract", "")[3:9] for d in result.decisions if d.get("outcome") == "filled"}
    assert "filled" in outcomes and {"open", "close", "hold"} <= actions and len(contracts) > 10
    assert "assigned" in actions
    assert all(c >= 0 for c in result.cash), "cash-secured: cash never goes negative"
    shares = [f for f in result.fills if f.symbol == "QQQ"]
    assert shares and all(f.quantity == 100 for f in shares)
    assert result.positions.get("QQQ", Decimal(0)) in (Decimal(0), Decimal(100))
    assert _run().ledger_hash == result.ledger_hash  # deterministic


def test_legs_held_to_expiry_expire_or_are_assigned() -> None:
    result = simulate(wheel_spec(), definition_of(wheel), WheelParams(take_profit=Decimal("0.99")), DATA)
    actions = [d["action"] for d in result.decisions]
    assert "expired" in actions and "assigned" in actions
    assert not any(d.get("action") == "close" and d.get("outcome") == "filled" for d in result.decisions)


def test_costs_and_haircuts_reduce_equity() -> None:
    base, expensive = _run(), _run(cost_multiplier=Decimal(3))
    assert expensive.equity[-1] < base.equity[-1]


def test_entry_cutoff_and_state_rules() -> None:
    late = wheel_spec(options={"underlyings": ["QQQ"], "entry_cutoff": "09:30"})
    result = simulate(late, definition_of(wheel), WheelParams(), DATA)
    assert not any(f.side == "sell" and len(f.symbol) > 6 for f in result.fills)
    assert {d.get("why") for d in result.decisions if d.get("outcome") == "refused"} >= {
        "OPTIONS_ENTRY_CUTOFF"
    }


@options_strategy(params=WheelParams, lookback=lambda p: p.trend_sessions)
def peeks_at_today(ctx: OptionsCtx, p: WheelParams) -> object:
    # A leak: at the open checkpoint, compare the (legitimate) spot with the close of the same day.
    closes = ctx.bars(p.underlying).close
    return OD.hold("HOLD_LEG", state={"x": float(closes[-1] > float(ctx.spot(p.underlying)))})


def test_contract_errors() -> None:
    with pytest.raises(EngineError, match="STRATEGY_KIND_MISMATCH"):
        simulate(wheel_spec(kind="equity_daily", options=None), definition_of(wheel), WheelParams(), DATA)

    @options_strategy(params=WheelParams, lookback=lambda p: 5)
    def undeclared(ctx: OptionsCtx, p: WheelParams):
        return OD.hold("NOT_DECLARED")

    with pytest.raises(EngineError, match="REASON_CODE_UNDECLARED"):
        simulate(wheel_spec(), definition_of(undeclared), WheelParams(), DATA)


def test_options_lookahead_check_catches_same_day_leaks() -> None:
    cut = len(DATA.sessions) // 2
    clean = _run()
    mutated = simulate(
        wheel_spec(),
        definition_of(wheel),
        WheelParams(),
        perturb_after(DATA, cut, 0.5, 0.8, keep_open_at_cut=True),
    )
    day = DATA.sessions[cut].isoformat()

    def known(d):
        return d["session"] < day or (d["session"] == day and d.get("checkpoint") == "open")

    assert [d for d in clean.decisions if known(d)] == [d for d in mutated.decisions if known(d)]


def _sqy(capsys, *argv):
    from signalquarry.cli.main import main

    code = main(["--json", *argv])
    return code, json.loads(capsys.readouterr().out)


def test_cli_check_backtest_evaluate_for_options(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    package = "wheel_" + uuid.uuid4().hex[:8]
    project = tmp_path / "wheel-lab"
    _sqy(capsys, "init", str(project), "--demo", "--package", package)
    target = project / "src" / package / "wheel_reference"
    target.mkdir()
    (target / "__init__.py").write_text("")
    (target / "strategy.py").write_text(
        (Path(__file__).parent / "wheel_reference.py")
        .read_text()
        .replace("from __future__ import annotations\n", "from __future__ import annotations\n", 1)
    )
    spec = wheel_spec().model_dump(mode="json", by_alias=True, exclude_none=True)
    import yaml

    (target / "strategy.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    config = project / "signalquarry.toml"
    config.write_text(
        config.read_text().replace(
            f'modules = ["{package}.sma_trend.strategy"]',
            f'modules = ["{package}.sma_trend.strategy", "{package}.wheel_reference.strategy"]',
        )
    )
    code, payload = _sqy(capsys, "check", "--strategy", "wheel-reference", "--project", str(project))
    assert code == 0, payload
    code, payload = _sqy(capsys, "backtest", "--strategy", "wheel-reference", "--project", str(project))
    assert code == 0 and payload["evidence"]["grade"] == "synthetic", payload
    code, payload = _sqy(capsys, "evaluate", "--strategy", "wheel-reference", "--project", str(project))
    assert "delay_1" not in payload["data"]["gates"]["G3_stress"]["scenarios"]
    assert payload["evidence"]["claim_level"] == "none"


def test_leaky_options_strategy_fails_check(tmp_path: Path) -> None:
    from signalquarry._internal.project.project import LoadedStrategy
    from signalquarry._internal.validation.conformance import run_checks

    fake = LoadedStrategy(
        wheel_spec(),
        definition_of(peeks_at_today),
        WheelParams(),
        __import__("tests.options.test_simulator"),
        Path(__file__).parent,
    )
    checks = {c.name: c.ok for c in run_checks(fake)}
    assert checks["lookahead"] is True  # comparing yesterday's close with today's open is legitimate
    _ = replace, CODES


def test_off_by_one_context_builder_is_caught_for_options(monkeypatch: pytest.MonkeyPatch) -> None:
    from signalquarry._internal.engine import backtest as engine
    from signalquarry._internal.project.project import LoadedStrategy
    from signalquarry._internal.validation.conformance import run_checks

    original = engine._Views.bars

    def leaky(self, symbol: str, start: int, stop: int):  # includes the bar of the session being decided
        return original(self, symbol, start + 1, min(stop + 1, len(self.dataset.sessions)))

    monkeypatch.setattr(engine._Views, "bars", leaky)
    strategy = LoadedStrategy(
        wheel_spec(),
        definition_of(wheel),
        WheelParams(),
        __import__("tests.options.wheel_reference"),
        Path(__file__).parent,
    )
    assert {c.name: c.ok for c in run_checks(strategy)}["lookahead"] is False
