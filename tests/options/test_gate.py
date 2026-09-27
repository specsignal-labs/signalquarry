# SPDX-License-Identifier: Apache-2.0
"""1.0 gate: plan invariants hold for every entry plan the reference wheel produces."""

from __future__ import annotations

import importlib.util
import sys
import uuid
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from signalquarry._internal.contracts.spec import StrategySpecV1
from signalquarry._internal.data.synthetic import synthetic_dataset
from signalquarry._internal.engine import options_sim
from signalquarry._internal.engine.run import simulate
from signalquarry._internal.options.resolver import plan_invariants
from signalquarry.sdk import definition_of

EXAMPLE = Path(__file__).parents[2] / "examples" / "wheel_reference"


def _example():
    name = f"wheel_reference_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, EXAMPLE / "strategy.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    body = yaml.safe_load((EXAMPLE / "strategy.yaml").read_text())
    return module, StrategySpecV1.model_validate(body)


@pytest.mark.parametrize("seed", [3, 7, 11])
def test_every_reference_wheel_plan_satisfies_the_invariants(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, spec = _example()
    plans = []
    original = options_sim.resolve

    def recording(selection, **kwargs):
        plan = original(selection, **kwargs)
        if not isinstance(plan, tuple):
            plans.append((plan, selection, kwargs["today"], kwargs["tick"]))
        return plan

    monkeypatch.setattr(options_sim, "resolve", recording)
    dataset = synthetic_dataset(date(2021, 1, 4), date(2024, 12, 31), seed=seed, symbols=("QQQ",))
    definition = definition_of(module.decide)
    for params in (
        module.WheelParams(),
        module.WheelParams(min_dte=1, max_dte=45, take_profit=Decimal("0.5")),
    ):
        result = simulate(spec, definition, params, dataset)
        assert result.decisions
    assert len(plans) > 50
    problems = [
        (plan.contract.symbol, code)
        for plan, selection, today, tick in plans
        for code in plan_invariants(plan, selection, today=today, tick=tick)
    ]
    assert problems == []


def test_example_matches_the_init_template() -> None:
    template = Path(__file__).parents[2] / "src/signalquarry/templates/strategies/wheel/strategy.py"
    rendered = template.read_text().replace("{{underlying}}", "QQQ")
    assert rendered == (EXAMPLE / "strategy.py").read_text()
