# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import pytest

from signalquarry._internal.validation.conformance import import_policy
from signalquarry._internal.validation.factor_conformance import run_factor_checks
from signalquarry.api.project import check_factor
from signalquarry.sdk import FactorCtx, Params, factor, factor_definition_of


class FactorParams(Params):
    period: int = 3


@factor(params=FactorParams, lookback=lambda params: params.period)
def momentum(ctx: FactorCtx, params: FactorParams) -> dict[str, float]:
    del params
    close = ctx.panel("close")
    return {symbol: float(close[-1, i] / close[0, i] - 1) for i, symbol in enumerate(ctx.universe)}


def test_factor_synthetic_conformance_passes() -> None:
    checks = run_factor_checks(factor_definition_of(momentum), FactorParams())
    assert {item.name: item.ok for item in checks} == {
        "contract": True,
        "determinism": True,
        "lookahead": True,
    }


def test_factor_check_catches_state_and_contract_errors() -> None:
    calls = 0

    @factor(params=FactorParams, lookback=lambda params: params.period)
    def stateful(ctx: FactorCtx, params: FactorParams) -> dict[str, float]:
        nonlocal calls
        del params
        calls += 1
        return {ctx.universe[0]: float(calls)}

    checks = run_factor_checks(factor_definition_of(stateful), FactorParams())
    assert {item.name: item.ok for item in checks} == {
        "contract": True,
        "determinism": False,
        "lookahead": False,
    }

    @factor(params=FactorParams, lookback=lambda params: params.period)
    def invalid(ctx: FactorCtx, params: FactorParams) -> dict[str, float]:
        del ctx, params
        return {"OUTSIDE": 1.0}

    assert run_factor_checks(factor_definition_of(invalid), FactorParams())[0].ok is False


@pytest.mark.parametrize("fail_on, failed_check", ((4, "determinism"), (7, "lookahead")))
def test_factor_check_reports_invalid_universe_and_later_exceptions(fail_on: int, failed_check: str) -> None:
    assert run_factor_checks(factor_definition_of(momentum), FactorParams(), universe=())[0].ok is False
    calls = 0

    @factor(params=FactorParams, lookback=lambda params: params.period)
    def raises_later(ctx: FactorCtx, params: FactorParams) -> dict[str, float]:
        nonlocal calls
        del params
        calls += 1
        if calls == fail_on:
            raise RuntimeError("synthetic failure")
        return {ctx.universe[0]: 1.0}

    checks = run_factor_checks(factor_definition_of(raises_later), FactorParams())
    assert checks[-1].name == failed_check
    assert not checks[-1].ok


def test_project_factor_check_reports_results(tmp_path: Path) -> None:
    package = tmp_path / "src" / "example_factors"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "signalquarry.toml").write_text('[project]\nname = "example"\n', encoding="utf-8")
    (package / "signal.py").write_text(
        "from signalquarry.sdk import FactorCtx, Params, factor\n"
        "class P(Params):\n    period: int = 2\n"
        "@factor(params=P, lookback=lambda p: p.period)\n"
        "def score(ctx: FactorCtx, p: P) -> dict[str, float]:\n"
        "    return {symbol: float(ctx.panel('close')[-1, i]) for i, symbol in enumerate(ctx.universe)}\n",
        encoding="utf-8",
    )
    outcome = check_factor("example_factors.signal", project=tmp_path, params_json='{"period":3}')
    assert outcome.status == "ok"
    assert [item["name"] for item in outcome.data["checks"]] == [
        "import_policy",
        "contract",
        "determinism",
        "lookahead",
    ]
    assert check_factor("example_factors.signal", project=tmp_path, params_json="[]").status == "invalid"


def test_import_policy_rejects_internal_engine_access(tmp_path: Path) -> None:
    (tmp_path / "factor.py").write_text(
        "import signalquarry\nfrom signalquarry._internal.data import panel\n"
        "from signalquarry.sdk import factor\n",
        encoding="utf-8",
    )
    result = import_policy(tmp_path, "example_factors", sdk_only=True)
    assert not result.ok
    assert "import signalquarry;" in result.detail
    assert "signalquarry._internal.data" in result.detail
