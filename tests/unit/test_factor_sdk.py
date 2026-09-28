# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import replace
from datetime import date
from types import MappingProxyType

import numpy as np
import pytest

from signalquarry._internal.data.panel import LoadedPanel
from signalquarry._internal.engine.factors import factor_context, run_factor
from signalquarry.sdk import FactorCtx, Params, factor, factor_definition_of, xs


class P(Params):
    period: int = 2


@factor(params=P, lookback=lambda p: p.period)
def latest_close(ctx: FactorCtx, p: P) -> dict[str, float]:
    del p
    close = ctx.panel("close")[-1]
    return {
        symbol: float(value) for symbol, value in zip(ctx.universe, close, strict=True) if np.isfinite(value)
    }


def _loaded() -> LoadedPanel:
    close = np.array([[1_000_000, 2_000_000], [3_000_000, 0], [5_000_000, 6_000_000]])
    micro = MappingProxyType({field: close.copy() for field in ("open", "high", "low", "close")})
    return LoadedPanel(
        dataset_identity="sha256:" + "0" * 64,
        sessions=(date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)),
        symbols=("A", "B"),
        micro=micro,
        volume=np.array([[10.0, 20.0], [30.0, 0.0], [50.0, 60.0]]),
        present=np.array([[True, True], [True, False], [True, True]]),
    )


def test_factor_uses_declared_history_and_explicit_universe() -> None:
    window = _loaded().window(decision_session=date(2024, 1, 4))
    context = factor_context(window, ("B", "A"), lookback=2)
    assert context.sessions == (date(2024, 1, 2), date(2024, 1, 3))
    assert context.universe == ("B", "A")
    assert np.isnan(context.panel("close")[-1, 0])
    assert context.panel("close")[-1, 1] == 3.0
    assert not context.panel("present")[-1, 0]
    assert context.panel("volume")[0, 0] == 20.0
    with pytest.raises(ValueError):
        context.panel("close").flags.writeable = True
    with pytest.raises(ValueError):
        context.panel("close")[-1, 1] = 999
    with pytest.raises(TypeError):
        context._panels["close"] = np.empty((0, 0))
    with pytest.raises(KeyError, match="FACTOR_PANEL_UNKNOWN"):
        context.panel("future")
    assert dict(run_factor(factor_definition_of(latest_close), P(), window, ("B", "A"))) == {"A": 3.0}


def test_future_prices_cannot_change_an_earlier_score() -> None:
    loaded = _loaded()
    decision = date(2024, 1, 4)
    before = run_factor(
        factor_definition_of(latest_close), P(), loaded.window(decision_session=decision), ("A",)
    )
    loaded.micro["close"][-1, :] = 999_000_000
    after = run_factor(
        factor_definition_of(latest_close), P(), loaded.window(decision_session=decision), ("A",)
    )
    assert before == after == {"A": 3.0}


def test_factor_contract_rejects_bad_context_and_scores() -> None:
    window = _loaded().window(decision_session=date(2024, 1, 4))
    definition = factor_definition_of(latest_close)
    with pytest.raises(TypeError, match="NOT_A_FACTOR"):
        factor_definition_of(lambda: None)
    with pytest.raises(TypeError, match="FACTOR_PARAMS_INVALID"):
        run_factor(definition, Params(), window, ("A",))
    with pytest.raises(ValueError, match="FACTOR_UNIVERSE_INVALID"):
        factor_context(window, ("A", "A"), lookback=1)
    with pytest.raises(ValueError, match="FACTOR_INSUFFICIENT_HISTORY"):
        factor_context(window, ("A",), lookback=3)
    with pytest.raises(ValueError, match="FACTOR_LOOKAHEAD"):
        factor_context(replace(window, sessions=(date(2024, 1, 4),)), ("A",), lookback=1)

    @factor(params=P, lookback=lambda p: 1)
    def undeclared(ctx: FactorCtx, p: P) -> dict[str, float]:
        return {"Z": 1.0}

    @factor(params=P, lookback=lambda p: 1)
    def nonfinite(ctx: FactorCtx, p: P) -> dict[str, float]:
        return {"A": float("nan")}

    with pytest.raises(ValueError, match="FACTOR_SYMBOL_UNDECLARED"):
        run_factor(factor_definition_of(undeclared), P(), window, ("A",))
    with pytest.raises(ValueError, match="FACTOR_SCORE_NONFINITE"):
        run_factor(factor_definition_of(nonfinite), P(), window, ("A",))


def test_cross_sectional_operators_handle_missing_and_ties() -> None:
    values = np.array([1.0, 2.0, 2.0, np.nan, 9.0])
    np.testing.assert_allclose(xs.rank(values), [0.125, 0.5, 0.5, np.nan, 0.875], equal_nan=True)
    np.testing.assert_allclose(xs.demean(values), [-2.5, -1.5, -1.5, np.nan, 5.5], equal_nan=True)
    assert np.isnan(xs.zscore(np.array([4.0, 4.0]))).all()
    np.testing.assert_allclose(xs.winsorize(values, 0.25, 0.75), [1.75, 2, 2, np.nan, 3.75], equal_nan=True)
    assert np.array_equal(values[:3], [1.0, 2.0, 2.0])


def test_neutralize_removes_linear_exposure_and_refuses_bad_design() -> None:
    exposure = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    score = np.array([3.0, 5.0, 7.0, 9.0, 99.0])
    np.testing.assert_allclose(xs.neutralize(score, exposure)[:4], 0.0, atol=1e-12)
    assert np.isnan(xs.neutralize(score, exposure)[-1])
    with pytest.raises(ValueError, match="FACTOR_EXPOSURE_RANK_DEFICIENT"):
        xs.neutralize(np.array([1.0, 2.0, 3.0]), np.ones(3))
    with pytest.raises(ValueError, match="FACTOR_EXPOSURE_SHAPE"):
        xs.neutralize(score, np.ones((3, 1)))
