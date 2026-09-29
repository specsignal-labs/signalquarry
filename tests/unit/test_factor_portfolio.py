# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal
from types import MappingProxyType
from typing import cast

import numpy as np
import pytest

from signalquarry._internal.contracts.reason_codes import lookup
from signalquarry.sdk import Bars, Ctx, FactorInput, FactorWeightSnapshot, Params, factor, factor_portfolio
from signalquarry.sdk.factors import FactorCtx


class P(Params):
    period: int = 2


@factor(params=P, lookback=lambda params: params.period)
def recent_close(ctx: FactorCtx, params: P) -> dict[str, float]:
    del params
    values = ctx.panel("close")[-1]
    return {
        symbol: float(value) for symbol, value in zip(ctx.universe, values, strict=True) if np.isfinite(value)
    }


@factor(params=P, lookback=lambda params: params.period)
def middle_name(ctx: FactorCtx, params: P) -> dict[str, float]:
    del params
    return {symbol: float(symbol == "B") for symbol in ctx.universe}


def _ctx(
    closes: dict[str, tuple[float, ...]],
    *,
    held: dict[str, Decimal] | None = None,
    session: date = date(2024, 1, 4),
) -> Ctx:
    bars = {}
    for symbol, values in closes.items():
        close = np.asarray(values, dtype=np.float64)
        sessions = np.array(["2024-01-02", "2024-01-03"][-len(close) :], dtype="datetime64[D]")
        bars[symbol] = Bars(
            symbol,
            sessions,
            close.copy(),
            close.copy(),
            close.copy(),
            close.copy(),
            np.full(len(close), 100_000.0),
        )
    return Ctx(
        decision_session=session,
        _bars=MappingProxyType(bars),
        positions=MappingProxyType({}),
        weights=MappingProxyType(held or {}),
        cash=Decimal("10000"),
        equity=Decimal("10000"),
        state=MappingProxyType({}),
    )


def test_equal_zscore_composition_returns_ordinary_capped_decision() -> None:
    ctx = _ctx({"A": (1, 3), "B": (1, 2), "C": (1, 1)})
    decision = factor_portfolio(
        ctx,
        (
            FactorInput("close", recent_close, P()),
            FactorInput("middle", middle_name, P()),
        ),
        top_quantile=0.25,
        max_weight_per_symbol=0.2,
    )

    assert decision.action == "target"
    assert decision.weights == {"B": Decimal("0.200000")}
    assert sum(decision.weights.values()) <= Decimal(1)


def test_rank_buffer_retains_current_holding_until_it_leaves_exit_band() -> None:
    ctx = _ctx(
        {"A": (1, 4), "B": (1, 3), "C": (1, 2), "D": (1, 1)},
        held={"B": Decimal("0.25")},
    )
    factor_input = (FactorInput("close", recent_close, P()),)
    retained = factor_portfolio(
        ctx,
        factor_input,
        top_quantile=0.25,
        turnover_buffer=0.25,
        max_weight_per_symbol=1.0,
    )
    assert retained.weights == {"B": Decimal("1.000000")}

    outside_band = _ctx(
        {"A": (1, 4), "B": (1, 3), "C": (1, 2), "D": (1, 1)},
        held={"C": Decimal("0.25")},
    )
    rotated = factor_portfolio(
        outside_band,
        factor_input,
        top_quantile=0.25,
        turnover_buffer=0.25,
        max_weight_per_symbol=1.0,
    )
    assert rotated.weights == {"A": Decimal("1.000000")}

    two_retained = _ctx(
        {"A": (1, 4), "B": (1, 3), "C": (1, 2), "D": (1, 1)},
        held={"A": Decimal("0.25"), "C": Decimal("0.25")},
    )
    filled = factor_portfolio(
        two_retained,
        factor_input,
        top_quantile=0.75,
        turnover_buffer=0.25,
        max_weight_per_symbol=1.0,
    )
    assert set(filled.weights) == {"A", "B", "C"}


def test_weight_schedule_uses_only_prior_knowledge() -> None:
    ctx = _ctx({"A": (1, 3), "B": (1, 2), "C": (1, 1)})
    factor_inputs = (
        FactorInput("close", recent_close, P()),
        FactorInput("middle", middle_name, P()),
    )
    future = FactorWeightSnapshot(
        known_at=date(2024, 1, 2),
        effective_from=date(2024, 1, 5),
        weights={"close": 1.0},
    )
    unavailable = factor_portfolio(
        ctx,
        factor_inputs,
        top_quantile=0.25,
        max_weight_per_symbol=1.0,
        weight_schedule=(future,),
    )
    assert unavailable.action == "unavailable"
    assert unavailable.reason_codes == ("FACTOR_PORTFOLIO_UNAVAILABLE",)

    prior = FactorWeightSnapshot(
        known_at=date(2024, 1, 2),
        effective_from=date(2024, 1, 3),
        weights={"close": 1.0, "middle": 0.0},
    )
    selected = factor_portfolio(
        ctx,
        factor_inputs,
        top_quantile=0.25,
        max_weight_per_symbol=1.0,
        weight_schedule=(prior,),
    )
    assert selected.weights == {"A": Decimal("1.000000")}


def test_factor_context_is_bounded_immutable_and_unavailable_without_scores() -> None:
    @factor(params=P, lookback=lambda params: params.period)
    def constant(ctx: FactorCtx, params: P) -> dict[str, float]:
        del params
        with pytest.raises(ValueError):
            ctx.panel("close")[-1, 0] = 999.0
        assert ctx.sessions == (date(2024, 1, 2), date(2024, 1, 3))
        assert all(session < ctx.decision_session for session in ctx.sessions)
        return {symbol: 1.0 for symbol in ctx.universe}

    result = factor_portfolio(
        _ctx({"A": (1, 1), "B": (1, 1)}),
        (FactorInput("constant", constant, P()),),
        top_quantile=0.5,
        max_weight_per_symbol=0.5,
    )
    assert result.action == "unavailable"
    assert result.reason_codes == ("FACTOR_PORTFOLIO_UNAVAILABLE",)


def test_invalid_schedule_and_config_fail_closed() -> None:
    ctx = _ctx({"A": (1, 2), "B": (1, 1)})
    factor_inputs = (FactorInput("close", recent_close, P()),)
    same_day = FactorWeightSnapshot(
        known_at=date(2024, 1, 4),
        effective_from=date(2024, 1, 4),
        weights={"close": 1.0},
    )
    with pytest.raises(ValueError, match="FACTOR_WEIGHT_SCHEDULE_INVALID"):
        factor_portfolio(
            ctx,
            factor_inputs,
            top_quantile=0.5,
            max_weight_per_symbol=1.0,
            weight_schedule=(same_day,),
        )
    with pytest.raises(ValueError, match="FACTOR_PORTFOLIO_CONFIG_INVALID"):
        factor_portfolio(ctx, factor_inputs, top_quantile=0, max_weight_per_symbol=1.0)
    with pytest.raises(ValueError, match="FACTOR_PORTFOLIO_CONFIG_INVALID"):
        factor_portfolio(ctx, factor_inputs, top_quantile=True, max_weight_per_symbol=1.0)
    with pytest.raises(ValueError, match="FACTOR_WEIGHT_CAP_TOO_SMALL"):
        factor_portfolio(
            ctx,
            factor_inputs,
            top_quantile=1.0,
            max_weight_per_symbol=0.0000005,
        )
    assert lookup("FACTOR_PORTFOLIO_UNAVAILABLE") is not None


def test_factor_context_fails_closed_on_empty_short_misaligned_and_future_data() -> None:
    factor_inputs = (FactorInput("close", recent_close, P()),)
    options = {"top_quantile": 0.5, "max_weight_per_symbol": 1.0}

    with pytest.raises(ValueError, match="FACTOR_UNIVERSE_INVALID"):
        factor_portfolio(_ctx({}), factor_inputs, **options)
    short = factor_portfolio(_ctx({"A": (3.0,)}), factor_inputs, **options)
    assert short.action == "unavailable"
    assert short.reason_codes == ("INSUFFICIENT_HISTORY",)

    aligned = _ctx({"A": (1, 2), "B": (2, 3)})
    bars = dict(aligned._bars)
    bars["B"] = replace(bars["B"], sessions=np.array(["2024-01-01", "2024-01-03"], dtype="datetime64[D]"))
    with pytest.raises(ValueError, match="FACTOR_PANEL_SESSIONS_INVALID"):
        factor_portfolio(replace(aligned, _bars=MappingProxyType(bars)), factor_inputs, **options)

    future = _ctx({"A": (1, 2), "B": (2, 3)}, session=date(2024, 1, 3))
    with pytest.raises(ValueError, match="FACTOR_LOOKAHEAD"):
        factor_portfolio(future, factor_inputs, **options)


def test_factor_inputs_and_lookback_validation() -> None:
    ctx = _ctx({"A": (1, 3), "B": (1, 2)})
    options = {"top_quantile": 0.5, "max_weight_per_symbol": 1.0}
    with pytest.raises(ValueError, match="FACTOR_INPUTS_EMPTY"):
        factor_portfolio(ctx, (), **options)
    with pytest.raises(TypeError, match="FACTOR_INPUT_INVALID"):
        factor_portfolio(ctx, cast(tuple[FactorInput, ...], ("invalid",)), **options)
    with pytest.raises(ValueError, match="FACTOR_PORTFOLIO_ID_INVALID"):
        factor_portfolio(ctx, (FactorInput("Bad ID", recent_close, P()),), **options)
    with pytest.raises(ValueError, match="FACTOR_PORTFOLIO_ID_DUPLICATE"):
        factor_portfolio(
            ctx,
            (FactorInput("close", recent_close, P()), FactorInput("close", middle_name, P())),
            **options,
        )
    with pytest.raises(TypeError, match="NOT_A_FACTOR"):
        factor_portfolio(ctx, (FactorInput("plain", lambda *_: {}, P()),), **options)
    with pytest.raises(TypeError, match="FACTOR_PARAMS_INVALID"):
        factor_portfolio(ctx, (FactorInput("close", recent_close, Params()),), **options)

    @factor(params=P, lookback=lambda params: 0)
    def invalid_lookback(ctx: FactorCtx, params: P) -> dict[str, float]:
        del ctx, params
        return {}

    with pytest.raises(ValueError, match="FACTOR_LOOKBACK_INVALID"):
        factor_portfolio(ctx, (FactorInput("bad", invalid_lookback, P()),), **options)


@pytest.mark.parametrize(
    ("weights", "error"),
    [
        ({}, "FACTOR_WEIGHT_SCHEDULE_INVALID"),
        ({"unknown": 1.0}, "FACTOR_WEIGHT_ID_UNKNOWN"),
        ({"close": -1.0}, "FACTOR_WEIGHT_INVALID"),
        ({"close": True}, "FACTOR_WEIGHT_INVALID"),
        ({"close": 0.0}, "FACTOR_WEIGHT_INVALID"),
    ],
)
def test_weight_rows_reject_empty_unknown_negative_and_zero_weights(
    weights: dict[str, object], error: str
) -> None:
    ctx = _ctx({"A": (1, 3), "B": (1, 2)})
    snapshot = FactorWeightSnapshot(
        known_at=date(2024, 1, 1),
        effective_from=date(2024, 1, 2),
        weights=cast(dict[str, float], weights),
    )
    with pytest.raises((TypeError, ValueError), match=error):
        factor_portfolio(
            ctx,
            (FactorInput("close", recent_close, P()),),
            top_quantile=0.5,
            max_weight_per_symbol=1.0,
            weight_schedule=(snapshot,),
        )


def test_weight_rows_reject_overflowing_totals_and_unsorted_dates() -> None:
    ctx = _ctx({"A": (1, 3), "B": (1, 2)})
    inputs = (FactorInput("close", recent_close, P()), FactorInput("middle", middle_name, P()))
    overflowing = FactorWeightSnapshot(
        known_at=date(2024, 1, 1),
        effective_from=date(2024, 1, 2),
        weights={"close": 1e308, "middle": 1e308},
    )
    with pytest.raises(ValueError, match="FACTOR_WEIGHT_INVALID"):
        factor_portfolio(
            ctx,
            inputs,
            top_quantile=0.5,
            max_weight_per_symbol=1.0,
            weight_schedule=(overflowing,),
        )
    later = FactorWeightSnapshot(date(2024, 1, 1), date(2024, 1, 3), {"close": 1.0})
    earlier = FactorWeightSnapshot(date(2024, 1, 1), date(2024, 1, 2), {"close": 1.0})
    with pytest.raises(ValueError, match="FACTOR_WEIGHT_SCHEDULE_INVALID"):
        factor_portfolio(
            ctx,
            inputs,
            top_quantile=0.5,
            max_weight_per_symbol=1.0,
            weight_schedule=(later, earlier),
        )

    with pytest.raises(TypeError, match="FACTOR_WEIGHT_SCHEDULE_INVALID"):
        factor_portfolio(
            ctx,
            inputs,
            top_quantile=0.5,
            max_weight_per_symbol=1.0,
            weight_schedule=cast(tuple[FactorWeightSnapshot, ...], (object(),)),
        )
