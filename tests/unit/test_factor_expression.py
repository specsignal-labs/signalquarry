# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from types import MappingProxyType

import numpy as np
import pytest

from signalquarry._internal.factors.expr import evaluate_expression, parse_expression
from signalquarry.sdk.factors import FactorCtx


def _context(
    *,
    close: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    sessions: tuple[date, ...] | None = None,
) -> FactorCtx:
    closes = np.array(
        [[1.0, 10.0], [2.0, 20.0], [4.0, 40.0], [8.0, 30.0], [16.0, 50.0]] if close is None else close,
        dtype=np.float64,
    )
    volumes = np.array(
        [[2.0, 1.0], [4.0, 2.0], [8.0, 3.0], [16.0, 4.0], [32.0, 5.0]] if volume is None else volume,
        dtype=np.float64,
    )
    if sessions is None:
        sessions = tuple(date(2024, 1, 2) + timedelta(days=index) for index in range(len(closes)))
    return FactorCtx(
        decision_session=date(2024, 1, 20),
        sessions=sessions,
        universe=("A", "B"),
        _panels=MappingProxyType(
            {
                "close": closes,
                "open": closes,
                "high": closes,
                "low": closes,
                "volume": volumes,
            }
        ),
    )


def _evaluate(
    source: str, context: FactorCtx | None = None, eligible: np.ndarray | None = None
) -> np.ndarray:
    selected = context or _context()
    if eligible is None:
        eligible = np.ones((len(selected.sessions), len(selected.universe)), dtype=np.bool_)
    return evaluate_expression(parse_expression(source), selected, eligible=eligible)


def test_expression_identity_is_stable_across_formatting() -> None:
    first = parse_expression("ts_mean(close, 3) / delay(close, 1)")
    second = parse_expression(" ts_mean( close,3)/delay( close,1 ) ")
    assert first.canonical == second.canonical
    assert first.identity == second.identity
    assert first.identity.startswith("sha256:")


def test_trailing_and_cross_sectional_operators_match_reference_vectors() -> None:
    context = _context()
    close = context.panel("close")

    delayed = _evaluate("delay(close, 2)", context)
    assert np.isnan(delayed[:2]).all()
    np.testing.assert_array_equal(delayed[2:], close[:-2])

    delta = _evaluate("delta(close, 2)", context)
    assert np.isnan(delta[:2]).all()
    np.testing.assert_array_equal(delta[2:], close[2:] - close[:-2])

    mean = _evaluate("ts_mean(close, 3)", context)
    assert np.isnan(mean[:2]).all()
    np.testing.assert_allclose(mean[2], np.mean(close[:3], axis=0))

    deviation = _evaluate("ts_std(close, 3)", context)
    np.testing.assert_allclose(deviation[2], np.std(close[:3], axis=0))

    rank = _evaluate("ts_rank(close, 3)", context)
    np.testing.assert_allclose(rank[2], [5 / 6, 5 / 6])

    correlation = _evaluate("ts_corr(close, volume, 3)", context)
    assert correlation[2, 0] == pytest.approx(1.0)

    np.testing.assert_allclose(_evaluate("rank(close)", context)[0], [0.25, 0.75])
    np.testing.assert_allclose(_evaluate("zscore(close)", context)[0], [-1.0, 1.0])


def test_arithmetic_log_abs_and_sign_are_vectorized() -> None:
    context = _context()
    close = context.panel("close")
    changes = _evaluate("log(abs(delta(close, 1)))", context)
    assert np.isnan(changes[0]).all()
    np.testing.assert_allclose(changes[1], np.log(np.abs(close[1] - close[0])))
    signed = _evaluate("sign(close - delay(close, 1))", context)
    assert np.isnan(signed[0]).all()
    np.testing.assert_array_equal(signed[1:], np.sign(close[1:] - close[:-1]))

    divided_by_zero = _evaluate("close / (close - close)", context)
    assert np.isnan(divided_by_zero).all()
    constant_division_by_zero = _evaluate("close + 1 / 0", context)
    assert np.isnan(constant_division_by_zero).all()
    np.testing.assert_array_equal(_evaluate("close + +1", context), close + 1)
    np.testing.assert_array_equal(_evaluate("-(+close * 2)", context), -close * 2)
    np.testing.assert_array_equal(_evaluate("high - low", context), np.zeros_like(close))


def test_long_windows_and_constant_correlation_remain_unavailable() -> None:
    context = _context()
    assert np.isnan(_evaluate("delay(close, 6)", context)).all()
    assert np.isnan(_evaluate("ts_mean(close, 6)", context)).all()
    assert np.isnan(_evaluate("ts_corr(close, close * 0 + 1, 3)", context)).all()


def test_rolling_operators_require_complete_finite_windows() -> None:
    close = np.array([[1.0, 10.0], [np.nan, 20.0], [4.0, 40.0], [8.0, 30.0], [16.0, 50.0]])
    context = _context(close=close)
    mean = _evaluate("ts_mean(close, 2)", context)
    assert np.isnan(mean[1, 0]) and np.isnan(mean[2, 0])
    assert mean[3, 0] == pytest.approx(6.0)
    assert mean[1, 1] == pytest.approx(15.0)

    ranked = _evaluate("rank(close)", context)
    assert np.isnan(ranked[1, 0])
    assert ranked[1, 1] == pytest.approx(0.5)
    ranked_time_series = _evaluate("ts_rank(close, 2)", context)
    assert np.isnan(ranked_time_series[1, 0]) and np.isnan(ranked_time_series[2, 0])

    all_missing = _context(close=np.full((5, 2), np.nan))
    assert np.isnan(_evaluate("ts_rank(close, 2)", all_missing)).all()


def test_cross_sectional_operators_use_dated_membership() -> None:
    context = _context()
    eligible = np.ones((len(context.sessions), len(context.universe)), dtype=np.bool_)
    eligible[0, 1] = False
    eligible[1, 0] = False

    rank = _evaluate("rank(close)", context, eligible)
    assert rank[0, 0] == pytest.approx(0.5) and np.isnan(rank[0, 1])
    assert np.isnan(rank[1, 0]) and rank[1, 1] == pytest.approx(0.5)
    assert np.isnan(_evaluate("zscore(close)", context, eligible)[:2]).all()

    raw = _evaluate("close", context, eligible)
    assert np.isnan(raw[0, 1]) and np.isnan(raw[1, 0])


def test_each_time_row_ignores_later_rows() -> None:
    original = _context()
    close = original.panel("close").copy()
    altered = close.copy()
    altered[4] = [900.0, 901.0]
    changed = _context(close=altered)
    source = "ts_mean(close, 3) - delay(close, 1)"
    before = _evaluate(source, original)
    after = _evaluate(source, changed)
    np.testing.assert_allclose(before[:4], after[:4], equal_nan=True)
    assert not np.allclose(before[4], after[4], equal_nan=True)


@pytest.mark.parametrize(
    "source",
    [
        "",
        "1.0",
        "close.__class__",
        "unknown_field",
        "__import__('os').system('true')",
        "close[0]",
        "close ** 2",
        "close and volume",
        "close + True",
        "ts_mean(close, 0)",
        "ts_mean(close, -1)",
        "ts_mean(close, 253)",
        "ts_mean(close, 1 + 1)",
        "ts_corr(close, volume)",
        "rank(close, 2)",
        "log(close, base=2)",
        "close + 1000001",
        "close + 1e999",
        "close +",
    ],
)
def test_parser_rejects_constructs_outside_the_bounded_grammar(source: str) -> None:
    with pytest.raises(ValueError, match="FACTOR_EXPRESSION_INVALID"):
        parse_expression(source)


def test_parser_bounds_source_size_depth_and_node_count() -> None:
    with pytest.raises(ValueError, match="FACTOR_EXPRESSION_INVALID"):
        parse_expression("close" + " " * 508)
    with pytest.raises(ValueError, match="FACTOR_EXPRESSION_INVALID"):
        parse_expression("abs(" * 16 + "close" + ")" * 16)
    with pytest.raises(ValueError, match="FACTOR_EXPRESSION_INVALID"):
        parse_expression(" + ".join(["close"] * 40))
    with pytest.raises(ValueError, match="FACTOR_EXPRESSION_INVALID"):
        parse_expression("close + " + "9" * 400)


def test_evaluator_rejects_future_or_misaligned_context() -> None:
    expression = parse_expression("close")
    context = _context()
    eligible = np.ones((len(context.sessions), len(context.universe)), dtype=np.bool_)
    future_sessions = (*context.sessions[:-1], date(2024, 1, 20))
    with pytest.raises(ValueError, match="FACTOR_EXPRESSION_CONTEXT_INVALID"):
        evaluate_expression(expression, replace(context, sessions=future_sessions), eligible=eligible)

    with pytest.raises(ValueError, match="FACTOR_EXPRESSION_CONTEXT_INVALID"):
        evaluate_expression(expression, replace(context, universe=("A", "A")), eligible=eligible)

    bad_shape = dict(context._panels)
    bad_shape["close"] = np.zeros((1, 2))
    with pytest.raises(ValueError, match="FACTOR_EXPRESSION_CONTEXT_INVALID"):
        evaluate_expression(
            expression, replace(context, _panels=MappingProxyType(bad_shape)), eligible=eligible
        )
    with pytest.raises(ValueError, match="FACTOR_EXPRESSION_CONTEXT_INVALID"):
        evaluate_expression(expression, replace(context, _panels=MappingProxyType({})), eligible=eligible)

    infinite = dict(context._panels)
    invalid_close = context.panel("close").copy()
    invalid_close[0, 0] = np.inf
    infinite["close"] = invalid_close
    with pytest.raises(ValueError, match="FACTOR_EXPRESSION_CONTEXT_INVALID"):
        evaluate_expression(
            expression, replace(context, _panels=MappingProxyType(infinite)), eligible=eligible
        )

    with pytest.raises(ValueError, match="FACTOR_EXPRESSION_CONTEXT_INVALID"):
        evaluate_expression(expression, context, eligible=np.ones((1, 1), dtype=np.bool_))
    with pytest.raises(ValueError, match="FACTOR_EXPRESSION_CONTEXT_INVALID"):
        evaluate_expression(expression, context, eligible=np.ones((5, 2), dtype=np.int8))


def test_evaluator_returns_an_immutable_score_panel() -> None:
    scores = _evaluate("close + 1")
    assert scores.shape == (5, 2)
    assert not scores.flags.writeable
    with pytest.raises(ValueError):
        scores[0, 0] = 0.0

    with pytest.raises(TypeError, match="FACTOR_EXPRESSION_INVALID"):
        evaluate_expression(
            object(),
            _context(),
            eligible=np.ones((5, 2), dtype=np.bool_),
        )  # type: ignore[arg-type]
