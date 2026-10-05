# SPDX-License-Identifier: Apache-2.0
"""Independent references and boundary cases for the factor expression interpreter.

A mutation pass (hardening track A5) over ``factors/expr.py``, ``sdk/factors.py`` and
``engine/factors.py`` left survivors that the existing tests could not see: the sliding-window
updates were only checked on the first window, ``ts_std`` was never run on data with volatility
below one, expression text was never compared with the formula it came from, the parser limits
were never probed at their exact boundary, and nothing asserted that factor code receives only
its declared lookback when more history exists.

Every expectation below comes from a naive loop, a hand-written string or a documented limit,
not from the code under test.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from datetime import date, timedelta
from types import MappingProxyType

import numpy as np
import pytest

from signalquarry._internal.data.dataset import FIELDS, MICRO
from signalquarry._internal.data.panel import PanelWindow
from signalquarry._internal.engine.factors import factor_context, run_factor
from signalquarry._internal.factors.expr import (
    evaluate_expression,
    expression_complexity,
    parse_expression,
)
from signalquarry.sdk.factors import FactorCtx, checked_scores, factor, factor_definition_of
from signalquarry.sdk.strategy import Params

INVALID = r"^FACTOR_EXPRESSION_INVALID$"
BAD_CONTEXT = r"^FACTOR_EXPRESSION_CONTEXT_INVALID$"


def _context(close: np.ndarray, volume: np.ndarray | None = None) -> FactorCtx:
    closes = np.asarray(close, dtype=np.float64)
    volumes = closes.copy() if volume is None else np.asarray(volume, dtype=np.float64)
    rows, columns = closes.shape
    sessions = tuple(date(2024, 1, 2) + timedelta(days=index) for index in range(rows))
    return FactorCtx(
        decision_session=sessions[-1] + timedelta(days=1),
        sessions=sessions,
        universe=tuple(f"S{index}" for index in range(columns)),
        _panels=MappingProxyType(
            {"open": closes, "high": closes, "low": closes, "close": closes, "volume": volumes}
        ),
    )


def _evaluate(source: str, context: FactorCtx) -> np.ndarray:
    eligible = np.ones((len(context.sessions), len(context.universe)), dtype=np.bool_)
    return evaluate_expression(parse_expression(source), context, eligible=eligible)


def _research_panels() -> tuple[np.ndarray, np.ndarray]:
    """40 sessions x 4 symbols of return-sized values: a gap, a constant column, correlated volume."""
    rng = np.random.default_rng(7)
    close = rng.normal(0.0, 0.02, (40, 4))
    volume = rng.normal(0.0, 0.02, (40, 4)) + 0.6 * close
    close[5:9, 1] = np.nan  # a gap that later leaves the window
    volume[12:14, 2] = np.nan
    close[:, 3] = 0.5  # no dispersion at all
    return close, volume


def _reference(
    values: np.ndarray, period: int, summarize: Callable[[np.ndarray], float], other: np.ndarray | None = None
) -> np.ndarray:
    """Naive trailing window: defined only when every observation in it is finite."""
    result = np.full(values.shape, np.nan)
    for end in range(period - 1, len(values)):
        for column in range(values.shape[1]):
            window = values[end - period + 1 : end + 1, column]
            if other is None:
                if np.isfinite(window).all():
                    result[end, column] = summarize(window)
            else:
                partner = other[end - period + 1 : end + 1, column]
                if np.isfinite(window).all() and np.isfinite(partner).all():
                    result[end, column] = summarize(np.stack([window, partner]))
    return result


def _correlation(pair: np.ndarray) -> float:
    if np.var(pair[0]) < 1e-12 or np.var(pair[1]) < 1e-12:
        return math.nan
    return float(np.corrcoef(pair[0], pair[1])[0, 1])


def _assert_same(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-10, equal_nan=True)


@pytest.mark.parametrize("period", [1, 2, 3, 7, 40])
def test_ts_mean_matches_a_naive_window_on_every_window(period: int) -> None:
    close, volume = _research_panels()
    _assert_same(
        _evaluate(f"ts_mean(close, {period})", _context(close, volume)), _reference(close, period, np.mean)
    )


@pytest.mark.parametrize("period", [1, 2, 3, 7, 40])
def test_ts_std_matches_a_naive_population_std_on_every_window(period: int) -> None:
    # Values far below one: a variance floor of one would show here.
    close, volume = _research_panels()
    _assert_same(
        _evaluate(f"ts_std(close, {period})", _context(close, volume)), _reference(close, period, np.std)
    )


@pytest.mark.parametrize("period", [1, 2, 3, 7, 40])
def test_ts_corr_matches_a_naive_window_and_is_undefined_without_dispersion(period: int) -> None:
    close, volume = _research_panels()
    context = _context(close, volume)
    expected = _reference(close, period, _correlation, other=volume)
    actual = _evaluate(f"ts_corr(close, volume, {period})", context)
    _assert_same(actual, expected)
    assert np.isnan(actual[:, 3]).all()  # the constant column never has a defined correlation
    if np.isfinite(actual).any():
        assert np.nanmax(np.abs(actual)) <= 1.0


def test_ts_corr_is_undefined_when_only_one_series_is_constant() -> None:
    rng = np.random.default_rng(3)
    moving = rng.normal(0.0, 0.02, (12, 2))
    flat = np.full((12, 2), 0.5)
    assert np.isnan(_evaluate("ts_corr(close, volume, 5)", _context(moving, flat))).all()
    assert np.isnan(_evaluate("ts_corr(close, volume, 5)", _context(flat, moving))).all()


def test_ts_corr_of_a_series_with_itself_and_its_negation() -> None:
    rng = np.random.default_rng(5)
    values = rng.normal(0.0, 0.02, (15, 3))
    own = _evaluate("ts_corr(close, close, 4)", _context(values, values))
    opposite = _evaluate("ts_corr(close, volume, 4)", _context(values, -values))
    np.testing.assert_allclose(own[3:], 1.0, atol=1e-9)
    np.testing.assert_allclose(opposite[3:], -1.0, atol=1e-9)
    assert np.isnan(own[:3]).all()


def test_ts_rank_needs_a_complete_window_per_symbol_and_skips_empty_windows() -> None:
    nan = np.nan
    close = np.array(
        [
            [nan, nan],
            [nan, nan],
            [nan, nan],  # several windows with no complete column, then valid ones follow
            [1.0, 2.0],
            [3.0, 1.0],
            [2.0, 1.0],
            [5.0, nan],  # only the second column is incomplete here
            [4.0, 3.0],
            [4.0, 4.0],
        ]
    )
    actual = _evaluate("ts_rank(close, 3)", _context(close))
    expected = np.full(close.shape, nan)
    for end in range(2, len(close)):
        for column in range(close.shape[1]):
            window = close[end - 2 : end + 1, column]
            if np.isfinite(window).all():
                current = window[-1]
                expected[end, column] = (np.sum(window < current) + np.sum(window == current) / 2.0) / 3.0
    _assert_same(actual, expected)
    assert np.isfinite(actual[5]).all() and np.isnan(actual[6, 1]) and np.isfinite(actual[6, 0])


def test_log_is_nan_for_nonpositive_inputs_and_exact_elsewhere() -> None:
    close = np.array([[-1.0, 0.0], [0.5, 1.0], [2.0, math.e]])
    actual = _evaluate("log(close)", _context(close))
    expected = np.array([[math.nan, math.nan], [math.log(0.5), 0.0], [math.log(2.0), 1.0]])
    _assert_same(actual, expected)


def test_division_by_zero_becomes_nan_not_infinity() -> None:
    close = np.array([[1.0, 2.0], [1.0, 3.0], [4.0, 3.0]])
    # delta(close, 1) is [nan, nan], [0, 1], [3, 0]: two genuine divisions by zero (1/0 and 3/0)
    actual = _evaluate("close / delta(close, 1)", _context(close))
    expected = np.array([[math.nan, math.nan], [math.nan, 3.0], [4.0 / 3.0, math.nan]])
    _assert_same(actual, expected)
    assert np.isfinite(actual[np.isfinite(actual)]).all()


# Constant windows. Window statistics are computed from centred windows, not running sums, so a
# flat stretch of a price that is not exactly representable (123.456) has exactly zero dispersion
# and its mean is the price itself; running sums left rounding noise there (grammar version 1).


def _drifting_with_a_flat_stretch() -> np.ndarray:
    rng = np.random.default_rng(1)
    prices = 100.0 + np.cumsum(rng.normal(0.0, 0.5, (400, 2)), axis=0)
    prices[200:230] = prices[200]
    return prices


def test_ts_std_of_a_flat_stretch_is_exactly_zero() -> None:
    context = _context(_drifting_with_a_flat_stretch())
    flat = _evaluate("ts_std(close, 10)", context)[215:225]
    assert (flat == 0.0).all()


def test_dividing_by_the_std_of_a_flat_stretch_is_nan() -> None:
    context = _context(_drifting_with_a_flat_stretch())
    ratio = _evaluate("close / ts_std(close, 10)", context)[215:225]
    assert np.isnan(ratio).all()


def test_ts_corr_against_a_flat_price_series_is_undefined() -> None:
    rng = np.random.default_rng(2)
    flat = np.full((40, 1), 123.456)
    moving = rng.normal(0.0, 1.0, (40, 1)) + 100.0
    assert np.isnan(_evaluate("ts_corr(close, volume, 10)", _context(flat, moving))).all()


@pytest.mark.parametrize(
    ("source", "canonical"),
    [
        ("close+volume", "(close + volume)"),
        ("close-volume", "(close - volume)"),
        ("close*volume", "(close * volume)"),
        ("close/volume", "(close / volume)"),
        ("-close", "(-close)"),
        ("+close", "(+close)"),
        ("close + 2.5", "(close + 2.5)"),
        ("close + 2", "(close + 2.0)"),
        ("ts_corr(close, volume, 5)", "ts_corr(close, volume, 5)"),
        ("ts_mean(-close, 3)", "ts_mean((-close), 3)"),
        ("rank(delta(close, 1) / ts_std(volume, 20))", "rank((delta(close, 1) / ts_std(volume, 20)))"),
    ],
)
def test_canonical_text_is_exactly_the_formula_that_was_parsed(source: str, canonical: str) -> None:
    parsed = parse_expression(source)
    assert parsed.canonical == canonical
    # What a reviewer reads must parse back to the very same expression.
    assert parse_expression(parsed.canonical).identity == parsed.identity


def test_every_operator_changes_the_identity() -> None:
    sources = [
        "close + volume",
        "close - volume",
        "close * volume",
        "close / volume",
        "-close",
        "+close",
        "abs(close)",
        "sign(close)",
        "log(close)",
        "rank(close)",
        "zscore(close)",
        "delay(close, 2)",
        "delta(close, 2)",
        "ts_mean(close, 2)",
        "ts_std(close, 2)",
        "ts_rank(close, 2)",
        "ts_corr(close, volume, 2)",
    ]
    assert len({parse_expression(source).identity for source in sources}) == len(sources)


def test_complexity_counts_every_node_but_not_window_lengths() -> None:
    counts = {
        "close": 1,
        "ts_mean(close, 3)": 2,
        "close + 1": 3,
        "-(close * volume)": 4,
        "ts_corr(close, volume, 5)": 3,
        "rank(ts_mean(close, 3) - delay(volume, 1))": 6,
    }
    for source, expected in counts.items():
        assert expression_complexity(parse_expression(source)) == expected, source
    with pytest.raises(TypeError, match=INVALID):
        expression_complexity("close")  # type: ignore[arg-type]


def _balanced_sum(terms: list[str]) -> str:
    while len(terms) > 1:
        terms = [f"({terms[index]} + {terms[index + 1]})" for index in range(0, len(terms), 2)]
    return terms[0]


def test_node_limit_is_exactly_sixty_four() -> None:
    plain = _balanced_sum(["close"] * 32)  # 63 nodes
    parse_expression(f"-({plain})")  # 64 nodes: accepted
    with pytest.raises(ValueError, match=INVALID):
        parse_expression(f"-(-({plain}))")  # 65


def test_window_lengths_count_toward_the_node_limit() -> None:
    windows = _balanced_sum(["ts_mean(close, 3)"] * 16)  # 16 x 3 nodes + 15 additions = 63
    parse_expression(f"-({windows})")  # 64: accepted
    with pytest.raises(ValueError, match=INVALID):
        parse_expression(f"-(-({windows}))")  # 65


_NESTERS: dict[str, Callable[[str], str]] = {
    "unary": lambda inner: f"(-{inner})",
    "binary-left": lambda inner: f"({inner} + 1)",
    "binary-right": lambda inner: f"(1 + {inner})",
    "call": lambda inner: f"abs({inner})",
    "window": lambda inner: f"ts_mean({inner}, 2)",
    "corr-left": lambda inner: f"ts_corr({inner}, volume, 2)",
    "corr-right": lambda inner: f"ts_corr(volume, {inner}, 2)",
}


@pytest.mark.parametrize("construct", sorted(_NESTERS))
def test_depth_limit_is_exactly_sixteen_through_every_construct(construct: str) -> None:
    def nested(levels: int) -> str:
        source = "close"
        for _ in range(levels):
            source = _NESTERS[construct](source)
        return source

    parse_expression(nested(15))  # the leaf sits at depth 16: accepted
    with pytest.raises(ValueError, match=INVALID):
        parse_expression(nested(16))  # depth 17


def test_source_length_constant_and_window_limits_are_exact() -> None:
    parse_expression("close" + " " * 507)  # 512 characters
    with pytest.raises(ValueError, match=INVALID):
        parse_expression("close" + " " * 508)
    parse_expression("close + 1000000")
    with pytest.raises(ValueError, match=INVALID):
        parse_expression("close + 1000001")
    parse_expression("ts_mean(close, 252)")
    parse_expression("ts_mean(close, 1)")
    for bad in ("ts_mean(close, 253)", "ts_mean(close, 0)", "ts_mean(close, 2.0)", "ts_mean(close, True)"):
        with pytest.raises(ValueError, match=INVALID):
            parse_expression(bad)


@pytest.mark.parametrize("source", ["", "   ", "1 + 2", "close.shift(1)", "close if volume else 1", "open_"])
def test_unsupported_sources_raise_the_documented_code(source: str) -> None:
    with pytest.raises(ValueError, match=INVALID):
        parse_expression(source)
    with pytest.raises(ValueError, match=INVALID):
        parse_expression(7)  # type: ignore[arg-type]


def test_evaluation_rejects_a_bad_expression_or_context_with_documented_codes() -> None:
    close, volume = _research_panels()
    context = _context(close, volume)
    eligible = np.ones(close.shape, dtype=np.bool_)
    expression = parse_expression("close")
    with pytest.raises(TypeError, match=INVALID):
        evaluate_expression("close", context, eligible=eligible)  # type: ignore[arg-type]

    sessions = context.sessions
    bad_contexts = {
        "no sessions": replace(context, sessions=()),
        "no universe": replace(context, universe=()),
        "unsorted": replace(context, sessions=(sessions[1], sessions[0], *sessions[2:])),
        "duplicate session": replace(context, sessions=(sessions[0], sessions[0], *sessions[2:])),
        "future session": replace(context, decision_session=sessions[-1]),
        "duplicate symbol": replace(context, universe=("S0", "S0", "S2", "S3")),
    }
    for label, bad in bad_contexts.items():
        with pytest.raises(ValueError, match=BAD_CONTEXT):
            evaluate_expression(expression, bad, eligible=eligible)
        assert label  # keeps the failing case visible in the report
    with pytest.raises(ValueError, match=BAD_CONTEXT):
        evaluate_expression(expression, context, eligible=eligible.astype(np.int64))
    with pytest.raises(ValueError, match=BAD_CONTEXT):
        evaluate_expression(expression, context, eligible=eligible[:-1])


def test_evaluation_rejects_malformed_panels() -> None:
    close, volume = _research_panels()
    context = _context(close, volume)
    eligible = np.ones(close.shape, dtype=np.bool_)
    missing = replace(context, _panels=MappingProxyType({"volume": volume}))
    with pytest.raises(ValueError, match=BAD_CONTEXT):
        evaluate_expression(parse_expression("close"), missing, eligible=eligible)
    infinite = close.copy()
    infinite[2, 1] = np.inf
    with pytest.raises(ValueError, match=BAD_CONTEXT):
        evaluate_expression(parse_expression("close"), _context(infinite, volume), eligible=eligible)
    wrong_shape = replace(context, _panels=MappingProxyType({**context._panels, "close": close[:-1]}))
    with pytest.raises(ValueError, match=BAD_CONTEXT):
        evaluate_expression(parse_expression("close"), wrong_shape, eligible=eligible)


def test_ineligible_members_are_nan_and_excluded_from_cross_sections() -> None:
    close = np.array([[1.0, 2.0, 3.0, 4.0]] * 3)
    context = _context(close)
    eligible = np.ones(close.shape, dtype=np.bool_)
    eligible[:, 3] = False
    ranked = evaluate_expression(parse_expression("rank(close)"), context, eligible=eligible)
    assert np.isnan(ranked[:, 3]).all()
    plain = evaluate_expression(parse_expression("rank(close)"), context, eligible=np.ones_like(eligible))
    assert not np.array_equal(ranked[:, :3], plain[:, :3])  # rank only counts eligible members


# ---- engine boundary: factor code receives exactly the declared lookback ----------------------


def _window(rows: int = 6, symbols: tuple[str, ...] = ("A", "B", "C")) -> PanelWindow:
    sessions = tuple(date(2024, 1, 2) + timedelta(days=index) for index in range(rows))
    count = len(symbols)
    base = (np.arange(rows * count, dtype=np.int64).reshape(rows, count) + 1) * MICRO
    present = np.ones((rows, count), dtype=np.bool_)
    present[1, 0] = False
    return PanelWindow(
        dataset_identity="sha256:" + "0" * 64,
        decision_session=sessions[-1] + timedelta(days=1),
        sessions=sessions,
        symbols=symbols,
        micro={field: base + offset for offset, field in enumerate(FIELDS)},
        volume=np.arange(rows * count, dtype=np.int64).reshape(rows, count) * 10,
        present=present,
    )


def test_factor_context_hands_over_only_the_declared_lookback_in_universe_order() -> None:
    window = _window()
    context = factor_context(window, ("C", "A"), lookback=3)
    assert context.sessions == window.sessions[-3:]
    assert context.universe == ("C", "A")
    close = context.panel("close")
    assert close.shape == (3, 2)
    # last three rows of columns C then A, converted from micro-units, using the close field
    expected = (window.micro["close"][-3:][:, [2, 0]] / MICRO).astype(np.float64)
    np.testing.assert_array_equal(close, expected)
    np.testing.assert_array_equal(context.panel("volume"), window.volume[-3:][:, [2, 0]].astype(float))
    assert context.panel("present").shape == (3, 2)
    for field in (*FIELDS, "volume", "present"):
        assert not context.panel(field).flags.writeable


def test_factor_context_masks_missing_bars_as_nan_inside_the_window() -> None:
    window = _window()
    context = factor_context(window, ("A", "B"), lookback=6)
    assert not context.panel("present")[1, 0]
    for field in (*FIELDS, "volume"):
        assert np.isnan(context.panel(field)[1, 0])
        assert np.isfinite(context.panel(field)[1, 1])


def test_factor_context_refuses_bad_inputs_with_documented_codes() -> None:
    window = _window()
    cases: list[tuple[str, Callable[[], object]]] = [
        ("FACTOR_LOOKBACK_INVALID", lambda: factor_context(window, ("A",), lookback=0)),
        ("FACTOR_LOOKBACK_INVALID", lambda: factor_context(window, ("A",), lookback=True)),  # type: ignore[arg-type]
        ("FACTOR_LOOKBACK_INVALID", lambda: factor_context(window, ("A",), lookback=2.0)),  # type: ignore[arg-type]
        ("FACTOR_INSUFFICIENT_HISTORY", lambda: factor_context(window, ("A",), lookback=7)),
        (
            "FACTOR_LOOKAHEAD",
            lambda: factor_context(replace(window, decision_session=window.sessions[-1]), ("A",), lookback=2),
        ),
        ("FACTOR_UNIVERSE_INVALID", lambda: factor_context(window, ("A", "A"), lookback=2)),
        ("FACTOR_UNIVERSE_INVALID", lambda: factor_context(window, ("A", "Z"), lookback=2)),
        (
            "FACTOR_PANEL_SHAPE",
            lambda: factor_context(replace(window, present=window.present[:-1]), ("A",), lookback=2),
        ),
        (
            "FACTOR_PANEL_SHAPE",
            lambda: factor_context(replace(window, volume=window.volume[:, :-1]), ("A",), lookback=2),
        ),
        (
            "FACTOR_PANEL_SHAPE",
            lambda: factor_context(
                replace(window, micro={**window.micro, "high": window.micro["high"][:-1]}),
                ("A",),
                lookback=2,
            ),
        ),
    ]
    for code, call in cases:
        with pytest.raises(
            ValueError if code != "FACTOR_LOOKBACK_INVALID" else ValueError, match=f"^{code}$"
        ):
            call()


class _Weight(Params):
    weight: float = 2.0


def test_run_factor_passes_the_parameters_through_and_checks_their_type() -> None:
    @factor(params=_Weight, lookback=lambda p: 2)
    def scores(ctx: FactorCtx, p: _Weight) -> dict[str, float]:
        return {
            symbol: p.weight * float(ctx.panel("close")[-1, index])
            for index, symbol in enumerate(ctx.universe)
        }

    definition = factor_definition_of(scores)
    window = _window()
    result = run_factor(definition, _Weight(weight=3.0), window, ("A", "B"))
    last = window.micro["close"][-1] / MICRO
    assert result == {"A": 3.0 * last[0], "B": 3.0 * last[1]}
    with pytest.raises(TypeError, match=r"^FACTOR_PARAMS_INVALID$"):
        run_factor(definition, Params(), window, ("A",))  # type: ignore[arg-type]


def test_factor_decorator_records_identity_and_rejects_non_params_classes() -> None:
    def my_score(ctx: FactorCtx, p: _Weight) -> dict[str, float]:
        return {}

    decorated = factor(params=_Weight, lookback=lambda p: 5)(my_score)
    definition = factor_definition_of(decorated)
    assert definition.name == "my_score"
    assert definition.module == my_score.__module__
    assert definition.params is _Weight
    assert definition.lookback(_Weight()) == 5
    for bad in (int, "text", _Weight()):
        with pytest.raises(TypeError, match=r"^FACTOR_PARAMS_MUST_SUBCLASS_PARAMS$"):
            factor(params=bad, lookback=lambda p: 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"^NOT_A_FACTOR$"):
        factor_definition_of(object())


def test_checked_scores_accepts_only_finite_numbers_for_declared_symbols() -> None:
    universe = ("A", "B", "C")
    assert checked_scores({"C": 1, "A": np.float32(0.5), "B": np.int64(2)}, universe) == {
        "A": 0.5,
        "B": 2.0,
        "C": 1.0,
    }
    assert list(checked_scores({"C": 1.0, "A": 2.0}, universe)) == ["A", "C"]  # sorted, partial coverage
    cases: list[tuple[type[Exception], str, object]] = [
        (TypeError, "FACTOR_SCORES_INVALID", [("A", 1.0)]),
        (ValueError, "FACTOR_SYMBOL_UNDECLARED", {"Z": 1.0}),
        (ValueError, "FACTOR_SYMBOL_UNDECLARED", {1: 1.0}),
        (TypeError, "FACTOR_SCORE_INVALID", {"A": True}),
        (TypeError, "FACTOR_SCORE_INVALID", {"A": "1.0"}),
        (TypeError, "FACTOR_SCORE_INVALID", {"A": None}),
        (ValueError, "FACTOR_SCORE_NONFINITE", {"A": math.nan}),
        (ValueError, "FACTOR_SCORE_NONFINITE", {"A": math.inf}),
        (ValueError, "FACTOR_SCORE_NONFINITE", {"A": -math.inf}),
    ]
    for error, code, scores in cases:
        with pytest.raises(error, match=f"^{code}$"):
            checked_scores(scores, universe)  # type: ignore[arg-type]


def test_ts_mean_of_a_flat_stretch_is_the_price_itself() -> None:
    context = _context(_drifting_with_a_flat_stretch())
    flat = _evaluate("ts_mean(close, 10)", context)[215:225]
    assert (flat == _drifting_with_a_flat_stretch()[215:225]).all()


def test_the_grammar_version_records_the_centred_window_statistics() -> None:
    from signalquarry._internal.factors.expr import GRAMMAR_VERSION

    assert GRAMMAR_VERSION == 2
