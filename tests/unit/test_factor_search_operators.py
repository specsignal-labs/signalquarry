# SPDX-License-Identifier: Apache-2.0
"""The genetic operators driven by a scripted generator, so each branch is pinned exactly."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pytest

from signalquarry._internal.factors.expr import _Node, parse_expression
from signalquarry._internal.factors.search import (
    _crossover,
    _mutate,
    _parse_root,
    _paths,
    _random_expression,
    _random_tree,
    _replace,
    _Scored,
    _terminal,
    _tournament,
)

_FIELDS = ("open", "high", "low", "close", "volume")
_CONSTANTS = (-1.0, -0.5, 0.5, 1.0, 2.0)
_PERIODS = (1, 2, 3, 5, 10, 20)


class Scripted:
    """Stands in for ``numpy.random.Generator``; every draw is dictated by the test."""

    def __init__(
        self, *, integers: Iterable[int] = (), randoms: Iterable[float] = (), picks: Iterable[Any] = ()
    ) -> None:
        self._integers = iter(integers)
        self._randoms = iter(randoms)
        self._picks = iter(picks)
        self.integer_bounds: list[tuple[int, int]] = []
        self.choice_calls: list[tuple[Any, dict[str, Any]]] = []

    def integers(self, low: int, high: int) -> int:
        value = next(self._integers)
        assert low <= value < high
        self.integer_bounds.append((low, high))
        return value

    def random(self) -> float:
        return next(self._randoms)

    def choice(self, options: Any, **kwargs: Any) -> Any:
        self.choice_calls.append((options, kwargs))
        position = next(self._picks)
        if isinstance(options, int):
            return np.asarray(position)
        return options[position]

    def finished(self) -> bool:
        return all(next(iterator, None) is None for iterator in (self._integers, self._randoms, self._picks))


def _node(source: str) -> _Node:
    return parse_expression(source)._root


def test_terminal_is_a_panel_field_below_the_boundary_and_a_constant_at_it() -> None:
    assert _terminal(Scripted(randoms=[0.0], picks=[3])) == _Node("panel", ("close",))  # type: ignore[arg-type]
    assert _terminal(Scripted(randoms=[0.8199], picks=[0])) == _Node("panel", ("open",))  # type: ignore[arg-type]
    assert _terminal(Scripted(randoms=[0.82], picks=[4])) == _Node("constant", (2.0,))  # type: ignore[arg-type]
    assert _terminal(Scripted(randoms=[0.99], picks=[0])) == _Node("constant", (-1.0,))  # type: ignore[arg-type]
    chosen = Scripted(randoms=[0.1], picks=[1])
    _terminal(chosen)  # type: ignore[arg-type]
    assert chosen.choice_calls[0][0] == _FIELDS
    constants = Scripted(randoms=[0.9], picks=[1])
    _terminal(constants)  # type: ignore[arg-type]
    assert constants.choice_calls[0][0] == _CONSTANTS


def test_random_tree_stops_at_the_depth_limit_without_drawing_a_branch_probability() -> None:
    rng = Scripted(randoms=[0.1], picks=[3])
    assert _random_tree(rng, 3, 3) == _Node("panel", ("close",))  # type: ignore[arg-type]
    assert rng.finished()
    deeper = Scripted(randoms=[0.1], picks=[0])
    assert _random_tree(deeper, 4, 3) == _Node("panel", ("open",))  # type: ignore[arg-type]
    assert deeper.finished()


def test_random_tree_turns_into_a_terminal_below_the_branch_probability_only() -> None:
    assert _random_tree(Scripted(randoms=[0.27, 0.1], picks=[0]), 1, 3) == _Node("panel", ("open",))  # type: ignore[arg-type]
    expanded = Scripted(randoms=[0.28, 0.1], integers=[0], picks=[0, 3])
    tree = _random_tree(expanded, 1, 2)  # type: ignore[arg-type]
    assert tree == _Node("positive", (_Node("panel", ("close",)),))
    assert expanded.integer_bounds == [(0, 4)]


def test_random_tree_builds_each_node_kind_from_its_own_operator_set_and_periods() -> None:
    unary = _random_tree(Scripted(randoms=[0.9, 0.1], integers=[0], picks=[5, 1]), 1, 2)  # type: ignore[arg-type]
    assert unary == _Node("rank", (_Node("panel", ("high",)),))

    window = _random_tree(Scripted(randoms=[0.9, 0.1], integers=[1], picks=[2, 3, 4]), 1, 2)  # type: ignore[arg-type]
    assert window == _Node("ts_mean", (_Node("panel", ("close",)), 10))

    binary = _random_tree(Scripted(randoms=[0.9, 0.1, 0.1], integers=[2], picks=[3, 0, 4]), 1, 2)  # type: ignore[arg-type]
    assert binary == _Node("divide", (_Node("panel", ("open",)), _Node("panel", ("volume",))))

    correlation = _random_tree(
        Scripted(randoms=[0.9, 0.1, 0.1], integers=[3], picks=[1, 2, 5]),  # type: ignore[arg-type]
        1,
        2,
    )
    assert correlation == _Node("ts_corr", (_Node("panel", ("high",)), _Node("panel", ("low",)), 20))


def test_random_tree_children_are_one_level_deeper_than_their_parent() -> None:
    # Parent at depth 1 with max_depth 2: both children sit at depth 2 and must be terminals.
    rng = Scripted(randoms=[0.9, 0.99, 0.99], integers=[2], picks=[0, 0, 0])
    tree = _random_tree(rng, 1, 2)  # type: ignore[arg-type]
    assert tree == _Node("add", (_Node("constant", (-1.0,)), _Node("constant", (-1.0,))))
    assert rng.finished()


def test_paths_list_the_root_first_then_each_node_child_in_order() -> None:
    root = _node("ts_corr(close + open, volume, 5)")
    found = _paths(root)
    assert [path for path, _ in found] == [(), (0,), (0, 0), (0, 1), (1,)]
    assert [node.op for _, node in found] == ["ts_corr", "add", "panel", "panel", "panel"]
    assert [path for path, _ in _paths(_node("ts_mean(close, 5)"))] == [(), (0,)]


def test_replace_swaps_exactly_one_subtree_and_ignores_non_node_arguments() -> None:
    root = _node("(close + open) * volume")
    swapped = _replace(root, (0, 1), _Node("panel", ("low",)))
    assert swapped == _node("(close + low) * volume")
    assert _replace(root, (), _Node("panel", ("low",))) == _Node("panel", ("low",))
    assert root == _node("(close + open) * volume")
    window = _node("ts_mean(close, 5)")
    assert _replace(window, (1,), _Node("panel", ("low",))) == window
    assert _replace(window, (0,), _Node("panel", ("low",))) == _node("ts_mean(low, 5)")


def test_parse_root_adds_a_panel_when_the_tree_has_none_and_rejects_invalid_trees() -> None:
    constant = _parse_root(_Node("constant", (2.0,)))
    assert constant is not None
    assert constant.identity == parse_expression("2 + close").identity
    kept = _parse_root(_node("rank(volume)"))
    assert kept is not None and kept.canonical == "rank(volume)"
    assert _parse_root(_Node("ts_mean", (_Node("panel", ("close",)), 999))) is None
    assert _parse_root(_Node("not_an_operator", (_Node("panel", ("close",)),))) is None


def test_random_expression_is_a_tree_from_depth_one_with_the_given_limit() -> None:
    rng = Scripted(randoms=[0.9, 0.1, 0.1], integers=[2], picks=[0, 3, 4])
    expression = _random_expression(rng, 2)  # type: ignore[arg-type]
    assert expression is not None and expression.canonical == "(close + volume)"


def _mutated(source: str, *, integers: list[int], randoms: list[float], picks: list[Any]) -> str | None:
    rng = Scripted(integers=integers, randoms=randoms, picks=picks)
    result = _mutate(parse_expression(source), rng)  # type: ignore[arg-type]
    assert rng.finished()
    return None if result is None else result.canonical


def test_mutating_a_panel_never_picks_the_same_field() -> None:
    assert _mutated("ts_mean(close, 5)", integers=[1], randoms=[0.9], picks=[0]) == "ts_mean(open, 5)"
    assert _mutated("ts_mean(close, 5)", integers=[1], randoms=[0.9], picks=[3]) == "ts_mean(volume, 5)"
    options = Scripted(integers=[1], randoms=[0.9], picks=[0])
    _mutate(parse_expression("ts_mean(close, 5)"), options)  # type: ignore[arg-type]
    assert options.choice_calls[0][0] == ("open", "high", "low", "volume")
    assert options.integer_bounds == [(0, 2)]


def test_mutating_a_constant_never_picks_the_same_value() -> None:
    assert _mutated("close * 2", integers=[2], randoms=[0.9], picks=[0]) == "(close * (-1.0))"
    assert _mutated("close * 2", integers=[2], randoms=[0.9], picks=[3]) == "(close * 1.0)"
    options = Scripted(integers=[2], randoms=[0.9], picks=[0])
    _mutate(parse_expression("close * 2"), options)  # type: ignore[arg-type]
    assert options.choice_calls[0][0] == (-1.0, -0.5, 0.5, 1.0)


def test_mutating_an_operator_keeps_its_arguments_and_picks_another_operator() -> None:
    assert _mutated("close + open", integers=[0], randoms=[0.9], picks=[0]) == "(close - open)"
    assert _mutated("close + open", integers=[0], randoms=[0.9], picks=[2]) == "(close / open)"
    assert _mutated("abs(close)", integers=[0], randoms=[0.9], picks=[2]) == "log(close)"
    binary = Scripted(integers=[0], randoms=[0.9], picks=[0])
    _mutate(parse_expression("close + open"), binary)  # type: ignore[arg-type]
    assert binary.choice_calls[0][0] == ("subtract", "multiply", "divide")
    unary = Scripted(integers=[0], randoms=[0.9], picks=[0])
    _mutate(parse_expression("abs(close)"), unary)  # type: ignore[arg-type]
    assert unary.choice_calls[0][0] == ("positive", "negative", "log", "sign", "rank", "zscore")


def test_mutating_a_window_changes_either_the_operator_or_the_period() -> None:
    assert _mutated("ts_mean(close, 5)", integers=[0], randoms=[0.9, 0.49], picks=[0]) == "delay(close, 5)"
    assert _mutated("ts_mean(close, 5)", integers=[0], randoms=[0.9, 0.49], picks=[2]) == "ts_std(close, 5)"
    assert _mutated("ts_mean(close, 5)", integers=[0], randoms=[0.9, 0.5], picks=[0]) == "ts_mean(close, 1)"
    assert _mutated("ts_mean(close, 5)", integers=[0], randoms=[0.9, 0.5], picks=[4]) == "ts_mean(close, 20)"
    operator = Scripted(integers=[0], randoms=[0.9, 0.1], picks=[0])
    _mutate(parse_expression("ts_mean(close, 5)"), operator)  # type: ignore[arg-type]
    assert operator.choice_calls[0][0] == ("delay", "delta", "ts_std", "ts_rank")
    period = Scripted(integers=[0], randoms=[0.9, 0.9], picks=[0])
    _mutate(parse_expression("ts_mean(close, 5)"), period)  # type: ignore[arg-type]
    assert period.choice_calls[0][0] == (1, 2, 3, 10, 20)


def test_mutating_a_correlation_changes_only_its_period() -> None:
    assert (
        _mutated("ts_corr(close, volume, 5)", integers=[0], randoms=[0.9], picks=[0])
        == "ts_corr(close, volume, 1)"
    )
    assert (
        _mutated("ts_corr(close, volume, 5)", integers=[0], randoms=[0.9], picks=[4])
        == "ts_corr(close, volume, 20)"
    )
    period = Scripted(integers=[0], randoms=[0.9], picks=[0])
    _mutate(parse_expression("ts_corr(close, volume, 5)"), period)  # type: ignore[arg-type]
    assert period.choice_calls[0][0] == (1, 2, 3, 10, 20)


def test_mutation_sometimes_replaces_the_node_with_a_fresh_tree() -> None:
    # 0.34 < 0.35 takes the random-tree branch, rooted at depth 1 with a depth limit of 3.
    result = _mutated("ts_mean(close, 5)", integers=[1], randoms=[0.34, 0.1, 0.1], picks=[1])
    assert result == "ts_mean(high, 5)"
    fresh = Scripted(integers=[0], randoms=[0.34, 0.1, 0.1], picks=[1])
    mutated = _mutate(parse_expression("ts_mean(close, 5)"), fresh)  # type: ignore[arg-type]
    assert mutated is not None and mutated.canonical == "high"  # replaced root; a panel is already present
    # 0.35 does not.
    kept = _mutated("ts_mean(close, 5)", integers=[0], randoms=[0.35, 0.1], picks=[0])
    assert kept == "delay(close, 5)"


def test_crossover_grafts_a_subtree_of_the_second_parent_into_the_first() -> None:
    left = parse_expression("close + open")
    right = parse_expression("rank(volume)")
    rng = Scripted(integers=[1, 1])
    child = _crossover(left, right, rng)  # type: ignore[arg-type]
    assert child is not None and child.canonical == "(volume + open)"
    assert rng.integer_bounds == [(0, 3), (0, 2)]
    whole = _crossover(left, right, Scripted(integers=[0, 0]))  # type: ignore[arg-type]
    assert whole is not None and whole.canonical == "rank(volume)"


def _scored(source: str, fitness: float) -> _Scored:
    expression = parse_expression(source)
    return _Scored(expression, 1, 100, 100.0, 0.1, 0.1, 1.0, 0.0, fitness)


def test_tournament_returns_the_fittest_of_three_distinct_picks() -> None:
    population = [
        _scored(f"close + {value}", fitness) for value, fitness in ((1, 5.0), (2, 4.0), (3, 3.0), (4, 2.0))
    ]
    # ranked by fitness: positions 0..3 are the 5.0, 4.0, 3.0, 2.0 entries.
    rng = Scripted(picks=[[3, 1, 2]])
    assert _tournament(population, rng) is population[1]  # type: ignore[arg-type]
    options, kwargs = rng.choice_calls[0]
    assert options == 4 and kwargs == {"size": 3, "replace": False}
    assert _tournament(population, Scripted(picks=[[3, 2, 0]])) is population[0]  # type: ignore[arg-type]
    # ranking ignores input order
    shuffled = [population[2], population[0], population[3], population[1]]
    assert _tournament(shuffled, Scripted(picks=[[0, 1, 3]])) is population[0]  # type: ignore[arg-type]


def test_tournament_on_a_small_population_draws_at_most_that_many_and_breaks_ties_by_identity() -> None:
    pair = [_scored("close + 1", 1.0), _scored("close + 2", 1.0)]
    rng = Scripted(picks=[[0, 1]])
    winner = _tournament(pair, rng)  # type: ignore[arg-type]
    assert rng.choice_calls[0][1]["size"] == 2
    assert winner is max(pair, key=lambda item: item.expression.identity)
    assert _tournament(pair, Scripted(picks=[[1, 0]])) is winner  # type: ignore[arg-type]
    solo = [_scored("close + 1", 0.0)]
    solo_rng = Scripted(picks=[[0]])
    assert _tournament(solo, solo_rng) is solo[0]  # type: ignore[arg-type]
    assert solo_rng.choice_calls[0][1]["size"] == 1


@pytest.mark.parametrize("seed", range(5))
def test_real_generator_draws_always_yield_valid_expressions_or_none(seed: int) -> None:
    rng = np.random.Generator(np.random.PCG64(seed))
    for _ in range(40):
        expression = _random_expression(rng)
        if expression is None:
            continue
        assert parse_expression(expression.canonical).identity == expression.identity
        mutated = _mutate(expression, rng)
        if mutated is not None:
            assert parse_expression(mutated.canonical).identity == mutated.identity
