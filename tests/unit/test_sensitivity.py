# SPDX-License-Identifier: Apache-2.0
"""Hand-worked reference cases for ordered-grid sensitivity summaries."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import pytest

from signalquarry._internal.validation.sensitivity import sensitivity


def test_one_axis_reference_uses_declared_order() -> None:
    # Declared neighbours of 10 are 30 and 20; 40 is two positions away.
    result = sensitivity(
        {"window": [30, 10, 20, 40]},
        [
            {"params": {"window": 20}, "sharpe": 3.0},
            {"params": {"window": 10}, "sharpe": 4.0},
            {"params": {"window": 40}, "sharpe": 0.5},
            {"params": {"window": 30}, "sharpe": 1.0},
        ],
    )
    assert result == {
        "metric": "sharpe",
        "direction": "higher",
        "points": 4,
        "scored": 4,
        "best": {"params": {"window": 10}, "value": 4.0},
        "neighbours": [
            {"params": {"window": 30}, "value": 1.0},
            {"params": {"window": 20}, "value": 3.0},
        ],
        "neighbour_median": 2.0,
        "plateau": 0.5,
        "worse_neighbours": 2,
        "axes": {"window": {"values": [30, 10, 20, 40], "median": [1.0, 4.0, 3.0, 0.5]}},
    }


def test_two_axis_reference_excludes_diagonals_and_orders_neighbours() -> None:
    # Rows x=1,2,3: [1,2,3], [4,10,6], [7,8,9]. Best is (2,b).
    # Axial neighbours [2,8,4,6] have median 5; diagonals never participate.
    points = [
        {"params": {"x": x, "y": y}, "sharpe": value}
        for x, values in [(3, [7, 8, 9]), (2, [4, 10, 6]), (1, [1, 2, 3])]
        for y, value in zip(["a", "b", "c"], values, strict=True)
    ]
    assert sensitivity({"x": [1, 2, 3], "y": ["a", "b", "c"]}, points) == {
        "metric": "sharpe",
        "direction": "higher",
        "points": 9,
        "scored": 9,
        "best": {"params": {"x": 2, "y": "b"}, "value": 10.0},
        "neighbours": [
            {"params": {"x": 1, "y": "b"}, "value": 2.0},
            {"params": {"x": 3, "y": "b"}, "value": 8.0},
            {"params": {"x": 2, "y": "a"}, "value": 4.0},
            {"params": {"x": 2, "y": "c"}, "value": 6.0},
        ],
        "neighbour_median": 5.0,
        "plateau": 0.5,
        "worse_neighbours": 4,
        "axes": {
            "x": {"values": [1, 2, 3], "median": [2.0, 6.0, 8.0]},
            "y": {"values": ["a", "b", "c"], "median": [4.0, 8.0, 6.0]},
        },
    }


def test_lower_direction_reference_with_negative_best() -> None:
    assert sensitivity(
        {"x": [1, 2, 3]},
        [
            {"params": {"x": 1}, "loss": -2},
            {"params": {"x": 2}, "loss": -4},
            {"params": {"x": 3}, "loss": -1},
        ],
        metric="loss",
        direction="lower",
    ) == {
        "metric": "loss",
        "direction": "lower",
        "points": 3,
        "scored": 3,
        "best": {"params": {"x": 2}, "value": -4.0},
        "neighbours": [
            {"params": {"x": 1}, "value": -2.0},
            {"params": {"x": 3}, "value": -1.0},
        ],
        "neighbour_median": -1.5,
        "plateau": 0.375,
        "worse_neighbours": 2,
        "axes": {"x": {"values": [1, 2, 3], "median": [-2.0, -4.0, -1.0]}},
    }


@pytest.mark.parametrize("direction", ["higher", "lower"])
def test_best_ties_keep_first_point_and_are_not_strictly_worse(direction: Any) -> None:
    result = sensitivity(
        {"x": [1, 2, 3]},
        [{"params": {"x": x}, "sharpe": 2} for x in [2, 3, 1]],
        direction=direction,
    )
    assert result["best"] == {"params": {"x": 2}, "value": 2.0}
    assert result["neighbours"] == [{"params": {"x": 1}, "value": 2.0}, {"params": {"x": 3}, "value": 2.0}]
    assert result["worse_neighbours"] == 0


def test_missing_none_and_non_finite_metrics_are_excluded() -> None:
    result = sensitivity(
        {"x": [0, 1, 2, 3, 4, 5]},
        [
            {"params": {"x": 0}},
            {"params": {"x": 1}, "sharpe": None},
            {"params": {"x": 2}, "sharpe": float("nan")},
            {"params": {"x": 3}, "sharpe": float("inf")},
            {"params": {"x": 4}, "sharpe": float("-inf")},
            {"params": {"x": 5}, "sharpe": 2},
        ],
    )
    assert result["points"] == 6
    assert result["scored"] == 1
    assert result["best"] == {"params": {"x": 5}, "value": 2.0}
    assert result["neighbours"] == []
    assert result["axes"]["x"]["median"] == [None, None, None, None, None, 2.0]
    json.dumps(result, allow_nan=False)


def test_unscored_adjacent_point_does_not_bridge_to_a_distant_point() -> None:
    result = sensitivity(
        {"x": [1, 2, 3]},
        [{"params": {"x": 1}, "sharpe": 4}, {"params": {"x": 3}, "sharpe": 3}],
    )
    assert result["neighbours"] == []
    assert result["plateau"] is None
    assert result["axes"]["x"]["median"] == [4.0, None, 3.0]


def test_single_scored_point_has_no_neighbours_or_plateau() -> None:
    result = sensitivity({"x": [1]}, [{"params": {"x": 1}, "sharpe": 2}])
    assert result["neighbours"] == []
    assert result["neighbour_median"] is None
    assert result["plateau"] is None
    assert result["worse_neighbours"] == 0


def test_no_scored_point_reference() -> None:
    assert sensitivity({"x": [1, 2]}, [{"params": {"x": 1}, "sharpe": None}]) == {
        "metric": "sharpe",
        "direction": "higher",
        "points": 1,
        "scored": 0,
        "best": None,
        "neighbours": [],
        "neighbour_median": None,
        "plateau": None,
        "worse_neighbours": 0,
        "axes": {"x": {"values": [1, 2], "median": [None, None]}},
    }


def test_empty_points_reference() -> None:
    assert sensitivity({}, []) == {
        "metric": "sharpe",
        "direction": "higher",
        "points": 0,
        "scored": 0,
        "best": None,
        "neighbours": [],
        "neighbour_median": None,
        "plateau": None,
        "worse_neighbours": 0,
        "axes": {},
    }


@pytest.mark.parametrize(
    ("direction", "values"),
    [("higher", [0, -1]), ("higher", [-1, -2]), ("lower", [0, 1]), ("lower", [1, 2])],
)
def test_plateau_is_none_for_zero_or_opposite_sign_best(direction: Any, values: list[int]) -> None:
    result = sensitivity(
        {"x": [1, 2]},
        [{"params": {"x": x}, "sharpe": value} for x, value in zip([1, 2], values, strict=True)],
        direction=direction,
    )
    assert result["neighbour_median"] == float(values[1])
    assert result["plateau"] is None


def test_plateau_rounds_to_six_decimals_without_rounding_scores() -> None:
    result = sensitivity(
        {"x": [1, 2]},
        [{"params": {"x": 1}, "sharpe": 3}, {"params": {"x": 2}, "sharpe": 1}],
    )
    assert result["plateau"] == 0.333333
    assert result["neighbour_median"] == 1.0


@pytest.mark.parametrize("point", [{}, {"params": {}}, {"params": {"x": 3}}, {"params": {"other": 1}}])
def test_off_grid_points_raise_even_without_a_metric(point: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="^SENSITIVITY_POINT_OFF_GRID$"):
        sensitivity({"x": [1, 2]}, [point])


def test_point_must_supply_every_axis() -> None:
    with pytest.raises(ValueError, match="^SENSITIVITY_POINT_OFF_GRID$"):
        sensitivity({"x": [1], "y": [2]}, [{"params": {"x": 1}, "sharpe": 1}])


def test_empty_axis_cannot_accept_a_point() -> None:
    with pytest.raises(ValueError, match="^SENSITIVITY_POINT_OFF_GRID$"):
        sensitivity({"x": []}, [{"params": {"x": 1}, "sharpe": 1}])


def test_extra_param_keys_are_ignored() -> None:
    result = sensitivity(
        {"x": [1, 2]},
        [{"params": {"x": 1, "extra": "a"}, "sharpe": 3}, {"params": {"x": 2, "extra": "b"}, "sharpe": 2}],
    )
    assert result["best"] == {"params": {"x": 1}, "value": 3.0}
    assert result["neighbours"] == [{"params": {"x": 2}, "value": 2.0}]


def test_axis_medians_pool_all_scored_points_sharing_each_value() -> None:
    result = sensitivity(
        {"x": [1, 2], "y": ["a", "b", "c"]},
        [
            {"params": {"x": 1, "y": "a"}, "sharpe": 1},
            {"params": {"x": 1, "y": "b"}, "sharpe": 4},
            {"params": {"x": 2, "y": "a"}, "sharpe": 7},
            {"params": {"x": 2, "y": "c"}, "sharpe": None},
        ],
    )
    assert result["axes"] == {
        "x": {"values": [1, 2], "median": [2.5, 7.0]},
        "y": {"values": ["a", "b", "c"], "median": [4.0, 4.0, None]},
    }


def test_non_hashable_json_values_are_supported() -> None:
    result = sensitivity(
        {"x": [[1], [2]]},
        [{"params": {"x": [1]}, "sharpe": 3}, {"params": {"x": [2]}, "sharpe": 2}],
    )
    assert result["neighbours"] == [{"params": {"x": [2]}, "value": 2.0}]
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_duplicate_point_is_not_its_own_neighbour() -> None:
    result = sensitivity(
        {"x": [1, 2]},
        [
            {"params": {"x": 1}, "sharpe": 4},
            {"params": {"x": 1}, "sharpe": 3},
            {"params": {"x": 2}, "sharpe": 2},
        ],
    )
    assert result["neighbours"] == [{"params": {"x": 2}, "value": 2.0}]
    assert result["axes"]["x"]["median"] == [3.5, 2.0]


def test_empty_grid_scored_point_has_no_neighbours() -> None:
    result = sensitivity({}, [{"params": {}, "sharpe": 2}])
    assert result["best"] == {"params": {}, "value": 2.0}
    assert result["axes"] == {}
    assert result["neighbours"] == []


def test_sensitivity_is_deterministic_and_does_not_mutate_inputs() -> None:
    grid = {"x": [1, 2]}
    points = [{"params": {"x": 1}, "sharpe": 2}, {"params": {"x": 2}, "sharpe": 1}]
    original = deepcopy((grid, points))
    assert sensitivity(grid, points) == sensitivity(grid, points)
    assert (grid, points) == original
