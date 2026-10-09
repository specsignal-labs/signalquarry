# SPDX-License-Identifier: Apache-2.0
"""Deterministic descriptive summaries of an ordered parameter grid, without I/O."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import median
from typing import Any, Literal


def sensitivity(
    grid: Mapping[str, Sequence[Any]],
    points: Sequence[Mapping[str, Any]],
    *,
    metric: str = "sharpe",
    direction: Literal["higher", "lower"] = "higher",
) -> dict[str, Any]:
    """Describe the best scored point, its adjacent points and axis medians.

    Only declared axes participate. Scored points retain their input order on
    ties; neighbours are ordered by axis, then by the declared value order.
    """
    axes = {axis: list(values) for axis, values in grid.items()}
    scored: list[tuple[tuple[int, ...], dict[str, Any], float]] = []
    for point in points:
        params = point.get("params", {})
        coordinates: list[int] = []
        for axis, values in axes.items():
            if axis not in params:
                raise ValueError("SENSITIVITY_POINT_OFF_GRID")
            try:
                coordinates.append(values.index(params[axis]))
            except ValueError:
                raise ValueError("SENSITIVITY_POINT_OFF_GRID") from None
        value = point.get(metric)
        if value is not None and math.isfinite(value):
            scored.append((tuple(coordinates), {axis: params[axis] for axis in axes}, float(value)))

    best = (max if direction == "higher" else min)(scored, key=lambda item: item[2], default=None)
    neighbours: list[dict[str, Any]] = []
    if best is not None:
        for axis_index in range(len(axes)):
            adjacent = [
                item
                for item in scored
                if abs(item[0][axis_index] - best[0][axis_index]) == 1
                and all(
                    position == axis_index or coordinate == best[0][position]
                    for position, coordinate in enumerate(item[0])
                )
            ]
            adjacent.sort(key=lambda item: item[0][axis_index])
            neighbours.extend({"params": params, "value": value} for _, params, value in adjacent)
    neighbour_median = round(float(median(item["value"] for item in neighbours)), 6) if neighbours else None
    good_sign = best is not None and (best[2] > 0 if direction == "higher" else best[2] < 0)
    axis_summaries: dict[str, dict[str, Any]] = {}
    for axis_index, (axis, values) in enumerate(axes.items()):
        medians: list[float | None] = []
        for value_index in range(len(values)):
            scores = [value for coordinates, _, value in scored if coordinates[axis_index] == value_index]
            medians.append(round(float(median(scores)), 6) if scores else None)
        axis_summaries[axis] = {"values": values, "median": medians}
    return {
        "metric": metric,
        "direction": direction,
        "points": len(points),
        "scored": len(scored),
        "best": None if best is None else {"params": best[1], "value": best[2]},
        "neighbours": neighbours,
        "neighbour_median": neighbour_median,
        "plateau": round(neighbour_median / best[2], 6)
        if neighbour_median is not None and good_sign and best is not None
        else None,
        "worse_neighbours": sum(
            item["value"] < best[2] if direction == "higher" else item["value"] > best[2]
            for item in neighbours
        )
        if best is not None
        else 0,
        "axes": axis_summaries,
    }
