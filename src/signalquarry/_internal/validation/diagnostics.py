# SPDX-License-Identifier: Apache-2.0
"""Pure descriptive summaries of cost curves and recorded walk-forward folds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import stdev
from typing import Any


def cost_curve_break_even(points: Sequence[tuple[float, float]]) -> float | None:
    """Interpolate the first crossing from positive return to zero or below.

    Multipliers are ascending. An empty curve, a non-positive starting return,
    or returns positive throughout have no break-even in the recorded range.
    """
    if not points or points[0][1] <= 0:
        return None
    for (left, positive), (right, value) in zip(points, points[1:], strict=False):
        if value <= 0:
            return round(left + (right - left) * positive / (positive - value), 6)
    return None


def fold_consistency(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize returns, with sample dispersion and an explicit sufficiency status."""
    returns = [float(fold["total_return"]) for fold in folds]
    count = len(returns)
    positive = sum(value > 0 for value in returns)
    benchmark = [float(fold["excess_return"]) for fold in folds if "excess_return" in fold]
    return {
        "folds": count,
        "positive": positive,
        "share_positive": round(positive / count, 6) if count else None,
        "ahead_of_benchmark": sum(value > 0 for value in benchmark) if benchmark else None,
        "best": round(max(returns), 8) if returns else None,
        "worst": round(min(returns), 8) if returns else None,
        "dispersion": round(stdev(returns), 8) if count >= 2 else None,
        "status": "ok" if count >= 3 else "insufficient",
    }
