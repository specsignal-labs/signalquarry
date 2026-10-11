# SPDX-License-Identifier: Apache-2.0
"""Pure comparisons of aligned backtest returns and parsed result documents."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import numpy as np

from signalquarry._internal.validation.evaluate import max_drawdown
from signalquarry._internal.validation.metrics import SESSIONS_PER_YEAR
from signalquarry._internal.validation.stats import moments, paired_block_bootstrap_sharpe_difference


@dataclass(frozen=True)
class RunSeries:
    document: Mapping[str, Any]
    sessions: tuple[date, ...]
    returns: np.ndarray


def configuration_diff(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, list[Any]]:
    """Sorted dotted-path differences; missing keys use None and lists stay whole."""
    differences: dict[str, list[Any]] = {}

    def visit(a: Mapping[str, Any], b: Mapping[str, Any], prefix: str) -> None:
        for key in a.keys() | b.keys():
            path = f"{prefix}.{key}" if prefix else key
            left_value, right_value = a.get(key), b.get(key)
            if key not in a or key not in b:
                differences[path] = [left_value, right_value]
            elif isinstance(left_value, Mapping) and isinstance(right_value, Mapping):
                visit(left_value, right_value, path)
            elif left_value != right_value:
                differences[path] = [left_value, right_value]

    visit(left, right, "")
    return dict(sorted(differences.items()))


def comparability(runs: Sequence[RunSeries]) -> dict[str, Any]:
    """Report sorted reasons that prevent comparisons on the supplied sessions."""
    reasons = []
    if runs:
        reference = runs[0]
        if any(
            run.document["dataset_identity"] != reference.document["dataset_identity"] for run in runs[1:]
        ):
            reasons.append("COMPARE_DATASET_DIFFERS")
        if any(run.sessions != reference.sessions for run in runs[1:]):
            reasons.append("COMPARE_WINDOW_DIFFERS")
        accounts = [run.document["spec"].get("account") for run in runs if "spec" in run.document]
        if accounts and any(account != accounts[0] for account in accounts[1:]):
            reasons.append("COMPARE_ACCOUNT_DIFFERS")
        if any(
            run.document["evidence"]["grade"] != reference.document["evidence"]["grade"] for run in runs[1:]
        ):
            reasons.append("COMPARE_GRADE_DIFFERS")
    return {"comparable": not reasons, "reasons": sorted(reasons)}


def compare_runs(
    runs: Sequence[RunSeries], *, block: int = 20, samples: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """Compare each run against the first, retaining pairing and configuration differences.

    Pair statistics use only jointly finite returns; ``sessions`` counts those rows.
    Result-document metrics are retained as supplied.
    """
    if len(runs) < 2:
        raise ValueError("COMPARE_NEEDS_TWO_RUNS")
    seen = set()
    for run in runs:
        run_id = run.document["run_id"]
        if run_id in seen:
            raise ValueError(f"COMPARE_RUN_DUPLICATE:{run_id}")
        seen.add(run_id)

    reference = runs[0]
    verdict = comparability(runs)
    differences = {}
    pairs = {}
    for run in runs[1:]:
        run_id = run.document["run_id"]
        differences[run_id] = {
            part: configuration_diff(reference.document[part], run.document[part])
            if part in reference.document and part in run.document
            else {}
            for part in ("params", "spec")
        }
        if not verdict["comparable"]:
            continue

        low, high = paired_block_bootstrap_sharpe_difference(
            run.returns, reference.returns, block=block, samples=samples, seed=seed
        )
        values = np.asarray(run.returns, dtype=np.float64)
        other = np.asarray(reference.returns, dtype=np.float64)
        finite = np.isfinite(values) & np.isfinite(other)
        values, other = values[finite], other[finite]
        correlation = None
        if len(values) > 1 and values.std(ddof=1) > 0 and other.std(ddof=1) > 0:
            correlation = round(float(np.corrcoef(values, other)[0, 1]), 6)
        sharpe, reference_sharpe = moments(values).sharpe, moments(other).sharpe
        sharpe_difference = None
        interval = None
        if math.isfinite(sharpe) and math.isfinite(reference_sharpe):
            annualization = math.sqrt(SESSIONS_PER_YEAR)
            sharpe_difference = round((sharpe - reference_sharpe) * annualization, 6)
            if math.isfinite(low) and math.isfinite(high):
                interval = [round(low * annualization, 6), round(high * annualization, 6)]
        pairs[run_id] = {
            "correlation": correlation,
            "sharpe_difference": sharpe_difference,
            "sharpe_difference_90": interval,
            "total_return_difference": round(
                run.document["metrics"]["total_return"] - reference.document["metrics"]["total_return"], 8
            ),
            "max_drawdown_difference": round(
                run.document["metrics"]["max_drawdown"] - reference.document["metrics"]["max_drawdown"], 8
            ),
            "sessions": len(values),
        }

    return {
        "reference": reference.document["run_id"],
        **verdict,
        "runs": [
            {
                "run_id": run.document["run_id"],
                "strategy_id": run.document["strategy_id"],
                "configuration_hash": run.document["configuration_hash"],
                "dataset_identity": run.document["dataset_identity"],
                "grade": run.document["evidence"]["grade"],
                "metrics": run.document["metrics"],
            }
            for run in runs
        ],
        "differences": differences,
        "pairs": pairs,
    }


STUDY_METRICS = (
    "total_return",
    "cagr",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "annual_volatility",
)


def metric_from_returns(metric: str, returns: np.ndarray) -> float:
    """One comparison metric from simple daily returns alone.

    Used to resample a metric; the figures shown for a run come from its result document.
    ``cagr`` annualizes over sessions (252 a year) because a resampled series has no calendar.
    A ratio whose denominator is zero is 0.
    """
    if metric not in STUDY_METRICS:
        raise ValueError(f"COMPARE_METRIC_UNKNOWN:{metric}")
    values = np.asarray(returns, dtype=np.float64)
    n = len(values)
    if n < 2:
        return float("nan")
    total = float(np.prod(1.0 + values) - 1.0)
    cagr = (1.0 + total) ** (SESSIONS_PER_YEAR / n) - 1.0 if total > -1.0 else -1.0
    deviation = float(values.std(ddof=1))
    mean = float(values.mean())
    if metric == "total_return":
        return total
    if metric == "cagr":
        return cagr
    if metric == "annual_volatility":
        return deviation * math.sqrt(SESSIONS_PER_YEAR)
    if metric == "sharpe":
        return mean / deviation * math.sqrt(SESSIONS_PER_YEAR) if deviation > 0 else 0.0
    if metric == "sortino":
        downside = values[values < 0]
        below = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else 0.0
        return mean / below * math.sqrt(SESSIONS_PER_YEAR) if below > 0 else 0.0
    drawdown = max_drawdown(values)
    if metric == "max_drawdown":
        return drawdown
    return cagr / drawdown if drawdown > 0 else 0.0


def paired_metric_interval(
    metric: str,
    returns: np.ndarray,
    other: np.ndarray,
    *,
    block: int = 20,
    samples: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """5th and 95th percentiles of ``metric(returns) - metric(other)``.

    A moving-block bootstrap that draws the same blocks from both series, like
    :func:`paired_block_bootstrap_sharpe_difference`. Fewer than ``2 * block`` jointly finite
    rows give ``(nan, nan)``.
    """
    values = np.asarray(returns, dtype=np.float64)
    paired = np.asarray(other, dtype=np.float64)
    if len(values) != len(paired):
        raise ValueError("PAIRED_RETURNS_NOT_ALIGNED")
    finite = np.isfinite(values) & np.isfinite(paired)
    values, paired = values[finite], paired[finite]
    n = len(values)
    if n < block * 2:
        return float("nan"), float("nan")
    rng = np.random.Generator(np.random.PCG64(seed))
    starts = np.arange(n - block + 1)
    count = math.ceil(n / block)
    differences = np.empty(samples)
    for k in range(samples):
        picks = rng.choice(starts, size=count)
        index = np.concatenate([np.arange(s, s + block) for s in picks])[:n]
        differences[k] = metric_from_returns(metric, values[index]) - metric_from_returns(
            metric, paired[index]
        )
    return float(np.percentile(differences, 5)), float(np.percentile(differences, 95))


def verdict(
    metric: str,
    direction: Literal["higher", "lower"],
    subject: RunSeries,
    baseline: RunSeries,
    *,
    comparable: bool,
    block: int = 20,
    samples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """What a predeclared comparison rule allows one to say about the subject.

    ``supported`` needs the subject to be better than the baseline on the recorded metric and
    the paired 90 percent interval of the difference to lie wholly on that side. Better with an
    interval that contains zero is ``insufficient``, as are arms that are not comparable, a
    metric that is missing, or too few sessions to resample. Not better is ``not_supported``.
    Descriptive: a verdict is not a gate and sets no claim level.
    """
    if direction not in ("higher", "lower"):
        raise ValueError(f"COMPARE_DIRECTION_UNKNOWN:{direction}")
    ours = subject.document.get("metrics", {}).get(metric)
    theirs = baseline.document.get("metrics", {}).get(metric)
    result: dict[str, Any] = {
        "metric": metric,
        "direction": direction,
        "subject": ours,
        "baseline": theirs,
        "difference": None,
        "interval_90": None,
    }
    if not comparable:
        return {**result, "outcome": "insufficient", "reason": "the arms are not comparable"}
    if ours is None or theirs is None:
        return {**result, "outcome": "insufficient", "reason": f"{metric} is not available for both arms"}
    difference = float(ours) - float(theirs)
    result["difference"] = round(difference, 8)
    better = difference > 0 if direction == "higher" else difference < 0
    low, high = paired_metric_interval(
        metric, subject.returns, baseline.returns, block=block, samples=samples, seed=seed
    )
    if math.isfinite(low) and math.isfinite(high):
        result["interval_90"] = [round(low, 8), round(high, 8)]
    if not better:
        return {**result, "outcome": "not_supported", "reason": "the subject is not better than the baseline"}
    if result["interval_90"] is None:
        return {**result, "outcome": "insufficient", "reason": "too few sessions to resample"}
    clear = low > 0 if direction == "higher" else high < 0
    if not clear:
        return {
            **result,
            "outcome": "insufficient",
            "reason": "the subject is better, but the interval of the difference contains zero",
        }
    return {
        **result,
        "outcome": "supported",
        "reason": "the subject is better and the interval excludes zero",
    }
