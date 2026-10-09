# SPDX-License-Identifier: Apache-2.0
"""Reference cases for resampled comparison metrics and the rule-bound study verdict."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from signalquarry._internal.validation.compare import (
    STUDY_METRICS,
    RunSeries,
    metric_from_returns,
    paired_metric_interval,
    verdict,
)

# +10%, -10%, +10%, -20%: the curve is 1.1, 0.99, 1.089, 0.8712; the peak is 1.1.
PATH = np.array([0.1, -0.1, 0.1, -0.2])


def test_metrics_from_returns_reference_vector() -> None:
    cagr = 0.8712 ** (252 / 4) - 1
    expected = {
        "total_return": 0.8712 - 1,
        "cagr": cagr,
        "annual_volatility": 0.15 * math.sqrt(252),
        "sharpe": -0.025 / 0.15 * math.sqrt(252),
        "sortino": -0.025 / math.sqrt(0.025) * math.sqrt(252),
        "max_drawdown": 1 - 0.8712 / 1.1,
        "calmar": cagr / (1 - 0.8712 / 1.1),
    }
    assert set(expected) == set(STUDY_METRICS)
    for metric, value in expected.items():
        assert metric_from_returns(metric, PATH) == pytest.approx(value, rel=1e-12), metric


def test_metric_boundaries() -> None:
    for metric in STUDY_METRICS:
        assert math.isnan(metric_from_returns(metric, np.array([0.1])))
    flat = np.zeros(5)
    assert metric_from_returns("sharpe", flat) == 0.0  # no dispersion
    assert metric_from_returns("sortino", np.array([0.1, 0.2])) == 0.0  # no losing session
    assert metric_from_returns("calmar", np.array([0.1, 0.2])) == 0.0  # no drawdown
    assert metric_from_returns("max_drawdown", np.array([0.1, 0.2])) == 0.0
    ruined = np.array([0.5, -1.0, 0.2])
    assert metric_from_returns("total_return", ruined) == -1.0
    assert metric_from_returns("cagr", ruined) == -1.0
    with pytest.raises(ValueError, match="COMPARE_METRIC_UNKNOWN:alpha"):
        metric_from_returns("alpha", PATH)


RNG = np.random.Generator(np.random.PCG64(21))
BASE = RNG.normal(0.0003, 0.01, 600)


@pytest.mark.parametrize("metric", STUDY_METRICS)
def test_a_series_against_itself_has_a_zero_interval(metric: str) -> None:
    assert paired_metric_interval(metric, BASE, BASE, samples=50) == (0.0, 0.0)


def test_the_interval_is_antisymmetric_and_follows_the_metric() -> None:
    better = BASE + 0.0005
    low, high = paired_metric_interval("sharpe", better, BASE, samples=200)
    assert 0 < low < high
    swapped = paired_metric_interval("sharpe", BASE, better, samples=200)
    assert swapped == pytest.approx((-high, -low))

    # Half the exposure: a smaller drawdown in every resample, and a lower volatility.
    half = BASE * 0.5
    assert paired_metric_interval("max_drawdown", half, BASE, samples=200)[1] < 0
    assert paired_metric_interval("annual_volatility", half, BASE, samples=200)[1] < 0
    assert paired_metric_interval("total_return", better, BASE, samples=200)[0] > 0


def test_interval_alignment_length_and_non_finite_rows() -> None:
    with pytest.raises(ValueError, match="PAIRED_RETURNS_NOT_ALIGNED"):
        paired_metric_interval("sharpe", BASE, BASE[:-1])
    low, high = paired_metric_interval("sharpe", BASE[:39], BASE[:39] + 0.001)
    assert math.isnan(low) and math.isnan(high)
    assert not math.isnan(paired_metric_interval("sharpe", BASE[:40], BASE[:40] + 0.001, samples=20)[0])
    holed = BASE.copy()
    holed[5] = np.nan
    kept = np.delete(BASE, 5)
    assert paired_metric_interval("sharpe", holed, BASE + 0.001, samples=30) == paired_metric_interval(
        "sharpe", kept, np.delete(BASE + 0.001, 5), samples=30
    )
    assert paired_metric_interval("sharpe", BASE, BASE + 0.001, samples=30, seed=1) != (
        paired_metric_interval("sharpe", BASE, BASE + 0.001, samples=30, seed=2)
    )


DAYS = tuple(date(2020, 1, 1) + timedelta(days=i) for i in range(len(BASE)))


def _run(returns: np.ndarray, **metrics: float | None) -> RunSeries:
    recorded = {name: metric_from_returns(name, returns) for name in STUDY_METRICS}
    return RunSeries({"metrics": {**recorded, **metrics}}, DAYS[: len(returns)], returns)


def test_supported_needs_better_and_an_interval_clear_of_zero() -> None:
    result = verdict("sharpe", "higher", _run(BASE + 0.0005), _run(BASE), comparable=True, samples=200)
    assert result["outcome"] == "supported"
    assert result["difference"] > 0 and result["interval_90"][0] > 0
    assert (result["metric"], result["direction"]) == ("sharpe", "higher")
    assert result["subject"] > result["baseline"]

    lower = verdict("max_drawdown", "lower", _run(BASE * 0.5), _run(BASE), comparable=True, samples=200)
    assert lower["outcome"] == "supported" and lower["difference"] < 0 and lower["interval_90"][1] < 0


def test_not_better_is_not_supported() -> None:
    result = verdict("sharpe", "higher", _run(BASE - 0.0005), _run(BASE), comparable=True, samples=100)
    assert result["outcome"] == "not_supported" and result["difference"] < 0
    assert result["interval_90"][1] < 0  # the interval is still reported
    worse = verdict("max_drawdown", "lower", _run(BASE), _run(BASE * 0.5), comparable=True, samples=100)
    assert worse["outcome"] == "not_supported"
    equal = verdict("sharpe", "higher", _run(BASE), _run(BASE), comparable=True, samples=50)
    assert equal["outcome"] == "not_supported" and equal["difference"] == 0.0


def test_better_but_not_distinguishable_is_insufficient() -> None:
    noisy = BASE + np.random.Generator(np.random.PCG64(4)).normal(0.00002, 0.004, len(BASE))
    subject, baseline = _run(noisy), _run(BASE)
    if subject.document["metrics"]["sharpe"] <= baseline.document["metrics"]["sharpe"]:
        subject, baseline = baseline, subject
    result = verdict("sharpe", "higher", subject, baseline, comparable=True, samples=300)
    low, high = result["interval_90"]
    assert low < 0 < high and result["difference"] > 0
    assert result["outcome"] == "insufficient" and "contains zero" in result["reason"]


def test_insufficient_without_comparability_a_metric_or_enough_sessions() -> None:
    better, base = _run(BASE + 0.0005), _run(BASE)
    apart = verdict("sharpe", "higher", better, base, comparable=False)
    assert apart["outcome"] == "insufficient" and apart["reason"] == "the arms are not comparable"
    assert apart["difference"] is None and apart["interval_90"] is None

    missing = verdict("sharpe", "higher", _run(BASE + 0.0005, sharpe=None), base, comparable=True)
    assert missing["outcome"] == "insufficient" and "not available" in missing["reason"]
    absent = verdict("sharpe", "higher", better, RunSeries({}, DAYS, BASE), comparable=True)
    assert absent["outcome"] == "insufficient"

    short = verdict("sharpe", "higher", _run(BASE[:30] + 0.002), _run(BASE[:30]), comparable=True)
    assert short["outcome"] == "insufficient" and short["reason"] == "too few sessions to resample"
    assert short["interval_90"] is None and short["difference"] > 0

    with pytest.raises(ValueError, match="COMPARE_DIRECTION_UNKNOWN:up"):
        verdict("sharpe", "up", better, base, comparable=True)  # type: ignore[arg-type]
