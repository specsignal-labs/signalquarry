# SPDX-License-Identifier: Apache-2.0
"""Independent references for the significance arithmetic used by formula search.

These formulas decide how hard it is to accept a discovered expression, so each is checked
against a separately written computation and against literal hand-worked values.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pytest

from signalquarry._internal.factors.search import (
    _benjamini_hochberg,
    _effective_observations,
    _expected_max_t,
    _t_statistic,
)

_GAMMA = 0.5772156649015329


def _autocorrelated(count: int, seed: int, persistence: float = 0.5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, count)
    values = np.empty(count)
    values[0] = noise[0]
    for index in range(1, count):
        values[index] = persistence * values[index - 1] + noise[index]
    return values


def _reference_effective(values: np.ndarray) -> float:
    """Bartlett-weighted long-run variance inflation, written lag by lag."""
    count = len(values)
    if count < 3:
        return float(count)
    centered = values - values.mean()
    gamma0 = float(np.sum(centered * centered))
    if gamma0 <= 0:
        return float(count)
    bandwidth = min(20, max(1, int(4 * (count / 100) ** (2 / 9))))
    inflation = 1.0
    for lag in range(1, min(bandwidth, count - 1) + 1):
        autocorrelation = sum(centered[i] * centered[i - lag] for i in range(lag, count)) / gamma0
        inflation += 2.0 * (1.0 - lag / (bandwidth + 1.0)) * autocorrelation
    inflation = max(1.0, inflation)
    return max(1.0, min(float(count), count / inflation))


@pytest.mark.parametrize("count", [3, 4, 10, 30, 99, 100, 250, 1_000])
@pytest.mark.parametrize("persistence", [0.0, 0.5, 0.9, -0.5])
def test_effective_observations_matches_the_lag_by_lag_reference(count: int, persistence: float) -> None:
    values = _autocorrelated(count, count, persistence)
    assert _effective_observations(values) == pytest.approx(_reference_effective(values), rel=1e-9)


def test_effective_observations_uses_the_bandwidth_cap_on_very_long_series() -> None:
    # 4 * (n / 100) ** (2 / 9) first exceeds 20 near n = 180,000: the cap must bind.
    values = _autocorrelated(200_000, 7, 0.3)
    centered = values - values.mean()
    gamma0 = float(np.dot(centered, centered))
    inflation = 1.0
    for lag in range(1, 21):
        inflation += 2.0 * (1.0 - lag / 21.0) * float(np.dot(centered[lag:], centered[:-lag])) / gamma0
    assert _effective_observations(values) == pytest.approx(len(values) / inflation, rel=1e-9)


def test_effective_observations_handles_tiny_constant_and_clipped_cases() -> None:
    assert _effective_observations(np.array([])) == 0.0
    assert _effective_observations(np.array([1.0])) == 1.0
    assert _effective_observations(np.array([1.0, 2.0])) == 2.0
    # constant series have no variance to scale by
    assert _effective_observations(np.full(50, 0.3)) == 50.0
    # negative autocorrelation cannot raise the effective count above the sample size
    alternating = np.tile([1.0, -1.0], 100)
    assert _effective_observations(alternating) == 200.0
    # a strongly persistent series cannot drop below one observation
    trend = np.cumsum(np.ones(60))
    assert _effective_observations(trend) >= 1.0
    assert _effective_observations(trend) < 60.0


def test_effective_observations_three_point_value_is_hand_worked() -> None:
    # centered = [-1, 0, 1]; gamma0 = 2; bandwidth 1; lag 1 autocorrelation = (0*-1 + 1*0)/2 = 0
    assert _effective_observations(np.array([1.0, 2.0, 3.0])) == 3.0
    # centered = [1, -2, 1]; gamma0 = 6; lag 1: (-2*1 + 1*-2)/6 = -2/3; weight 1/2 -> inflation 1 - 2/3 < 1 -> 1
    assert _effective_observations(np.array([1.0, -2.0, 1.0])) == 3.0
    # centered = [-1, -1, 2]; gamma0 = 6; lag 1: (-1*-1 + 2*-1)/6 = -1/6 -> clipped to 1
    assert _effective_observations(np.array([0.0, 0.0, 3.0])) == 3.0


def test_t_statistic_requires_thirty_observations_and_ignores_missing_days() -> None:
    thirty_less_one = tuple(float(i % 5) - 2.0 for i in range(29))
    assert _t_statistic(thirty_less_one) == (29, 29.0, None, None)
    assert _t_statistic((None,) * 40 + thirty_less_one) == (29, 29.0, None, None)
    series = tuple(0.01 + 0.02 * ((i * 7) % 11 - 5) / 5 for i in range(30))
    count, effective, icir, statistic = _t_statistic(series)
    values = np.array(series)
    assert count == 30
    assert effective == pytest.approx(_reference_effective(values), rel=1e-9)
    assert icir == pytest.approx(values.mean() / values.std(ddof=1), rel=1e-12)
    assert statistic == pytest.approx(icir * math.sqrt(effective), rel=1e-12)
    # Missing days are dropped before counting, not treated as zeros.
    assert _t_statistic((None, *series, None)) == (count, effective, icir, statistic)


def test_t_statistic_without_dispersion_is_undefined() -> None:
    assert _t_statistic((0.05,) * 30) == (30, 30.0, None, None)


def test_t_statistic_uses_the_sample_not_population_deviation() -> None:
    series = (0.1, -0.1) * 15
    _, _, icir, _ = _t_statistic(series)
    assert icir == pytest.approx(0.0, abs=1e-15)
    shifted = tuple(0.5 + value for value in series)
    _, _, icir, _ = _t_statistic(shifted)
    sample_deviation = math.sqrt(30 * 0.01 / 29)
    assert icir == pytest.approx(0.5 / sample_deviation, rel=1e-12)


def test_expected_max_t_matches_the_euler_mascheroni_approximation() -> None:
    assert _expected_max_t(0) == 0.0
    assert _expected_max_t(1) == 0.0
    normal = NormalDist()
    for trials in (2, 3, 10, 100, 5_000):
        expected = (1 - _GAMMA) * normal.inv_cdf(1 - 1 / trials) + _GAMMA * normal.inv_cdf(
            1 - 1 / (trials * math.e)
        )
        assert _expected_max_t(trials) == pytest.approx(expected, rel=1e-12)
    # trials=2: the first quantile is the median (0); only the second term remains
    assert _expected_max_t(2) == pytest.approx(_GAMMA * normal.inv_cdf(1 - 1 / (2 * math.e)), rel=1e-12)
    assert _expected_max_t(2) > 0.0
    # Hand-checked magnitudes: the expected maximum of 100 standard normals is about 2.5.
    assert 2.4 < _expected_max_t(100) < 2.6
    assert _expected_max_t(10) < _expected_max_t(100) < _expected_max_t(5_000)


def test_benjamini_hochberg_matches_the_textbook_adjustment() -> None:
    assert _benjamini_hochberg(()) == ()
    # sorted p: .005 (1), .01 (2), .03 (3), .04 (4) -> raw .02, .02, .04, .04 -> already monotone
    assert _benjamini_hochberg((0.01, 0.04, 0.03, 0.005)) == pytest.approx((0.02, 0.04, 0.04, 0.02))
    # a smaller p behind a larger raw value is lifted to the running minimum: raw .02, .15, .3
    assert _benjamini_hochberg((0.01, 0.1, 0.1)) == pytest.approx((0.03, 0.1, 0.1))
    # adjusted values never exceed one
    assert _benjamini_hochberg((0.9, 0.95, 0.99)) == pytest.approx((0.99, 0.99, 0.99))
    assert _benjamini_hochberg((0.5,)) == (0.5,)
    for adjusted, raw in zip(_benjamini_hochberg((0.2, 0.2, 0.9)), (0.2, 0.2, 0.9), strict=True):
        assert adjusted >= raw
