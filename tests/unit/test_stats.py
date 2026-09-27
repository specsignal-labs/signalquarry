# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pytest

from signalquarry._internal.validation.stats import (
    ReturnMoments,
    block_bootstrap_sharpe,
    deflated_sharpe,
    expected_max_sharpe,
    min_track_record_length,
    moments,
    probabilistic_sharpe,
)


def test_psr_reduces_to_normal_case_by_hand() -> None:
    m = ReturnMoments(n=253, sharpe=0.1, skew=0.0, kurtosis=3.0)
    expected = NormalDist().cdf(0.1 * math.sqrt(252) / math.sqrt(1 + 0.5 * 0.01))
    assert probabilistic_sharpe(m) == pytest.approx(expected, rel=1e-12)


def test_negative_skew_and_fat_tails_lower_confidence() -> None:
    normal = ReturnMoments(1000, 0.05, 0.0, 3.0)
    ugly = ReturnMoments(1000, 0.05, -2.0, 12.0)
    assert probabilistic_sharpe(ugly) < probabilistic_sharpe(normal)


def test_more_trials_raise_the_bar() -> None:
    m = ReturnMoments(1250, 0.08, 0.0, 3.0)
    assert expected_max_sharpe(1, 0.001) == 0.0
    bars = [expected_max_sharpe(n, 0.001) for n in (2, 10, 100, 1000)]
    assert bars == sorted(bars) and bars[0] > 0
    assert deflated_sharpe(m, 1000, 0.001) < deflated_sharpe(m, 10, 0.001) < probabilistic_sharpe(m)


def test_min_track_record_length() -> None:
    m = ReturnMoments(500, 0.1, 0.0, 3.0)
    z = NormalDist().inv_cdf(0.95)
    assert min_track_record_length(m) == pytest.approx(1 + (1 + 0.25 * 2 * 0.01) * (z / 0.1) ** 2)
    assert min_track_record_length(ReturnMoments(500, -0.1, 0.0, 3.0)) == math.inf


def test_moments_and_bootstrap_on_known_series() -> None:
    rng = np.random.Generator(np.random.PCG64(1))
    returns = rng.normal(0.001, 0.01, 5000)
    m = moments(returns)
    assert (
        m.sharpe == pytest.approx(0.1, abs=0.02)
        and abs(m.skew) < 0.1
        and m.kurtosis == pytest.approx(3.0, abs=0.2)
    )
    low, high = block_bootstrap_sharpe(returns, samples=300)
    assert low < m.sharpe < high
