# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pytest

from signalquarry._internal.validation.stats import (
    ReturnMoments,
    _denominator,
    block_bootstrap_sharpe,
    deflated_sharpe,
    expected_max_sharpe,
    min_track_record_length,
    moments,
    pbo_cscv,
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


def test_block_bootstrap_matches_small_fixed_seed_reference() -> None:
    # For PCG64(7), the eight two-block draws start at
    # (2,1), (2,2), (1,2), (2,0), (0,0), (0,2), (2,0), (1,2).
    # Compute each draw's sample Sharpe independently with sample stdev, then
    # linearly interpolate the 5th and 95th percentiles.
    series = np.array([0.01, 0.02, -0.01, 0.03])
    assert block_bootstrap_sharpe(series, block=2, samples=8, seed=7) == pytest.approx(
        (0.3638034375544994, 1.944923306528644), abs=1e-12
    )


# The tests below close gaps a mutation-testing pass (hardening track A5) found: each
# one is written against a hand-computed expectation independent of the formula under
# test, not a directional/self-referential check, because the survivors they kill are
# exactly the mutations a directional check cannot see (a coefficient or an operator
# swapped for one that still points the same way).


def test_moments_nan_boundary_is_exactly_at_n_equals_3() -> None:
    # n=2 must be the degenerate (NaN) case and n=3 the first real one -- distinguishes
    # `n < 3` from `n <= 3`, and confirms the NaN path returns NaN for every moment
    # (not just the ones a directional test happens to look at).
    below = moments(np.array([0.01, 0.02]))
    assert below.n == 2
    assert math.isnan(below.sharpe) and math.isnan(below.skew) and math.isnan(below.kurtosis)
    at = moments(np.array([0.01, 0.02, 0.015]))
    assert at.n == 3
    assert math.isfinite(at.sharpe) and math.isfinite(at.skew) and math.isfinite(at.kurtosis)


def test_denominator_exact_value_distinguishes_multiply_from_divide() -> None:
    # 1 - skew*sharpe + (kurtosis-1)/4*sharpe**2, hand-computed: a `skew/sharpe` typo
    # gives a materially different (and directionally plausible, so hard to notice)
    # number here, not just a slightly-off one.
    m = ReturnMoments(n=100, sharpe=2.0, skew=-1.0, kurtosis=5.0)
    assert _denominator(m) == pytest.approx(math.sqrt(7.0), rel=1e-12)  # not sqrt(5.5)


def test_denominator_floors_at_1e_minus_12_when_the_expression_goes_negative() -> None:
    # skew*sharpe alone already exceeds 1, driving the expression well below zero;
    # nothing exercises this floor otherwise, so `max(value, 1e-12)` could silently
    # become `max(value, 1.000000000001)` (also always positive) without any test
    # noticing.
    m = ReturnMoments(n=100, sharpe=10.0, skew=5.0, kurtosis=1.0)
    assert _denominator(m) == pytest.approx(1e-6, rel=1e-9)  # sqrt(1e-12)


def test_probabilistic_sharpe_nan_paths_are_independent() -> None:
    # `n < 3 or not isfinite(sharpe)`: each side must independently trigger NaN, or an
    # `or`-to-`and` typo would only fail when unusual inputs coincide on both sides at
    # once -- which nothing here otherwise constructs.
    short_but_finite = ReturnMoments(n=2, sharpe=0.5, skew=0.0, kurtosis=3.0)
    assert math.isnan(probabilistic_sharpe(short_but_finite))
    long_but_nan = ReturnMoments(n=100, sharpe=float("nan"), skew=0.0, kurtosis=3.0)
    assert math.isnan(probabilistic_sharpe(long_but_nan))


def test_expected_max_sharpe_is_exactly_zero_at_zero_variance() -> None:
    # `sharpe_variance <= 0` vs `< 0`: zero variance must itself return 0.0, not just
    # negative variance.
    assert expected_max_sharpe(10, 0.0) == 0.0


def test_min_track_record_length_is_infinite_when_sharpe_equals_the_benchmark() -> None:
    # `sharpe <= benchmark` vs `< benchmark`: a strategy exactly at the benchmark can
    # never demonstrate it is *above* it, however long the track record.
    m = ReturnMoments(n=100, sharpe=0.3, skew=0.0, kurtosis=3.0)
    assert min_track_record_length(m, benchmark_sharpe=0.3) == math.inf


def test_pbo_rank_denominator_is_configurations_plus_one() -> None:
    # A minimal, hand-verified 4-period/2-configuration/2-block matrix. Config 0 is
    # picked as the in-sample best in both splits and ranks first out-of-sample too
    # (rank 2/3 -> logit=+ln(2), so PBO=0). A `configurations + 1` -> `+ 2` typo in the
    # rank formula moves the same split to a boundary logit of exactly 0 (<=0 counts as
    # overfit), flipping PBO to 1.0 -- a difference no directional "noise vs. a
    # dominant configuration" check would notice, since both scenarios here already
    # favour config 0.
    matrix = np.array(
        [
            [1.0, 5.0],
            [3.0, 1.0],
            [2.0, 8.0],
            [6.0, 2.0],
        ]
    )
    result = pbo_cscv(matrix, blocks=2)
    assert result["splits"] == 2
    assert result["configurations"] == 2
    assert result["pbo"] == 0.0
    assert result["median_logit"] == pytest.approx(math.log(2.0), abs=5e-6)


def test_pbo_cscv_accepts_exactly_two_configurations() -> None:
    # `values.shape[1] < 2` vs `<= 2`: two configurations is the documented minimum
    # ("N configurations of returns", N >= 2), not an off-by-one short of it.
    matrix = np.zeros((8, 2))
    matrix[:, 0] = 1.0
    result = pbo_cscv(matrix, blocks=2)
    assert result["splits"] > 0


def test_pbo_ranks_each_configuration_by_its_own_volatility() -> None:
    # In the first half, configuration 0 has a lower mean but far less variation.
    # It has the higher Sharpe in both halves, so both out-of-sample ranks are top.
    matrix = np.array([[1.0, 2.0], [1.1, 10.0], [3.0, 1.0], [3.1, 1.1]])
    result = pbo_cscv(matrix, blocks=2)
    assert result == {
        "pbo": 0.0,
        "splits": 2,
        "configurations": 2,
        "median_logit": 0.693147,
    }


def test_pbo_middle_out_of_sample_rank_counts_as_overfit() -> None:
    # The winning configuration in each half ranks second of three in the other.
    # Its relative rank is 1/2, hence logit zero in both CSCV splits.
    matrix = np.array([[2.9, 1.9, 0.9], [3.1, 2.1, 1.1], [1.9, 2.9, 0.9], [2.1, 3.1, 1.1]])
    assert pbo_cscv(matrix, blocks=2) == {
        "pbo": 1.0,
        "splits": 2,
        "configurations": 3,
        "median_logit": 0.0,
    }


def test_pbo_invalid_matrix_reports_the_observed_configuration_count() -> None:
    one_column = pbo_cscv(np.zeros((8, 1)), blocks=2)
    one_dimensional = pbo_cscv(np.zeros(8), blocks=2)
    assert math.isnan(one_column.pop("pbo"))
    assert one_column == {"splits": 0, "configurations": 1}
    assert math.isnan(one_dimensional.pop("pbo"))
    assert one_dimensional == {"splits": 0, "configurations": 0}
