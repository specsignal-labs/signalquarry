# SPDX-License-Identifier: Apache-2.0
"""PSR, DSR and MinTRL cross-checked against the authors' own published worked examples.

Unlike ``tests/unit/test_stats.py`` (which re-derives its expectations from the same
formula, so it cannot catch a bug shared between the test and the code), every value
below is copied from the paper itself, independent of this implementation:

* Bailey & Lopez de Prado, "The Sharpe Ratio Efficient Frontier", Journal of Risk,
  2012 -- Section 3 (Figure 6/7) and Appendix A.3 (Figure 11's exact reference
  implementation, whose own worked example this test reproduces line for line).
* Bailey & Lopez de Prado, "The Deflated Sharpe Ratio", Journal of Portfolio
  Management, 2014 -- "A Numerical Example" and Exhibit 2.
"""

from __future__ import annotations

import math

import pytest

from signalquarry._internal.validation.stats import (
    ReturnMoments,
    deflated_sharpe,
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe,
)


def test_psr_matches_the_hedge_fund_track_record_example() -> None:
    # Sharpe Ratio Efficient Frontier, Section 3: a monthly track record over 2 years
    # (n=24), SR=0.458. PSR(0) is 0.982 assuming Normality, and 0.913 once the track
    # record's actual skew (-2.448) and kurtosis (10.164) are taken into account.
    m = ReturnMoments(n=24, sharpe=0.458, skew=0.0, kurtosis=3.0)
    assert probabilistic_sharpe(m, 0.0) == pytest.approx(0.982, abs=5e-4)
    m = ReturnMoments(n=24, sharpe=0.458, skew=-2.448, kurtosis=10.164)
    assert probabilistic_sharpe(m, 0.0) == pytest.approx(0.913, abs=5e-4)


def test_min_track_record_length_matches_the_reference_implementation() -> None:
    # Appendix A.3's own reference Python code, run with its own stated inputs:
    # SR-hat=2/sqrt(12), skew=-0.72, kurtosis=5.78, SR*=1/sqrt(12) (all monthly).
    # MinTRL(0.95) = 59.895 months (~4.99 years, matching Figure 11's table).
    sr, sr_star = 2 / math.sqrt(12), 1 / math.sqrt(12)
    m = ReturnMoments(n=1, sharpe=sr, skew=-0.72, kurtosis=5.78)  # n unused by MinTRL
    trl = min_track_record_length(m, benchmark_sharpe=sr_star, confidence=0.95)
    assert trl == pytest.approx(59.895, abs=5e-4)
    # The paper's own corroboration: PSR at exactly that sample length is 0.95.
    m = ReturnMoments(n=trl, sharpe=sr, skew=-0.72, kurtosis=5.78)
    assert probabilistic_sharpe(m, benchmark_sharpe=sr_star) == pytest.approx(0.95, abs=5e-4)


def test_dsr_matches_the_treasury_seasonality_example() -> None:
    # Deflated Sharpe Ratio, "A Numerical Example": N=100 independent trials,
    # V[{SR_n}]=1/2 (annualized), T=1250 daily observations (250/year), skew=-3,
    # kurtosis=10, selected annualized SR=2.5. DSR = 0.9004 < 0.95: not significant.
    #
    # sharpe_variance is the variance of the TRIALS' Sharpe ratios, in the same
    # per-period units as `sharpe` itself (see the module docstring) -- exactly like
    # `sharpe`, it must be de-annualized by dividing by the number of periods per
    # year, here 250. Passing the paper's annualized 0.5 directly would silently
    # answer a different, much harder question and understate DSR by a wide margin.
    sr = 2.5 / math.sqrt(250)
    daily_variance = 0.5 / 250
    m = ReturnMoments(n=1250, sharpe=sr, skew=-3.0, kurtosis=10.0)

    sr0 = expected_max_sharpe(100, daily_variance)
    assert sr0 == pytest.approx(0.1132, abs=5e-4)
    assert deflated_sharpe(m, 100, daily_variance) == pytest.approx(0.9004, abs=5e-4)

    # The paper's own boundary check: at N=46 trials DSR crosses back above 0.95.
    assert deflated_sharpe(m, 46, daily_variance) == pytest.approx(0.9505, abs=5e-4)
    # And with Normal returns (skew=0, kurtosis=3) instead, the same crossing point
    # moves out to N=88 trials -- non-Normality alone accounts for the difference.
    m_normal = ReturnMoments(n=1250, sharpe=sr, skew=0.0, kurtosis=3.0)
    assert deflated_sharpe(m_normal, 88, daily_variance) == pytest.approx(0.9505, abs=5e-4)
