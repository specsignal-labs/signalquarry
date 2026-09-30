# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from signalquarry.sdk import Decision, ta


def test_target_normalizes_and_validates() -> None:
    decision = Decision.target({"spy": 0.5, "QQQ": "0.25", "IWM": 0}, "GO")
    assert decision.weights == {"QQQ": Decimal("0.250000"), "SPY": Decimal("0.500000")}
    equal = Decision.target({"SPY": "0.2", "QQQ": Decimal("0.200000")})
    assert equal.weights["SPY"] is equal.weights["QQQ"]
    with pytest.raises(ValueError, match="EXCEED_ONE"):
        Decision.target({"SPY": 0.7, "QQQ": 0.4})
    with pytest.raises(ValueError, match="NEGATIVE"):
        Decision.target({"SPY": -0.1})
    with pytest.raises(ValueError, match="SYMBOL_INVALID"):
        Decision.target({"S P Y": 0.1})
    with pytest.raises(ValueError, match="REASON_CODE_INVALID"):
        Decision.hold("not a code")


def test_ta_functions() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.isnan(ta.sma(values, 3)[1]) and ta.sma(values, 3)[-1] == pytest.approx(4.0)
    assert ta.returns(values)[-1] == pytest.approx(0.25)
    assert ta.ema(values, 3)[2] == pytest.approx(2.0) and ta.ema(values, 3)[-1] == pytest.approx(4.0)
    assert ta.rolling_std(values, 5)[-1] == pytest.approx(np.std(values, ddof=1))
    up = ta.rsi(np.arange(1.0, 30.0), 14)
    assert up[-1] == pytest.approx(100.0)


@pytest.mark.parametrize("period", [1, 2, 5, 17, 50])
def test_sma_matches_zero_prefixed_cumulative_reference(period: int) -> None:
    rng = np.random.default_rng(927)
    values = rng.normal(100, 3, 73)
    expected = np.full(values.shape, np.nan)
    csum = np.cumsum(np.insert(values, 0, 0.0))
    expected[period - 1 :] = (csum[period:] - csum[:-period]) / period
    np.testing.assert_array_equal(ta.sma(values, period), expected)


def test_sma_short_history_and_nonfinite_values() -> None:
    assert np.isnan(ta.sma(np.array([1.0, 2.0]), 3)).all()
    values = np.array([1.0, np.nan, 3.0, np.inf])
    with np.errstate(invalid="ignore"):
        old = np.cumsum(np.insert(values, 0, 0.0))
        expected = np.full(values.shape, np.nan)
        expected[1:] = (old[2:] - old[:-2]) / 2
        np.testing.assert_array_equal(ta.sma(values, 2), expected)
