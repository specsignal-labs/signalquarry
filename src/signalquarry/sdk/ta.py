# SPDX-License-Identifier: Apache-2.0
"""Technical-analysis helpers over completed-bar arrays. Pure numpy, no look-ahead.

Each function returns an array aligned with its input; positions without
enough history are NaN. Use ``[-1]`` for the value as of the last completed bar.
"""

from __future__ import annotations

import numpy as np


def _as_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def sma(values: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average."""
    data = _as_array(values)
    out = np.full(data.shape, np.nan)
    if period < 1:
        raise ValueError("TA_PERIOD_INVALID")
    if len(data) >= period:
        # Preallocate the zero prefix instead of np.insert, which copies and
        # reshapes on every call in a cross-sectional backtest.
        csum = np.empty(len(data) + 1, dtype=np.float64)
        csum[0] = 0.0
        np.cumsum(data, out=csum[1:])
        out[period - 1 :] = (csum[period:] - csum[:-period]) / period
    return out


def ema(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average seeded with the first ``period``-bar SMA."""
    data = _as_array(values)
    out = np.full(data.shape, np.nan)
    if period < 1:
        raise ValueError("TA_PERIOD_INVALID")
    if len(data) < period:
        return out
    alpha = 2.0 / (period + 1.0)
    out[period - 1] = data[:period].mean()
    for index in range(period, len(data)):
        out[index] = alpha * data[index] + (1.0 - alpha) * out[index - 1]
    return out


def returns(values: np.ndarray, periods: int = 1) -> np.ndarray:
    """Simple returns over ``periods`` bars."""
    data = _as_array(values)
    out = np.full(data.shape, np.nan)
    if len(data) > periods:
        out[periods:] = data[periods:] / data[:-periods] - 1.0
    return out


def rolling_std(values: np.ndarray, period: int) -> np.ndarray:
    """Rolling sample standard deviation."""
    data = _as_array(values)
    out = np.full(data.shape, np.nan)
    if period < 2:
        raise ValueError("TA_PERIOD_INVALID")
    for index in range(period - 1, len(data)):
        out[index] = np.std(data[index - period + 1 : index + 1], ddof=1)
    return out


def zscore(values: np.ndarray, period: int) -> np.ndarray:
    """Distance from the rolling mean in rolling standard deviations."""
    data = _as_array(values)
    return (data - sma(data, period)) / rolling_std(data, period)


def rsi(values: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's relative strength index."""
    data = _as_array(values)
    out = np.full(data.shape, np.nan)
    if len(data) <= period:
        return out
    delta = np.diff(data)
    gains, losses = np.clip(delta, 0, None), np.clip(-delta, 0, None)
    avg_gain, avg_loss = gains[:period].mean(), losses[:period].mean()
    for index in range(period, len(data)):
        if index > period:
            avg_gain = (avg_gain * (period - 1) + gains[index - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[index - 1]) / period
        out[index] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out
