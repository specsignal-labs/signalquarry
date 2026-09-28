# SPDX-License-Identifier: Apache-2.0
"""Pure cross-sectional operators. NaN marks unavailable observations."""

from __future__ import annotations

import numpy as np


def _vector(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError("FACTOR_VECTOR_SHAPE")
    return data


def rank(values: np.ndarray) -> np.ndarray:
    """Average-tie percentile ranks in (0, 1); missing values stay NaN."""
    data = _vector(values)
    out = np.full(data.shape, np.nan)
    indices = np.flatnonzero(np.isfinite(data))
    order = indices[np.argsort(data[indices], kind="stable")]
    count = len(order)
    start = 0
    while start < count:
        stop = start + 1
        while stop < count and data[order[stop]] == data[order[start]]:
            stop += 1
        out[order[start:stop]] = (start + stop) / (2 * count)
        start = stop
    return out


def demean(values: np.ndarray) -> np.ndarray:
    data = _vector(values)
    out = np.full(data.shape, np.nan)
    mask = np.isfinite(data)
    if mask.any():
        out[mask] = data[mask] - data[mask].mean()
    return out


def zscore(values: np.ndarray) -> np.ndarray:
    """Population z-scores; zero cross-sectional dispersion is unavailable."""
    data = demean(values)
    mask = np.isfinite(data)
    if mask.any():
        scale = np.sqrt(np.mean(data[mask] ** 2))
        if scale > 0:
            data[mask] /= scale
        else:
            data[mask] = np.nan
    return data


def winsorize(values: np.ndarray, lower: float = 0.01, upper: float = 0.99) -> np.ndarray:
    if not (0 <= lower <= upper <= 1):
        raise ValueError("FACTOR_WINSOR_LIMITS")
    data = _vector(values)
    out = np.full(data.shape, np.nan)
    mask = np.isfinite(data)
    if mask.any():
        floor, ceiling = np.quantile(data[mask], [lower, upper])
        out[mask] = np.clip(data[mask], floor, ceiling)
    return out


def neutralize(values: np.ndarray, exposures: np.ndarray) -> np.ndarray:
    """OLS residuals against numeric exposure columns and an intercept.

    Rows missing a score or any exposure stay NaN. Rank-deficient designs
    fail rather than silently returning an arbitrary decomposition.
    """
    data = _vector(values)
    x = np.asarray(exposures, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] != len(data) or x.shape[1] < 1:
        raise ValueError("FACTOR_EXPOSURE_SHAPE")
    out = np.full(data.shape, np.nan)
    mask = np.isfinite(data) & np.isfinite(x).all(axis=1)
    design = np.column_stack((np.ones(mask.sum()), x[mask]))
    if len(design) <= design.shape[1]:
        raise ValueError("FACTOR_EXPOSURE_INSUFFICIENT")
    coefficients, _, matrix_rank, _ = np.linalg.lstsq(design, data[mask], rcond=None)
    if matrix_rank != design.shape[1]:
        raise ValueError("FACTOR_EXPOSURE_RANK_DEFICIENT")
    out[mask] = data[mask] - design @ coefficients
    return out
