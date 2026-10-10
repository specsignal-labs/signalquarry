# SPDX-License-Identifier: Apache-2.0
"""Descriptive OLS exposure to passive return series, with Newey–West uncertainty.

No fit here supplies evidence for a gate, claim or trial. A design with condition
number above 1e12 is treated as collinear rather than reporting unstable betas.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from signalquarry._internal.validation.metrics import MIN_RELATIVE_OBSERVATIONS, SESSIONS_PER_YEAR

MAX_REFERENCES = 8


def _aligned(returns: np.ndarray, references: Mapping[str, np.ndarray]) -> np.ndarray:
    if not references or len(references) > MAX_REFERENCES:
        raise ValueError("EXPOSURE_REFERENCES_INVALID")
    arrays = [returns, *references.values()]
    if any(
        not isinstance(array, np.ndarray)
        or array.ndim != 1
        or array.shape != returns.shape
        or array.dtype.kind != "f"
        for array in arrays
    ):
        raise ValueError("EXPOSURE_RETURNS_NOT_ALIGNED")
    return np.column_stack(arrays)


def _fit(rows: np.ndarray) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    rows = rows[np.isfinite(rows).all(axis=1)]
    n, columns = rows.shape
    summary: dict[str, Any] = {"observations": n, "status": "insufficient"}
    y, x = rows[:, 0], np.column_stack((np.ones(n), rows[:, 1:]))
    coefficients = residuals = np.array([])
    if n < max(MIN_RELATIVE_OBSERVATIONS, 10 * columns):
        return summary, x, coefficients, residuals
    if np.linalg.matrix_rank(x) != columns or np.linalg.cond(x) > 1e12:
        summary["status"] = "collinear"
        return summary, x, coefficients, residuals
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ coefficients
    # Numerical noise in an exact fit should not become a spurious enormous t.
    if np.linalg.norm(residuals) <= 32 * np.finfo(float).eps * np.linalg.norm(y):
        residuals = np.zeros(n)
    squared_error = float(residuals @ residuals)
    total_variation = float(np.sum((y - np.mean(y)) ** 2))
    varies = bool(np.ptp(y) > 0)
    r_squared = 1 - squared_error / total_variation if varies else None
    summary.update(
        status="ok",
        r_squared=None if r_squared is None else round(r_squared, 6),
        adjusted_r_squared=None
        if r_squared is None
        else round(1 - (1 - r_squared) * (n - 1) / (n - columns), 6),
        residual_volatility=round(float(np.sqrt(squared_error / (n - columns) * SESSIONS_PER_YEAR)), 8),
        mean_return_annual=round(float(np.mean(y) * SESSIONS_PER_YEAR), 8),
    )
    return summary, x, coefficients, residuals


def _newey_west(x: np.ndarray, residuals: np.ndarray, lag: int) -> np.ndarray:
    scores = x * residuals[:, None]
    meat = scores.T @ scores
    for offset in range(1, lag + 1):
        cross = scores[offset:].T @ scores[:-offset]
        meat += (1 - offset / (lag + 1)) * (cross + cross.T)
    # QR avoids squaring the design's condition number when forming the inverse.
    inverse_r = np.linalg.inv(np.linalg.qr(x, mode="reduced")[1])
    bread = inverse_r @ inverse_r.T
    n, columns = x.shape
    covariance = bread @ meat @ bread * n / (n - columns)
    return np.sqrt(np.maximum(np.diag(covariance), 0))


def _t(coefficient: float, standard_error: float) -> float | None:
    return None if standard_error == 0 else round(float(coefficient / standard_error), 6)


def exposures(returns: np.ndarray, references: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Regress aligned daily float arrays on an intercept and the references in input order.

    Nonfinite rows are dropped jointly. Annual contributions use arithmetic daily
    means, so together with annual alpha they sum to the annual mean return.
    """
    summary, x, coefficients, residuals = _fit(_aligned(returns, references))
    if summary["status"] != "ok":
        return summary
    n = summary["observations"]
    lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    errors = _newey_west(x, residuals, lag)
    summary.update(
        lag=lag,
        alpha_annual=round(float(coefficients[0] * SESSIONS_PER_YEAR), 8),
        alpha_t=_t(coefficients[0], errors[0]),
        references=[
            {
                "name": name,
                "beta": round(float(coefficients[index]), 6),
                "standard_error": round(float(errors[index]), 6),
                "t": _t(coefficients[index], errors[index]),
                "contribution_annual": round(
                    float(coefficients[index] * np.mean(x[:, index]) * SESSIONS_PER_YEAR), 8
                ),
            }
            for index, name in enumerate(references, 1)
        ],
    )
    return summary


def rolling_exposures(
    returns: np.ndarray,
    references: Mapping[str, np.ndarray],
    *,
    window: int = 126,
    step: int = 21,
) -> list[dict[str, Any]]:
    """Fit each window of original rows; ``end`` is its zero-based last row index."""
    if window < 2 or step < 1:
        raise ValueError("EXPOSURE_WINDOW_INVALID")
    rows = _aligned(returns, references)
    result: list[dict[str, Any]] = []
    for end in range(window - 1, len(rows), step):
        summary, _, coefficients, _ = _fit(rows[end - window + 1 : end + 1])
        entry: dict[str, Any] = {"end": end, "status": summary["status"]}
        if summary["status"] == "ok":
            entry.update(
                betas={
                    name: round(float(coefficients[index]), 6) for index, name in enumerate(references, 1)
                },
                r_squared=summary["r_squared"],
            )
        result.append(entry)
    return result
