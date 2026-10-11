# SPDX-License-Identifier: Apache-2.0
"""Descriptive exposures, including an independent Bartlett HAC calculation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from signalquarry._internal.validation.exposure import (
    MAX_REFERENCES,
    _newey_west,
    exposures,
    rolling_exposures,
)
from signalquarry._internal.validation.metrics import MIN_RELATIVE_OBSERVATIONS, SESSIONS_PER_YEAR


def _model(n: int = 500) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(82)
    a = rng.normal(0.0008, 0.012, n)
    b = rng.normal(-0.0002, 0.009, n)
    return 0.0003 + 0.65 * a - 0.35 * b, {"A": a, "B": b}


def test_planted_model_exact_fit_contributions_and_zero_t() -> None:
    y, references = _model()
    result = exposures(y, references)
    assert result["status"] == "ok"
    assert result["alpha_annual"] == pytest.approx(0.0003 * SESSIONS_PER_YEAR, abs=1e-8)
    assert [row["name"] for row in result["references"]] == ["A", "B"]
    assert [row["beta"] for row in result["references"]] == [0.65, -0.35]
    assert result["r_squared"] == result["adjusted_r_squared"] == 1
    assert result["residual_volatility"] == 0
    assert result["alpha_t"] is None
    assert all(row["t"] is None and row["standard_error"] == 0 for row in result["references"])
    assert result["alpha_annual"] + sum(
        row["contribution_annual"] for row in result["references"]
    ) == pytest.approx(result["mean_return_annual"], abs=2e-8)


def test_noisy_planted_model_recovers_coefficients() -> None:
    y, references = _model(12000)
    y += np.random.default_rng(37).normal(0, 0.0008, len(y))
    result = exposures(y, references)
    assert [r["beta"] for r in result["references"]] == pytest.approx([0.65, -0.35], abs=0.003)
    assert result["alpha_annual"] == pytest.approx(0.0756, abs=0.004)
    assert result["alpha_annual"] + sum(
        r["contribution_annual"] for r in result["references"]
    ) == pytest.approx(result["mean_return_annual"], abs=2e-8)


def _independent_hac(x: np.ndarray, residuals: np.ndarray, lag: int) -> np.ndarray:
    """Sum outer products explicitly, independent of implementation vectorization."""
    n, p = x.shape
    meat = np.zeros((p, p))
    for t in range(n):
        meat += residuals[t] ** 2 * np.outer(x[t], x[t])
    for lag_index in range(1, lag + 1):
        weight = 1 - lag_index / (lag + 1)
        for t in range(lag_index, n):
            pair = residuals[t] * residuals[t - lag_index] * np.outer(x[t], x[t - lag_index])
            meat += weight * (pair + pair.T)
    inverse = np.linalg.inv(x.T @ x)
    return inverse @ meat @ inverse * n / (n - p)


def test_newey_west_against_independent_calculation_and_rounding() -> None:
    y, references = _model(400)
    rng = np.random.default_rng(5)
    noise = rng.normal(0, 0.002, len(y)) * (1 + abs(references["A"]) * 50)
    for t in range(1, len(noise)):
        noise[t] += 0.4 * noise[t - 1]
    y += noise
    x = np.column_stack((np.ones(len(y)), *references.values()))
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ coef
    lag = math.floor(4 * (len(y) / 100) ** (2 / 9))
    errors = np.sqrt(np.diag(_independent_hac(x, residuals, lag)))
    result = exposures(y, references)
    assert result["alpha_t"] == round(coef[0] / errors[0], 6)
    assert result["alpha_annual"] == round(coef[0] * 252, 8)
    assert result["mean_return_annual"] == round(np.mean(y) * 252, 8)
    r2 = 1 - sum(residuals**2) / sum((y - y.mean()) ** 2)
    assert result["r_squared"] == round(r2, 6)
    assert result["adjusted_r_squared"] == round(1 - (1 - r2) * (len(y) - 1) / (len(y) - 3), 6)
    assert result["residual_volatility"] == round(np.sqrt(sum(residuals**2) / (len(y) - 3) * 252), 8)
    for i, row in enumerate(result["references"], 1):
        assert row["beta"] == round(coef[i], 6)
        assert row["standard_error"] == round(errors[i], 6)
        assert row["t"] == round(coef[i] / errors[i], 6)
        assert row["contribution_annual"] == round(coef[i] * x[:, i].mean() * 252, 8)


def test_zero_lag_is_heteroskedasticity_consistent_small_sample_covariance() -> None:
    # The public minimum is 20, whose prescribed lag is already 2. Exercise the
    # covariance primitive at lag 0 rather than changing the public lag formula.
    y, references = _model(12)
    y += np.random.default_rng(1).normal(0, 0.001, len(y))
    x = np.column_stack((np.ones(len(y)), *references.values()))
    residuals = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    inverse = np.linalg.inv(x.T @ x)
    hc1 = inverse @ x.T @ np.diag(residuals**2) @ x @ inverse * len(y) / (len(y) - 3)
    assert _newey_west(x, residuals, 0) == pytest.approx(np.sqrt(np.diag(hc1)), rel=1e-12)


@pytest.mark.parametrize("n", [20, 30, 100, 252, 1000])
def test_lag_formula(n: int) -> None:
    y, refs = _model(n)
    result = exposures(y, {"A": refs["A"]})
    assert result["lag"] == math.floor(4 * (n / 100) ** (2 / 9))


@pytest.mark.parametrize("k", [1, 2, MAX_REFERENCES])
def test_minimum_observation_boundary(k: int) -> None:
    minimum = max(MIN_RELATIVE_OBSERVATIONS, 10 * (k + 1))
    rng = np.random.default_rng(7)
    refs = {str(i): rng.normal(0, 0.01, minimum) for i in range(k)}
    y = rng.normal(0, 0.01, minimum)
    assert exposures(y[:-1], {name: x[:-1] for name, x in refs.items()}) == {
        "status": "insufficient",
        "observations": minimum - 1,
    }
    assert exposures(y, refs)["status"] == "ok"


def test_nonfinite_rows_dropped_jointly_and_counted() -> None:
    y, refs = _model(34)
    y[0] = np.nan
    refs["A"][1] = np.inf
    refs["B"][2] = -np.inf
    result = exposures(y, refs)
    assert result == exposures(y[3:], {name: x[3:] for name, x in refs.items()})
    assert result["observations"] == 31
    y[3:5] = np.nan
    assert exposures(y, refs) == {"status": "insufficient", "observations": 29}
    assert exposures(np.full(34, np.nan), refs) == {"status": "insufficient", "observations": 0}


@pytest.mark.parametrize("cause", ["constant", "repeated", "combination", "ill_conditioned"])
def test_collinear_designs(cause: str) -> None:
    y, refs = _model(200)
    if cause == "constant":
        refs["B"] = np.ones(len(y))
    elif cause == "repeated":
        refs["B"] = refs["A"]
    elif cause == "combination":
        refs["C"] = refs["A"] + 2 * refs["B"]
    else:
        refs["B"] = refs["A"] + np.random.default_rng(12).normal(0, 5e-13, len(y))
        x = np.column_stack((np.ones(len(y)), *refs.values()))
        assert np.linalg.matrix_rank(x) == 3 and np.linalg.cond(x) > 1e12
    assert exposures(y, refs) == {"status": "collinear", "observations": len(y)}


@pytest.mark.parametrize("value", [0.0, 0.001])
def test_returns_do_not_vary(value: float) -> None:
    y, refs = _model()
    result = exposures(np.full_like(y, value), refs)
    assert result["status"] == "ok"
    assert result["r_squared"] is result["adjusted_r_squared"] is None
    assert result["residual_volatility"] == 0


@pytest.mark.parametrize(
    ("y", "refs", "code"),
    [
        (np.zeros(30), {}, "EXPOSURE_REFERENCES_INVALID"),
        (np.zeros(30), {str(i): np.zeros(30) for i in range(9)}, "EXPOSURE_REFERENCES_INVALID"),
        (np.zeros((30, 1)), {"A": np.zeros(30)}, "EXPOSURE_RETURNS_NOT_ALIGNED"),
        (np.zeros(30), {"A": np.zeros((30, 1))}, "EXPOSURE_RETURNS_NOT_ALIGNED"),
        (np.zeros(30), {"A": np.zeros(29)}, "EXPOSURE_RETURNS_NOT_ALIGNED"),
        (np.arange(30), {"A": np.zeros(30)}, "EXPOSURE_RETURNS_NOT_ALIGNED"),
        (np.zeros(30), {"A": np.arange(30)}, "EXPOSURE_RETURNS_NOT_ALIGNED"),
        ([0.0] * 30, {"A": np.zeros(30)}, "EXPOSURE_RETURNS_NOT_ALIGNED"),
        (np.zeros(30), {"A": [0.0] * 30}, "EXPOSURE_RETURNS_NOT_ALIGNED"),
    ],
)
def test_invalid_arrays(y, refs, code: str) -> None:
    for function in (exposures, rolling_exposures):
        with pytest.raises(ValueError, match=f"^{code}$"):
            function(y, refs)


def test_rolling_windows_and_slice_equivalence() -> None:
    y, refs = _model(200)
    rows = rolling_exposures(y, refs)
    assert len(rows) == 4 and [row["end"] for row in rows] == [125, 146, 167, 188]
    for row in rows:
        end = row["end"]
        result = exposures(y[end - 125 : end + 1], {name: x[end - 125 : end + 1] for name, x in refs.items()})
        assert row == {
            "end": end,
            "status": "ok",
            "betas": {r["name"]: r["beta"] for r in result["references"]},
            "r_squared": result["r_squared"],
        }
    assert rolling_exposures(y[:125], {name: x[:125] for name, x in refs.items()}) == []
    assert rolling_exposures(y, refs, window=200)[-1]["end"] == 199
    assert rolling_exposures(y, refs, window=2, step=200) == [{"end": 1, "status": "insufficient"}]
    refs["B"][:126] = refs["A"][:126]
    rows = rolling_exposures(y, refs)
    assert rows[0] == {"end": 125, "status": "collinear"}
    assert rows[-1]["status"] == "ok"
    refs["A"][80:] = np.nan
    assert rolling_exposures(y, refs)[-1] == {"end": 188, "status": "insufficient"}


@pytest.mark.parametrize(("window", "step"), [(1, 21), (0, 21), (126, 0), (126, -1)])
def test_invalid_rolling_arguments(window: int, step: int) -> None:
    y, refs = _model()
    with pytest.raises(ValueError, match="^EXPOSURE_WINDOW_INVALID$"):
        rolling_exposures(y, refs, window=window, step=step)
