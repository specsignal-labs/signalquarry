# SPDX-License-Identifier: Apache-2.0
"""Statistics for judging a backtest (Bailey & López de Prado). numpy + NormalDist only.

All Sharpe ratios here are per-period (daily), non-annualized, as the formulas
require. ``kurtosis`` is Pearson kurtosis (normal = 3).

* PSR(SR*) = Φ((SR - SR*) · √(n-1) / √(1 - γ3·SR + (γ4-1)/4 · SR²))
* DSR = PSR(SR*) with SR* = √V · ((1-γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e))),
  N = number of independent trials, V = variance of their Sharpe ratios,
  γ = Euler–Mascheroni constant.
* MinTRL = 1 + (1 - γ3·SR + (γ4-1)/4 · SR²) · (Φ⁻¹(1-α) / (SR - SR*))²
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

EULER_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()


@dataclass(frozen=True)
class ReturnMoments:
    n: int
    sharpe: float  # per period
    skew: float
    kurtosis: float  # Pearson (normal = 3)


def moments(returns: np.ndarray) -> ReturnMoments:
    values = np.asarray(returns, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 3:
        return ReturnMoments(n, float("nan"), float("nan"), float("nan"))
    std = values.std(ddof=1)
    if std == 0:
        return ReturnMoments(n, 0.0, 0.0, 3.0)
    centered = values - values.mean()
    m2 = np.mean(centered**2)
    skew = float(np.mean(centered**3) / m2**1.5)
    kurt = float(np.mean(centered**4) / m2**2)
    return ReturnMoments(n, float(values.mean() / std), skew, kurt)


def _denominator(m: ReturnMoments) -> float:
    value = 1.0 - m.skew * m.sharpe + (m.kurtosis - 1.0) / 4.0 * m.sharpe**2
    return math.sqrt(max(value, 1e-12))


def probabilistic_sharpe(m: ReturnMoments, benchmark_sharpe: float = 0.0) -> float:
    if m.n < 3 or not math.isfinite(m.sharpe):
        return float("nan")
    z = (m.sharpe - benchmark_sharpe) * math.sqrt(m.n - 1) / _denominator(m)
    return _NORMAL.cdf(z)


def expected_max_sharpe(trials: int, sharpe_variance: float) -> float:
    """SR* : the Sharpe ratio expected from the best of ``trials`` unskilled attempts."""
    if trials <= 1 or sharpe_variance <= 0:
        return 0.0
    a = _NORMAL.inv_cdf(1.0 - 1.0 / trials)
    b = _NORMAL.inv_cdf(1.0 - 1.0 / (trials * math.e))
    return math.sqrt(sharpe_variance) * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b)


def deflated_sharpe(m: ReturnMoments, trials: int, sharpe_variance: float) -> float:
    return probabilistic_sharpe(m, expected_max_sharpe(trials, sharpe_variance))


def min_track_record_length(
    m: ReturnMoments, benchmark_sharpe: float = 0.0, confidence: float = 0.95
) -> float:
    if not math.isfinite(m.sharpe) or m.sharpe <= benchmark_sharpe:
        return float("inf")
    z = _NORMAL.inv_cdf(confidence)
    return 1.0 + _denominator(m) ** 2 * (z / (m.sharpe - benchmark_sharpe)) ** 2


def block_bootstrap_sharpe(
    returns: np.ndarray, *, block: int = 20, samples: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """5th and 95th percentiles of the per-period Sharpe under a moving-block bootstrap."""
    values = np.asarray(returns, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < block * 2:
        return float("nan"), float("nan")
    rng = np.random.Generator(np.random.PCG64(seed))
    starts = np.arange(n - block + 1)
    count = math.ceil(n / block)
    stats = np.empty(samples)
    for k in range(samples):
        picks = rng.choice(starts, size=count)
        sample = np.concatenate([values[s : s + block] for s in picks])[:n]
        std = sample.std(ddof=1)
        stats[k] = sample.mean() / std if std > 0 else 0.0
    return float(np.percentile(stats, 5)), float(np.percentile(stats, 95))


def pbo_cscv(matrix: np.ndarray, *, blocks: int = 10) -> dict[str, float | int]:
    """Probability of backtest overfitting by combinatorially symmetric cross-validation.

    ``matrix`` is T periods × N configurations of returns over a common window
    (Bailey, Borwein, López de Prado and Zhu, 2017). The rows are split into ``blocks``
    contiguous blocks; for every choice of half the blocks as in-sample, the
    configuration with the best in-sample Sharpe is ranked out of sample. PBO is the
    share of splits in which it ranks at or below the median (logit ≤ 0).
    """
    from itertools import combinations

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or blocks % 2 or values.shape[0] < blocks * 2:
        return {
            "pbo": float("nan"),
            "splits": 0,
            "configurations": int(values.shape[1] if values.ndim == 2 else 0),
        }
    periods, configurations = values.shape
    edges = np.linspace(0, periods, blocks + 1).astype(int)
    parts = [values[edges[k] : edges[k + 1]] for k in range(blocks)]

    def sharpe(rows: np.ndarray) -> np.ndarray:
        std = rows.std(axis=0, ddof=1)
        return np.divide(rows.mean(axis=0), std, out=np.zeros(configurations), where=std > 0)

    logits = []
    for chosen in combinations(range(blocks), blocks // 2):
        inside = np.concatenate([parts[k] for k in chosen])
        outside = np.concatenate([parts[k] for k in range(blocks) if k not in chosen])
        best = int(np.argmax(sharpe(inside)))
        oos = sharpe(outside)
        rank = float((oos < oos[best]).sum() + 1) / (configurations + 1)  # relative rank in (0, 1)
        logits.append(math.log(rank / (1.0 - rank)))
    lambdas = np.array(logits)
    return {
        "pbo": round(float((lambdas <= 0).mean()), 6),
        "splits": len(logits),
        "configurations": configurations,
        "median_logit": round(float(np.median(lambdas)), 6),
    }
