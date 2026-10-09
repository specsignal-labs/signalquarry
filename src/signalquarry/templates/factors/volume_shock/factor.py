# SPDX-License-Identifier: Apache-2.0
"""Rank the last completed volume relative to its trailing mean."""

import numpy as np

from signalquarry.sdk import FactorCtx, Field, Params, factor, xs


class VolumeShockParams(Params):
    lookback: int = Field(20, ge=2, le=252)


@factor(params=VolumeShockParams, lookback=lambda p: p.lookback)
def volume_shock(ctx: FactorCtx, p: VolumeShockParams) -> dict[str, float]:
    volume = ctx.panel("volume")
    mean = volume[-p.lookback :].mean(axis=0)
    shock = np.divide(volume[-1], mean, out=np.full(mean.shape, np.nan), where=mean > 0)
    scores = xs.rank(shock)
    return {
        symbol: float(score) for symbol, score in zip(ctx.universe, scores, strict=True) if np.isfinite(score)
    }
