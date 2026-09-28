# SPDX-License-Identifier: Apache-2.0
"""Dated factor scores and descriptive rank IC on explicitly synthetic labels.

The factor sees only completed bars through ``run_factor``. Membership timing
is checked locally, but a higher layer must verify the source manifests before
any real-data claim. This module deliberately emits no evidence grade.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Literal

import numpy as np

from signalquarry._internal.canonical import canonical_hash
from signalquarry._internal.data.panel import LoadedPanel
from signalquarry._internal.engine.factors import run_factor
from signalquarry.sdk.factors import FactorDef
from signalquarry.sdk.strategy import Params
from signalquarry.sdk.xs import rank

_IDENTITY = re.compile(r"sha256:[0-9a-f]{64}\Z")
MIN_PAIRS = 5


def _immutable(values: np.ndarray) -> np.ndarray:
    return np.frombuffer(values.tobytes(order="C"), dtype=values.dtype).reshape(values.shape)


@dataclass(frozen=True)
class UniverseAt:
    """Membership observed no later than the decision's explicit cutoff."""

    session: date
    observed_at: datetime
    decision_cutoff: datetime
    symbols: tuple[str, ...]
    identity: str


@dataclass(frozen=True)
class ScorePanel:
    dataset_identity: str
    universe_identity: str
    sessions: tuple[date, ...]
    symbols: tuple[str, ...]
    scores: np.ndarray
    eligible: np.ndarray


@dataclass(frozen=True)
class SyntheticLabels:
    """Forward returns from synthetic outcomes; values are not an executable P&L claim."""

    dataset_identity: str
    label_identity: str
    sessions: tuple[date, ...]
    symbols: tuple[str, ...]
    forward_returns: Mapping[int, np.ndarray]
    predecision_adv: np.ndarray | None = None
    outcome_end_sessions: Mapping[int, tuple[date | None, ...]] | None = None


@dataclass(frozen=True)
class HorizonIC:
    horizon: int
    observations: int
    scored_pairs: int
    eligible_pairs: int
    mean_ic: float | None
    icir: float | None
    chronological_blocks: tuple[float | None, ...]
    daily_ic: tuple[float | None, ...]
    quintile_returns: tuple[float | None, ...]
    quintile_monotonicity: float | None
    top_quintile_net_return: float | None
    long_short_spread: float | None
    top_quintile_turnover: float | None
    max_share_of_adv: float | None


@dataclass(frozen=True)
class DiagnosticReport:
    scope: Literal["synthetic"]
    dataset_identity: str
    universe_identity: str
    label_identity: str
    horizons: tuple[HorizonIC, ...]


@dataclass(frozen=True)
class Redundancy:
    accepted_factor: str
    observations: int
    mean_rank_correlation: float | None


def score_factor(
    definition: FactorDef, params: Params, panel: LoadedPanel, membership: Sequence[UniverseAt]
) -> ScorePanel:
    """Score dated, observed universes through pre-decision panel windows."""
    if not _IDENTITY.fullmatch(panel.dataset_identity) or not membership:
        raise ValueError("FACTOR_EVALUATION_IDENTITY_INVALID")
    dates = tuple(item.session for item in membership)
    if dates != tuple(sorted(set(dates))) or not set(dates) <= set(panel.sessions):
        raise ValueError("FACTOR_EVALUATION_SESSIONS_INVALID")
    columns = {symbol: index for index, symbol in enumerate(panel.symbols)}
    scores = np.full((len(membership), len(columns)), np.nan, dtype=np.float64)
    eligible = np.zeros(scores.shape, dtype=np.bool_)
    identity_rows = []
    for row, item in enumerate(membership):
        if (
            item.observed_at.utcoffset() is None
            or item.decision_cutoff.utcoffset() is None
            or item.observed_at > item.decision_cutoff
            or item.decision_cutoff.astimezone(UTC).date() >= item.session
            or not _IDENTITY.fullmatch(item.identity)
        ):
            raise ValueError("FACTOR_UNIVERSE_TIMING_INVALID")
        if (
            not item.symbols
            or len(set(item.symbols)) != len(item.symbols)
            or not set(item.symbols) <= columns.keys()
        ):
            raise ValueError("FACTOR_UNIVERSE_INVALID")
        for symbol in item.symbols:
            eligible[row, columns[symbol]] = True
        observed = run_factor(definition, params, panel.window(decision_session=item.session), item.symbols)
        for symbol, value in observed.items():
            scores[row, columns[symbol]] = value
        identity_rows.append(
            [item.session, item.observed_at, item.decision_cutoff, item.symbols, item.identity]
        )
    return ScorePanel(
        panel.dataset_identity,
        canonical_hash(identity_rows),
        dates,
        panel.symbols,
        _immutable(scores),
        _immutable(eligible),
    )


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    x, y = rank(left), rank(right)
    if len(x) < MIN_PAIRS or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else None


def rank_ic(
    scores: ScorePanel,
    labels: SyntheticLabels,
    *,
    blocks: int = 6,
    cost_bps: float = 10.0,
    capital: float = 1_000_000.0,
) -> DiagnosticReport:
    """Describe score/forward-return association without significance or alpha claims."""
    if (
        scores.dataset_identity != labels.dataset_identity
        or scores.sessions != labels.sessions
        or scores.symbols != labels.symbols
        or not _IDENTITY.fullmatch(labels.label_identity)
        or scores.scores.shape != scores.eligible.shape
        or scores.scores.shape != (len(scores.sessions), len(scores.symbols))
        or type(blocks) is not int
        or blocks < 1
        or not math.isfinite(cost_bps)
        or not 0 <= cost_bps < 10_000
        or not math.isfinite(capital)
        or capital <= 0
        or (labels.predecision_adv is not None and labels.predecision_adv.shape != scores.scores.shape)
    ):
        raise ValueError("FACTOR_LABEL_ALIGNMENT_INVALID")
    if not labels.forward_returns:
        raise ValueError("FACTOR_LABEL_HORIZONS_INVALID")
    reports = []
    for horizon, outcomes in sorted(labels.forward_returns.items()):
        if (
            type(horizon) is not int
            or horizon < 1
            or outcomes.shape != scores.scores.shape
            or np.isinf(outcomes).any()
            or np.any(np.isfinite(outcomes) & (outcomes < -1))
        ):
            raise ValueError("FACTOR_LABEL_HORIZONS_INVALID")
        daily: list[float | None] = []
        scored_pairs = eligible_pairs = 0
        quintiles: list[list[float]] = [[] for _ in range(5)]
        top_net: list[float] = []
        spreads: list[float] = []
        turnover: list[float] = []
        capacity: list[float] = []
        capacity_complete = labels.predecision_adv is not None
        prior_weights: np.ndarray | None = None
        for row in range(len(scores.sessions)):
            available = scores.eligible[row]
            paired = available & np.isfinite(scores.scores[row]) & np.isfinite(outcomes[row])
            eligible_pairs += int(available.sum())
            scored_pairs += int(paired.sum())
            daily.append(_correlation(scores.scores[row, paired], outcomes[row, paired]))
            if paired.sum() < MIN_PAIRS:
                continue
            columns = np.flatnonzero(paired)
            bucket = np.minimum((rank(scores.scores[row, columns]) * 5).astype(int), 4)
            if len(set(bucket.tolist())) != 5:
                continue
            means = [float(outcomes[row, columns[bucket == q]].mean()) for q in range(5)]
            for q, value in enumerate(means):
                quintiles[q].append(value)
            top = columns[bucket == 4]
            top_net.append((1 + means[4]) * (1 - cost_bps / 10_000) ** 2 - 1)
            spreads.append(means[4] - means[0])
            weights = np.zeros(len(scores.symbols), dtype=np.float64)
            weights[top] = 1 / len(top)
            if prior_weights is not None:
                turnover.append(float(np.abs(weights - prior_weights).sum() / 2))
            prior_weights = weights
            if labels.predecision_adv is not None:
                adv = labels.predecision_adv[row, top]
                if np.isfinite(adv).all() and (adv > 0).all():
                    capacity.append(float(np.max(capital / len(top) / adv)))
                else:
                    capacity_complete = False
        valid = np.array([value for value in daily if value is not None], dtype=np.float64)
        mean_ic = float(valid.mean()) if len(valid) else None
        std = float(valid.std(ddof=1)) if len(valid) >= 3 else 0.0
        icir = float(valid.mean() / std) if std > 0 else None
        fold_means = []
        for indices in np.array_split(np.arange(len(daily)), blocks):
            values = [value for index in indices if (value := daily[int(index)]) is not None]
            fold_means.append(float(np.mean(values)) if values else None)
        quintile_means = tuple(float(np.mean(values)) if values else None for values in quintiles)
        monotonicity = (
            _correlation(np.arange(5, dtype=np.float64), np.array(quintile_means, dtype=np.float64))
            if all(value is not None for value in quintile_means)
            else None
        )
        reports.append(
            HorizonIC(
                horizon,
                len(valid),
                scored_pairs,
                eligible_pairs,
                mean_ic,
                icir,
                tuple(fold_means),
                tuple(daily),
                quintile_means,
                monotonicity,
                float(np.mean(top_net)) if top_net else None,
                float(np.mean(spreads)) if spreads else None,
                float(np.mean(turnover)) if turnover else None,
                max(capacity) if capacity_complete and capacity else None,
            )
        )
    return DiagnosticReport(
        "synthetic", scores.dataset_identity, scores.universe_identity, labels.label_identity, tuple(reports)
    )


def redundancy(scores: ScorePanel, accepted: Mapping[str, ScorePanel]) -> tuple[Redundancy, ...]:
    """Descriptive score correlation against accepted factors on identical dates/universe."""
    results = []
    for name, other in sorted(accepted.items()):
        if (
            scores.dataset_identity != other.dataset_identity
            or scores.universe_identity != other.universe_identity
            or scores.sessions != other.sessions
            or scores.symbols != other.symbols
            or scores.scores.shape != other.scores.shape
            or scores.eligible.shape != other.eligible.shape
        ):
            raise ValueError("FACTOR_REDUNDANCY_ALIGNMENT_INVALID")
        daily = []
        for row in range(len(scores.sessions)):
            paired = (
                scores.eligible[row]
                & other.eligible[row]
                & np.isfinite(scores.scores[row])
                & np.isfinite(other.scores[row])
            )
            value = _correlation(scores.scores[row, paired], other.scores[row, paired])
            if value is not None:
                daily.append(value)
        results.append(Redundancy(name, len(daily), float(np.mean(daily)) if daily else None))
    return tuple(results)
