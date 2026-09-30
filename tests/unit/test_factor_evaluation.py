# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from types import MappingProxyType

import numpy as np
import pytest

from signalquarry._internal.data.panel import LoadedPanel
from signalquarry._internal.factors.evaluate import (
    SyntheticLabels,
    UniverseAt,
    rank_ic,
    redundancy,
    score_factor,
)
from signalquarry.sdk import FactorCtx, Params, factor, factor_definition_of


class P(Params):
    period: int = 2


@factor(params=P, lookback=lambda p: p.period)
def previous_close(ctx: FactorCtx, p: P) -> dict[str, float]:
    del p
    return {symbol: float(ctx.panel("close")[-1, index]) for index, symbol in enumerate(ctx.universe)}


def _panel() -> LoadedPanel:
    sessions = tuple(date(2024, 1, 1) + timedelta(days=i) for i in range(55))
    symbols = tuple(f"S{i:02}" for i in range(10))
    prices = np.tile(np.arange(10, 20, dtype=np.int64) * 1_000_000, (len(sessions), 1))
    return LoadedPanel(
        dataset_identity="sha256:" + "1" * 64,
        sessions=sessions,
        symbols=symbols,
        micro=MappingProxyType({field: prices.copy() for field in ("open", "high", "low", "close")}),
        volume=np.ones(prices.shape),
        present=np.ones(prices.shape, dtype=bool),
    )


def _membership(panel: LoadedPanel) -> tuple[UniverseAt, ...]:
    return tuple(
        UniverseAt(
            session=session,
            observed_at=datetime.combine(session - timedelta(days=1), datetime.min.time(), UTC),
            decision_cutoff=datetime.combine(session, datetime.min.time(), UTC) + timedelta(hours=14),
            symbols=panel.symbols,
            identity="sha256:" + "2" * 64,
        )
        for session in panel.sessions[2:45]
    )


def _scores():
    panel = _panel()
    return score_factor(factor_definition_of(previous_close), P(), panel, _membership(panel))


def test_scoring_uses_only_completed_bars_and_dated_membership() -> None:
    panel = _panel()
    membership = _membership(panel)
    scores = score_factor(factor_definition_of(previous_close), P(), panel, membership)
    assert scores.scores.shape == (43, 10)
    assert scores.scores[0, 0] == 10.0
    assert scores.eligible.all()
    assert scores.universe_identity.startswith("sha256:")
    with pytest.raises(ValueError):
        scores.scores.flags.writeable = True

    panel.micro["close"][2:, :] = 999_000_000
    earlier = score_factor(factor_definition_of(previous_close), P(), panel, membership[:1])
    assert earlier.scores[0, 0] == 10.0


def test_rank_ic_finds_planted_signal_and_reports_null() -> None:
    scores = _scores()
    planted = np.tile(np.linspace(-0.05, 0.05, 10), (len(scores.sessions), 1))
    rng = np.random.Generator(np.random.PCG64(42))
    null = rng.normal(0, 0.01, planted.shape)
    labels = SyntheticLabels(
        scores.dataset_identity,
        "sha256:" + "3" * 64,
        scores.sessions,
        scores.symbols,
        {1: planted, 5: null},
        np.full(planted.shape, 1_000_000.0),
    )
    report = rank_ic(scores, labels)
    assert report.scope == "synthetic"
    assert report.dataset_identity == scores.dataset_identity
    assert report.universe_identity == scores.universe_identity
    assert report.label_identity == labels.label_identity
    first, second = report.horizons
    assert first.horizon == 1 and first.observations == 43
    assert first.mean_ic == pytest.approx(1.0)
    assert first.icir is None  # a constant IC has no measurable dispersion
    assert first.scored_pairs == first.eligible_pairs == 430
    assert first.chronological_blocks == pytest.approx((1.0,) * 6)
    assert first.quintile_monotonicity == pytest.approx(1.0)
    assert first.quintile_returns[0] < first.quintile_returns[-1]
    assert first.top_quintile_net_return == pytest.approx((1 + first.quintile_returns[-1]) * 0.999**2 - 1)
    assert first.long_short_spread == pytest.approx(first.quintile_returns[-1] - first.quintile_returns[0])
    assert first.top_quintile_turnover == pytest.approx(0.0)
    assert first.max_share_of_adv == pytest.approx(0.5)
    assert second.horizon == 5
    assert abs(second.mean_ic or 0) < 0.2


def test_factor_evaluation_rejects_bad_timing_membership_and_labels() -> None:
    panel = _panel()
    membership = _membership(panel)
    definition = factor_definition_of(previous_close)
    with pytest.raises(ValueError, match="FACTOR_EVALUATION_IDENTITY_INVALID"):
        score_factor(definition, P(), panel, ())
    with pytest.raises(ValueError, match="FACTOR_UNIVERSE_TIMING_INVALID"):
        score_factor(
            definition,
            P(),
            panel,
            (replace(membership[0], observed_at=membership[0].decision_cutoff + timedelta(seconds=1)),),
        )
    with pytest.raises(ValueError, match="FACTOR_UNIVERSE_INVALID"):
        score_factor(definition, P(), panel, (replace(membership[0], symbols=("S00", "S00")),))
    with pytest.raises(ValueError, match="FACTOR_EVALUATION_SESSIONS_INVALID"):
        score_factor(definition, P(), panel, (membership[1], membership[0]))
    scores = _scores()
    outcome = np.ones(scores.scores.shape)
    labels = SyntheticLabels(
        scores.dataset_identity, "sha256:" + "3" * 64, scores.sessions, scores.symbols, {1: outcome}
    )
    assert rank_ic(scores, labels).horizons[0].observations == 0
    with pytest.raises(ValueError, match="FACTOR_LABEL_ALIGNMENT_INVALID"):
        rank_ic(scores, replace(labels, dataset_identity="sha256:" + "9" * 64))
    with pytest.raises(ValueError, match="FACTOR_LABEL_HORIZONS_INVALID"):
        rank_ic(scores, replace(labels, forward_returns={0: outcome}))
    with pytest.raises(ValueError, match="FACTOR_LABEL_HORIZONS_INVALID"):
        rank_ic(scores, replace(labels, forward_returns={}))
    with pytest.raises(ValueError, match="FACTOR_LABEL_HORIZONS_INVALID"):
        rank_ic(scores, replace(labels, forward_returns={1: np.full(outcome.shape, -1.1)}))
    with pytest.raises(ValueError, match="FACTOR_LABEL_ALIGNMENT_INVALID"):
        rank_ic(scores, labels, cost_bps=-1)


def test_rank_ic_needs_five_non_tied_pairs() -> None:
    scores = _scores()
    values = np.full(scores.scores.shape, np.nan)
    values[:, :4] = np.arange(4)
    labels = SyntheticLabels(
        scores.dataset_identity, "sha256:" + "4" * 64, scores.sessions, scores.symbols, {1: values}
    )
    result = rank_ic(scores, labels).horizons[0]
    assert result.observations == 0 and result.mean_ic is None and result.icir is None
    assert result.scored_pairs == 4 * len(scores.sessions)
    tied = rank_ic(
        replace(scores, scores=np.ones(scores.scores.shape)),
        replace(labels, forward_returns={1: np.zeros(values.shape)}),
    ).horizons[0]
    assert tied.quintile_returns == (None,) * 5
    no_capacity = rank_ic(
        scores,
        replace(
            labels,
            forward_returns={1: np.tile(np.linspace(-0.05, 0.05, 10), (len(scores.sessions), 1))},
            predecision_adv=np.zeros(scores.scores.shape),
        ),
    ).horizons[0]
    assert no_capacity.max_share_of_adv is None


def test_redundancy_requires_aligned_provenance() -> None:
    scores = _scores()
    reports = {
        item.accepted_factor: item
        for item in redundancy(scores, {"same": scores, "inverse": replace(scores, scores=-scores.scores)})
    }
    same, inverse = reports["same"], reports["inverse"]
    assert (same.accepted_factor, same.observations, same.mean_rank_correlation) == (
        "same",
        len(scores.sessions),
        1.0,
    )
    assert inverse.mean_rank_correlation == pytest.approx(-1.0)
    missing = redundancy(scores, {"flat": replace(scores, scores=np.ones(scores.scores.shape))})[0]
    assert missing.observations == 0 and missing.mean_rank_correlation is None
    with pytest.raises(ValueError, match="FACTOR_REDUNDANCY_ALIGNMENT_INVALID"):
        redundancy(scores, {"other": replace(scores, universe_identity="sha256:" + "0" * 64)})
