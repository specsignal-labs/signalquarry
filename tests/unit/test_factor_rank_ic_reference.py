# SPDX-License-Identifier: Apache-2.0
"""Rank-IC diagnostics against hand-worked panels and an independent per-day reference."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest

from signalquarry._internal.factors.evaluate import (
    HorizonIC,
    ScorePanel,
    SyntheticLabels,
    rank_ic,
    redundancy,
)

_DATASET = "sha256:" + "1" * 64
_UNIVERSE = "sha256:" + "2" * 64
_LABELS = "sha256:" + "3" * 64


def _panels(
    scores: np.ndarray,
    outcomes: np.ndarray,
    *,
    eligible: np.ndarray | None = None,
    adv: np.ndarray | None = None,
    horizon: int = 1,
) -> tuple[ScorePanel, SyntheticLabels]:
    rows, columns = scores.shape
    sessions = tuple(date(2024, 1, 2) + timedelta(days=index) for index in range(rows))
    symbols = tuple(f"S{index:02}" for index in range(columns))
    panel = ScorePanel(
        _DATASET,
        _UNIVERSE,
        sessions,
        symbols,
        scores.astype(np.float64),
        np.ones(scores.shape, dtype=np.bool_) if eligible is None else eligible,
    )
    labels = SyntheticLabels(
        _DATASET,
        _LABELS,
        sessions,
        symbols,
        {horizon: outcomes.astype(np.float64)},
        predecision_adv=adv,
    )
    return panel, labels


def _percentile(values: np.ndarray) -> np.ndarray:
    count = len(values)
    return np.array([(np.sum(values < value) + np.sum(values == value) / 2) / count for value in values])


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 5:
        return None
    x, y = _percentile(left), _percentile(right)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    xc, yc = x - x.mean(), y - y.mean()
    return float((xc * yc).sum() / math.sqrt((xc * xc).sum() * (yc * yc).sum()))


def _reference(
    panel: ScorePanel,
    outcomes: np.ndarray,
    adv: np.ndarray | None,
    *,
    blocks: int,
    cost_bps: float,
    capital: float,
) -> HorizonIC:
    daily: list[float | None] = []
    buckets: list[list[float]] = [[] for _ in range(5)]
    top_net: list[float] = []
    spreads: list[float] = []
    turnover: list[float] = []
    capacity: list[float] = []
    complete = adv is not None
    previous: set[int] | None = None
    scored = eligible = 0
    for row in range(len(panel.sessions)):
        mask = panel.eligible[row] & np.isfinite(panel.scores[row]) & np.isfinite(outcomes[row])
        eligible += int(panel.eligible[row].sum())
        scored += int(mask.sum())
        columns = np.flatnonzero(mask)
        daily.append(_spearman(panel.scores[row, columns], outcomes[row, columns]))
        if len(columns) < 5:
            continue
        share = _percentile(panel.scores[row, columns])
        group = np.minimum((share * 5).astype(int), 4)
        if len(set(group.tolist())) != 5:
            continue
        means = [float(np.mean(outcomes[row, columns][group == q])) for q in range(5)]
        for q in range(5):
            buckets[q].append(means[q])
        top = {int(c) for c in columns[group == 4]}
        top_net.append((1 + means[4]) * (1 - cost_bps / 10_000) ** 2 - 1)
        spreads.append(means[4] - means[0])
        if previous is not None:
            moved = len(top ^ previous)
            weight_change = (
                sum(
                    abs((1 / len(top) if c in top else 0) - (1 / len(previous) if c in previous else 0))
                    for c in top | previous
                )
                / 2
            )
            assert moved >= 0
            turnover.append(weight_change)
        previous = top
        if adv is not None:
            values = adv[row, sorted(top)]
            if np.isfinite(values).all() and (values > 0).all():
                capacity.append(float(np.max(capital / len(top) / values)))
            else:
                complete = False
    valid = [value for value in daily if value is not None]
    mean = float(np.mean(valid)) if valid else None
    deviation = float(np.std(valid, ddof=1)) if len(valid) >= 3 else 0.0
    icir = mean / deviation if mean is not None and deviation > 0 else None
    folds = []
    for indices in np.array_split(np.arange(len(daily)), blocks):
        members = [daily[int(i)] for i in indices if daily[int(i)] is not None]
        folds.append(float(np.mean(members)) if members else None)
    quintiles = tuple(float(np.mean(item)) if item else None for item in buckets)
    monotonic = None
    if all(value is not None for value in quintiles):
        monotonic = _spearman(np.arange(5.0), np.array(quintiles, dtype=np.float64))
    return HorizonIC(
        1,
        len(valid),
        scored,
        eligible,
        mean,
        icir,
        tuple(folds),
        tuple(daily),
        quintiles,
        monotonic,
        float(np.mean(top_net)) if top_net else None,
        float(np.mean(spreads)) if spreads else None,
        float(np.mean(turnover)) if turnover else None,
        max(capacity) if complete and capacity else None,
    )


def _close(actual: object, expected: object) -> None:
    if expected is None:
        assert actual is None
    elif isinstance(expected, tuple):
        assert isinstance(actual, tuple) and len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _close(left, right)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12)
    else:
        assert actual == expected


def _assert_report(actual: HorizonIC, expected: HorizonIC) -> None:
    for name in HorizonIC.__dataclass_fields__:
        _close(getattr(actual, name), getattr(expected, name))


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("blocks", [1, 4, 7])
def test_rank_ic_matches_the_per_day_reference_with_ties_gaps_and_ineligible_names(
    seed: int, blocks: int
) -> None:
    rng = np.random.default_rng(seed)
    rows, columns = 31, 14
    scores = np.round(rng.normal(0, 1, (rows, columns)), 1)
    outcomes = np.round(0.4 * scores + rng.normal(0, 1, (rows, columns)), 2) / 50
    scores[rng.random(scores.shape) < 0.08] = np.nan
    outcomes[rng.random(outcomes.shape) < 0.08] = np.nan
    eligible = rng.random(scores.shape) > 0.1
    eligible[3, :] = False
    adv = rng.uniform(1_000, 50_000, scores.shape)
    adv[rng.random(adv.shape) < 0.01] = np.nan
    panel, labels = _panels(scores, outcomes, eligible=eligible, adv=adv)
    actual = rank_ic(panel, labels, blocks=blocks, cost_bps=7.5, capital=250_000.0).horizons[0]
    expected = _reference(panel, outcomes, adv, blocks=blocks, cost_bps=7.5, capital=250_000.0)
    _assert_report(actual, expected)


def test_rank_ic_without_adv_reports_no_capacity_and_matches_the_reference() -> None:
    rng = np.random.default_rng(11)
    scores = rng.normal(0, 1, (20, 12))
    outcomes = 0.01 * scores + rng.normal(0, 0.02, scores.shape)
    panel, labels = _panels(scores, outcomes)
    actual = rank_ic(panel, labels).horizons[0]
    assert actual.max_share_of_adv is None
    _assert_report(actual, _reference(panel, outcomes, None, blocks=6, cost_bps=10.0, capital=1_000_000.0))


def _ladder(rows: int = 2) -> tuple[np.ndarray, np.ndarray]:
    scores = np.tile(np.arange(10.0), (rows, 1))
    outcomes = np.tile(np.arange(10.0) / 100, (rows, 1))
    return scores, outcomes


def test_a_perfect_ranking_is_worked_by_hand() -> None:
    scores, outcomes = _ladder()
    adv = np.tile(np.arange(10.0) + 1, (2, 1)) * 1_000
    panel, labels = _panels(scores, outcomes, adv=adv)
    report = rank_ic(panel, labels).horizons[0]
    assert report.daily_ic == (pytest.approx(1.0), pytest.approx(1.0))
    assert report.mean_ic == pytest.approx(1.0)
    assert report.observations == 2 and report.scored_pairs == 20 and report.eligible_pairs == 20
    # quintile means of pairs (0,1)(2,3)(4,5)(6,7)(8,9) / 100
    assert report.quintile_returns == pytest.approx((0.005, 0.025, 0.045, 0.065, 0.085))
    assert report.quintile_monotonicity == pytest.approx(1.0)
    assert report.long_short_spread == pytest.approx(0.08)
    assert report.top_quintile_net_return == pytest.approx(1.085 * (1 - 0.001) ** 2 - 1)
    assert report.top_quintile_turnover == 0.0
    # top quintile = last two names, weight 1/2 each; capital / 2 / smallest adv in the top pair
    assert report.max_share_of_adv == pytest.approx(1_000_000 / 2 / 9_000)
    assert report.icir is None  # fewer than three daily values: no dispersion estimate
    assert report.chronological_blocks == (pytest.approx(1.0), pytest.approx(1.0), None, None, None, None)


def test_swapping_the_whole_top_quintile_is_full_turnover() -> None:
    scores, outcomes = _ladder(3)
    scores[1] = scores[1][::-1]
    panel, labels = _panels(scores, outcomes)
    report = rank_ic(panel, labels).horizons[0]
    # top sets: {8,9} -> {0,1} -> {8,9}: each change moves all weight, so half the absolute change is 1
    assert report.top_quintile_turnover == pytest.approx(1.0)
    partial, _ = _ladder(3)
    partial[1, 8] = -1.0  # top pair becomes {7, 9}: one of two names replaced
    swapped = rank_ic(*_panels(partial, outcomes)).horizons[0]
    # {8,9} -> {7,9}: |0 - .5| + |.5 - 0| = 1, halved; then {7,9} -> {8,9} again 0.5; mean 0.5
    assert swapped.top_quintile_turnover == pytest.approx(0.5)


def test_icir_uses_the_sample_deviation_of_three_or_more_daily_values() -> None:
    rng = np.random.default_rng(3)
    scores = rng.normal(0, 1, (12, 20))
    outcomes = (0.3 * scores + rng.normal(0, 1, scores.shape)) / 50
    panel, labels = _panels(scores, outcomes)
    report = rank_ic(panel, labels).horizons[0]
    values = np.array([value for value in report.daily_ic if value is not None])
    assert report.icir == pytest.approx(values.mean() / values.std(ddof=1), rel=1e-12)
    two = rank_ic(*_panels(scores[:2], outcomes[:2])).horizons[0]
    assert two.observations == 2 and two.icir is None
    three = rank_ic(*_panels(scores[:3], outcomes[:3])).horizons[0]
    assert three.observations == 3 and three.icir is not None


def test_five_pairs_are_the_minimum_for_a_daily_correlation() -> None:
    scores = np.array([[1.0, 2, 3, 4, 5, np.nan, np.nan, np.nan]])
    outcomes = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
    assert rank_ic(*_panels(scores, outcomes)).horizons[0].daily_ic == (pytest.approx(1.0),)
    four = np.array([[1.0, 2, 3, 4, np.nan, np.nan, np.nan, np.nan]])
    assert rank_ic(*_panels(four, outcomes)).horizons[0].daily_ic == (None,)


def test_the_spread_and_net_return_need_every_quintile_populated() -> None:
    # five pairs, one per quintile
    scores = np.array([[1.0, 2, 3, 4, 5]])
    outcomes = np.array([[0.01, 0.02, 0.03, 0.04, 0.05]])
    report = rank_ic(*_panels(scores, outcomes)).horizons[0]
    assert report.long_short_spread == pytest.approx(0.04)
    assert report.quintile_returns == pytest.approx((0.01, 0.02, 0.03, 0.04, 0.05))
    # a tie across the quintile boundary collapses one bucket
    tied = np.array([[1.0, 1, 3, 4, 5]])
    collapsed = rank_ic(*_panels(tied, outcomes)).horizons[0]
    assert collapsed.long_short_spread is None and collapsed.quintile_returns == (None,) * 5


def test_zero_or_missing_adv_in_the_top_quintile_blocks_the_capacity_figure() -> None:
    scores, outcomes = _ladder(2)
    good = np.full((2, 10), 500.0)
    assert rank_ic(*_panels(scores, outcomes, adv=good)).horizons[0].max_share_of_adv == pytest.approx(
        1_000_000 / 2 / 500
    )
    zero = good.copy()
    zero[0, 9] = 0.0
    assert rank_ic(*_panels(scores, outcomes, adv=zero)).horizons[0].max_share_of_adv is None
    tiny = good.copy()
    tiny[:, 8:] = 0.5
    assert rank_ic(*_panels(scores, outcomes, adv=tiny)).horizons[0].max_share_of_adv == pytest.approx(
        1_000_000 / 2 / 0.5
    )
    missing = good.copy()
    missing[1, 8] = np.nan
    assert rank_ic(*_panels(scores, outcomes, adv=missing)).horizons[0].max_share_of_adv is None


def test_total_loss_outcomes_are_allowed_but_not_worse() -> None:
    scores, outcomes = _ladder(2)
    outcomes[:, 0] = -1.0
    rank_ic(*_panels(scores, outcomes))
    outcomes[:, 0] = -1.0000001
    with pytest.raises(ValueError, match="^FACTOR_LABEL_HORIZONS_INVALID$"):
        rank_ic(*_panels(scores, outcomes))


@pytest.mark.parametrize(
    "options",
    [
        {"blocks": 0},
        {"blocks": True},
        {"blocks": 2.0},
        {"cost_bps": -0.01},
        {"cost_bps": 10_000},
        {"cost_bps": math.nan},
        {"cost_bps": math.inf},
        {"capital": 0.0},
        {"capital": -1.0},
        {"capital": math.nan},
        {"capital": math.inf},
    ],
)
def test_invalid_diagnostic_options_use_the_stable_code(options: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="^FACTOR_LABEL_ALIGNMENT_INVALID$"):
        rank_ic(*_panels(*_ladder()), **options)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "options",
    [
        {"blocks": 1},
        {"blocks": 100},
        {"cost_bps": 0},
        {"cost_bps": 9_999.99},
        {"capital": 1e-9},
        {"capital": 1},
    ],
)
def test_diagnostic_option_limits_are_inclusive_where_stated(options: dict[str, object]) -> None:
    rank_ic(*_panels(*_ladder()), **options)  # type: ignore[arg-type]


def test_cost_is_charged_on_both_entry_and_exit() -> None:
    scores, outcomes = _ladder(2)
    free = rank_ic(*_panels(scores, outcomes), cost_bps=0).horizons[0].top_quintile_net_return
    charged = rank_ic(*_panels(scores, outcomes), cost_bps=100).horizons[0].top_quintile_net_return
    assert free == pytest.approx(0.085)
    assert charged == pytest.approx(1.085 * 0.99**2 - 1)


def test_misaligned_inputs_are_rejected() -> None:
    panel, labels = _panels(*_ladder())
    for broken in (
        replace(labels, dataset_identity="sha256:" + "9" * 64),
        replace(labels, sessions=labels.sessions[:-1]),
        replace(labels, symbols=labels.symbols[:-1]),
        replace(labels, label_identity="sha256:xyz"),
        replace(labels, label_identity="sha256:" + "A" * 64),
        replace(labels, predecision_adv=np.ones((1, 1))),
    ):
        with pytest.raises(ValueError, match="^FACTOR_LABEL_ALIGNMENT_INVALID$"):
            rank_ic(panel, broken)
    with pytest.raises(ValueError, match="^FACTOR_LABEL_ALIGNMENT_INVALID$"):
        rank_ic(replace(panel, eligible=panel.eligible[:, :-1]), labels)
    with pytest.raises(ValueError, match="^FACTOR_LABEL_ALIGNMENT_INVALID$"):
        rank_ic(panel, object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="^FACTOR_LABEL_HORIZONS_INVALID$"):
        rank_ic(panel, replace(labels, forward_returns={}))
    with pytest.raises(ValueError, match="^FACTOR_LABEL_HORIZONS_INVALID$"):
        rank_ic(panel, replace(labels, forward_returns={1: labels.forward_returns[1][:, :-1]}))
    bad = labels.forward_returns[1].copy()
    bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="^FACTOR_LABEL_HORIZONS_INVALID$"):
        rank_ic(panel, replace(labels, forward_returns={1: bad}))


def test_redundancy_is_the_mean_daily_rank_correlation_over_shared_eligible_names() -> None:
    rng = np.random.default_rng(9)
    base = rng.normal(0, 1, (8, 12))
    other = 0.5 * base + rng.normal(0, 1, base.shape)
    other[2, :] = np.nan
    eligible = np.ones(base.shape, dtype=bool)
    eligible[4, :9] = False
    panel, _ = _panels(base, base, eligible=eligible)
    twin, _ = _panels(other, other, eligible=eligible)
    result = redundancy(panel, {"zeta": twin, "alpha": panel})
    assert [item.accepted_factor for item in result] == ["alpha", "zeta"]
    assert result[0].mean_rank_correlation == pytest.approx(1.0) and result[0].observations == 7
    daily = []
    for row in range(8):
        mask = eligible[row] & np.isfinite(base[row]) & np.isfinite(other[row])
        value = _spearman(base[row, mask], other[row, mask])
        if value is not None:
            daily.append(value)
    assert result[1].observations == len(daily) == 6
    assert result[1].mean_rank_correlation == pytest.approx(float(np.mean(daily)), rel=1e-12)
    # an ineligible name on either side drops out of the pair
    only_other = replace(twin, eligible=np.zeros(base.shape, dtype=bool))
    assert redundancy(panel, {"none": only_other})[0].mean_rank_correlation is None


@pytest.mark.parametrize(
    "change",
    [
        {"dataset_identity": "sha256:" + "9" * 64},
        {"universe_identity": "sha256:" + "9" * 64},
        {"sessions": ()},
        {"symbols": ("A",)},
    ],
)
def test_redundancy_rejects_unaligned_panels(change: dict[str, object]) -> None:
    panel, _ = _panels(*_ladder())
    with pytest.raises(ValueError, match="^FACTOR_REDUNDANCY_ALIGNMENT_INVALID$"):
        redundancy(panel, {"other": replace(panel, **change)})
    with pytest.raises(ValueError, match="^FACTOR_REDUNDANCY_ALIGNMENT_INVALID$"):
        redundancy(panel, {"other": replace(panel, scores=panel.scores[:, :-1])})
    with pytest.raises(ValueError, match="^FACTOR_REDUNDANCY_ALIGNMENT_INVALID$"):
        redundancy(panel, {"other": replace(panel, eligible=panel.eligible[:, :-1])})
