# SPDX-License-Identifier: Apache-2.0
"""Deterministic panel structure, causal volume signal and dated memberships."""

from __future__ import annotations

import re
from dataclasses import FrozenInstanceError
from datetime import date, datetime
from decimal import Decimal

import numpy as np
import pytest

from signalquarry._internal.data.dataset import FIELDS, MICRO
from signalquarry._internal.data.synthetic import (
    SyntheticPanel,
    synthetic_dataset,
    synthetic_memberships,
    synthetic_panel,
    weekday_sessions,
)


def test_existing_synthetic_dataset_identity_is_unchanged() -> None:
    dataset = synthetic_dataset(date(2014, 1, 2), date(2025, 12, 31), symbols=("SYNA",))
    assert dataset.identity() == "sha256:6229c5b059545c501be654386fcc482441ec74bac8085c53bb87145489022947"


def test_synthetic_panel_is_deterministic_and_signal_is_read_only() -> None:
    kwargs = dict(n_symbols=20, n_sectors=4, planted_ic=0.1, seed=17)
    first = synthetic_panel(date(2020, 1, 1), date(2020, 12, 31), **kwargs)
    other = synthetic_panel(date(2020, 1, 1), date(2020, 12, 31), **{**kwargs, "seed": 18})
    # An intervening generation with different arguments cannot advance a shared RNG.
    second = synthetic_panel(date(2020, 1, 1), date(2020, 12, 31), **kwargs)
    assert first.dataset.identity() == second.dataset.identity()
    assert first.dataset.identity() != other.dataset.identity()
    np.testing.assert_array_equal(first.signal, second.signal)
    assert not np.array_equal(first.signal, other.signal)
    assert first.sectors == second.sectors
    assert first.listed == second.listed
    assert first.dataset.source == "synthetic-panel:seed=17:n=20:ic=0.1"
    assert first.signal.dtype == np.float64
    assert first.signal.shape == (len(first.dataset.sessions), 20)
    assert not first.signal.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        first.signal[0, 0] = 0.0
    with pytest.raises(FrozenInstanceError):
        first.symbols = ()


@pytest.fixture(scope="module")
def structured_panel() -> SyntheticPanel:
    return synthetic_panel(
        date(2020, 1, 1),
        date(2021, 12, 31),
        n_symbols=80,
        n_sectors=5,
        late_listing_fraction=0.25,
        delisting_fraction=0.20,
        missing_rate=0.0,
        split_fraction=0.20,
        dividend_fraction=0.40,
    )


def test_synthetic_panel_structure_and_actions(structured_panel: SyntheticPanel) -> None:
    panel = structured_panel
    dataset = panel.dataset
    assert dataset.sessions == weekday_sessions(date(2020, 1, 1), date(2021, 12, 31))
    assert panel.symbols == dataset.symbols == tuple(f"S{i:04d}" for i in range(1, 81))
    assert all(re.fullmatch(r"[A-Z][A-Z0-9.]{0,9}", symbol) for symbol in panel.symbols)
    assert set(panel.sectors) == set(panel.symbols)
    assert set(panel.sectors.values()) == {f"SEC{i}" for i in range(5)}
    assert sum(first > dataset.sessions[0] for first, _ in panel.listed.values()) == 20
    assert sum(last is not None for _, last in panel.listed.values()) == 16
    for symbol, item in dataset.series.items():
        first, last = panel.listed[symbol]
        observed = np.flatnonzero(item.present)
        assert dataset.sessions[observed[0]] == first
        assert dataset.sessions[observed[-1]] == (last or dataset.sessions[-1])
        assert item.present.dtype == np.bool_
        assert np.all(item.volume[item.present] > 0)
        assert np.all(item.volume[~item.present] == 0)
        prices = {name: item.micro[name][item.present] for name in FIELDS}
        for name in FIELDS:
            assert item.micro[name].dtype == np.int64
            assert np.all(prices[name] > 0)
            assert np.all(prices[name] % (MICRO // 100) == 0)
            assert np.all(item.micro[name][~item.present] == 0)
        assert np.all(prices["low"] <= prices["open"])
        assert np.all(prices["low"] <= prices["close"])
        assert np.all(prices["high"] >= prices["open"])
        assert np.all(prices["high"] >= prices["close"])

    assert len(dataset.splits) == len({action.symbol for action in dataset.splits}) == 16
    for split in dataset.splits:
        assert split.symbol in panel.symbols
        assert split.ratio == Decimal(2)
        ex = dataset.index_of(split.ex_date)
        item = dataset.series[split.symbol]
        assert item.present[ex - 1] and item.present[ex]
        raw_ratio = item.micro["close"][ex] / item.micro["close"][ex - 1]
        assert 0.40 < raw_ratio < 0.60
        factor = dataset.cumulative_split(split.symbol)
        np.testing.assert_array_equal(factor[:ex], 1.0)
        np.testing.assert_array_equal(factor[ex:], 2.0)

    assert len({action.symbol for action in dataset.dividends}) == 32
    for dividend in dataset.dividends:
        assert dividend.symbol in panel.symbols
        assert dividend.amount > 0
        assert dividend.ex_date <= dividend.pay_date <= dataset.sessions[-1]
        assert dataset.series[dividend.symbol].present[dataset.index_of(dividend.ex_date)]
    for symbol in {action.symbol for action in dataset.dividends}:
        ex_indices = [dataset.index_of(d.ex_date) for d in dataset.dividends if d.symbol == symbol]
        assert np.all(np.diff(ex_indices) == 63)


def test_missing_bars_and_listing_boundaries() -> None:
    panel = synthetic_panel(date(2020, 1, 1), date(2020, 12, 31), n_symbols=40, n_sectors=4, missing_rate=0.2)
    dropped = available = 0
    for symbol in panel.symbols:
        item = panel.dataset.series[symbol]
        first, last = panel.listed[symbol]
        left = panel.dataset.index_of(first)
        right = panel.dataset.index_of(last) if last else len(item.present) - 1
        assert item.present[left] and item.present[right]
        assert not item.present[:left].any()
        assert not item.present[right + 1 :].any()
        interior = item.present[left + 1 : right]
        dropped += int((~interior).sum())
        available += len(interior)
    assert dropped / available == pytest.approx(0.2, abs=0.02)


def test_market_and_sector_structure_in_returns() -> None:
    panel = synthetic_panel(
        date(2020, 1, 1),
        date(2021, 12, 31),
        n_symbols=50,
        n_sectors=5,
        late_listing_fraction=0,
        delisting_fraction=0,
        missing_rate=0,
        split_fraction=0,
        dividend_fraction=0,
    )
    close = np.column_stack([panel.dataset.series[s].micro["close"] for s in panel.symbols])
    returns = np.diff(np.log(close), axis=0)
    correlation = np.corrcoef(returns, rowvar=False)
    pairs = np.triu(np.ones(correlation.shape, dtype=bool), k=1)
    sectors = np.array([panel.sectors[s] for s in panel.symbols])
    same_sector = sectors[:, None] == sectors[None, :]
    across = float(correlation[pairs & ~same_sector].mean())
    within = float(correlation[pairs & same_sector].mean())
    assert across > 0.02
    assert within > across + 0.01
    assert 0.015 < float(np.median(returns.std(axis=0))) < 0.03


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average zero-based ranks, including ties from volume/price rounding."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts + 1) / 2)[inverse]


def _mean_daily_spearman(scores: np.ndarray, returns: np.ndarray, mask: np.ndarray) -> float:
    daily = []
    for score, outcome, eligible in zip(scores, returns, mask, strict=True):
        if eligible.sum() < 5:
            continue
        x, y = _ranks(score[eligible]), _ranks(outcome[eligible])
        x -= x.mean()
        y -= y.mean()
        daily.append(float(np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y))))
    assert len(daily) > 700
    return float(np.mean(daily))


@pytest.mark.parametrize("planted_ic", [0.0, 0.1])
def test_synthetic_panel_signal_and_observable_volume_route(planted_ic: float) -> None:
    panel = synthetic_panel(date(2018, 1, 1), date(2020, 11, 13), n_symbols=300, planted_ic=planted_ic)
    dataset = panel.dataset
    close = np.column_stack(
        [dataset.series[s].micro["close"] * dataset.cumulative_split(s) for s in panel.symbols]
    )
    present = np.column_stack([dataset.series[s].present for s in panel.symbols])
    volume = np.column_stack([dataset.series[s].volume for s in panel.symbols])
    forward = np.zeros_like(close[:-1])
    np.divide(close[1:], close[:-1], out=forward, where=close[:-1] > 0)
    forward -= 1
    paired = present[:-1] & present[1:]
    ic = _mean_daily_spearman(panel.signal[:-1], forward, paired)
    if planted_ic:
        assert ic == pytest.approx(planted_ic, abs=0.02)
    else:
        assert abs(ic) < 0.01

    totals = np.vstack([np.zeros(300), np.cumsum(volume, axis=0)])
    counts = np.vstack([np.zeros(300, dtype=int), np.cumsum(present, axis=0)])
    rows = np.arange(20, len(dataset.sessions) - 1)
    trailing_mean = (totals[rows] - totals[rows - 20]) / 20
    full_history = (counts[rows] - counts[rows - 20]) == 20
    relative_volume = np.zeros_like(trailing_mean)
    np.divide(volume[20:-1], trailing_mean, out=relative_volume, where=trailing_mean > 0)
    volume_ic = _mean_daily_spearman(relative_volume, forward[20:], paired[20:] & full_history)
    if planted_ic:
        assert volume_ic >= planted_ic / 2
    else:
        assert abs(volume_ic) < 0.01
    assert float(panel.signal.mean()) == pytest.approx(0, abs=0.01)
    assert float(panel.signal.std()) == pytest.approx(1, abs=0.01)


def test_memberships_use_prior_session_and_prior_present_history() -> None:
    panel = synthetic_panel(
        date(2020, 1, 1),
        date(2020, 12, 31),
        n_symbols=30,
        n_sectors=3,
        late_listing_fraction=0.3,
        delisting_fraction=0.3,
        missing_rate=0.2,
    )
    every, history = 7, 20
    decisions = synthetic_memberships(panel, every=every, min_history=history)
    sessions = panel.dataset.sessions
    assert tuple(sessions.index(day) for day, _ in decisions) == tuple(range(history, len(sessions), every))
    for day, members in decisions:
        index = sessions.index(day)
        expected = tuple(
            symbol
            for symbol in sorted(panel.symbols)
            if panel.dataset.series[symbol].present[index - 1]
            and panel.dataset.series[symbol].present[:index].sum() >= history
        )
        assert members == expected
        for symbol, (_, last) in panel.listed.items():
            if last is not None and index > sessions.index(last) + 1:
                assert symbol not in members
    assert synthetic_memberships(panel, min_history=len(sessions)) == ()
    defaults = synthetic_memberships(panel)
    assert tuple(sessions.index(day) for day, _ in defaults) == tuple(range(20, len(sessions), 21))


def test_membership_excludes_the_decision_bar_and_keeps_empty_decisions() -> None:
    panel = synthetic_panel(
        date(2020, 1, 1),
        date(2020, 2, 28),
        n_symbols=2,
        n_sectors=1,
        late_listing_fraction=0,
        delisting_fraction=0,
        missing_rate=0,
    )
    for item in panel.dataset.series.values():
        item.present[:] = False
        item.present[:3] = True
    decisions = synthetic_memberships(panel, every=1, min_history=3)
    assert decisions[0] == (panel.dataset.sessions[3], panel.symbols)
    assert all(members == () for _, members in decisions[1:])


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("start", "2020-01-01"),
        ("start", datetime(2020, 1, 1)),
        ("end", None),
        ("end", date(2019, 12, 31)),
        ("end", date(2020, 1, 2)),
        ("end", date.max),
        ("n_symbols", 1),
        ("n_symbols", 3001),
        ("n_symbols", 2.5),
        ("n_symbols", True),
        ("n_sectors", 0),
        ("n_sectors", 17),
        ("n_sectors", "2"),
        ("seed", 1.5),
        ("seed", False),
        ("planted_ic", -0.1),
        ("planted_ic", 0.5001),
        ("planted_ic", np.nan),
        ("planted_ic", np.inf),
        ("planted_ic", "0.1"),
        ("planted_ic", True),
        *[
            (name, value)
            for name in (
                "late_listing_fraction",
                "delisting_fraction",
                "missing_rate",
                "split_fraction",
                "dividend_fraction",
            )
            for value in (-0.01, 1.01, np.nan, None, False)
        ],
    ],
)
def test_panel_argument_validation(name: str, value: object) -> None:
    kwargs = dict(start=date(2020, 1, 1), end=date(2020, 12, 31), n_symbols=16, n_sectors=4)
    kwargs[name] = value
    with pytest.raises(ValueError, match=f"^SYNTHETIC_PANEL_ARGUMENT_INVALID:{name}$"):
        synthetic_panel(**kwargs)


@pytest.mark.parametrize(
    ("name", "value"), [("every", 0), ("every", 1.5), ("min_history", 0), ("min_history", True)]
)
def test_membership_argument_validation(name: str, value: object) -> None:
    panel = synthetic_panel(date(2020, 1, 1), date(2020, 1, 31), n_symbols=8)
    with pytest.raises(ValueError, match=f"^SYNTHETIC_PANEL_ARGUMENT_INVALID:{name}$"):
        synthetic_memberships(panel, **{name: value})


def test_argument_boundaries_and_fraction_counts_round_down() -> None:
    panel = synthetic_panel(
        date(2020, 1, 1),
        date(2020, 1, 3),
        n_symbols=2,
        n_sectors=2,
        seed=-1,
        planted_ic=0.5,
        late_listing_fraction=1,
        delisting_fraction=1,
        missing_rate=1,
        split_fraction=1,
        dividend_fraction=1,
    )
    assert len(panel.dataset.splits) == len(panel.dataset.dividends) == 2
    assert all(item.present.tolist() == [False, True, False] for item in panel.dataset.series.values())
    small = synthetic_panel(date(2020, 1, 1), date(2020, 1, 31), n_symbols=9, n_sectors=1)
    assert all(first == small.dataset.sessions[0] and last is None for first, last in small.listed.values())
    assert small.dataset.splits == ()
    assert len({d.symbol for d in small.dataset.dividends}) == 2
    with pytest.raises(ValueError, match="^SYNTHETIC_PANEL_ARGUMENT_INVALID:panel$"):
        synthetic_memberships(None)
