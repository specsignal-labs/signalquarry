# SPDX-License-Identifier: Apache-2.0
"""Forward-return labels against an independent day-by-day reference and hand-worked boundaries."""

from __future__ import annotations

import math
from datetime import date, timedelta
from decimal import Decimal

import numpy as np
import pytest

from signalquarry._internal.data.dataset import MICRO, Dataset, Dividend, Split, SymbolSeries
from signalquarry._internal.factors.labels import derive_forward_return_labels


def _dataset(
    closes: list[str],
    *,
    present: list[bool] | None = None,
    volumes: list[float] | None = None,
    splits: tuple[Split, ...] = (),
    dividends: tuple[Dividend, ...] = (),
) -> Dataset:
    sessions = tuple(date(2024, 1, 1) + timedelta(days=index) for index in range(len(closes)))
    close = np.array([int(Decimal(value) * MICRO) for value in closes], dtype=np.int64)
    series = SymbolSeries(
        micro={name: close.copy() for name in ("open", "high", "low", "close")},
        volume=np.ones(len(closes)) if volumes is None else np.array(volumes, dtype=np.float64),
        present=np.ones(len(closes), dtype=bool) if present is None else np.array(present, dtype=bool),
    )
    return Dataset(sessions, {"AAA": series}, splits=splits, dividends=dividends, source="synthetic:labels")


def _day(dataset: Dataset, index: int) -> date:
    return dataset.sessions[index]


def _returns(
    dataset: Dataset, horizons: tuple[int, ...] = (1,), decisions: slice | None = None
) -> dict[int, np.ndarray]:
    sessions = dataset.sessions if decisions is None else dataset.sessions[decisions]
    labels = derive_forward_return_labels(
        dataset, decision_sessions=sessions, symbols=("AAA",), horizons=horizons
    )
    return {horizon: values[:, 0] for horizon, values in labels.forward_returns.items()}


def _reference(dataset: Dataset, decision: int, horizon: int) -> float:
    series = dataset.series["AAA"]
    entry, exit_ = decision - 1, decision + horizon - 1
    if decision == 0 or exit_ >= len(dataset.sessions):
        return math.nan
    if not series.present[entry : exit_ + 1].all() or (series.micro["close"][entry : exit_ + 1] <= 0).any():
        return math.nan
    shares, cash = Decimal(1), Decimal(0)
    for day in range(entry + 1, exit_ + 1):
        today = dataset.sessions[day]
        for dividend in dataset.dividends:
            if dividend.ex_date == today:
                cash += shares * dividend.amount
        for split in dataset.splits:
            if split.ex_date == today:
                shares *= split.ratio
    start = Decimal(int(series.micro["close"][entry])) / Decimal(MICRO)
    end = Decimal(int(series.micro["close"][exit_])) / Decimal(MICRO)
    return float((shares * end + cash) / start - 1)


def _reference_adv(dataset: Dataset, decision: int, lookback: int) -> float:
    series = dataset.series["AAA"]
    if decision - lookback < 0:
        return math.nan
    window = slice(decision - lookback, decision)
    if not series.present[window].all() or (series.micro["close"][window] <= 0).any():
        return math.nan
    if (series.volume[window] < 0).any() or not np.isfinite(series.volume[window]).all():
        return math.nan
    return float(np.mean(series.micro["close"][window].astype(np.float64) / MICRO * series.volume[window]))


def test_labels_match_the_day_by_day_reference_on_a_dense_random_history() -> None:
    rng = np.random.default_rng(5)
    rows = 40
    closes = [str(round(float(value), 4)) for value in 50 + np.cumsum(rng.normal(0, 1.0, rows))]
    present = [bool(value) for value in rng.random(rows) > 0.08]
    volumes = [float(value) for value in rng.uniform(0, 1_000, rows)]
    split_days = rng.choice(rows, size=4, replace=False)
    dividend_days = [*rng.choice(rows, size=5, replace=False), 0, rows - 1]
    ratios = [Decimal(2), Decimal("0.5"), Decimal(3), Decimal(1)]
    base = _dataset(closes, present=present, volumes=volumes)
    splits = tuple(Split("AAA", base.sessions[int(day)], ratios[i]) for i, day in enumerate(split_days))
    dividends = tuple(
        Dividend(
            "AAA",
            base.sessions[int(day)],
            base.sessions[min(rows - 1, int(day) + 1)],
            Decimal(str(round(0.1 + i * 0.3, 2))),
        )
        for i, day in enumerate(dict.fromkeys(dividend_days))
    )
    dataset = Dataset(base.sessions, base.series, splits=splits, dividends=dividends, source=base.source)
    horizons = (1, 2, 3, 5, 10)
    labels = derive_forward_return_labels(
        dataset, decision_sessions=dataset.sessions, symbols=("AAA",), horizons=horizons, adv_lookback=5
    )
    for horizon in horizons:
        for decision in range(rows):
            expected = _reference(dataset, decision, horizon)
            actual = labels.forward_returns[horizon][decision, 0]
            if math.isnan(expected):
                assert math.isnan(actual), (horizon, decision)
            else:
                assert actual == pytest.approx(expected, rel=1e-12, abs=1e-15), (horizon, decision)
    for decision in range(rows):
        expected_adv = _reference_adv(dataset, decision, 5)
        actual_adv = labels.predecision_adv[decision, 0]
        if math.isnan(expected_adv):
            assert math.isnan(actual_adv), decision
        else:
            assert actual_adv == pytest.approx(expected_adv, rel=1e-12)


def test_actions_on_the_entry_session_are_excluded_and_on_the_exit_session_included() -> None:
    closes = ["10", "10", "10", "10"]
    base = _dataset(closes)
    entry_day, exit_day = _day(base, 1), _day(base, 2)
    on_entry = Dataset(
        base.sessions, base.series, dividends=(Dividend("AAA", entry_day, exit_day, Decimal(1)),)
    )
    on_exit = Dataset(
        base.sessions, base.series, dividends=(Dividend("AAA", exit_day, exit_day, Decimal(1)),)
    )
    # decision at index 2, horizon 1: entry close of session 1, exit close of session 2
    assert _returns(on_entry, decisions=slice(2, 3))[1][0] == pytest.approx(0.0)
    assert _returns(on_exit, decisions=slice(2, 3))[1][0] == pytest.approx(0.1)


def test_an_action_on_the_last_session_counts_for_the_horizon_that_ends_there() -> None:
    base = _dataset(["10", "10", "10"])
    last = _day(base, 2)
    dataset = Dataset(base.sessions, base.series, dividends=(Dividend("AAA", last, last, Decimal(2)),))
    assert _returns(dataset, decisions=slice(2, 3))[1][0] == pytest.approx(0.2)
    split = Dataset(base.sessions, base.series, splits=(Split("AAA", last, Decimal(3)),))
    assert _returns(split, decisions=slice(2, 3))[1][0] == pytest.approx(2.0)


def test_same_day_dividend_is_paid_on_pre_split_shares_whatever_the_amounts() -> None:
    base = _dataset(["10", "10", "10"])
    day = _day(base, 1)
    dataset = Dataset(
        base.sessions,
        base.series,
        splits=(Split("AAA", day, Decimal(2)),),
        dividends=(Dividend("AAA", day, day, Decimal(3)),),
    )
    # one pre-split share receives 3; two shares are then worth 10 each: (2*10 + 3)/10 - 1
    assert _returns(dataset, decisions=slice(1, 2))[1][0] == pytest.approx(1.3)


def test_dividends_accumulate_cash_and_splits_compound_shares_across_days() -> None:
    base = _dataset(["10", "10", "10", "10", "10"])
    dataset = Dataset(
        base.sessions,
        base.series,
        splits=(Split("AAA", _day(base, 1), Decimal(2)), Split("AAA", _day(base, 3), Decimal(3))),
        dividends=(
            Dividend("AAA", _day(base, 2), _day(base, 2), Decimal(1)),
            Dividend("AAA", _day(base, 4), _day(base, 4), Decimal(1)),
        ),
    )
    # shares 1 -> 2 (day 1); cash 2 (day 2); shares 6 (day 3); cash 2 + 6 = 8 (day 4)
    assert _returns(dataset, horizons=(4,), decisions=slice(1, 2))[4][0] == pytest.approx(
        (6 * 10 + 8) / 10 - 1
    )


def test_zero_dividend_and_unit_split_are_valid_and_change_nothing() -> None:
    base = _dataset(["10", "11"])
    day = _day(base, 1)
    dataset = Dataset(
        base.sessions,
        base.series,
        splits=(Split("AAA", day, Decimal(1)),),
        dividends=(Dividend("AAA", day, day, Decimal(0)),),
    )
    assert _returns(dataset, decisions=slice(1, 2))[1][0] == pytest.approx(0.1)


def test_a_dividend_may_be_paid_on_its_ex_date() -> None:
    base = _dataset(["10", "10"])
    day = _day(base, 1)
    dataset = Dataset(base.sessions, base.series, dividends=(Dividend("AAA", day, day, Decimal(1)),))
    assert _returns(dataset, decisions=slice(1, 2))[1][0] == pytest.approx(0.1)


def test_a_zero_price_inside_the_horizon_leaves_the_outcome_unavailable() -> None:
    values = _returns(_dataset(["10", "0", "10", "10"]), horizons=(1, 2))
    assert np.isnan(values[1][1]) and np.isnan(values[1][2])  # entry or exit at zero
    assert np.isnan(values[2][1]) and np.isnan(values[2][2])
    assert values[1][3] == pytest.approx(0.0)


def test_the_first_session_has_no_prior_close_so_no_outcome() -> None:
    values = _returns(_dataset(["10", "11", "12"]), horizons=(1, 2))
    assert np.isnan(values[1][0]) and np.isnan(values[2][0])
    assert values[1][1] == pytest.approx(0.1)


def test_a_horizon_that_exactly_reaches_the_last_session_is_available() -> None:
    values = _returns(_dataset(["10", "11", "12", "13"]), horizons=(1, 2, 3))
    assert values[1][3] == pytest.approx(13 / 12 - 1)
    assert values[2][2] == pytest.approx(13 / 11 - 1)
    assert np.isnan(values[2][3]) and np.isnan(values[3][3])
    assert values[3][1] == pytest.approx(13 / 10 - 1)


def test_adv_is_the_mean_prior_dollar_volume_and_allows_zero_volume_and_tiny_prices() -> None:
    dataset = _dataset(["10", "20", "0.000001", "5"], volumes=[100.0, 0.0, 5.0, 7.0])
    labels = derive_forward_return_labels(
        dataset, decision_sessions=dataset.sessions, symbols=("AAA",), horizons=(1,), adv_lookback=2
    )
    adv = labels.predecision_adv[:, 0]
    assert np.isnan(adv[0]) and np.isnan(adv[1])
    assert adv[2] == pytest.approx((10 * 100 + 20 * 0) / 2)
    assert adv[3] == pytest.approx((20 * 0 + 0.000001 * 5) / 2, rel=1e-9)
    one = derive_forward_return_labels(
        dataset, decision_sessions=dataset.sessions, symbols=("AAA",), horizons=(1,), adv_lookback=1
    )
    assert one.predecision_adv[1, 0] == pytest.approx(1000.0)
    assert one.predecision_adv[3, 0] == pytest.approx(0.000001 * 5, rel=1e-9)


def test_adv_with_a_missing_zero_priced_or_negative_volume_day_is_unavailable() -> None:
    for kwargs in (
        {"present": [True, False, True, True]},
        {"closes": ["10", "0", "10", "10"]},
        {"volumes": [1.0, -1.0, 1.0, 1.0]},
    ):
        closes = kwargs.pop("closes", ["10", "10", "10", "10"])  # type: ignore[arg-type]
        dataset = _dataset(closes, **kwargs)  # type: ignore[arg-type]
        labels = derive_forward_return_labels(
            dataset, decision_sessions=dataset.sessions, symbols=("AAA",), horizons=(1,), adv_lookback=2
        )
        assert np.isnan(labels.predecision_adv[2, 0])
        assert np.isnan(labels.predecision_adv[3, 0]) or kwargs.get("present") is None


@pytest.mark.parametrize("horizon", [1, 252])
def test_horizon_limits_are_inclusive(horizon: int) -> None:
    dataset = _dataset(["10", "11", "12"])
    labels = derive_forward_return_labels(
        dataset, decision_sessions=dataset.sessions[1:], symbols=("AAA",), horizons=(horizon,)
    )
    assert set(labels.forward_returns) == {horizon}


@pytest.mark.parametrize(
    ("horizons", "code"),
    [
        ((0,), "FACTOR_LABEL_HORIZONS_INVALID"),
        ((253,), "FACTOR_LABEL_HORIZONS_INVALID"),
        ((-1, 2), "FACTOR_LABEL_HORIZONS_INVALID"),
        ((True,), "FACTOR_LABEL_HORIZONS_INVALID"),
        ((1.0,), "FACTOR_LABEL_HORIZONS_INVALID"),
        ((2, 1), "FACTOR_LABEL_HORIZONS_INVALID"),
        ((1, 1), "FACTOR_LABEL_HORIZONS_INVALID"),
        ((), "FACTOR_LABEL_HORIZONS_INVALID"),
    ],
)
def test_invalid_horizons_use_the_stable_code(horizons: tuple[object, ...], code: str) -> None:
    dataset = _dataset(["10", "11", "12"])
    with pytest.raises(ValueError, match=f"^{code}$"):
        derive_forward_return_labels(
            dataset,
            decision_sessions=dataset.sessions,
            symbols=("AAA",),
            horizons=horizons,  # type: ignore[arg-type]
        )


def test_alignment_boundaries() -> None:
    dataset = _dataset(["10", "11", "12"])
    ok = derive_forward_return_labels(
        dataset, decision_sessions=dataset.sessions, symbols=("AAA",), horizons=(1,), adv_lookback=1
    )
    assert ok.sessions == dataset.sessions
    for adv_lookback in (0, -1, True, 1.0):
        with pytest.raises(ValueError, match="^FACTOR_LABEL_ALIGNMENT_INVALID$"):
            derive_forward_return_labels(
                dataset,
                decision_sessions=dataset.sessions,
                symbols=("AAA",),
                horizons=(1,),
                adv_lookback=adv_lookback,  # type: ignore[arg-type]
            )
    for sessions in (
        (),
        (date(2030, 1, 1),),
        (dataset.sessions[1], dataset.sessions[0]),
        (dataset.sessions[0],) * 2,
        ("x",),
    ):
        with pytest.raises(ValueError, match="^FACTOR_LABEL_ALIGNMENT_INVALID$"):
            derive_forward_return_labels(dataset, decision_sessions=sessions, symbols=("AAA",), horizons=(1,))  # type: ignore[arg-type]
    for symbols in ((), ("ZZZ",), ("AAA", "AAA"), ("",), (3,)):
        with pytest.raises(ValueError, match="^FACTOR_LABEL_ALIGNMENT_INVALID$"):
            derive_forward_return_labels(
                dataset, decision_sessions=dataset.sessions, symbols=symbols, horizons=(1,)
            )  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", ["split", "dividend"])
def test_two_records_for_the_same_symbol_date_and_kind_are_rejected(kind: str) -> None:
    base = _dataset(["10", "10", "10"])
    day = _day(base, 1)
    if kind == "split":
        dataset = Dataset(
            base.sessions, base.series, splits=(Split("AAA", day, Decimal(2)), Split("AAA", day, Decimal(3)))
        )
    else:
        dataset = Dataset(
            base.sessions,
            base.series,
            dividends=(Dividend("AAA", day, day, Decimal(1)), Dividend("AAA", day, day, Decimal(2))),
        )
    with pytest.raises(ValueError, match="^FACTOR_LABEL_ACTIONS_INVALID$"):
        derive_forward_return_labels(
            dataset, decision_sessions=dataset.sessions, symbols=("AAA",), horizons=(1,)
        )


def test_label_identity_binds_every_request_field_and_is_pinned() -> None:
    dataset = _dataset(["10", "11", "12", "13"])
    base = derive_forward_return_labels(
        dataset, decision_sessions=dataset.sessions[1:], symbols=("AAA",), horizons=(1, 2)
    )
    assert base.label_identity == "sha256:8bc306aac44bf64aba5842a78b27696144548794595d0a84195cd2b3dd0276c8"
    variants = [
        derive_forward_return_labels(
            dataset, decision_sessions=dataset.sessions[2:], symbols=("AAA",), horizons=(1, 2)
        ),
        derive_forward_return_labels(
            dataset, decision_sessions=dataset.sessions[1:], symbols=("AAA",), horizons=(1,)
        ),
        derive_forward_return_labels(
            dataset, decision_sessions=dataset.sessions[1:], symbols=("AAA",), horizons=(1, 2), adv_lookback=3
        ),
        derive_forward_return_labels(
            _dataset(["10", "11", "12", "14"]),
            decision_sessions=dataset.sessions[1:],
            symbols=("AAA",),
            horizons=(1, 2),
        ),
    ]
    identities = {base.label_identity, *(item.label_identity for item in variants)}
    assert len(identities) == 5
