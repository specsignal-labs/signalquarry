# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import numpy as np
import pytest

from signalquarry._internal.data.dataset import MICRO, Dataset, Dividend, Split, SymbolSeries
from signalquarry._internal.factors.labels import derive_forward_return_labels


def _dataset(
    prices: tuple[str, ...],
    *,
    present: tuple[bool, ...] | None = None,
    volumes: tuple[float, ...] | None = None,
    splits: tuple[Split, ...] = (),
    dividends: tuple[Dividend, ...] = (),
) -> Dataset:
    sessions = tuple(date(2024, 1, 1) + timedelta(days=index) for index in range(len(prices)))
    close = np.array([int(Decimal(value) * MICRO) for value in prices], dtype=np.int64)
    available = np.ones(len(sessions), dtype=bool) if present is None else np.array(present, dtype=bool)
    volume = np.ones(len(sessions), dtype=np.float64) if volumes is None else np.array(volumes)
    series = SymbolSeries(
        micro={name: close.copy() for name in ("open", "high", "low", "close")},
        volume=volume,
        present=available,
    )
    return Dataset(sessions, {"AAA": series}, splits=splits, dividends=dividends, source="synthetic:labels")


def test_forward_labels_apply_split_and_dividend_on_ex_date() -> None:
    base = _dataset(("100", "49.5", "50", "52", "53"))
    ex_date = base.sessions[1]
    dataset = Dataset(
        base.sessions,
        base.series,
        splits=(Split("AAA", ex_date, Decimal(2)),),
        dividends=(Dividend("AAA", ex_date, base.sessions[3], Decimal(1)),),
        source=base.source,
    )

    labels = derive_forward_return_labels(
        dataset,
        decision_sessions=dataset.sessions[1:3],
        symbols=("AAA",),
        horizons=(1, 2),
    )

    assert labels.forward_returns[1][:, 0] == pytest.approx((0, 50 / 49.5 - 1))
    assert labels.forward_returns[2][:, 0] == pytest.approx((0.01, 52 / 49.5 - 1))
    assert labels.outcome_end_sessions[1] == dataset.sessions[1:3]
    assert labels.outcome_end_sessions[2] == dataset.sessions[2:4]
    assert labels.dataset_identity == dataset.identity()
    assert labels.label_identity.startswith("sha256:")
    assert labels.provenance_verified is False
    with pytest.raises(ValueError):
        labels.forward_returns[1].flags.writeable = True
    with pytest.raises(ValueError):
        labels.predecision_adv.flags.writeable = True


def test_forward_labels_leave_missing_and_right_censored_outcomes_unavailable() -> None:
    dataset = _dataset(("10", "11", "12", "13"), present=(True, True, False, True))
    labels = derive_forward_return_labels(
        dataset,
        decision_sessions=(dataset.sessions[1], dataset.sessions[3]),
        symbols=("AAA",),
        horizons=(1, 2, 3),
    )

    assert labels.forward_returns[1][0, 0] == pytest.approx(0.1)
    assert np.isnan(labels.forward_returns[2][0, 0])
    assert np.isnan(labels.forward_returns[3][0, 0])
    assert np.isnan(labels.forward_returns[1][1, 0])
    assert labels.outcome_end_sessions[1] == (dataset.sessions[1], dataset.sessions[3])
    assert labels.outcome_end_sessions[2] == (dataset.sessions[2], None)
    assert labels.outcome_end_sessions[3] == (dataset.sessions[3], None)


def test_forward_labels_compute_only_predecision_adv() -> None:
    dataset = _dataset(
        tuple("10" for _ in range(25)),
        volumes=tuple(100.0 + index for index in range(25)),
    )
    labels = derive_forward_return_labels(
        dataset,
        decision_sessions=(dataset.sessions[20], dataset.sessions[24]),
        symbols=("AAA",),
        horizons=(1,),
    )

    assert labels.predecision_adv[:, 0] == pytest.approx((1_095.0, 1_135.0))


def test_forward_labels_leave_adv_unavailable_for_invalid_or_overflowing_volume() -> None:
    prices = tuple("10" for _ in range(25))
    invalid_volumes = [100.0] * 25
    invalid_volumes[5] = -1.0
    invalid = _dataset(prices, volumes=tuple(invalid_volumes))
    invalid_labels = derive_forward_return_labels(
        invalid,
        decision_sessions=(invalid.sessions[20],),
        symbols=("AAA",),
        horizons=(1,),
    )
    assert np.isnan(invalid_labels.predecision_adv[0, 0])

    overflowing = _dataset(prices, volumes=tuple(1e308 for _ in range(25)))
    overflow_labels = derive_forward_return_labels(
        overflowing,
        decision_sessions=(overflowing.sessions[20],),
        symbols=("AAA",),
        horizons=(1,),
    )
    assert np.isnan(overflow_labels.predecision_adv[0, 0])


def test_forward_labels_need_a_completed_bar_before_the_decision() -> None:
    dataset = _dataset(("10", "11", "12"))
    labels = derive_forward_return_labels(
        dataset,
        decision_sessions=(dataset.sessions[0],),
        symbols=("AAA",),
        horizons=(1,),
    )

    assert np.isnan(labels.forward_returns[1][0, 0])


def test_forward_labels_apply_actions_in_ex_date_order() -> None:
    base = _dataset(("100", "50", "50", "50"))
    dataset = Dataset(
        base.sessions,
        base.series,
        splits=(Split("AAA", base.sessions[1], Decimal(2)),),
        dividends=(Dividend("AAA", base.sessions[2], base.sessions[3], Decimal(1)),),
        source=base.source,
    )
    labels = derive_forward_return_labels(
        dataset,
        decision_sessions=(dataset.sessions[1],),
        symbols=("AAA",),
        horizons=(3,),
    )

    # The split doubles the position before the later $1-per-share dividend.
    assert labels.forward_returns[3][0, 0] == pytest.approx(0.02)


def test_forward_labels_ignore_unselected_and_out_of_window_actions() -> None:
    base = _dataset(("10", "11", "12", "13"))
    dataset = Dataset(
        base.sessions,
        base.series,
        splits=(
            Split("BBB", base.sessions[1], Decimal(2)),
            Split("AAA", base.sessions[0], Decimal(2)),
            Split("AAA", base.sessions[2], Decimal(2)),
        ),
        dividends=(
            Dividend("BBB", base.sessions[1], base.sessions[2], Decimal(1)),
            Dividend("AAA", base.sessions[0], base.sessions[1], Decimal(1)),
            Dividend("AAA", base.sessions[2], base.sessions[3], Decimal(1)),
        ),
        source=base.source,
    )
    labels = derive_forward_return_labels(
        dataset,
        decision_sessions=(dataset.sessions[1],),
        symbols=("AAA",),
        horizons=(1,),
    )

    assert labels.forward_returns[1][0, 0] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"decision_sessions": ()}, "FACTOR_LABEL_ALIGNMENT_INVALID"),
        ({"decision_sessions": (date(2030, 1, 1),)}, "FACTOR_LABEL_ALIGNMENT_INVALID"),
        ({"symbols": ("MISSING",)}, "FACTOR_LABEL_ALIGNMENT_INVALID"),
        ({"horizons": ()}, "FACTOR_LABEL_HORIZONS_INVALID"),
        ({"horizons": (2, 1)}, "FACTOR_LABEL_HORIZONS_INVALID"),
        ({"adv_lookback": True}, "FACTOR_LABEL_ALIGNMENT_INVALID"),
    ],
)
def test_forward_labels_reject_invalid_requests(changes, code: str) -> None:
    dataset = _dataset(("10", "11", "12"))
    request = {
        "decision_sessions": (dataset.sessions[1],),
        "symbols": ("AAA",),
        "horizons": (1,),
        "adv_lookback": 20,
    }
    request.update(changes)

    with pytest.raises(ValueError, match=code):
        derive_forward_return_labels(dataset, **request)


def test_forward_labels_reject_conflicting_action_records() -> None:
    base = _dataset(("10", "11", "12"))
    action = Split("AAA", base.sessions[1], Decimal(2))
    dataset = Dataset(
        base.sessions,
        base.series,
        splits=(action, action),
        source=base.source,
    )

    with pytest.raises(ValueError, match="FACTOR_LABEL_ACTIONS_INVALID"):
        derive_forward_return_labels(
            dataset,
            decision_sessions=(dataset.sessions[1],),
            symbols=("AAA",),
            horizons=(1,),
        )


@pytest.mark.parametrize(
    "action",
    [
        Split("AAA", date(2024, 1, 2), Decimal(0)),
        Dividend("AAA", date(2024, 1, 2), date(2024, 1, 1), Decimal(1)),
        Dividend("AAA", date(2024, 1, 2), date(2024, 1, 3), Decimal(-1)),
        Dividend("AAA", date(2024, 1, 2), date(2024, 1, 3), Decimal("NaN")),
    ],
)
def test_forward_labels_reject_invalid_action_terms(action) -> None:
    dataset = _dataset(("10", "11", "12"))
    if isinstance(action, Split):
        invalid = Dataset(dataset.sessions, dataset.series, splits=(action,), source=dataset.source)
    else:
        invalid = Dataset(dataset.sessions, dataset.series, dividends=(action,), source=dataset.source)

    with pytest.raises(ValueError, match="FACTOR_LABEL_ACTIONS_INVALID"):
        derive_forward_return_labels(
            invalid,
            decision_sessions=(invalid.sessions[1],),
            symbols=("AAA",),
            horizons=(1,),
        )


@pytest.mark.parametrize(
    "action",
    [
        Split("AAA", "2024-01-02", Decimal(2)),
        Split("AAA", date(2024, 1, 2), 2),
        Split("AAA", date(2024, 1, 2), Decimal("NaN")),
        Dividend("AAA", "2024-01-02", "2024-01-03", Decimal(1)),
    ],
)
def test_forward_labels_reject_invalid_action_types_and_dates(action) -> None:
    dataset = _dataset(("10", "11", "12"))
    if isinstance(action, Split):
        invalid = Dataset(dataset.sessions, dataset.series, splits=(action,), source=dataset.source)
    else:
        invalid = Dataset(dataset.sessions, dataset.series, dividends=(action,), source=dataset.source)

    with pytest.raises(ValueError, match="FACTOR_LABEL_ACTIONS_INVALID"):
        derive_forward_return_labels(
            invalid,
            decision_sessions=(invalid.sessions[1],),
            symbols=("AAA",),
            horizons=(1,),
        )


def test_forward_labels_block_float_overflow() -> None:
    base = _dataset(("0.000001", "0.000001", "0.000001"))
    dataset = Dataset(
        base.sessions,
        base.series,
        dividends=(Dividend("AAA", base.sessions[1], base.sessions[2], Decimal("1e400")),),
        source=base.source,
    )

    with pytest.raises(ValueError, match="FACTOR_LABEL_HORIZONS_INVALID"):
        derive_forward_return_labels(
            dataset,
            decision_sessions=(dataset.sessions[1],),
            symbols=("AAA",),
            horizons=(1,),
        )
