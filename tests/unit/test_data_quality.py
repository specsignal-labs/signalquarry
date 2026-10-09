# SPDX-License-Identifier: Apache-2.0
"""Hand-built reference cases for pure daily-dataset quality assessment."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import date
from decimal import Decimal

import numpy as np
import pytest

from signalquarry._internal.calendar.nyse import MAX_YEAR, MIN_YEAR
from signalquarry._internal.data.dataset import FIELDS, MICRO, Dataset, Dividend, Split, SymbolSeries
from signalquarry._internal.data.quality import QUALITY_SCHEMA, QualityThresholds, assess
from tests.helpers import dataset, weekdays


def test_clean_dataset_exact_report() -> None:
    recorded = dataset(weekdays(date(2024, 2, 5), 3), {"XYZ": {"open": [100, 101, 102]}})
    assert assess(recorded) == {
        "schema": QUALITY_SCHEMA,
        "dataset_identity": recorded.identity(),
        "source": "test",
        "thresholds": {"max_gap_sessions": 3, "stale_close_sessions": 5, "extreme_move": 0.4},
        "sessions": {
            "count": 3,
            "first": "2024-02-05",
            "last": "2024-02-07",
            "calendar": "nyse",
            "weekend_sessions": [],
            "holiday_sessions": [],
            "missing_trading_days": [],
        },
        "symbols": {
            "XYZ": {
                "present": 3,
                "first": "2024-02-05",
                "last": "2024-02-07",
                "coverage": 1.0,
                "leading_missing": 0,
                "trailing_missing": 0,
                "gaps": [],
                "longest_gap": 0,
                "ohlc_violations": 0,
                "nonpositive_prices": 0,
                "zero_volume_sessions": 0,
                "stale_close_runs": [],
                "unexplained_moves": [],
                "splits": 0,
                "dividends": 0,
                "findings": [],
            }
        },
        "common_window": {"first": "2024-02-05", "last": "2024-02-07", "sessions": 3},
        "findings": [],
        "ok": True,
    }


def test_thresholds_are_frozen() -> None:
    thresholds = QualityThresholds()
    with pytest.raises(FrozenInstanceError):
        thresholds.max_gap_sessions = 4


def test_weekend_finding_in_isolation() -> None:
    result = assess(dataset((date(2024, 2, 10),), {"XYZ": {"open": [100]}}))
    assert result["sessions"]["weekend_sessions"] == ["2024-02-10"]
    assert result["sessions"]["holiday_sessions"] == []
    assert result["findings"] == ["QUALITY_SESSION_ON_WEEKEND"]
    assert result["symbols"]["XYZ"]["findings"] == []
    assert result["ok"] is False


@pytest.mark.parametrize("session", [date(2024, 7, 4), date(2025, 1, 9)])
def test_holiday_finding_in_isolation(session: date) -> None:
    result = assess(dataset((session,), {"XYZ": {"open": [100]}}))
    assert result["sessions"]["holiday_sessions"] == [session.isoformat()]
    assert result["sessions"]["weekend_sessions"] == []
    assert result["findings"] == ["QUALITY_SESSION_ON_HOLIDAY"]


def test_missing_trading_days_are_sorted_and_exclude_closures() -> None:
    recorded = dataset((date(2024, 7, 3), date(2024, 7, 9)), {"XYZ": {"open": [100, 101]}})
    result = assess(recorded)
    assert result["sessions"]["missing_trading_days"] == ["2024-07-05", "2024-07-08"]
    assert result["findings"] == ["QUALITY_TRADING_DAY_MISSING"]
    assert result["symbols"]["XYZ"]["gaps"] == []


@pytest.mark.parametrize("year", [MIN_YEAR - 1, MAX_YEAR + 1])
def test_outside_calendar_range_has_no_holiday_or_missing_day_findings(year: int) -> None:
    sessions = tuple(day for day in weekdays(date(year, 12, 24), 5) if day.day != 26)
    recorded = dataset(sessions, {"XYZ": {"open": list(range(100, 100 + len(sessions)))}})
    result = assess(recorded)
    assert result["sessions"]["calendar"] == "weekdays_only"
    assert result["sessions"]["holiday_sessions"] == []
    assert result["sessions"]["missing_trading_days"] == []
    assert result["ok"] is True


def test_partial_calendar_checks_only_the_tracked_range() -> None:
    sessions = (date(MIN_YEAR - 1, 12, 31), date(MIN_YEAR, 1, 5))
    result = assess(dataset(sessions, {"XYZ": {"open": [100, 101]}}))
    assert result["sessions"]["calendar"] == "partial"
    assert result["sessions"]["missing_trading_days"] == [f"{MIN_YEAR}-01-04"]
    assert result["findings"] == ["QUALITY_TRADING_DAY_MISSING"]


def test_symbol_empty_finding_in_isolation() -> None:
    recorded = dataset(weekdays(date(2024, 2, 5), 3), {"XYZ": {"open": [0, 0, 0], "present": [False] * 3}})
    result = assess(recorded)
    assert result["findings"] == ["QUALITY_SYMBOL_EMPTY"]
    report = result["symbols"]["XYZ"]
    assert (report["present"], report["first"], report["last"], report["coverage"]) == (0, None, None, None)
    assert (report["leading_missing"], report["trailing_missing"], report["gaps"], report["longest_gap"]) == (
        3,
        3,
        [],
        0,
    )
    assert result["common_window"] == {"first": None, "last": None, "sessions": 0}


def test_empty_dataset_and_empty_selection() -> None:
    result = assess(Dataset((), {}))
    assert result["sessions"] == {
        "count": 0,
        "first": None,
        "last": None,
        "calendar": "weekdays_only",
        "weekend_sessions": [],
        "holiday_sessions": [],
        "missing_trading_days": [],
    }
    assert result["symbols"] == {}
    assert result["common_window"] == {"first": None, "last": None, "sessions": 0}
    assert result["ok"] is True
    recorded = dataset((date(2024, 2, 5),), {"XYZ": {"open": [100]}})
    result = assess(recorded, symbols=())
    assert result["symbols"] == {}
    assert result["common_window"] == {"first": None, "last": None, "sessions": 0}


@pytest.mark.parametrize("gap, findings", [(3, []), (4, ["QUALITY_GAP"])])
def test_gap_threshold_boundary(gap: int, findings: list[str]) -> None:
    sessions = weekdays(date(2024, 2, 5), gap + 2)
    recorded = dataset(
        sessions, {"XYZ": {"open": list(range(100, 102 + gap)), "present": [True, *([False] * gap), True]}}
    )
    result = assess(recorded)
    report = result["symbols"]["XYZ"]
    assert report["gaps"] == [
        {"start": sessions[1].isoformat(), "end": sessions[-2].isoformat(), "sessions": gap}
    ]
    assert report["longest_gap"] == gap
    assert report["coverage"] == round(2 / (gap + 2), 6)
    assert report["findings"] == result["findings"] == findings


def test_coverage_and_every_internal_gap_exclude_missing_edges() -> None:
    sessions = weekdays(date(2024, 2, 5), 9)
    recorded = dataset(
        sessions,
        {
            "XYZ": {
                "open": list(range(100, 109)),
                "present": [False, True, False, True, False, False, True, False, False],
            }
        },
    )
    result = assess(recorded)
    report = result["symbols"]["XYZ"]
    assert (report["present"], report["first"], report["last"], report["coverage"]) == (
        3,
        "2024-02-06",
        "2024-02-13",
        0.5,
    )
    assert (report["leading_missing"], report["trailing_missing"], report["longest_gap"]) == (1, 2, 2)
    assert report["gaps"] == [
        {"start": "2024-02-07", "end": "2024-02-07", "sessions": 1},
        {"start": "2024-02-09", "end": "2024-02-12", "sessions": 2},
    ]
    assert result["ok"] is True


@pytest.mark.parametrize(
    "values",
    [(99, 101, 100, 100), (102, 101, 99, 100), (100, 101, 101, 100), (100, 99, 99, 100), (100, 99, 101, 100)],
)
def test_ohlc_inconsistent_finding_in_isolation(values: tuple[int, ...]) -> None:
    series = SymbolSeries(
        {
            field: np.array([value * MICRO], dtype=np.int64)
            for field, value in zip(FIELDS, values, strict=True)
        },
        np.array([1e6]),
        np.array([True]),
    )
    result = assess(Dataset((date(2024, 2, 5),), {"XYZ": series}))
    assert result["symbols"]["XYZ"]["ohlc_violations"] == 1
    assert result["findings"] == ["QUALITY_OHLC_INCONSISTENT"]


@pytest.mark.parametrize("price", [0, -1])
def test_nonpositive_price_finding_in_isolation(price: int) -> None:
    result = assess(dataset((date(2024, 2, 5),), {"XYZ": {"open": [price]}}))
    assert result["symbols"]["XYZ"]["nonpositive_prices"] == 1
    assert result["findings"] == ["QUALITY_PRICE_NONPOSITIVE"]


@pytest.mark.parametrize("volume", [0.0, -1.0])
def test_zero_volume_finding_in_isolation(volume: float) -> None:
    series = SymbolSeries(
        {field: np.array([100 * MICRO], dtype=np.int64) for field in FIELDS},
        np.array([volume]),
        np.array([True]),
    )
    result = assess(Dataset((date(2024, 2, 5),), {"XYZ": series}))
    assert result["symbols"]["XYZ"]["zero_volume_sessions"] == 1
    assert result["findings"] == ["QUALITY_ZERO_VOLUME"]


def test_invalid_absent_values_are_ignored() -> None:
    series = SymbolSeries(
        {field: np.array([-1, 100 * MICRO], dtype=np.int64) for field in FIELDS},
        np.array([-1, 1e6]),
        np.array([False, True]),
    )
    result = assess(Dataset(weekdays(date(2024, 2, 5), 2), {"XYZ": series}))
    assert result["ok"] is True
    assert result["symbols"]["XYZ"]["nonpositive_prices"] == 0
    assert result["symbols"]["XYZ"]["zero_volume_sessions"] == 0


@pytest.mark.parametrize("count", [4, 5])
def test_stale_close_threshold_boundary(count: int) -> None:
    sessions = weekdays(date(2024, 2, 5), count)
    result = assess(
        dataset(sessions, {"XYZ": {"open": list(range(100, 100 + count)), "close": [100] * count}})
    )
    expected = (
        [{"start": sessions[0].isoformat(), "end": sessions[-1].isoformat(), "sessions": count}]
        if count == 5
        else []
    )
    assert result["symbols"]["XYZ"]["stale_close_runs"] == expected
    assert result["findings"] == (["QUALITY_STALE_CLOSE"] if count == 5 else [])


def test_absence_breaks_stale_close_runs() -> None:
    recorded = dataset(
        weekdays(date(2024, 2, 5), 9),
        {"XYZ": {"open": [100] * 9, "present": [True] * 4 + [False] + [True] * 4}},
    )
    result = assess(recorded)
    assert result["symbols"]["XYZ"]["stale_close_runs"] == []
    assert result["ok"] is True


def test_multiple_stale_close_runs_are_sorted() -> None:
    sessions = weekdays(date(2024, 2, 5), 6)
    result = assess(
        dataset(sessions, {"XYZ": {"open": [100, 100, 100, 101, 101, 101]}}),
        thresholds=QualityThresholds(stale_close_sessions=3),
    )
    assert result["symbols"]["XYZ"]["stale_close_runs"] == [
        {"start": "2024-02-05", "end": "2024-02-07", "sessions": 3},
        {"start": "2024-02-08", "end": "2024-02-12", "sessions": 3},
    ]
    assert result["findings"] == ["QUALITY_STALE_CLOSE"]


@pytest.mark.parametrize("later, direction", [(140, "up"), (60, "down")])
def test_extreme_move_inclusive_boundary(later: int, direction: str) -> None:
    result = assess(dataset(weekdays(date(2024, 2, 5), 2), {"XYZ": {"open": [100, later]}}))
    assert result["symbols"]["XYZ"]["unexplained_moves"] == [
        {"session": "2024-02-06", "direction": direction}
    ]
    assert result["findings"] == ["QUALITY_UNEXPLAINED_MOVE"]


@pytest.mark.parametrize("later", [139.999999, 60.000001])
def test_move_just_below_threshold(later: float) -> None:
    result = assess(dataset(weekdays(date(2024, 2, 5), 2), {"XYZ": {"open": [100, later]}}))
    assert result["symbols"]["XYZ"]["unexplained_moves"] == []
    assert result["ok"] is True


def test_moves_compare_consecutive_present_closes_across_a_gap() -> None:
    recorded = dataset(
        weekdays(date(2024, 2, 5), 3), {"XYZ": {"open": [100, 0, 150], "present": [True, False, True]}}
    )
    result = assess(recorded)
    assert result["symbols"]["XYZ"]["unexplained_moves"] == [{"session": "2024-02-07", "direction": "up"}]
    assert result["findings"] == ["QUALITY_UNEXPLAINED_MOVE"]


@pytest.mark.parametrize("record_split", [True, False])
def test_two_for_one_split_explains_halving_price(record_split: bool) -> None:
    sessions = weekdays(date(2024, 2, 5), 2)
    splits = (Split("XYZ", sessions[1], Decimal("2")),) if record_split else ()
    result = assess(dataset(sessions, {"XYZ": {"open": [100, 50]}}, splits=splits))
    assert result["symbols"]["XYZ"]["unexplained_moves"] == (
        [] if record_split else [{"session": "2024-02-06", "direction": "down"}]
    )
    assert result["symbols"]["XYZ"]["splits"] == int(record_split)
    assert result["findings"] == ([] if record_split else ["QUALITY_UNEXPLAINED_MOVE"])


def test_cumulative_splits_on_absent_sessions_and_reverse_splits() -> None:
    sessions = weekdays(date(2024, 2, 5), 4)
    splits = (Split("XYZ", sessions[1], Decimal("2")), Split("XYZ", sessions[3], Decimal("0.5")))
    recorded = dataset(
        sessions, {"XYZ": {"open": [100, 0, 50, 100], "present": [True, False, True, True]}}, splits=splits
    )
    result = assess(recorded)
    assert result["symbols"]["XYZ"]["unexplained_moves"] == []
    assert result["symbols"]["XYZ"]["splits"] == 2
    assert result["ok"] is True


def test_residual_extreme_move_after_split_is_reported() -> None:
    sessions = weekdays(date(2024, 2, 5), 2)
    recorded = dataset(
        sessions, {"XYZ": {"open": [100, 75]}}, splits=(Split("XYZ", sessions[1], Decimal("2")),)
    )
    result = assess(recorded)
    assert result["symbols"]["XYZ"]["unexplained_moves"] == [{"session": "2024-02-06", "direction": "up"}]
    assert result["findings"] == ["QUALITY_UNEXPLAINED_MOVE"]


def test_common_window_counts_dataset_sessions_in_overlap() -> None:
    sessions = weekdays(date(2024, 2, 5), 5)
    recorded = dataset(
        sessions,
        {
            "XYZ": {"open": [100, 101, 102, 103, 104], "present": [True, True, False, True, False]},
            "ABC": {"open": [100, 101, 102, 103, 104], "present": [False, True, True, True, True]},
        },
    )
    result = assess(recorded)
    assert list(result["symbols"]) == ["ABC", "XYZ"]
    assert result["common_window"] == {"first": "2024-02-06", "last": "2024-02-08", "sessions": 3}
    assert result["ok"] is True


@pytest.mark.parametrize("second_present", [[False, False, True], [False, False, False]])
def test_common_window_is_empty_for_disjoint_or_empty_symbols(second_present: list[bool]) -> None:
    recorded = dataset(
        weekdays(date(2024, 2, 5), 3),
        {
            "XYZ": {"open": [100, 101, 102], "present": [True, False, False]},
            "ABC": {"open": [100, 101, 102], "present": second_present},
        },
    )
    assert assess(recorded)["common_window"] == {"first": None, "last": None, "sessions": 0}


def test_symbol_selection_restricts_findings_and_common_window() -> None:
    recorded = dataset(
        weekdays(date(2024, 2, 5), 2),
        {
            "XYZ": {"open": [100, 101]},
            "ABC": {"open": [0, 0], "present": [False, False]},
        },
    )
    result = assess(recorded, symbols=("XYZ",))
    assert list(result["symbols"]) == ["XYZ"]
    assert result["common_window"] == {"first": "2024-02-05", "last": "2024-02-06", "sessions": 2}
    assert result["dataset_identity"] == recorded.identity()
    assert result["ok"] is True


def test_unknown_symbol_is_rejected() -> None:
    recorded = dataset((date(2024, 2, 5),), {"XYZ": {"open": [100]}})
    with pytest.raises(ValueError, match="^QUALITY_SYMBOL_UNKNOWN:ABC$"):
        assess(recorded, symbols=("XYZ", "ABC"))


def test_corporate_action_counts_are_per_symbol() -> None:
    sessions = weekdays(date(2024, 2, 5), 2)
    dividend = Dividend("XYZ", sessions[1], sessions[1], Decimal("0.5"))
    recorded = dataset(
        sessions, {"XYZ": {"open": [100, 101]}, "ABC": {"open": [100, 101]}}, dividends=(dividend,)
    )
    result = assess(recorded)
    assert result["symbols"]["XYZ"]["dividends"] == 1
    assert result["symbols"]["ABC"]["dividends"] == 0
    assert result["ok"] is True


def test_findings_union_is_sorted_and_deduplicated() -> None:
    result = assess(dataset((date(2024, 2, 10),), {"XYZ": {"open": [0]}, "ABC": {"open": [0]}}))
    assert result["findings"] == ["QUALITY_PRICE_NONPOSITIVE", "QUALITY_SESSION_ON_WEEKEND"]


def test_custom_thresholds_are_used_and_reported() -> None:
    thresholds = QualityThresholds(max_gap_sessions=0, stale_close_sessions=2, extreme_move=0.1)
    result = assess(
        dataset(weekdays(date(2024, 2, 5), 2), {"XYZ": {"open": [100, 110]}}), thresholds=thresholds
    )
    assert result["thresholds"] == {"max_gap_sessions": 0, "stale_close_sessions": 2, "extreme_move": 0.1}
    assert result["findings"] == ["QUALITY_UNEXPLAINED_MOVE"]


def test_report_is_deterministic_json_serializable_and_does_not_disclose_values() -> None:
    recorded = dataset(
        weekdays(date(2024, 2, 5), 3), {"XYZ": {"open": [123456.789123, 123457.789123, 123458.789123]}}
    )
    identity = recorded.identity()
    arrays = {field: values.copy() for field, values in recorded.series["XYZ"].micro.items()}
    split_factors = recorded.cumulative_split("XYZ").copy()
    first, second = assess(recorded), assess(recorded)
    assert first == second
    dumped = json.dumps(first, allow_nan=False)
    assert "123456.789123" not in dumped
    assert "123456789123" not in dumped
    assert '"volume"' not in dumped and '"return"' not in dumped and '"close"' not in dumped
    assert recorded.identity() == identity
    for field, values in arrays.items():
        np.testing.assert_array_equal(recorded.series["XYZ"].micro[field], values)
    np.testing.assert_array_equal(recorded.cumulative_split("XYZ"), split_factors)
