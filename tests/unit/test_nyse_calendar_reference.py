# SPDX-License-Identifier: Apache-2.0
"""Independent references for the calendar's date arithmetic across the whole tracked range."""

from __future__ import annotations

import calendar
from datetime import date, timedelta

import pytest

from signalquarry._internal.calendar import nyse

EASTER = {
    2016: (3, 27), 2017: (4, 16), 2018: (4, 1), 2019: (4, 21), 2020: (4, 12), 2021: (4, 4),
    2022: (4, 17), 2023: (4, 9), 2024: (3, 31), 2025: (4, 20), 2026: (4, 5), 2027: (3, 28),
    2028: (4, 16), 2029: (4, 1), 2030: (4, 21), 2031: (4, 13), 2032: (3, 28), 2033: (4, 17),
    2034: (4, 9), 2035: (3, 25),
}  # fmt: skip


def _oudin_easter(year: int) -> date:
    """Oudin's (1940) algorithm, written differently from the module's Meeus/Jones/Butcher."""
    century = year // 100
    n = year % 19
    k = (century - 17) // 25
    i = (century - century // 4 - (century - k) // 3 + 19 * n + 15) % 30
    i -= (i // 28) * (1 - (i // 28) * (29 // (i + 1)) * ((21 - n) // 11))
    j = (year + year // 4 + i + 2 - century + century // 4) % 7
    offset = i - j
    month = 3 + (offset + 40) // 44
    day = offset + 28 - 31 * (month // 4)
    return date(year, month, day)


@pytest.mark.parametrize("year", sorted(EASTER))
def test_easter_matches_the_published_table_for_every_tracked_year(year: int) -> None:
    assert nyse._easter_sunday(year) == date(year, *EASTER[year])
    assert nyse.holidays(year)["good_friday"] == date(year, *EASTER[year]) - timedelta(days=2)


def test_easter_matches_an_independent_algorithm_over_three_centuries() -> None:
    for year in range(1900, 2200):
        computed = nyse._easter_sunday(year)
        assert computed == _oudin_easter(year), year
        assert computed.weekday() == 6, year


def test_nth_and_last_weekdays_match_the_calendar_module_in_every_month() -> None:
    for year in (2016, 2019, 2024, 2025, 2028, 2035):
        for month in range(1, 13):
            weeks = calendar.monthcalendar(year, month)
            for weekday in range(7):
                days = [week[weekday] for week in weeks if week[weekday]]
                assert nyse._last_weekday(year, month, weekday) == date(year, month, days[-1]), (
                    year,
                    month,
                    weekday,
                )
                for n in range(1, 5):
                    assert nyse._nth_weekday(year, month, weekday, n) == date(year, month, days[n - 1])


def test_the_last_monday_of_december_rolls_into_the_next_year_correctly() -> None:
    assert nyse._last_weekday(2025, 12, 0) == date(2025, 12, 29)
    assert nyse._last_weekday(2024, 12, 1) == date(2024, 12, 31)
    assert nyse._last_weekday(2023, 12, 6) == date(2023, 12, 31)
    assert nyse._last_weekday(2026, 5, 0) == date(2026, 5, 25)
    assert nyse._last_weekday(2024, 2, 3) == date(2024, 2, 29)  # leap day


def test_weekend_shift_moves_saturday_back_and_sunday_forward_only() -> None:
    assert nyse._weekend_shift(date(2026, 6, 19)) == date(2026, 6, 19)  # Friday
    assert nyse._weekend_shift(date(2027, 6, 19)) == date(2027, 6, 18)  # Saturday
    assert nyse._weekend_shift(date(2022, 12, 25)) == date(2022, 12, 26)  # Sunday
    assert nyse._weekend_shift(date(2024, 12, 25)) == date(2024, 12, 25)  # Wednesday


def test_the_tracked_range_is_inclusive_at_both_ends() -> None:
    assert (nyse.MIN_YEAR, nyse.MAX_YEAR) == (2016, 2035)
    first_half = nyse.half_days(2016)["day_after_thanksgiving"]
    last_half = nyse.half_days(2035)["day_after_thanksgiving"]
    assert nyse.is_half_day(first_half) and nyse.is_half_day(last_half)
    assert not nyse.is_half_day(date(2015, 11, 27)) and not nyse.is_half_day(date(2036, 11, 28))
    assert nyse.is_holiday(date(2016, 1, 1)) and nyse.is_holiday(date(2035, 12, 25))
    # outside the range only weekends and the explicit one-offs close the exchange
    assert not nyse.is_holiday(date(2015, 12, 25)) and not nyse.is_holiday(date(2036, 12, 25))
    assert nyse.is_holiday(date(2036, 12, 27)) and nyse.is_holiday(date(2015, 12, 26))


def test_one_off_closures_are_holidays_and_never_half_days(monkeypatch: pytest.MonkeyPatch) -> None:
    assert nyse.is_holiday(date(2018, 12, 5)) and nyse.is_holiday(date(2025, 1, 9))
    assert not nyse.is_half_day(date(2018, 12, 5))
    day = nyse.half_days(2024)["day_after_thanksgiving"]
    assert nyse.is_half_day(day)
    monkeypatch.setattr(nyse, "EXTRA_HOLIDAYS", frozenset({day}))
    assert not nyse.is_half_day(day)
    assert nyse.is_holiday(day)


def test_the_previous_trading_day_skips_one_calendar_day_at_a_time() -> None:
    assert nyse.previous_trading_day(date(2026, 1, 20)) == date(2026, 1, 16)  # Tue after MLK: Fri
    assert nyse.previous_trading_day(date(2026, 1, 21)) == date(2026, 1, 20)
    assert nyse.previous_trading_day(date(2026, 4, 6)) == date(2026, 4, 2)  # Mon after Good Friday
    assert nyse.previous_trading_day(date(2026, 4, 3)) == date(2026, 4, 2)  # the holiday itself
    assert nyse.previous_trading_day(date(2025, 1, 10)) == date(2025, 1, 8)  # mourning day 1/9
    assert nyse.previous_trading_day(date(2024, 12, 26)) == date(2024, 12, 24)
    assert nyse.previous_trading_day(date(2021, 1, 4)) == date(2020, 12, 31)  # weekend and New Year


def test_every_tracked_year_has_the_expected_holiday_count() -> None:
    for year in range(nyse.MIN_YEAR, nyse.MAX_YEAR + 1):
        names = nyse.holidays(year)
        expected = 7 + (year >= 2022) + (date(year, 7, 4).weekday() < 5) + (date(year, 1, 1).weekday() != 5)
        # mlk, presidents, good friday, memorial, labor, thanksgiving, christmas = 7
        assert len(names) == expected, year
        assert all(nyse.is_holiday(day) and day.weekday() < 5 for day in names.values()), year
