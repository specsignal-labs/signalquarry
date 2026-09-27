# SPDX-License-Identifier: Apache-2.0
"""Golden fixture for the NYSE calendar: every date below is cross-checked against
NYSE's own 3-year-ahead press releases (see ``_internal/calendar/nyse.py`` for the
sources), covering nine consecutive years and every edge case that occurred in them.
"""

from __future__ import annotations

from datetime import date

import pytest

from signalquarry._internal.calendar import nyse

# fmt: off
HOLIDAYS = {
    2019: {"new_years_day": "2019-01-01", "mlk_day": "2019-01-21", "washingtons_birthday": "2019-02-18",
           "good_friday": "2019-04-19", "memorial_day": "2019-05-27", "labor_day": "2019-09-02",
           "thanksgiving": "2019-11-28", "christmas": "2019-12-25"},
    2020: {"new_years_day": "2020-01-01", "mlk_day": "2020-01-20", "washingtons_birthday": "2020-02-17",
           "good_friday": "2020-04-10", "memorial_day": "2020-05-25", "labor_day": "2020-09-07",
           "thanksgiving": "2020-11-26", "christmas": "2020-12-25"},
    2021: {"new_years_day": "2021-01-01", "mlk_day": "2021-01-18", "washingtons_birthday": "2021-02-15",
           "good_friday": "2021-04-02", "memorial_day": "2021-05-31", "labor_day": "2021-09-06",
           "thanksgiving": "2021-11-25", "christmas": "2021-12-24"},  # Dec 25 is a Saturday
    2022: {"mlk_day": "2022-01-17", "washingtons_birthday": "2022-02-21", "good_friday": "2022-04-15",
           "memorial_day": "2022-05-30", "juneteenth": "2022-06-20", "independence_day": "2022-07-04",
           "labor_day": "2022-09-05", "thanksgiving": "2022-11-24",
           "christmas": "2022-12-26"},  # Dec 25 is a Sunday; no new_years_day (Jan 1 is a Saturday)
    2023: {"new_years_day": "2023-01-02"},  # Jan 1 is a Sunday
    2027: {"christmas": "2027-12-24"},  # Dec 25 is a Saturday
}

HALF_DAYS = {
    2019: {"day_after_thanksgiving": "2019-11-29", "independence_day_eve": "2019-07-03",
           "christmas_eve": "2019-12-24"},
    2020: {"day_after_thanksgiving": "2020-11-27", "independence_day_eve": "2020-07-03",
           "christmas_eve": "2020-12-24"},
    2021: {"day_after_thanksgiving": "2021-11-26",
           "independence_day_eve": "2021-07-05"},  # July 4 is a Sunday; no christmas_eve (Dec 24 is the
                                                     # observed Christmas holiday itself)
    2022: {"day_after_thanksgiving": "2022-11-25"},  # no independence_day_eve (July 4 is a Monday: the
                                                       # preceding day is Sunday); no christmas_eve (Dec 24
                                                       # is a Saturday)
    2023: {"day_after_thanksgiving": "2023-11-24", "independence_day_eve": "2023-07-03"},
    2024: {"day_after_thanksgiving": "2024-11-29", "independence_day_eve": "2024-07-03",
           "christmas_eve": "2024-12-24"},
    2025: {"day_after_thanksgiving": "2025-11-28", "independence_day_eve": "2025-07-03",
           "christmas_eve": "2025-12-24"},
    2026: {"day_after_thanksgiving": "2026-11-27",
           "independence_day_eve": "2026-07-03",  # July 4 is a Saturday: shifts to the preceding Friday
           "christmas_eve": "2026-12-24"},
    2027: {"day_after_thanksgiving": "2027-11-26",
           "independence_day_eve": "2027-07-05"},  # July 4 is a Sunday: shifts to the following Monday;
                                                     # no christmas_eve (Dec 25 is a Saturday, see HOLIDAYS)
}
# fmt: on


@pytest.mark.parametrize("year", sorted(HOLIDAYS))
def test_holidays_match_the_nyse_press_release(year: int) -> None:
    actual = {name: d.isoformat() for name, d in nyse.holidays(year).items()}
    for name, expected in HOLIDAYS[year].items():
        assert actual.get(name) == expected, f"{year} {name}"


@pytest.mark.parametrize("year", sorted(HALF_DAYS))
def test_half_days_match_the_nyse_press_release(year: int) -> None:
    actual = {name: d.isoformat() for name, d in nyse.half_days(year).items()}
    assert actual == HALF_DAYS[year], year


def test_new_years_day_is_not_observed_when_it_falls_on_a_saturday() -> None:
    # NYSE Rule 7.2: the exchange stays open on Dec 31 (a settlement day), unlike
    # every other holiday here, which does shift to the preceding Friday.
    assert "new_years_day" not in nyse.holidays(2022)
    assert not nyse.is_holiday(date(2021, 12, 31))


def test_independence_day_never_gets_a_substitute_full_holiday() -> None:
    for year in (2020, 2026):  # July 4 on a Saturday
        assert "independence_day" not in nyse.holidays(year)
    for year in (2021, 2027):  # July 4 on a Sunday
        assert "independence_day" not in nyse.holidays(year)


def test_juneteenth_only_applies_from_2022() -> None:
    assert "juneteenth" not in nyse.holidays(2021)
    assert "juneteenth" in nyse.holidays(2022)


def test_one_off_closures_cannot_be_derived_and_are_listed_explicitly() -> None:
    assert nyse.is_holiday(date(2018, 12, 5))  # George H.W. Bush
    assert nyse.is_holiday(date(2025, 1, 9))  # Jimmy Carter
    assert not nyse.is_half_day(date(2025, 1, 9))  # a full closure, not a half day


def test_weekends_are_holidays() -> None:
    assert nyse.is_holiday(date(2024, 11, 30))  # Saturday
    assert nyse.is_holiday(date(2024, 12, 1))  # Sunday


def test_ordinary_trading_day_is_neither() -> None:
    tuesday = date(2024, 11, 26)
    assert not nyse.is_holiday(tuesday)
    assert not nyse.is_half_day(tuesday)


def test_outside_the_tracked_range_defaults_to_the_ordinary_session() -> None:
    # A date this module hasn't verified must never be silently guessed at as a
    # holiday or a half day -- the safe default is the ordinary full session.
    far_future = date(2041, 6, 12)  # a plain Wednesday, well past MAX_YEAR
    assert far_future.year > nyse.MAX_YEAR
    assert far_future.weekday() == 2
    assert not nyse.is_half_day(far_future)
    assert not nyse.is_holiday(far_future)


def test_dst_transition_weeks_are_ordinary_sessions() -> None:
    # The calendar is date-only; DST is a UTC-offset concern handled separately by
    # zoneinfo when a checkpoint's local time is combined with a real timezone.
    for d in (date(2024, 3, 11), date(2024, 11, 4)):  # the Monday after each 2024 change
        assert not nyse.is_holiday(d)
        assert not nyse.is_half_day(d)
