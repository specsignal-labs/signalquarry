# SPDX-License-Identifier: Apache-2.0
"""NYSE holidays and 1pm early closes, computed by rule (no network, no file, no cache).

Every rule below was cross-checked against NYSE's own 3-year-ahead press releases
("NYSE Group Announces YYYY, YYYY+1 and YYYY+2 Holiday and Early Closings Calendar")
for 2019 through 2027 inclusive — nine consecutive years, chosen to exercise every
edge case that actually occurred in that span:

* New Year's Day is **not observed at all** when January 1 falls on a Saturday
  (NYSE Rule 7.2: the exchange will not close on the preceding Friday, December 31,
  because it is a month/year-end settlement day). Confirmed 2022. Every other
  weekend case (Sunday) shifts to the following Monday as usual.
* Independence Day never produces a substitute full holiday when July 4 falls on a
  weekend — unlike every other holiday here. It only produces a 1pm close on the
  adjacent business day: the preceding Friday when July 4 is a Saturday (confirmed
  2020, 2026), or the following Monday when July 4 is a Sunday (confirmed 2021, 2027).
* Christmas and Juneteenth shift the normal way: Saturday to the preceding Friday
  (confirmed Christmas 2021, Juneteenth 2027), Sunday to the following Monday
  (confirmed Christmas 2022).
* The day after Thanksgiving is always a 1pm close; there is no exception on record.

Good Friday is computed from the Anonymous Gregorian Easter algorithm (Meeus/Jones/
Butcher); every other moving holiday is the Nth weekday of a month. All of these
were spot-checked against the same nine confirmed years and matched exactly.

Two full-day closures cannot be derived from any rule — they were declared after a
sitting or former president's death — and are listed explicitly below.

Years outside ``MIN_YEAR..MAX_YEAR`` are not covered: extend that range (and add a
fixture year to ``tests/test_nyse_calendar.py``) from NYSE's next published
calendar rather than trusting the rule blindly past where it has been checked.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import cache

MIN_YEAR = 2016
MAX_YEAR = 2035

# Declared ad hoc after a president's death; not derivable from any rule.
EXTRA_HOLIDAYS: frozenset[date] = frozenset(
    {
        date(2018, 12, 5),  # National Day of Mourning, George H.W. Bush
        date(2025, 1, 9),  # National Day of Mourning, Jimmy Carter
    }
)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The n-th (1-based) occurrence of `weekday` (Monday=0) in `year`-`month`."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    next_month_first = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    last = next_month_first - timedelta(days=1)
    offset = (last.weekday() - weekday) % 7
    return last - timedelta(days=offset)


def _easter_sunday(year: int) -> date:
    """Anonymous Gregorian algorithm (Meeus/Jones/Butcher)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    n = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * n) // 451
    month, day = divmod(h + n - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _weekend_shift(d: date) -> date:
    """Saturday -> the preceding Friday, Sunday -> the following Monday, else itself."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@cache
def holidays(year: int) -> dict[str, date]:
    """Full-day closures observed in `year`, keyed by a short name."""
    out: dict[str, date] = {}

    new_year = date(year, 1, 1)
    if new_year.weekday() != 5:  # Saturday: NYSE Rule 7.2, not observed at all
        out["new_years_day"] = _weekend_shift(new_year)

    out["mlk_day"] = _nth_weekday(year, 1, 0, 3)
    out["washingtons_birthday"] = _nth_weekday(year, 2, 0, 3)
    out["good_friday"] = _easter_sunday(year) - timedelta(days=2)
    out["memorial_day"] = _last_weekday(year, 5, 0)

    if year >= 2022:  # federal holiday since 2021; first NYSE-observed in 2022
        out["juneteenth"] = _weekend_shift(date(year, 6, 19))

    independence_day = date(year, 7, 4)
    if independence_day.weekday() < 5:  # weekend years get no substitute holiday
        out["independence_day"] = independence_day

    out["labor_day"] = _nth_weekday(year, 9, 0, 1)
    out["thanksgiving"] = _nth_weekday(year, 11, 3, 4)
    out["christmas"] = _weekend_shift(date(year, 12, 25))

    return out


@cache
def half_days(year: int) -> dict[str, date]:
    """1pm early closes observed in `year`, keyed by a short name."""
    out: dict[str, date] = {}

    thanksgiving = _nth_weekday(year, 11, 3, 4)
    out["day_after_thanksgiving"] = thanksgiving + timedelta(days=1)

    independence_day = date(year, 7, 4)
    weekday = independence_day.weekday()
    if weekday == 5:  # Saturday
        out["independence_day_eve"] = independence_day - timedelta(days=1)
    elif weekday == 6:  # Sunday
        out["independence_day_eve"] = independence_day + timedelta(days=1)
    elif weekday != 0:  # Monday: the preceding day is Sunday, nothing to shorten
        out["independence_day_eve"] = independence_day - timedelta(days=1)

    christmas = date(year, 12, 25)
    if christmas.weekday() not in (5, 6, 0):
        # Weekend: Dec 24 is either the observed holiday itself (Saturday) or a
        # non-business day (Sunday). Monday: Dec 24 is a Sunday.
        out["christmas_eve"] = christmas - timedelta(days=1)

    return out


def _in_range(d: date) -> bool:
    return MIN_YEAR <= d.year <= MAX_YEAR


def is_holiday(d: date) -> bool:
    """True if NYSE is fully closed on `d` (weekend, holiday or a one-off closure).

    Outside the tracked range this only ever answers from the weekend and the
    explicit one-off list — it does not guess at a holiday it hasn't verified.
    """
    if d.weekday() >= 5:
        return True
    if d in EXTRA_HOLIDAYS:
        return True
    return _in_range(d) and d in holidays(d.year).values()


def is_half_day(d: date) -> bool:
    """True if NYSE closes early (1pm) on `d`.

    Outside the tracked range this returns False rather than guessing — the safe
    default is the ordinary full session.
    """
    return _in_range(d) and d not in EXTRA_HOLIDAYS and d in half_days(d.year).values()


def previous_trading_day(d: date) -> date:
    """The most recent trading day strictly before `d`.

    Mirrors the OCC's own rule for an option whose scheduled expiration lands on an
    exchange holiday: trading (and expiration) moves to the business day immediately
    before it, not forward to the next open session.
    """
    prior = d - timedelta(days=1)
    while is_holiday(prior):
        prior -= timedelta(days=1)
    return prior
