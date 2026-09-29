# SPDX-License-Identifier: Apache-2.0
"""The options simulator must follow the NYSE calendar for checkpoints and expiries.

Before this fix, the close checkpoint was hard-coded to 15:55 on every session. On a
half day that is three hours after the market actually shut, and it silently changed a
trading rule: the default ``entry_cutoff`` (15:15) never arrives before a half day's
1pm close, so a strategy allowed to open near the real close (as it would in the live
paper runner, whose poller simply stops once the market is shut) was instead always
refused with ``OPTIONS_ENTRY_CUTOFF`` -- a backtest/paper parity gap that only showed up
on the ~2 sessions a year that close early.

Weekly expirations were also generated with no holiday awareness at all, so a
contract scheduled to expire on Good Friday (the only NYSE holiday that can land on a
Friday) would only ever have expired on the next actual trading session -- backwards
from the OCC's own rule, which moves such a contract's expiration to the business day
immediately before it.
"""

from __future__ import annotations

from datetime import date, time

from signalquarry._internal.calendar.nyse import holidays, is_half_day
from signalquarry._internal.data.synthetic import synthetic_dataset
from signalquarry._internal.engine.run import simulate
from signalquarry._internal.options.chains import fridays, session_checkpoints
from signalquarry.sdk import definition_of
from signalquarry.sdk.options import OD, OptionsCtx, Strike, options_strategy, sell_put
from tests.options.wheel_reference import CODES, WheelParams, wheel_spec

HALF_DAY = date(2024, 11, 29)  # day after Thanksgiving: NYSE closes at 1pm
NORMAL_DAY = date(2024, 11, 26)  # the Tuesday before: a plain full session
DATA = synthetic_dataset(date(2024, 11, 1), date(2024, 12, 6), symbols=("QQQ",))


@options_strategy(params=WheelParams, lookback=lambda p: 1)
def always_open_put(ctx: OptionsCtx, p: WheelParams):
    # A far-dated, spread-intolerant request that resolve() can never fill (the same
    # combination `test_simulator_edges.py::test_refusals` uses for the same reason),
    # so the wheel stays FLAT throughout: the cutoff check is isolated from every
    # other reason an open can be refused.
    return OD.open(sell_put("QQQ", dte=(100, 110), strike=Strike.otm("0.02")), "ALWAYS_TRY")


def _close_checkpoint(session: date):
    spec = wheel_spec(
        options={"underlyings": ["QQQ"], "max_relative_spread": "0.0001"},
        reason_codes={**CODES, "ALWAYS_TRY": "Always tries to open, for this test only."},
    )
    definition, params = definition_of(always_open_put), WheelParams()
    result = simulate(spec, definition, params, DATA, start=session, end=session)
    return next(d for d in result.decisions if d["checkpoint"] == "close")


def test_confirmed_against_the_nyse_calendar() -> None:
    assert is_half_day(HALF_DAY)
    assert not is_half_day(NORMAL_DAY)


def test_close_checkpoint_moves_to_1255_on_a_half_day() -> None:
    assert dict(session_checkpoints(NORMAL_DAY))["close"] == time(15, 55)
    assert dict(session_checkpoints(HALF_DAY))["close"] == time(12, 55)


def test_entry_cutoff_does_not_fire_before_a_half_days_real_close() -> None:
    normal_close = _close_checkpoint(NORMAL_DAY)
    assert normal_close["outcome"] == "refused"
    assert normal_close["why"] == "OPTIONS_ENTRY_CUTOFF"

    half_day_close = _close_checkpoint(HALF_DAY)
    assert half_day_close.get("why") != "OPTIONS_ENTRY_CUTOFF"


def test_a_weekly_expiration_on_good_friday_moves_to_the_preceding_thursday() -> None:
    good_friday = holidays(2024)["good_friday"]
    assert good_friday.isoformat() == "2024-03-29"
    expiries = fridays(date(2024, 3, 1), 40)
    assert good_friday not in expiries
    assert date(2024, 3, 28) in expiries  # the Thursday before, per the OCC's rule
    # every other generated date is still an ordinary Friday
    assert all(day.weekday() in (3, 4) for day in expiries)
