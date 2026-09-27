# SPDX-License-Identifier: Apache-2.0
"""Synthetic option chains (Black-Scholes) for the low-evidence simulator.

Volatility is the underlying's trailing 20-session realized volatility times 1.1
(floored at 10%); quotes are the model price ± a half-spread of 2% of the price
(at least one tick). Weekly expirations fall on Fridays. Nothing here is market data:
results that use these chains are graded ``low_evidence_options``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import ROUND_HALF_UP, Decimal
from statistics import NormalDist

import numpy as np

from signalquarry._internal.calendar.nyse import is_half_day
from signalquarry._internal.options.contracts import OptionsError, Quote, contract, parse_occ
from signalquarry._internal.options.resolver import Candidate

_N = NormalDist()
RATE = 0.04

OPEN_CHECKPOINT = time(9, 35)
_NORMAL_CLOSE_CHECKPOINT = time(15, 55)
_HALF_DAY_CLOSE_CHECKPOINT = time(12, 55)


def session_checkpoints(session: date) -> tuple[tuple[str, time], tuple[str, time]]:
    """The (open, close) checkpoint moments for `session`.

    Both are 5 minutes before the exchange's actual session close, so that quotes
    are never modelled at the closing bell itself. The close checkpoint moves from
    15:55 to 12:55 on an NYSE 1pm early close (``_internal.calendar.nyse``); every
    other checkpoint concern (holiday sessions in a synthetic or demo dataset, dates
    outside the calendar's tracked range) falls back to the ordinary full session,
    since ``is_half_day`` only ever answers "yes, definitely a half day".
    """
    close = _HALF_DAY_CLOSE_CHECKPOINT if is_half_day(session) else _NORMAL_CLOSE_CHECKPOINT
    return (("open", OPEN_CHECKPOINT), ("close", close))


def black_scholes(
    spot: float, strike: float, years: float, sigma: float, right: str, rate: float = RATE
) -> float:
    if years <= 0:
        return max(0.0, (strike - spot) if right == "PUT" else (spot - strike))
    root = sigma * math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate + sigma * sigma / 2) * years) / root
    d2 = d1 - root
    if right == "CALL":
        return spot * _N.cdf(d1) - strike * math.exp(-rate * years) * _N.cdf(d2)
    return strike * math.exp(-rate * years) * _N.cdf(-d2) - spot * _N.cdf(-d1)


def realized_sigma(closes: np.ndarray) -> float:
    values = np.asarray(closes, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return 0.2
    returns = np.diff(np.log(values[-21:]))
    return max(0.10, float(np.std(returns, ddof=1) * math.sqrt(252)) * 1.1)


def fridays(start: date, days: int) -> list[date]:
    first = start + timedelta(days=(4 - start.weekday()) % 7)
    return [
        first + timedelta(days=7 * k)
        for k in range((days // 7) + 1)
        if (first + timedelta(days=7 * k) - start).days <= days
    ]


def _strike_step(spot: float) -> Decimal:
    return Decimal("1") if spot < 250 else Decimal("1") if spot < 1000 else Decimal("5")


def model_quote(
    spot: Decimal,
    strike: Decimal,
    expiration: date,
    right: str,
    *,
    session: date,
    at: datetime,
    sigma: float,
    tick: Decimal,
) -> Quote:
    years = max((expiration - session).days, 0) / 365.0 + 6.5 / (24 * 365)
    theo = black_scholes(float(spot), float(strike), years, sigma, right)
    half = max(float(tick), 0.02 * theo)
    bid = Decimal(str(max(0.0, theo - half))).quantize(tick, rounding=ROUND_HALF_UP)
    ask = Decimal(str(theo + half)).quantize(tick, rounding=ROUND_HALF_UP)
    return Quote(bid, max(ask, bid + tick), at)


class ChainModel:
    """A lazily priced synthetic chain: quotes only what is asked for."""

    def __init__(
        self, underlying: str, *, spot: Decimal, session: date, at: datetime, sigma: float, tick: Decimal
    ) -> None:
        self.underlying, self.spot, self.session, self.at, self.sigma, self.tick = (
            underlying,
            spot,
            session,
            at,
            sigma,
            tick,
        )

    def quote(self, symbol_contract) -> Quote:  # noqa: ANN001 - an OptionContract
        return model_quote(
            self.spot,
            symbol_contract.strike,
            symbol_contract.expiration,
            symbol_contract.right,
            session=self.session,
            at=self.at,
            sigma=self.sigma,
            tick=self.tick,
        )

    def candidates(self, right: str, min_dte: int, max_dte: int, target: Decimal) -> list[Candidate]:
        """Contracts of ``right`` inside the DTE window, strikes within 10 steps of ``target`` on each side."""
        step = _strike_step(float(self.spot))
        centre = (target / step).to_integral_value() * step
        strikes = [centre + k * step for k in range(-10, 11) if centre + k * step > 0]
        out = []
        for expiry in fridays(self.session, max_dte):
            if not min_dte <= (expiry - self.session).days <= max_dte:
                continue
            for strike in strikes:
                item = contract(self.underlying, expiry, right, strike)  # type: ignore[arg-type]
                out.append(Candidate(item, self.quote(item)))
        return out


class RecordedChain:
    """Quotes recorded from the provider on this session, with the model for anything missing.

    A recording is one snapshot per session; its bid and ask stand in for every checkpoint
    that session (re-stamped to the checkpoint time). Contracts the recording lacks are
    priced by the fallback model, and ``recorded``/``modelled`` count which was used.
    """

    def __init__(self, quotes: Mapping[str, tuple[Decimal, Decimal]], fallback: ChainModel) -> None:
        self.quotes, self.fallback = dict(quotes), fallback
        self.recorded = 0
        self.modelled = 0

    def quote(self, symbol_contract) -> Quote:  # noqa: ANN001 - an OptionContract
        found = self.quotes.get(symbol_contract.symbol)
        if found is None:
            self.modelled += 1
            return self.fallback.quote(symbol_contract)
        self.recorded += 1
        return Quote(found[0], found[1], self.fallback.at)

    def candidates(self, right: str, min_dte: int, max_dte: int, target: Decimal) -> list[Candidate]:
        session = self.fallback.session
        out = []
        for symbol in sorted(self.quotes):
            try:
                item = parse_occ(symbol)
            except OptionsError:
                continue
            if item.right == right and min_dte <= (item.expiration - session).days <= max_dte:
                out.append(Candidate(item, self.quote(item)))
        return out or self.fallback.candidates(right, min_dte, max_dte, target)
