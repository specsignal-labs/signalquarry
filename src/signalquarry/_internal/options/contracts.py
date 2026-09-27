# SPDX-License-Identifier: Apache-2.0
"""OCC option symbols, contracts and quotes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Literal

Right = Literal["CALL", "PUT"]
_OCC = re.compile(r"^(?P<root>[A-Z]{1,6})(?P<date>\d{6})(?P<right>[CP])(?P<strike>\d{8})$")


class OptionsError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}:{detail}" if detail else code)
        self.code = code


@dataclass(frozen=True, order=True)
class OptionContract:
    symbol: str
    underlying: str
    right: Right
    strike: Decimal
    expiration: date

    def dte(self, today: date) -> int:
        return (self.expiration - today).days


def occ_symbol(underlying: str, expiration: date, right: Right, strike: Decimal) -> str:
    """Alpaca/OCC compact form, e.g. ``QQQ261002P00480000`` (strike × 1000, 8 digits)."""
    thousandths = strike * 1000
    if thousandths != thousandths.to_integral_value() or not (0 < thousandths < 10**8):
        raise OptionsError("OPTION_STRIKE_INVALID", str(strike))
    return f"{underlying.upper()}{expiration:%y%m%d}{right[0]}{int(thousandths):08d}"


def parse_occ(symbol: str) -> OptionContract:
    match = _OCC.fullmatch(symbol.upper().replace(" ", ""))
    if match is None:
        raise OptionsError("OPTION_SYMBOL_INVALID", symbol)
    expiration = datetime.strptime(match["date"], "%y%m%d").date()
    right: Right = "CALL" if match["right"] == "C" else "PUT"
    strike = Decimal(int(match["strike"])).scaleb(-3)
    strike = strike.quantize(Decimal(1)) if strike == strike.to_integral_value() else strike.normalize()
    return OptionContract(match.group(0), match["root"], right, strike, expiration)


def contract(underlying: str, expiration: date, right: Right, strike: Decimal) -> OptionContract:
    return OptionContract(
        occ_symbol(underlying, expiration, right, strike), underlying.upper(), right, strike, expiration
    )


@dataclass(frozen=True)
class Quote:
    bid: Decimal
    ask: Decimal
    timestamp: datetime

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2

    @property
    def relative_spread(self) -> Decimal:
        mid = self.mid
        return (self.ask - self.bid) / mid if mid > 0 else Decimal("Infinity")


def quote_violations(
    quote: Quote | None,
    *,
    now: datetime,
    max_age_seconds: int,
    min_bid: Decimal,
    max_relative_spread: Decimal,
    max_future_skew_seconds: int = 2,
) -> tuple[str, ...]:
    """Reasons a quote may not be used (empty when usable)."""
    if quote is None:
        return ("QUOTE_MISSING",)
    reasons = []
    age = (now - quote.timestamp).total_seconds()
    if age > max_age_seconds:
        reasons.append("QUOTE_STALE")
    if age < -max_future_skew_seconds:
        reasons.append("QUOTE_FROM_FUTURE")
    if quote.ask < quote.bid:
        reasons.append("QUOTE_CROSSED")
    if quote.bid < min_bid:
        reasons.append("QUOTE_BID_TOO_LOW")
    if quote.ask <= 0 or quote.relative_spread > max_relative_spread:
        reasons.append("QUOTE_SPREAD_TOO_WIDE")
    return tuple(reasons)
