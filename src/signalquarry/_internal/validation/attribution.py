# SPDX-License-Identifier: Apache-2.0
"""Descriptive close-to-close attribution from recorded positions and fills.

Currency amounts stay exact Decimals (serialized as strings); only shares and ratios
are rounded. Cash flows are never assigned to a symbol without supporting evidence.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date
from decimal import Decimal
from typing import Any

PositionRow = tuple[date, str, Decimal, Decimal, Decimal]
FillRow = tuple[date, str, str, Decimal, Decimal, Decimal]
ZERO = Decimal(0)
REMAINDER_NOTE = (
    "Dividends (including accrued receivables), cash rounding and any other cash flows "
    "not explained by marked holdings or fill fees. Settlement transfers do not create P&L. "
    "These inputs cannot separate the remainder's components."
)


def _mean(values: list[Decimal], digits: int = 6) -> float | None:
    return round(float(sum(values, ZERO) / len(values)), digits) if values else None


def attribution(
    sessions: Sequence[date],
    equity: Sequence[Decimal],
    initial: Decimal,
    positions: Iterable[PositionRow],
    fills: Iterable[FillRow],
) -> dict[str, Any]:
    """Split equity changes by symbol, fees and residual cash flows, including by year.

    A holding's weight uses total closing equity; concentration uses weights normalized
    over invested value (so cash does not reduce concentration). Nonpositive equity has
    undefined equity weights and is excluded from their averages. Ties for the largest
    weight use the earliest session, then the alphabetically first symbol.
    """
    if (
        len(sessions) != len(equity)
        or list(sessions) != sorted(set(sessions))
        or not initial.is_finite()
        or initial <= 0
        or any(not value.is_finite() for value in equity)
    ):
        raise ValueError("Sessions, equity and initial cash are invalid or not aligned.")
    if not sessions:
        return {"status": "insufficient", "reason": "No sessions."}
    values: dict[date, dict[str, Decimal]] = {session: {} for session in sessions}
    for session, symbol, quantity, mark, value in positions:
        if (
            session not in values
            or not symbol
            or symbol in values[session]
            or any(not item.is_finite() for item in (quantity, mark, value))
            or quantity <= 0
            or mark < 0
            or value != quantity * mark
        ):
            raise ValueError("Positions are invalid, duplicated or not aligned.")
        values[session][symbol] = value
    if not any(values.values()):
        return {"status": "insufficient", "reason": "No positions rows."}
    net: dict[date, dict[str, Decimal]] = {session: {} for session in sessions}
    fees: dict[date, Decimal] = dict.fromkeys(sessions, ZERO)
    for session, symbol, side, quantity, price, fee in fills:
        if (
            session not in net
            or not symbol
            or side not in ("buy", "sell")
            or any(not item.is_finite() for item in (quantity, price, fee))
            or quantity <= 0
            or price < 0
            or fee < 0
        ):
            raise ValueError("Fills are invalid or not aligned.")
        amount = quantity * price * (1 if side == "buy" else -1)
        net[session][symbol] = net[session].get(symbol, ZERO) + amount
        fees[session] -= fee

    pnl: dict[str, Decimal] = {}
    held: dict[str, int] = {}
    weights: dict[str, list[Decimal]] = {}
    years: dict[int, dict[str, Any]] = {}
    previous: dict[str, Decimal] = {}
    previous_equity = initial
    invested_shares: list[Decimal] = []
    concentration: list[Decimal] = []
    counts: list[int] = []
    largest: tuple[Decimal, date, str] | None = None
    for session, level in zip(sessions, equity, strict=True):
        current = values[session]
        yearly = years.setdefault(session.year, {"by_symbol": {}, "fees": ZERO, "total": ZERO})
        for symbol in sorted(current.keys() | previous.keys() | net[session].keys()):
            change = current.get(symbol, ZERO) - previous.get(symbol, ZERO) - net[session].get(symbol, ZERO)
            pnl[symbol] = pnl.get(symbol, ZERO) + change
            yearly["by_symbol"][symbol] = yearly["by_symbol"].get(symbol, ZERO) + change
        yearly["fees"] += fees[session]
        yearly["total"] += level - previous_equity
        counts.append(len(current))
        invested = sum(current.values(), ZERO)
        if level > 0:
            invested_shares.append(invested / level)
        if invested > 0:
            concentration.append(sum(((value / invested) ** 2 for value in current.values()), ZERO))
        for symbol in sorted(current):
            held[symbol] = held.get(symbol, 0) + 1
            weights.setdefault(symbol, [])
            if level > 0:
                weight = current[symbol] / level
                weights[symbol].append(weight)
                if largest is None or weight > largest[0]:
                    largest = (weight, session, symbol)
        previous, previous_equity = current, level

    def split(total: Decimal, symbols: Decimal, fee: Decimal) -> dict[str, str]:
        return {
            "total": str(total),
            "symbols": str(symbols),
            "fees": str(fee),
            "remainder": str(total - symbols - fee),
        }

    by_year = []
    for year, row in sorted(years.items()):
        by_year.append(
            {
                "year": year,
                **split(row["total"], sum(row["by_symbol"].values(), ZERO), row["fees"]),
                "by_symbol": {symbol: str(value) for symbol, value in sorted(row["by_symbol"].items())},
            }
        )
    return {
        "status": "ok",
        "note": "Descriptive and in-sample; this attribution changes no claim, gate or trial count.",
        "by_symbol": [
            {
                "symbol": symbol,
                "pnl": str(value),
                "share_initial": round(float(value / initial), 8),
                "sessions_held": held.get(symbol, 0),
                "average_weight": _mean(weights.get(symbol, [])),
            }
            for symbol, value in sorted(pnl.items())
        ],
        "reconciliation": split(equity[-1] - initial, sum(pnl.values(), ZERO), sum(fees.values(), ZERO)),
        "remainder_note": REMAINDER_NOTE,
        "by_year": by_year,
        "exposure": {
            "average_invested_share": _mean(invested_shares),
            "position_session_share": round(sum(count > 0 for count in counts) / len(sessions), 6),
            "average_holdings": round(sum(counts) / len(sessions), 6),
            "maximum_holdings": max(counts),
            "average_herfindahl": _mean(concentration),
            "largest_weight": None
            if largest is None
            else {
                "weight": round(float(largest[0]), 6),
                "session": largest[1].isoformat(),
                "symbol": largest[2],
            },
            "note": "Herfindahl weights are normalized over invested value. Equity weights exclude nonpositive equity sessions.",
        },
    }
