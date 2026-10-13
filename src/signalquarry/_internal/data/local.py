# SPDX-License-Identifier: Apache-2.0
"""Pure, strict CSV parsing for the local daily-bar replay format (version 1)."""

from __future__ import annotations

import csv
import io
import math
import re
from collections.abc import Iterator
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from signalquarry._internal.calendar.nyse import MAX_YEAR, MIN_YEAR, is_holiday
from signalquarry._internal.data.alpaca import RawPage
from signalquarry._internal.data.dataset import FIELDS, MICRO, Dividend, Split

PARSER = "local-csv/v1"
BARS_ENDPOINT = "local:bars.csv"
ACTIONS_ENDPOINT = "local:actions.csv"
BARS_HEADER = ("session", "symbol", "open", "high", "low", "close", "volume")
ACTIONS_HEADER = ("symbol", "type", "ex_date", "pay_date", "ratio", "amount")
_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)\Z")
_SYMBOL = re.compile(r"[A-Z][A-Z0-9.]{0,9}\Z")
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")


class ImportError(ValueError):
    """A coded refusal with the file role and physical CSV line, without market values."""

    def __init__(self, code: str, role: str, line: int, detail: str) -> None:
        super().__init__(f"{role} line {line}: {detail}")
        self.code = code


def _rows(body: bytes, header: tuple[str, ...], role: str) -> Iterator[tuple[int, dict[str, str]]]:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        line = body[: exc.start].count(b"\n") + 1
        raise ImportError("DATA_IMPORT_VALUE_INVALID", role, line, "expected UTF-8 CSV") from exc
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        first = next(reader, None)
        if first is None:
            raise ImportError("DATA_IMPORT_EMPTY", role, 1, "empty file")
        if tuple(first) != header:
            raise ImportError("DATA_IMPORT_HEADER_INVALID", role, 1, "expected " + ",".join(header))
        for row in reader:
            if len(row) != len(header):
                raise ImportError("DATA_IMPORT_VALUE_INVALID", role, reader.line_num, "wrong field count")
            yield reader.line_num, dict(zip(header, row, strict=True))
    except csv.Error as exc:
        raise ImportError(
            "DATA_IMPORT_VALUE_INVALID", role, max(reader.line_num, 1), "malformed CSV"
        ) from exc


def _day(text: str, role: str, line: int, field: str) -> date:
    try:
        if not _DATE.fullmatch(text):
            raise ValueError
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ImportError("DATA_IMPORT_VALUE_INVALID", role, line, f"{field} needs YYYY-MM-DD") from exc


def _number(text: str, role: str, line: int, field: str, *, price: bool = False) -> Decimal:
    if price and not _NUMBER.fullmatch(text):
        raise ImportError("DATA_IMPORT_VALUE_INVALID", role, line, f"{field} needs finite decimal text")
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise ImportError("DATA_IMPORT_VALUE_INVALID", role, line, f"{field} needs a number") from exc
    if not value.is_finite():
        raise ImportError("DATA_IMPORT_VALUE_INVALID", role, line, f"{field} must be finite")
    if price and len(text.partition(".")[2]) > 6:
        raise ImportError("DATA_IMPORT_PRECISION", role, line, f"{field} exceeds six decimal places")
    if price and value <= 0:
        raise ImportError("DATA_IMPORT_PRICE_NONPOSITIVE", role, line, f"{field} must be positive")
    if price and value > Decimal(2**63 - 1) / MICRO:
        raise ImportError("DATA_IMPORT_VALUE_INVALID", role, line, f"{field} exceeds micro-dollar storage")
    return value


def _symbol(text: str, role: str, line: int) -> str:
    if not _SYMBOL.fullmatch(text):
        raise ImportError("DATA_IMPORT_VALUE_INVALID", role, line, "symbol needs 1–10 uppercase characters")
    return text


def bars_from_pages(pages: list[RawPage]) -> dict[str, list[dict[str, Any]]]:
    bars: dict[str, dict[date, dict[str, Any]]] = {}
    for page in pages:
        for line, row in _rows(page.body, BARS_HEADER, "bars"):
            session = _day(row["session"], "bars", line, "session")
            if not MIN_YEAR <= session.year <= MAX_YEAR or is_holiday(session):
                raise ImportError(
                    "DATA_IMPORT_SESSION_INVALID",
                    "bars",
                    line,
                    f"session must be an NYSE session within {MIN_YEAR}–{MAX_YEAR}",
                )
            symbol = _symbol(row["symbol"], "bars", line)
            prices = {name: _number(row[name], "bars", line, name, price=True) for name in FIELDS}
            if not (
                prices["low"] <= prices["open"] <= prices["high"]
                and prices["low"] <= prices["close"] <= prices["high"]
            ):
                raise ImportError(
                    "DATA_IMPORT_OHLC_INVALID", "bars", line, "low/high must enclose open/close"
                )
            number = _number(row["volume"], "bars", line, "volume")
            volume = float(number)
            if not math.isfinite(volume) or number < 0:
                raise ImportError(
                    "DATA_IMPORT_VALUE_INVALID", "bars", line, "volume must be finite and nonnegative"
                )
            existing = bars.setdefault(symbol, {})
            if session in existing:
                raise ImportError("DATA_IMPORT_DUPLICATE", "bars", line, "duplicate session/symbol")
            existing[session] = {"session": session, **prices, "volume": volume}
    if not bars:
        raise ImportError("DATA_IMPORT_EMPTY", "bars", 1, "no bar rows")
    return {symbol: [rows[key] for key in sorted(rows)] for symbol, rows in sorted(bars.items())}


def actions_from_pages(
    pages: list[RawPage], symbols: set[str]
) -> tuple[tuple[Split, ...], tuple[Dividend, ...]]:
    splits: dict[tuple[str, date], Split] = {}
    dividends: dict[tuple[str, date], Dividend] = {}
    for page in pages:
        for line, row in _rows(page.body, ACTIONS_HEADER, "actions"):
            symbol = _symbol(row["symbol"], "actions", line)
            if symbol not in symbols:
                raise ImportError("DATA_IMPORT_ACTION_SYMBOL", "actions", line, "symbol absent from bars")
            ex_date = _day(row["ex_date"], "actions", line, "ex_date")
            key = symbol, ex_date
            if row["type"] == "split":
                ratio = _number(row["ratio"], "actions", line, "ratio")
                if ratio <= 0 or not 0 < float(ratio) < math.inf:
                    raise ImportError(
                        "DATA_IMPORT_VALUE_INVALID", "actions", line, "ratio must be positive and finite"
                    )
                if row["pay_date"] or row["amount"]:
                    raise ImportError(
                        "DATA_IMPORT_VALUE_INVALID", "actions", line, "split needs empty pay_date and amount"
                    )
                if key in splits:
                    raise ImportError(
                        "DATA_IMPORT_DUPLICATE", "actions", line, "duplicate split symbol/ex_date"
                    )
                splits[key] = Split(symbol, ex_date, ratio)
            elif row["type"] == "dividend":
                pay_date = _day(row["pay_date"], "actions", line, "pay_date")
                amount = _number(row["amount"], "actions", line, "amount")
                if amount < 0:
                    raise ImportError(
                        "DATA_IMPORT_VALUE_INVALID", "actions", line, "amount must be nonnegative"
                    )
                if pay_date < ex_date or row["ratio"]:
                    raise ImportError(
                        "DATA_IMPORT_VALUE_INVALID",
                        "actions",
                        line,
                        "dividend needs pay_date on/after ex_date and empty ratio",
                    )
                if key in dividends:
                    raise ImportError(
                        "DATA_IMPORT_DUPLICATE", "actions", line, "duplicate dividend symbol/ex_date"
                    )
                dividends[key] = Dividend(symbol, ex_date, pay_date, amount)
            else:
                raise ImportError(
                    "DATA_IMPORT_VALUE_INVALID", "actions", line, "type needs split or dividend"
                )
    return tuple(splits[key] for key in sorted(splits)), tuple(dividends[key] for key in sorted(dividends))
