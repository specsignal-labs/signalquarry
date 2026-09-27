# SPDX-License-Identifier: Apache-2.0
"""Canonical v2 is a published encoding: these vectors must never change."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum

import pytest
from pydantic import BaseModel

from signalquarry._internal.canonical import canonical_hash, canonical_json, hash_without


class Side(Enum):
    BUY = "buy"


class Order(BaseModel):
    symbol: str
    quantity: Decimal
    at: datetime


VECTORS = [
    ({"b": 1, "a": [True, None, "é"]}, '{"a":[true,null,"é"],"b":1}'),
    (
        {"d": Decimal("1.0"), "e": Decimal("0.10"), "f": Decimal("-2.5E+3"), "g": Decimal("-0.00")},
        '{"d":"1","e":"0.1","f":"-2500","g":"0"}',
    ),
    ({"x": 1.5, "y": 0.1, "z": 1e-7}, '{"x":1.5,"y":0.1,"z":1e-07}'),
    (
        {"t": datetime(2026, 9, 25, 9, 30, tzinfo=timezone(timedelta(hours=-4)))},
        '{"t":"2026-09-25T13:30:00Z"}',
    ),
    (
        {"t": datetime(2026, 1, 2, 3, 4, 5, 600, tzinfo=UTC), "d": date(2026, 1, 2)},
        '{"d":"2026-01-02","t":"2026-01-02T03:04:05.000600Z"}',
    ),
    ({"side": Side.BUY, "legs": ("a", "b")}, '{"legs":["a","b"],"side":"buy"}'),
    (
        Order(symbol="SPY", quantity=Decimal("10.500"), at=datetime(2026, 1, 2, tzinfo=UTC)),
        '{"at":"2026-01-02T00:00:00Z","quantity":"10.500","symbol":"SPY"}',
    ),
]
# Pydantic dumps Decimal as a string in JSON mode, so model fields keep their scale; bare Decimals are normalized.


@pytest.mark.parametrize(("value", "expected"), VECTORS)
def test_canonical_json_vectors(value: object, expected: str) -> None:
    assert canonical_json(value) == expected


def test_canonical_hash_vector() -> None:
    assert canonical_hash({"b": 1, "a": [True, None, "é"]}) == (
        "sha256:" + __import__("hashlib").sha256('{"a":[true,null,"é"],"b":1}'.encode()).hexdigest()
    )


def test_numerically_equal_decimals_hash_equally() -> None:
    assert canonical_hash({"w": Decimal("0.950")}) == canonical_hash({"w": Decimal("0.95")})


@pytest.mark.parametrize(
    "value",
    [
        {"t": datetime(2026, 1, 1)},
        {"x": float("nan")},
        {"x": float("inf")},
        {"x": Decimal("NaN")},
    ],
)
def test_rejects_ambiguous_values(value: object) -> None:
    with pytest.raises(ValueError):
        canonical_json(value)


@pytest.mark.parametrize("value", [{1: "a"}, {"s": {1, 2}}, {"o": object()}])
def test_rejects_unsupported_types(value: object) -> None:
    with pytest.raises(TypeError):
        canonical_json(value)


def test_hash_without_excludes_self_referential_field() -> None:
    body = {"a": 1, "hash": "x"}
    assert hash_without(body, "hash") == canonical_hash({"a": 1})
