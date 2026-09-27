# SPDX-License-Identifier: Apache-2.0
"""Canonical JSON (v2) and SHA-256 identities. Standard library only.

Every identity in SignalQuarry — configuration, dataset, decision, ledger and
bundle hashes — is ``sha256:`` over this encoding, so it must never change
without a new version. Rules:

* Objects: keys are strings, sorted by code point; no insignificant whitespace.
* Strings: UTF-8, non-ASCII kept as-is (``ensure_ascii=False``).
* Integers and booleans: JSON literals. ``None`` is ``null``.
* Floats: must be finite; encoded with Python's shortest round-trip ``repr``.
* ``Decimal``: must be finite; encoded as a JSON **string** of the normalized
  plain decimal (no exponent, no trailing fractional zeros, ``-0`` → ``"0"``),
  so numerically equal decimals hash equally.
* ``datetime``: must be timezone-aware; converted to UTC,
  ``YYYY-MM-DDTHH:MM:SS[.ffffff]Z``. ``date``: ``YYYY-MM-DD``.
* ``Enum``: its value, canonicalized. Tuples and lists: arrays.
* Objects exposing ``model_dump`` (pydantic models) are dumped in JSON mode.

Anything else raises ``TypeError``. This module is vendored by other projects,
so it must stay dependency-free.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

CANONICAL_VERSION = "v2"
HASH_PREFIX = "sha256:"


def _decimal_text(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("CANONICAL_DECIMAL_NOT_FINITE")
    if value.is_zero():
        return "0"
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _datetime_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("CANONICAL_DATETIME_NAIVE")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def to_canonical(value: Any) -> Any:
    """Return the JSON-ready canonical form of ``value``."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("CANONICAL_FLOAT_NOT_FINITE")
        return value
    if isinstance(value, Decimal):
        return _decimal_text(value)
    if isinstance(value, datetime):
        return _datetime_text(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return to_canonical(value.value)
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"CANONICAL_KEY_NOT_STRING:{type(key).__name__}")
            result[key] = to_canonical(item)
        return result
    if isinstance(value, (list, tuple)):
        return [to_canonical(item) for item in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return to_canonical(dump(mode="json"))
    raise TypeError(f"CANONICAL_TYPE_UNSUPPORTED:{type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return the single canonical JSON text of ``value``."""
    return json.dumps(
        to_canonical(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def canonical_hash(value: Any) -> str:
    """Return ``sha256:<hex>`` over the canonical JSON of ``value``."""
    return HASH_PREFIX + hashlib.sha256(canonical_bytes(value)).hexdigest()


def hash_without(value: Any, *fields: str) -> str:
    """Hash an object after removing its self-referential fields (e.g. its own hash)."""
    body = to_canonical(value)
    if not isinstance(body, dict):
        raise TypeError("CANONICAL_HASH_WITHOUT_REQUIRES_OBJECT")
    return canonical_hash({key: item for key, item in body.items() if key not in fields})


def file_sha256(path: Any) -> str:
    """Return the hex SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "CANONICAL_VERSION",
    "HASH_PREFIX",
    "canonical_bytes",
    "canonical_hash",
    "canonical_json",
    "file_sha256",
    "hash_without",
    "to_canonical",
]
