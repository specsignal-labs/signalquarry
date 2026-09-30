# SPDX-License-Identifier: Apache-2.0
"""What a strategy returns for one decision session."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Any, Literal

_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.]{0,9}$")
_CODE = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")
_WEIGHT_QUANTUM = Decimal("0.000001")

Action = Literal["target", "hold", "unavailable"]


def _weight(value: Decimal | float | int | str) -> Decimal:
    if isinstance(value, bool):
        raise TypeError("DECISION_WEIGHT_INVALID")
    number = value if isinstance(value, Decimal) else Decimal(str(value))
    if not number.is_finite():
        raise ValueError("DECISION_WEIGHT_INVALID")
    return number.quantize(_WEIGHT_QUANTUM, rounding=ROUND_HALF_EVEN)


@dataclass(frozen=True)
class Decision:
    """A strategy's output. Build it with :meth:`target`, :meth:`hold` or :meth:`unavailable`."""

    action: Action
    weights: dict[str, Decimal] = field(default_factory=dict[str, Decimal])
    reason_codes: tuple[str, ...] = ()
    state: dict[str, Any] | None = None

    @classmethod
    def target(
        cls,
        weights: dict[str, Decimal | float | int | str],
        *reason_codes: str,
        state: dict[str, Any] | None = None,
    ) -> Decision:
        """Hold these portfolio weights (fractions of equity). Symbols not listed are sold."""
        normalized: dict[str, Decimal] = {}
        shared_weights: dict[Decimal, Decimal] = {}
        for symbol, value in weights.items():
            key = str(symbol).upper()
            if not _SYMBOL.fullmatch(key):
                raise ValueError(f"DECISION_SYMBOL_INVALID:{symbol}")
            if key in normalized:
                raise ValueError(f"DECISION_SYMBOL_DUPLICATE:{key}")
            weight = _weight(value)
            if weight < 0:
                raise ValueError(f"DECISION_WEIGHT_NEGATIVE:{key}")
            if weight > 0:
                # Decimal is immutable. Share equal quantized weights within this
                # decision instead of retaining one object per symbol in the ledger.
                normalized[key] = shared_weights.setdefault(weight, weight)
        if sum(normalized.values(), Decimal(0)) > 1:
            raise ValueError("DECISION_WEIGHTS_EXCEED_ONE")
        return cls("target", dict(sorted(normalized.items())), checked_codes(reason_codes), state)

    @classmethod
    def hold(cls, *reason_codes: str, state: dict[str, Any] | None = None) -> Decision:
        """Keep current holdings unchanged."""
        return cls("hold", {}, checked_codes(reason_codes), state)

    @classmethod
    def unavailable(cls, *reason_codes: str) -> Decision:
        """No decision can be made (e.g. missing data). Holdings and state stay unchanged."""
        return cls("unavailable", {}, checked_codes(reason_codes), None)


def checked_codes(codes: tuple[str, ...]) -> tuple[str, ...]:
    for code in codes:
        if not _CODE.fullmatch(code):
            raise ValueError(f"DECISION_REASON_CODE_INVALID:{code}")
    return tuple(codes)
