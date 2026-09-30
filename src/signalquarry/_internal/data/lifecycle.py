# SPDX-License-Identifier: Apache-2.0
"""Normalized corporate-action terms, independent of any provider's wire format.

The Alpaca adapter does not emit these events yet. A caller must verify every
term, identity and provenance field before constructing one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from signalquarry._internal.data.identity import AssetKey

ActionKind = Literal["rename", "split", "worthless_removal", "cash_merger", "stock_merger", "mixed_merger"]
FractionPolicy = Literal["retain", "reject_noninteger"]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.]{0,15}$")


class LifecycleError(ValueError):
    """An event's terms or ordering do not permit a deterministic transition."""

    def __init__(self, detail: str) -> None:
        super().__init__(f"CORPORATE_ACTION_UNSUPPORTED:{detail}")
        self.code = "CORPORATE_ACTION_UNSUPPORTED"


@dataclass(frozen=True)
class LifecycleEvent:
    """Fully resolved economic terms for one action on one source asset.

    ``sequence`` orders multiple actions on one effective date. Cash amounts
    and ratios are per old share. Fees must be explicitly confirmed zero in
    this first model; nonzero or unknown fees remain unsupported.
    """

    event_id: str
    kind: ActionKind
    source: AssetKey
    source_symbol: str
    effective_date: date
    process_date: date
    observed_at: datetime
    page_hashes: tuple[str, ...]
    normalization_version: int
    sequence: int
    fee_per_old_share: Decimal
    target: AssetKey | None = None
    target_symbol: str | None = None
    share_ratio: Decimal | None = None
    fraction_policy: FractionPolicy | None = None
    cash_per_old_share: Decimal | None = None
    cash_pay_date: date | None = None

    def __post_init__(self) -> None:
        if not self.event_id or self.event_id != self.event_id.strip():
            raise LifecycleError("event id missing")
        if self.kind not in (
            "rename",
            "split",
            "worthless_removal",
            "cash_merger",
            "stock_merger",
            "mixed_merger",
        ):
            raise LifecycleError("event kind")
        if not _SYMBOL.fullmatch(self.source_symbol):
            raise LifecycleError("source symbol")
        if self.target_symbol is not None and not _SYMBOL.fullmatch(self.target_symbol):
            raise LifecycleError("target symbol")
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise LifecycleError("observation time")
        if not self.page_hashes or any(not _SHA256.fullmatch(value) for value in self.page_hashes):
            raise LifecycleError("source page hashes")
        if self.normalization_version < 1 or self.sequence < 0:
            raise LifecycleError("normalization version or sequence")
        if not isinstance(self.fee_per_old_share, Decimal) or not self.fee_per_old_share.is_finite():
            raise LifecycleError("fee terms missing or invalid")
        if self.fee_per_old_share != 0:
            raise LifecycleError("unmodelled fee")
        if self.fraction_policy is not None and self.fraction_policy not in ("retain", "reject_noninteger"):
            raise LifecycleError("fraction policy")

        has_target = self.target is not None and self.target_symbol is not None
        has_ratio = self.share_ratio is not None and self.fraction_policy is not None
        has_cash = self.cash_per_old_share is not None and self.cash_pay_date is not None
        if self.kind == "rename":
            if not self.target_symbol or self.target_symbol == self.source_symbol:
                raise LifecycleError("rename target symbol")
            if self.target is not None or has_ratio or has_cash:
                raise LifecycleError("rename terms")
        elif self.kind == "split":
            if self.target is not None or has_cash or not has_ratio:
                raise LifecycleError("split terms")
        elif self.kind == "worthless_removal":
            if self.target is not None or self.target_symbol is not None or has_ratio:
                raise LifecycleError("worthless removal terms")
            if self.cash_per_old_share != 0 or self.cash_pay_date is not None:
                raise LifecycleError("worthless payoff must be explicit zero")
        elif self.kind == "cash_merger":
            if self.target is not None or self.target_symbol is not None or has_ratio or not has_cash:
                raise LifecycleError("cash merger terms")
        elif self.kind == "stock_merger":
            if not has_target or not has_ratio or has_cash:
                raise LifecycleError("stock merger terms")
        else:  # mixed_merger
            if not has_target or not has_ratio or not has_cash:
                raise LifecycleError("mixed merger terms")

        if self.kind in ("stock_merger", "mixed_merger") and self.target == self.source:
            raise LifecycleError("successor must have a distinct asset key")
        if self.target is not None and self.target.provider != self.source.provider:
            raise LifecycleError("successor provider identity is not reconciled")
        if self.share_ratio is not None and (
            not isinstance(self.share_ratio, Decimal)
            or not self.share_ratio.is_finite()
            or self.share_ratio <= 0
        ):
            raise LifecycleError("share ratio")
        if self.cash_per_old_share is not None and (
            not isinstance(self.cash_per_old_share, Decimal)
            or not self.cash_per_old_share.is_finite()
            or self.cash_per_old_share < 0
        ):
            raise LifecycleError("cash consideration")
        if self.cash_pay_date is not None and self.cash_pay_date < self.effective_date:
            raise LifecycleError("cash payment before effective date")
        if (self.share_ratio is None) != (self.fraction_policy is None):
            raise LifecycleError("fraction policy missing or extraneous")
        if self.kind != "worthless_removal" and (
            (self.cash_per_old_share is None) != (self.cash_pay_date is None)
        ):
            raise LifecycleError("cash payment terms incomplete")
