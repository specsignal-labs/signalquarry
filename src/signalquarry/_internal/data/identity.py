# SPDX-License-Identifier: Apache-2.0
"""Provider-neutral, point-in-time asset identity and dated ticker aliases.

This is a pure normalization model. A current asset list cannot reconstruct
historical aliases; callers must supply frozen observations with provenance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROVIDER = re.compile(r"^[a-z][a-z0-9_-]*$")
_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.]{0,15}$")


class IdentityError(ValueError):
    """The observed alias history cannot resolve a unique asset identity."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}:{detail}")
        self.code = code


@dataclass(frozen=True, order=True)
class AssetKey:
    """An asset's stable key within a provider namespace; never a ticker."""

    provider: str
    asset_id: str

    def __post_init__(self) -> None:
        if (
            not _PROVIDER.fullmatch(self.provider)
            or not self.asset_id
            or self.asset_id != self.asset_id.strip()
        ):
            raise IdentityError("PROVIDER_RESPONSE_INVALID", "asset key")


@dataclass(frozen=True)
class AliasObservation:
    """One immutable observation of a dated alias record.

    ``record_id`` identifies the record across revisions. ``symbol=None`` is a
    deletion; its latest observation removes that record from the as-of view.
    ``effective_until`` is exclusive. Process and observation times may follow
    the effective date, and are retained separately.
    """

    record_id: str
    asset: AssetKey
    symbol: str | None
    effective_from: date
    effective_until: date | None
    process_date: date
    observed_at: datetime
    page_hashes: tuple[str, ...]
    normalization_version: int

    def __post_init__(self) -> None:
        if not self.record_id or self.record_id != self.record_id.strip():
            raise IdentityError("PROVIDER_RESPONSE_INVALID", "alias record id")
        if self.symbol is not None and not _SYMBOL.fullmatch(self.symbol):
            raise IdentityError("PROVIDER_RESPONSE_INVALID", "alias symbol")
        if self.effective_until is not None and self.effective_until <= self.effective_from:
            raise IdentityError("PROVIDER_RESPONSE_INVALID", "alias interval")
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise IdentityError("PROVIDER_RESPONSE_INVALID", "alias observation time")
        if not self.page_hashes or any(not _SHA256.fullmatch(value) for value in self.page_hashes):
            raise IdentityError("PROVIDER_RESPONSE_INVALID", "alias page hashes")
        if self.normalization_version < 1:
            raise IdentityError("PROVIDER_RESPONSE_INVALID", "alias normalization version")


@dataclass(frozen=True)
class AliasBook:
    """A frozen set of observations, queried by session and information cutoff."""

    observations: tuple[AliasObservation, ...]

    def as_of(self, session: date, known_at: datetime) -> dict[str, AssetKey]:
        if known_at.tzinfo is None or known_at.utcoffset() is None:
            raise IdentityError("PROVIDER_RESPONSE_INVALID", "alias cutoff time")
        cutoff = known_at.astimezone(UTC)
        latest: dict[tuple[str, str], AliasObservation] = {}
        for item in self.observations:
            if item.observed_at.astimezone(UTC) > cutoff:
                continue
            record_key = (item.asset.provider, item.record_id)
            previous = latest.get(record_key)
            if previous is not None and previous.asset != item.asset:
                raise IdentityError("CORPORATE_ACTION_UNSUPPORTED", "alias record changed asset")
            if previous is None or previous.observed_at.astimezone(UTC) < item.observed_at.astimezone(UTC):
                latest[record_key] = item
            elif (
                previous.observed_at.astimezone(UTC) == item.observed_at.astimezone(UTC) and previous != item
            ):
                raise IdentityError("PROVIDER_RESPONSE_INVALID", "conflicting alias observations")

        symbols: dict[str, AssetKey] = {}
        assets: dict[AssetKey, str] = {}
        for item in sorted(latest.values(), key=lambda entry: entry.record_id):
            if item.symbol is None or session < item.effective_from:
                continue
            if item.effective_until is not None and session >= item.effective_until:
                continue
            old_asset = symbols.get(item.symbol)
            old_symbol = assets.get(item.asset)
            if old_asset is not None and old_asset != item.asset:
                raise IdentityError("CORPORATE_ACTION_UNSUPPORTED", f"ambiguous symbol {item.symbol}")
            if old_symbol is not None and old_symbol != item.symbol:
                raise IdentityError("CORPORATE_ACTION_UNSUPPORTED", f"ambiguous asset {item.asset.provider}")
            symbols[item.symbol] = item.asset
            assets[item.asset] = item.symbol
        return symbols

    def resolve_symbol(self, symbol: str, session: date, known_at: datetime) -> AssetKey | None:
        """Return the exact asset, or None; never infer continuity from a ticker."""
        return self.as_of(session, known_at).get(symbol)

    def resolve_asset(self, asset: AssetKey, session: date, known_at: datetime) -> str | None:
        """Return the dated display symbol for an exact asset key."""
        return next((s for s, key in self.as_of(session, known_at).items() if key == asset), None)
