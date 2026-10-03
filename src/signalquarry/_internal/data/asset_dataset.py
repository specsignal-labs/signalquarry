# SPDX-License-Identifier: Apache-2.0
"""Versioned raw-price data keyed by stable assets and dated alias evidence.

This manifest is separate from the legacy symbol-keyed ``Dataset`` so adding
lifecycle history cannot change existing dataset identities or backtest results.
It records observation cutoffs and provenance; it does not fetch or infer them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from types import MappingProxyType
from typing import Any

import numpy as np

from signalquarry._internal.canonical import canonical_hash
from signalquarry._internal.data.dataset import FIELDS
from signalquarry._internal.data.identity import AliasBook, AliasObservation, AssetKey
from signalquarry._internal.data.lifecycle import LifecycleEvent

ASSET_DATASET_SCHEMA = "signalquarry.asset-dataset/v1"


class AssetDatasetError(ValueError):
    """An asset-keyed dataset is incomplete, inconsistent or unsupported."""

    code = "ASSET_DATASET_INVALID"

    def __init__(self, detail: str) -> None:
        super().__init__(f"{self.code}:{detail}")


@dataclass(frozen=True)
class AssetSeries:
    """Raw, unadjusted daily bars for one stable asset identity."""

    micro: Mapping[str, np.ndarray]
    volume: np.ndarray
    present: np.ndarray

    def __post_init__(self) -> None:
        if set(self.micro) != set(FIELDS):
            raise AssetDatasetError("series fields")
        copied: dict[str, np.ndarray] = {}
        for name in FIELDS:
            values = np.asarray(self.micro[name])
            if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
                raise AssetDatasetError(f"invalid raw {name} array")
            if values.size and (
                values.min() < 0
                or values.max() > np.iinfo(np.int64).max
                or values.min() < np.iinfo(np.int64).min
            ):
                raise AssetDatasetError(f"raw {name} outside int64")
            array = values.astype(np.int64, copy=True)
            array.setflags(write=False)
            copied[name] = array

        volume = np.asarray(self.volume, dtype=np.float64)
        present = np.asarray(self.present)
        if volume.ndim != 1 or not np.isfinite(volume).all() or (volume < 0).any():
            raise AssetDatasetError("invalid volume array")
        if present.ndim != 1 or present.dtype != np.dtype(bool):
            raise AssetDatasetError("invalid present array")
        if any(len(copied[name]) != len(volume) or len(present) != len(volume) for name in FIELDS):
            raise AssetDatasetError("series array lengths")
        if any((copied[name][present] <= 0).any() for name in FIELDS):
            raise AssetDatasetError("present raw prices must be positive")

        volume = volume.copy()
        present = present.copy()
        volume.setflags(write=False)
        present.setflags(write=False)
        object.__setattr__(self, "micro", MappingProxyType(copied))
        object.__setattr__(self, "volume", volume)
        object.__setattr__(self, "present", present)

    def price(self, field_name: str, index: int) -> Decimal | None:
        """Return the raw, unadjusted mark, or ``None`` when no bar is present."""
        if field_name not in FIELDS:
            raise KeyError(f"DATASET_FIELD_UNKNOWN:{field_name}")
        if not self.present[index]:
            return None
        return Decimal(int(self.micro[field_name][index])).scaleb(-6)


@dataclass(frozen=True)
class AssetDatasetV1:
    """Immutable asset-keyed bars plus the frozen alias and lifecycle evidence.

    ``decision_cutoffs[i]`` is the information cutoff for a decision made on
    ``sessions[i]``. ``observation_cutoff`` is the latest time represented by
    this frozen manifest. Every alias and event retains its own first-observed
    timestamp and page hashes so a replay can reject look-ahead or late terms.
    """

    sessions: tuple[date, ...]
    series: Mapping[AssetKey, AssetSeries]
    aliases: AliasBook
    events: tuple[LifecycleEvent, ...]
    decision_cutoffs: tuple[datetime, ...]
    observation_cutoff: datetime
    source: str = "synthetic"
    normalization_version: int = 1
    manifest_version: int = 1

    def __post_init__(self) -> None:
        sessions = tuple(self.sessions)
        if not sessions or list(sessions) != sorted(set(sessions)):
            raise AssetDatasetError("sessions must be strictly increasing")
        if self.manifest_version != 1 or self.normalization_version < 1:
            raise AssetDatasetError("unsupported version")
        if not self.source or self.source != self.source.strip():
            raise AssetDatasetError("source")
        if not isinstance(self.aliases, AliasBook):
            raise AssetDatasetError("alias book")
        if self.observation_cutoff.tzinfo is None or self.observation_cutoff.utcoffset() is None:
            raise AssetDatasetError("observation cutoff must be timezone-aware")
        observed_until = self.observation_cutoff.astimezone(UTC)
        cutoffs = tuple(self.decision_cutoffs)
        if len(cutoffs) != len(sessions):
            raise AssetDatasetError("one decision cutoff is required per session")
        normalized_cutoffs: list[datetime] = []
        for cutoff in cutoffs:
            if cutoff.tzinfo is None or cutoff.utcoffset() is None:
                raise AssetDatasetError("decision cutoff must be timezone-aware")
            normalized = cutoff.astimezone(UTC)
            if normalized > observed_until:
                raise AssetDatasetError("decision cutoff exceeds observation cutoff")
            if normalized_cutoffs and normalized <= normalized_cutoffs[-1]:
                raise AssetDatasetError("decision cutoffs must be strictly increasing")
            normalized_cutoffs.append(normalized)

        series = dict(self.series)
        if not series or any(not isinstance(asset, AssetKey) for asset in series):
            raise AssetDatasetError("asset series keys")
        for asset, item in series.items():
            if not isinstance(item, AssetSeries) or len(item.present) != len(sessions):
                raise AssetDatasetError(f"series length for {asset.provider}:{asset.asset_id}")

        records: dict[tuple[str, str], AliasObservation] = {}
        for observation in self.aliases.observations:
            if observation.asset not in series:
                raise AssetDatasetError("alias references an asset without raw series")
            if observation.observed_at.astimezone(UTC) > observed_until:
                raise AssetDatasetError("alias observation exceeds manifest cutoff")
            key = (observation.asset.provider, observation.record_id)
            previous = records.get(key)
            if previous is not None:
                if previous.asset != observation.asset:
                    raise AssetDatasetError("alias record changed asset")
                before = previous.observed_at.astimezone(UTC)
                after = observation.observed_at.astimezone(UTC)
                if before == after and previous != observation:
                    raise AssetDatasetError("conflicting alias observations at one cutoff")
            if previous is None or previous.observed_at.astimezone(UTC) < observation.observed_at.astimezone(
                UTC
            ):
                records[key] = observation

        events = tuple(self.events)
        event_versions: dict[tuple[str, str, datetime], LifecycleEvent] = {}
        for event in events:
            if event.source not in series or (event.target is not None and event.target not in series):
                raise AssetDatasetError("event references an asset without raw series")
            if event.observed_at.astimezone(UTC) > observed_until:
                raise AssetDatasetError("event observation exceeds manifest cutoff")
            version_key = (event.source.provider, event.event_id, event.observed_at.astimezone(UTC))
            previous = event_versions.get(version_key)
            if previous is not None:
                if previous != event:
                    raise AssetDatasetError("conflicting event versions at one cutoff")
                raise AssetDatasetError("duplicate event observation")
            event_versions[version_key] = event

        object.__setattr__(self, "sessions", sessions)
        object.__setattr__(self, "series", MappingProxyType(series))
        object.__setattr__(
            self,
            "events",
            tuple(
                sorted(
                    events,
                    key=lambda item: (
                        item.source.provider,
                        item.event_id,
                        item.observed_at.astimezone(UTC),
                        item.effective_date,
                        item.sequence,
                    ),
                )
            ),
        )
        object.__setattr__(self, "decision_cutoffs", tuple(normalized_cutoffs))
        object.__setattr__(self, "observation_cutoff", observed_until)

    def price(self, asset: AssetKey, field_name: str, index: int) -> Decimal | None:
        """Return a raw mark for the exact asset; never look up by ticker."""
        try:
            item = self.series[asset]
        except KeyError:
            raise AssetDatasetError(f"unknown asset {asset.provider}:{asset.asset_id}") from None
        return item.price(field_name, index)

    def aliases_as_of(self, session: date, known_at: datetime) -> dict[str, AssetKey]:
        """Resolve aliases using only observations present at the requested cutoff."""
        return self.aliases.as_of(session, self._checked_cutoff(known_at))

    def events_as_of(self, known_at: datetime) -> tuple[LifecycleEvent, ...]:
        """Return the latest immutable version of each event known at ``known_at``."""
        cutoff = self._checked_cutoff(known_at)
        latest: dict[tuple[str, str], LifecycleEvent] = {}
        for event in self.events:
            observed_at = event.observed_at.astimezone(UTC)
            if observed_at > cutoff:
                continue
            event_key = (event.source.provider, event.event_id)
            previous = latest.get(event_key)
            if previous is None or previous.observed_at.astimezone(UTC) < observed_at:
                latest[event_key] = event
            elif previous.observed_at.astimezone(UTC) == observed_at and previous != event:
                raise AssetDatasetError("conflicting event versions at one cutoff")

        ordered = tuple(
            sorted(latest.values(), key=lambda item: (item.effective_date, item.sequence, item.event_id))
        )
        keys: set[tuple[date, int]] = set()
        for event in ordered:
            key = (event.effective_date, event.sequence)
            if key in keys:
                raise AssetDatasetError("ambiguous effective-date sequence")
            keys.add(key)
        return ordered

    def _checked_cutoff(self, known_at: datetime) -> datetime:
        if known_at.tzinfo is None or known_at.utcoffset() is None:
            raise AssetDatasetError("query cutoff must be timezone-aware")
        cutoff = known_at.astimezone(UTC)
        if cutoff > self.observation_cutoff:
            raise AssetDatasetError("query cutoff exceeds manifest cutoff")
        return cutoff

    def manifest_document(self) -> dict[str, Any]:
        """Return the canonical, versioned content covered by ``identity()``."""
        return {
            "schema": ASSET_DATASET_SCHEMA,
            "manifest_version": self.manifest_version,
            "normalization_version": self.normalization_version,
            "source": self.source,
            "sessions": [session.isoformat() for session in self.sessions],
            "decision_cutoffs": list(self.decision_cutoffs),
            "observation_cutoff": self.observation_cutoff,
            "series": [
                {
                    "asset": _asset_document(asset),
                    "micro": {name: item.micro[name].tolist() for name in FIELDS},
                    "volume": item.volume.tolist(),
                    "present": item.present.tolist(),
                }
                for asset, item in sorted(self.series.items())
            ],
            "aliases": [
                _alias_document(item)
                for item in sorted(
                    self.aliases.observations,
                    key=lambda value: (
                        value.asset.provider,
                        value.record_id,
                        value.observed_at.astimezone(UTC),
                        value.effective_from,
                        value.symbol or "",
                    ),
                )
            ],
            "events": [_event_document(item) for item in self.events],
        }

    def identity(self) -> str:
        """Stable content identity, versioned separately from legacy Dataset hashes."""
        return canonical_hash(self.manifest_document())

    def manifest(self) -> dict[str, Any]:
        """Return the complete manifest document with its content identity."""
        document = self.manifest_document()
        return {**document, "identity": canonical_hash(document)}


def _asset_document(asset: AssetKey) -> dict[str, str]:
    return {"provider": asset.provider, "asset_id": asset.asset_id}


def _alias_document(item: AliasObservation) -> dict[str, Any]:
    return {
        "record_id": item.record_id,
        "asset": _asset_document(item.asset),
        "symbol": item.symbol,
        "effective_from": item.effective_from,
        "effective_until": item.effective_until,
        "process_date": item.process_date,
        "observed_at": item.observed_at,
        "page_hashes": item.page_hashes,
        "normalization_version": item.normalization_version,
    }


def _event_document(item: LifecycleEvent) -> dict[str, Any]:
    return {
        "event_id": item.event_id,
        "kind": item.kind,
        "source": _asset_document(item.source),
        "source_symbol": item.source_symbol,
        "effective_date": item.effective_date,
        "process_date": item.process_date,
        "observed_at": item.observed_at,
        "page_hashes": item.page_hashes,
        "normalization_version": item.normalization_version,
        "sequence": item.sequence,
        "fee_per_old_share": item.fee_per_old_share,
        "target": _asset_document(item.target) if item.target is not None else None,
        "target_symbol": item.target_symbol,
        "share_ratio": item.share_ratio,
        "fraction_policy": item.fraction_policy,
        "cash_per_old_share": item.cash_per_old_share,
        "cash_pay_date": item.cash_pay_date,
    }
