# SPDX-License-Identifier: Apache-2.0
"""Synthetic asset-keyed manifest vectors; no provider data or broker orders."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal

import numpy as np
import pytest

from signalquarry._internal.contracts.reason_codes import REASON_CODES
from signalquarry._internal.data.asset_dataset import AssetDatasetError, AssetDatasetV1, AssetSeries
from signalquarry._internal.data.identity import AliasBook, AliasObservation, AssetKey, IdentityError
from signalquarry._internal.data.lifecycle import LifecycleEvent

OLD = AssetKey("synthetic", "asset-old")
NEW = AssetKey("synthetic", "asset-new")
SESSIONS = (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4))
CUTOFFS = (
    datetime(2024, 1, 2, 13, 30, tzinfo=UTC),
    datetime(2024, 1, 3, 13, 30, tzinfo=UTC),
    datetime(2024, 1, 4, 13, 30, tzinfo=UTC),
)
OBSERVATION_CUTOFF = datetime(2024, 1, 4, 15, tzinfo=UTC)


def series(prices: tuple[int, ...], present: tuple[bool, ...]) -> AssetSeries:
    micro = {name: np.asarray(prices, dtype=np.int64) for name in ("open", "high", "low", "close")}
    return AssetSeries(micro, np.asarray([1000.0] * len(prices)), np.asarray(present, dtype=bool))


def alias(
    record_id: str,
    asset: AssetKey,
    symbol: str,
    effective_from: date,
    effective_until: date | None,
    observed_at: datetime,
) -> AliasObservation:
    return AliasObservation(
        record_id,
        asset,
        symbol,
        effective_from,
        effective_until,
        date(2024, 1, 1),
        observed_at,
        ("a" * 64,),
        1,
    )


def merger(*, observed_at: datetime | None = None, event_id: str = "synthetic-merger") -> LifecycleEvent:
    return LifecycleEvent(
        event_id=event_id,
        kind="stock_merger",
        source=OLD,
        source_symbol="SYNA",
        effective_date=SESSIONS[1],
        process_date=SESSIONS[0],
        observed_at=observed_at or datetime(2024, 1, 3, 14, tzinfo=UTC),
        page_hashes=("b" * 64,),
        normalization_version=1,
        sequence=0,
        fee_per_old_share=Decimal(0),
        target=NEW,
        target_symbol="SYNA",
        share_ratio=Decimal("0.5"),
        fraction_policy="retain",
    )


def dataset(
    *,
    events: tuple[LifecycleEvent, ...] | None = None,
    aliases: tuple[AliasObservation, ...] | None = None,
    observation_cutoff: datetime = OBSERVATION_CUTOFF,
    decision_cutoffs: tuple[datetime, ...] = CUTOFFS,
) -> AssetDatasetV1:
    return AssetDatasetV1(
        sessions=SESSIONS,
        series={
            OLD: series((100_000_000, 0, 0), (True, False, False)),
            NEW: series((0, 0, 250_000_000), (False, False, True)),
        },
        aliases=AliasBook(
            aliases
            or (
                alias("old-symbol", OLD, "SYNA", SESSIONS[0], SESSIONS[1], datetime(2023, 12, 1, tzinfo=UTC)),
                alias("new-symbol", NEW, "SYNA", SESSIONS[1], None, datetime(2024, 1, 3, 12, tzinfo=UTC)),
            )
        ),
        events=(merger(),) if events is None else events,
        decision_cutoffs=decision_cutoffs,
        observation_cutoff=observation_cutoff,
    )


def test_asset_series_returns_exact_raw_asset_marks_and_is_immutable() -> None:
    values = np.asarray([100_000_000, 0, 0], dtype=np.int64)
    item = AssetSeries(
        {name: values for name in ("open", "high", "low", "close")},
        np.asarray([100.0, 0.0, 0.0]),
        np.asarray([True, False, False]),
    )
    values[0] = 1
    assert item.price("close", 0) == Decimal("100")
    assert item.price("close", 1) is None
    with pytest.raises(ValueError):
        item.micro["close"][0] = 2


def test_manifest_resolves_reused_ticker_to_different_asset_keys_by_date() -> None:
    frozen = dataset()
    assert frozen.aliases.resolve_symbol("SYNA", SESSIONS[0], CUTOFFS[0]) == OLD
    assert frozen.aliases.resolve_symbol("SYNA", SESSIONS[1], CUTOFFS[1]) == NEW
    assert frozen.price(OLD, "close", 0) == Decimal("100")
    assert frozen.price(NEW, "close", 2) == Decimal("250")


def test_manifest_hash_covers_raw_marks_cutoffs_alias_revisions_and_event_provenance() -> None:
    original = dataset()
    reversed_order = dataset(aliases=tuple(reversed(original.aliases.observations)))
    assert original.identity() == reversed_order.identity()
    assert original.manifest()["identity"] == original.identity()
    assert original.manifest()["schema"] == "signalquarry.asset-dataset/v1"

    later_snapshot = dataset(observation_cutoff=datetime(2024, 1, 5, tzinfo=UTC))
    assert later_snapshot.identity() != original.identity()

    changed_event = replace(merger(), page_hashes=("c" * 64,))
    assert dataset(events=(changed_event,)).identity() != original.identity()


def test_late_event_remains_identified_as_later_than_the_decision_cutoff() -> None:
    frozen = dataset()
    event = frozen.events[0]
    assert event.effective_date == SESSIONS[1]
    assert event.observed_at > frozen.decision_cutoffs[1]
    assert event.observed_at <= frozen.observation_cutoff
    assert frozen.events_as_of(frozen.decision_cutoffs[1]) == ()
    assert frozen.events_as_of(frozen.observation_cutoff) == (event,)


def test_event_revisions_are_preserved_and_selected_as_of_their_observation_time() -> None:
    first = merger(observed_at=datetime(2024, 1, 3, 12, tzinfo=UTC))
    revision = replace(
        first,
        observed_at=datetime(2024, 1, 3, 14, tzinfo=UTC),
        share_ratio=Decimal("0.4"),
        page_hashes=("c" * 64,),
    )
    frozen = dataset(events=(revision, first))
    assert len(frozen.events) == 2
    assert frozen.events_as_of(CUTOFFS[1]) == (first,)
    assert frozen.events_as_of(OBSERVATION_CUTOFF) == (revision,)
    assert frozen.identity() != dataset(events=(first,)).identity()


def test_late_alias_revision_blocks_ticker_reuse_until_the_revision_is_known() -> None:
    first = alias("old-symbol", OLD, "SYNA", SESSIONS[0], None, datetime(2023, 12, 1, tzinfo=UTC))
    revision = replace(
        first,
        effective_until=SESSIONS[1],
        observed_at=datetime(2024, 1, 3, 14, tzinfo=UTC),
        page_hashes=("d" * 64,),
    )
    frozen = dataset(aliases=(first, revision, dataset().aliases.observations[1]))
    with pytest.raises(IdentityError, match="ambiguous symbol SYNA"):
        frozen.aliases_as_of(SESSIONS[1], CUTOFFS[1])
    assert frozen.aliases_as_of(SESSIONS[1], CUTOFFS[2]) == {"SYNA": NEW}


def test_asset_dataset_copies_series_mapping_and_rejects_mutation() -> None:
    source = {
        OLD: series((100_000_000, 0, 0), (True, False, False)),
        NEW: series((0, 0, 1), (False, False, True)),
    }
    frozen = replace(dataset(), series=source)
    source.pop(OLD)
    assert OLD in frozen.series
    with pytest.raises(TypeError):
        frozen.series[OLD] = source[NEW]


@pytest.mark.parametrize(
    "change",
    [
        {"observation_cutoff": datetime(2024, 1, 5)},
        {
            "decision_cutoffs": (
                CUTOFFS[0],
                datetime(2024, 1, 4, 16, tzinfo=UTC),
                datetime(2024, 1, 5, 13, 30, tzinfo=UTC),
            )
        },
        {"decision_cutoffs": (CUTOFFS[0], CUTOFFS[0], CUTOFFS[2])},
        {"manifest_version": 2},
        {"normalization_version": 0},
    ],
)
def test_invalid_manifest_versions_and_cutoffs_fail_closed(change: dict) -> None:
    with pytest.raises(AssetDatasetError):
        replace(dataset(), **change)


def test_manifest_rejects_late_observation_beyond_snapshot_cutoff() -> None:
    late_alias = alias(
        "new-symbol",
        NEW,
        "SYNA",
        SESSIONS[1],
        None,
        datetime(2024, 1, 5, tzinfo=UTC),
    )
    with pytest.raises(AssetDatasetError, match="alias observation exceeds manifest cutoff"):
        dataset(aliases=(dataset().aliases.observations[0], late_alias))


def test_manifest_rejects_duplicate_or_ambiguous_event_order() -> None:
    with pytest.raises(AssetDatasetError, match="duplicate event observation"):
        dataset(events=(merger(), merger()))
    other = merger(event_id="synthetic-second-merger")
    ambiguous = dataset(events=(other, merger()))
    with pytest.raises(AssetDatasetError, match="ambiguous effective-date sequence"):
        ambiguous.events_as_of(OBSERVATION_CUTOFF)


def test_manifest_rejects_events_without_raw_source_or_successor_marks() -> None:
    unknown = AssetKey("synthetic", "missing")
    with pytest.raises(AssetDatasetError, match="event references an asset without raw series"):
        dataset(events=(replace(merger(), target=unknown),))


def test_asset_dataset_error_code_is_registered() -> None:
    assert AssetDatasetError.code in REASON_CODES
