# SPDX-License-Identifier: Apache-2.0
"""Synthetic reference cases for dated asset aliases and observation cutoffs."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from signalquarry._internal.data.identity import AliasBook, AliasObservation, AssetKey, IdentityError

OLD = AssetKey("synthetic", "asset-old")
NEW = AssetKey("synthetic", "asset-new")
OTHER = AssetKey("synthetic", "asset-other")
PAGE = ("a" * 64,)


def observed(day: int) -> datetime:
    return datetime(2024, 1, day, 15, 0, tzinfo=UTC)


def alias(
    record_id: str,
    asset: AssetKey,
    symbol: str | None,
    start: date,
    end: date | None,
    *,
    seen: datetime | None = None,
) -> AliasObservation:
    return AliasObservation(
        record_id, asset, symbol, start, end, date(2024, 1, 12), seen or observed(12), PAGE, 1
    )


def test_rename_keeps_asset_identity_and_does_not_backfill_unknown_alias() -> None:
    book = AliasBook(
        (
            alias("old", OLD, "SYNA", date(2024, 1, 1), date(2024, 1, 10)),
            alias("new", OLD, "SYNB", date(2024, 1, 10), None, seen=observed(15)),
        )
    )
    assert book.resolve_symbol("SYNA", date(2024, 1, 9), observed(15)) == OLD
    assert book.resolve_symbol("SYNA", date(2024, 1, 10), observed(15)) is None
    assert book.resolve_symbol("SYNB", date(2024, 1, 10), observed(15)) == OLD
    assert book.resolve_asset(OLD, date(2024, 1, 10), observed(15)) == "SYNB"
    assert book.resolve_asset(OLD, date(2024, 1, 10), observed(14)) is None


def test_reused_ticker_is_a_different_asset_after_a_gap() -> None:
    book = AliasBook(
        (
            alias("first", OLD, "REUSE", date(2024, 1, 1), date(2024, 1, 8)),
            alias("second", NEW, "REUSE", date(2024, 1, 10), None),
        )
    )
    assert book.resolve_symbol("REUSE", date(2024, 1, 7), observed(12)) == OLD
    assert book.resolve_symbol("REUSE", date(2024, 1, 9), observed(12)) is None
    assert book.resolve_symbol("REUSE", date(2024, 1, 10), observed(12)) == NEW
    assert book.resolve_asset(OLD, date(2024, 1, 10), observed(12)) is None


def test_late_revision_changes_only_later_information_sets() -> None:
    first = alias("one", OLD, "SYNA", date(2024, 1, 1), None)
    revised = replace(
        first, effective_until=date(2024, 1, 8), observed_at=observed(20), page_hashes=("b" * 64,)
    )
    book = AliasBook((revised, first))  # input order cannot change the answer
    assert book.resolve_symbol("SYNA", date(2024, 1, 10), observed(19)) == OLD
    assert book.resolve_symbol("SYNA", date(2024, 1, 10), observed(20)) is None
    assert book.resolve_symbol("SYNA", date(2024, 1, 7), observed(20)) == OLD
    assert (
        book.resolve_symbol("SYNA", date(2024, 1, 10), observed(20).astimezone(timezone(timedelta(hours=-5))))
        is None
    )


def test_deletion_removes_a_record_only_after_it_is_observed() -> None:
    first = alias("one", OLD, "SYNA", date(2024, 1, 1), None)
    deleted = replace(first, symbol=None, observed_at=observed(20))
    book = AliasBook((first, deleted))
    assert book.resolve_symbol("SYNA", date(2024, 1, 5), observed(19)) == OLD
    assert book.resolve_symbol("SYNA", date(2024, 1, 5), observed(20)) is None


@pytest.mark.parametrize(
    "records",
    [
        (alias("a", OLD, "SAME", date(2024, 1, 1), None), alias("b", NEW, "SAME", date(2024, 1, 1), None)),
        (alias("a", OLD, "SYNA", date(2024, 1, 1), None), alias("b", OLD, "SYNB", date(2024, 1, 1), None)),
    ],
)
def test_overlapping_aliases_refuse_ambiguous_identity(records: tuple[AliasObservation, ...]) -> None:
    with pytest.raises(IdentityError) as info:
        AliasBook(records).as_of(date(2024, 1, 15), observed(20))
    assert info.value.code == "CORPORATE_ACTION_UNSUPPORTED"


def test_conflicting_revisions_at_one_observation_time_are_invalid() -> None:
    first = alias("one", OLD, "SYNA", date(2024, 1, 1), None)
    conflicting = replace(first, symbol="SYNB")
    with pytest.raises(IdentityError) as info:
        AliasBook((first, conflicting)).as_of(date(2024, 1, 15), observed(20))
    assert info.value.code == "PROVIDER_RESPONSE_INVALID"


def test_one_alias_record_cannot_silently_change_asset() -> None:
    first = alias("one", OLD, "SYNA", date(2024, 1, 1), None)
    conflicting = replace(first, asset=OTHER, observed_at=observed(20))
    with pytest.raises(IdentityError) as info:
        AliasBook((first, conflicting)).as_of(date(2024, 1, 15), observed(20))
    assert info.value.code == "CORPORATE_ACTION_UNSUPPORTED"


def test_record_ids_are_scoped_to_the_provider() -> None:
    other_provider = AssetKey("other", "asset-one")
    book = AliasBook(
        (
            alias("one", OLD, "SYNA", date(2024, 1, 1), None),
            alias("one", other_provider, "SYNB", date(2024, 1, 1), None),
        )
    )
    assert book.as_of(date(2024, 1, 15), observed(20)) == {"SYNA": OLD, "SYNB": other_provider}


@pytest.mark.parametrize(
    "change",
    [
        {"symbol": "bad symbol"},
        {"effective_until": date(2024, 1, 1)},
        {"observed_at": datetime(2024, 1, 12, 15, 0)},
        {"page_hashes": ()},
        {"normalization_version": 0},
    ],
)
def test_invalid_alias_observations_are_rejected(change: dict) -> None:
    first = alias("one", OLD, "SYNA", date(2024, 1, 1), None)
    with pytest.raises(IdentityError) as info:
        replace(first, **change)
    assert info.value.code == "PROVIDER_RESPONSE_INVALID"
