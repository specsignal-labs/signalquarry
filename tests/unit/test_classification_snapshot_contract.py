# SPDX-License-Identifier: Apache-2.0
"""Exact failure detail and boundaries of the security-classification snapshot contract."""

from __future__ import annotations

import copy
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pytest

from signalquarry._internal.data.library import LibraryError
from signalquarry._internal.data.universe_build import verify_classification_snapshot

from .test_universe_build import ASSET_IDS, KNOWN_AT, OBSERVED_AT, SYMBOLS, _inputs


def _verify(
    edit: Callable[[dict[str, Any]], None] | None = None,
    *,
    known_at: datetime = KNOWN_AT,
    max_age: timedelta | None = None,
) -> tuple[dict[str, dict[str, Any]], str]:
    assets, _, classification, _, _ = _inputs()
    snapshot = copy.deepcopy(classification)
    if edit is not None:
        edit(snapshot)
    options: dict[str, Any] = {} if max_age is None else {"max_age": max_age}
    return verify_classification_snapshot(snapshot, assets, known_at=known_at, **options)


def _stamp(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def _fail(edit: Callable[[dict[str, Any]], None], code: str, detail: str, **options: Any) -> None:
    with pytest.raises(LibraryError, match=f"^{code}:{detail}$"):
        _verify(edit, **options)


def test_a_valid_snapshot_returns_dated_records_and_a_content_hash() -> None:
    records, digest = _verify()
    assert set(records) == set(ASSET_IDS)
    assert records[ASSET_IDS[0]] == {"security_type": "common_stock", "listed_at": date(2020, 1, 1)}
    assert records[ASSET_IDS[3]] == {"security_type": "other", "listed_at": None}
    assert records[ASSET_IDS[4]]["listed_at"] == date(2026, 9, 20)
    assert digest.startswith("sha256:") and len(digest) == 71
    assert _verify()[1] == digest

    def changed(snapshot: dict[str, Any]) -> None:
        snapshot["source"] = "another-source"

    assert _verify(changed)[1] != digest


def test_the_cutoff_must_be_aware_and_the_maximum_age_positive() -> None:
    for kwargs in (
        {"known_at": datetime(2026, 9, 28, 21)},
        {"max_age": timedelta(0)},
        {"max_age": timedelta(seconds=-1)},
    ):
        with pytest.raises(LibraryError, match="^DATA_MANIFEST_INVALID:classification cutoff$"):
            _verify(**kwargs)
    assert _verify(max_age=timedelta(microseconds=1) + (KNOWN_AT - OBSERVED_AT))


def test_observation_may_equal_the_cutoff_and_be_exactly_max_age_old() -> None:
    detail = "classification snapshot outside cutoff"

    def observed(moment: datetime) -> Callable[[dict[str, Any]], None]:
        return lambda snapshot: snapshot.update(observed_at=_stamp(moment))

    assert _verify(observed(KNOWN_AT))
    _fail(observed(KNOWN_AT + timedelta(seconds=1)), "UNIVERSE_INPUT_UNAVAILABLE", detail)
    assert _verify(observed(KNOWN_AT - timedelta(days=31)))
    _fail(observed(KNOWN_AT - timedelta(days=31, seconds=1)), "UNIVERSE_INPUT_UNAVAILABLE", detail)
    assert _verify(observed(KNOWN_AT - timedelta(days=5)), max_age=timedelta(days=5))
    _fail(
        observed(KNOWN_AT - timedelta(days=5, seconds=1)),
        "UNIVERSE_INPUT_UNAVAILABLE",
        detail,
        max_age=timedelta(days=5),
    )


def test_the_cutoff_is_compared_in_utc() -> None:
    zone = timezone(timedelta(hours=9))
    assert _verify(known_at=KNOWN_AT.astimezone(zone))
    with pytest.raises(
        LibraryError, match="^UNIVERSE_INPUT_UNAVAILABLE:classification snapshot outside cutoff$"
    ):
        _verify(known_at=(OBSERVED_AT - timedelta(seconds=1)).astimezone(zone))


@pytest.mark.parametrize("value", [None, 5, "yesterday", "2026-09-28T20:00:00", "", "2026-09-28"])
def test_observed_at_must_be_an_aware_iso_timestamp(value: object) -> None:
    _fail(
        lambda snapshot: snapshot.update(observed_at=value), "UNIVERSE_CLASSIFICATION_INVALID", "observed_at"
    )


def test_observed_at_accepts_z_and_explicit_offsets() -> None:
    assert _verify(lambda snapshot: snapshot.update(observed_at="2026-09-28T20:00:00+00:00"))
    assert _verify(lambda snapshot: snapshot.update(observed_at="2026-09-28T22:00:00+02:00"))


def test_top_level_fields_and_header_are_exact() -> None:
    fields = "snapshot fields"
    _fail(lambda snapshot: snapshot.update(extra=1), "UNIVERSE_CLASSIFICATION_INVALID", fields)
    _fail(lambda snapshot: snapshot.pop("source"), "UNIVERSE_CLASSIFICATION_INVALID", fields)
    _fail(lambda snapshot: snapshot.pop("records"), "UNIVERSE_CLASSIFICATION_INVALID", fields)
    with pytest.raises(LibraryError, match="^UNIVERSE_CLASSIFICATION_INVALID:snapshot fields$"):
        verify_classification_snapshot([], _inputs()[0], known_at=KNOWN_AT)  # type: ignore[arg-type]
    header = "snapshot header"
    _fail(lambda snapshot: snapshot.update(schema="other"), "UNIVERSE_CLASSIFICATION_INVALID", header)
    _fail(lambda snapshot: snapshot.update(source=5), "UNIVERSE_CLASSIFICATION_INVALID", header)
    _fail(lambda snapshot: snapshot.update(source="   "), "UNIVERSE_CLASSIFICATION_INVALID", header)
    _fail(lambda snapshot: snapshot.update(records="x"), "UNIVERSE_CLASSIFICATION_INVALID", header)


def _record(index: int = 0) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return lambda snapshot: snapshot["records"][index]


@pytest.mark.parametrize(
    ("name", "edit", "detail"),
    [
        ("not-a-dict", lambda s: s["records"].__setitem__(0, "x"), "record fields"),
        ("extra-field", lambda s: s["records"][0].update(extra=1), "record fields"),
        ("missing-field", lambda s: s["records"][0].pop("listed_at"), "record fields"),
        ("both-keys", lambda s: s["records"][0].update(symbol="AAA"), "record fields"),
        ("key-not-text", lambda s: s["records"][0].update(asset_id=5), "record values"),
        ("type-not-text", lambda s: s["records"][0].update(security_type=5), "record values"),
        ("not-a-uuid", lambda s: s["records"][0].update(asset_id="nope"), "asset_id"),
        (
            "uppercase-uuid",
            lambda s: s["records"][0].update(asset_id="00000000-0000-4000-8000-00000000000A"),
            "asset_id",
        ),
        ("braced-uuid", lambda s: s["records"][0].update(asset_id="{" + ASSET_IDS[0] + "}"), "asset_id"),
        ("unknown-type", lambda s: s["records"][0].update(security_type="etf"), "record identity"),
        ("duplicate-id", lambda s: s["records"][1].update(asset_id=ASSET_IDS[0]), "record identity"),
        ("bad-date", lambda s: s["records"][0].update(listed_at="2020-1-1"), "listed_at"),
        ("date-with-time", lambda s: s["records"][0].update(listed_at="2020-01-01T00:00:00"), "listed_at"),
        ("date-not-text", lambda s: s["records"][0].update(listed_at=20200101), "listed_at"),
        (
            "common-without-date",
            lambda s: s["records"][0].update(listed_at=None),
            "common stock listing date",
        ),
    ],
)
def test_each_record_defect_reports_its_own_detail(
    name: str, edit: Callable[[dict[str, Any]], None], detail: str
) -> None:
    del name
    _fail(edit, "UNIVERSE_CLASSIFICATION_INVALID", detail)


def test_other_securities_may_have_no_listing_date_or_one() -> None:
    def with_date(snapshot: dict[str, Any]) -> None:
        snapshot["records"][3]["listed_at"] = "2019-05-06"

    records, _ = _verify(with_date)
    assert records[ASSET_IDS[3]]["listed_at"] == date(2019, 5, 6)


def test_coverage_must_be_complete_and_records_sorted() -> None:
    _fail(lambda s: s["records"].pop(), "UNIVERSE_CLASSIFICATION_INVALID", "incomplete asset coverage")

    def unknown_id(snapshot: dict[str, Any]) -> None:
        snapshot["records"][0]["asset_id"] = "00000000-0000-4000-8000-0000000000ff"

    _fail(unknown_id, "UNIVERSE_CLASSIFICATION_INVALID", "incomplete asset coverage")
    _fail(lambda s: s["records"].reverse(), "UNIVERSE_CLASSIFICATION_INVALID", "record order")


def _by_symbol(snapshot: dict[str, Any]) -> None:
    for record, symbol in zip(snapshot["records"], SYMBOLS, strict=True):
        record["symbol"] = symbol
        del record["asset_id"]
    snapshot["records"].sort(key=lambda item: item["symbol"])


def test_records_may_be_keyed_by_symbol_when_symbols_are_unambiguous() -> None:
    records, _ = _verify(_by_symbol)
    assert set(records) == set(ASSET_IDS)
    assert records[ASSET_IDS[2]]["listed_at"] == date(2020, 1, 1)


def test_symbol_keys_must_be_known_clean_unmixed_and_sorted() -> None:
    def unknown(snapshot: dict[str, Any]) -> None:
        _by_symbol(snapshot)
        snapshot["records"][0]["symbol"] = "AAA0"

    _fail(unknown, "UNIVERSE_CLASSIFICATION_INVALID", "unknown symbol")

    def padded(snapshot: dict[str, Any]) -> None:
        _by_symbol(snapshot)
        snapshot["records"][0]["symbol"] = " AAA"

    _fail(padded, "UNIVERSE_CLASSIFICATION_INVALID", "ambiguous symbol key")

    def empty(snapshot: dict[str, Any]) -> None:
        _by_symbol(snapshot)
        snapshot["records"][0]["symbol"] = ""

    _fail(empty, "UNIVERSE_CLASSIFICATION_INVALID", "ambiguous symbol key")

    def mixed(snapshot: dict[str, Any]) -> None:
        _by_symbol(snapshot)
        snapshot["records"][1]["asset_id"] = ASSET_IDS[1]
        del snapshot["records"][1]["symbol"]

    _fail(mixed, "UNIVERSE_CLASSIFICATION_INVALID", "mixed record keys")

    def unsorted(snapshot: dict[str, Any]) -> None:
        _by_symbol(snapshot)
        snapshot["records"].reverse()

    _fail(unsorted, "UNIVERSE_CLASSIFICATION_INVALID", "record order")


def test_symbol_keys_are_refused_when_two_assets_share_a_symbol() -> None:
    assets, _, classification, _, _ = _inputs()
    twin = type(assets[0])(**{**assets[0].__dict__, "asset_id": "00000000-0000-4000-8000-0000000000aa"})
    snapshot = copy.deepcopy(classification)
    _by_symbol(snapshot)
    with pytest.raises(LibraryError, match="^UNIVERSE_CLASSIFICATION_INVALID:ambiguous symbol key$"):
        verify_classification_snapshot(snapshot, (*assets, twin), known_at=KNOWN_AT)
