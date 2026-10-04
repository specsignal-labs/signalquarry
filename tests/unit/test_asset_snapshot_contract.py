# SPDX-License-Identifier: Apache-2.0
"""Exact failure detail, identity and storage behaviour of the asset snapshot contract."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from signalquarry._internal.canonical import hash_without
from signalquarry._internal.data.library import LibraryError
from signalquarry._internal.data.universe import (
    ASSETS_PARAMS,
    ASSETS_PATH,
    AssetPage,
    AssetSnapshotStore,
    assets_from_page,
    make_asset_manifest,
    verify_asset_manifest,
)

OBSERVED = datetime(2026, 9, 28, 12, 30, 15, 123456, tzinfo=UTC)
ID_A = "00000000-0000-4000-8000-00000000000a"
ID_B = "00000000-0000-4000-8000-00000000000b"
ID_C = "00000000-0000-4000-8000-00000000000c"


def _row(asset_id: str = ID_A, **changes: Any) -> dict[str, Any]:
    row = {
        "id": asset_id,
        "class": "us_equity",
        "symbol": "AAA",
        "exchange": "NYSE",
        "status": "active",
        "tradable": True,
    }
    row.update(changes)
    return row


def _page(rows: Any) -> AssetPage:
    body = json.dumps(rows, sort_keys=True).encode()
    return AssetPage(ASSETS_PATH, dict(ASSETS_PARAMS), body, hashlib.sha256(body).hexdigest())


def _raw_page(body: bytes) -> AssetPage:
    return AssetPage(ASSETS_PATH, dict(ASSETS_PARAMS), body, hashlib.sha256(body).hexdigest())


def _fail(code: str, detail: str, call: Callable[[], object]) -> None:
    with pytest.raises(LibraryError, match=f"^{code}:{detail}$"):
        call()


def test_assets_are_returned_sorted_by_canonical_id_and_ticker_reuse_is_allowed() -> None:
    upper = ID_C.upper()
    page = _page([_row(upper, symbol="DUP"), _row(ID_A, symbol="DUP", status="inactive", tradable=False)])
    assets = assets_from_page(page)
    assert [item.asset_id for item in assets] == [ID_A, ID_C]
    assert [item.symbol for item in assets] == ["DUP", "DUP"]
    assert [item.status for item in assets] == ["inactive", "active"]
    assert [item.tradable for item in assets] == [False, True]
    assert assets[0].exchange == "NYSE"


def test_the_request_and_page_hash_must_match() -> None:
    page = _page([_row()])
    _fail(
        "PROVIDER_RESPONSE_INVALID",
        "asset request",
        lambda: assets_from_page(replace(page, endpoint="/v2/other")),
    )
    _fail("PROVIDER_RESPONSE_INVALID", "asset request", lambda: assets_from_page(replace(page, params={})))
    _fail(
        "PROVIDER_RESPONSE_INVALID",
        "asset request",
        lambda: assets_from_page(replace(page, params={"asset_class": "us_equity", "status": "active"})),
    )
    _fail("DATA_PAGE_CORRUPT", "asset page hash", lambda: assets_from_page(replace(page, sha256="0" * 64)))
    _fail(
        "DATA_PAGE_CORRUPT", "asset page hash", lambda: assets_from_page(replace(page, body=page.body + b" "))
    )


def test_the_response_must_be_a_nonempty_json_list_of_objects() -> None:
    _fail("PROVIDER_RESPONSE_INVALID", "asset JSON", lambda: assets_from_page(_raw_page(b"{not json")))
    _fail("PROVIDER_RESPONSE_INVALID", "asset JSON", lambda: assets_from_page(_raw_page(b"\xff\xfe")))
    _fail("PROVIDER_RESPONSE_INVALID", "asset list", lambda: assets_from_page(_page([])))
    _fail("PROVIDER_RESPONSE_INVALID", "asset list", lambda: assets_from_page(_page({"id": ID_A})))
    _fail("PROVIDER_RESPONSE_INVALID", "asset row", lambda: assets_from_page(_page([_row(), "x"])))


@pytest.mark.parametrize(
    "row",
    [
        {key: value for key, value in _row().items() if key != "id"},
        {key: value for key, value in _row().items() if key != "symbol"},
        {key: value for key, value in _row().items() if key != "exchange"},
        {key: value for key, value in _row().items() if key != "status"},
        {key: value for key, value in _row().items() if key != "class"},
        {key: value for key, value in _row().items() if key != "tradable"},
        _row(id=5),
        _row(id="not-a-uuid"),
        _row(symbol=5),
        _row(symbol=""),
        _row(symbol=" AAA"),
        _row(symbol="AAA "),
        _row(exchange=""),
        _row(exchange=None),
        _row(status=""),
        _row(status=3),
        _row(**{"class": "crypto"}),
        _row(tradable=1),
        _row(tradable="true"),
    ],
)
def test_each_malformed_asset_field_is_rejected(row: dict[str, Any]) -> None:
    _fail("PROVIDER_RESPONSE_INVALID", "asset fields", lambda: assets_from_page(_page([row])))


def test_duplicate_asset_ids_are_rejected_even_when_cased_differently() -> None:
    _fail(
        "PROVIDER_RESPONSE_INVALID", "duplicate asset id", lambda: assets_from_page(_page([_row(), _row()]))
    )
    _fail(
        "PROVIDER_RESPONSE_INVALID",
        "duplicate asset id",
        lambda: assets_from_page(_page([_row(ID_A), _row(ID_A.upper())])),
    )


def _rows() -> list[dict[str, Any]]:
    return [
        _row(ID_A, status="active"),
        _row(ID_B, status="inactive", tradable=False, symbol="BBB"),
        _row(ID_C, status="delisted", tradable=False, symbol="CCC"),
        _row("00000000-0000-4000-8000-00000000000d", status="active", symbol="DDD"),
    ]


def test_the_manifest_counts_and_identifies_the_observation() -> None:
    page = _page(_rows())
    manifest = make_asset_manifest(page, OBSERVED)
    assert manifest["schema"] == "signalquarry.asset-snapshot/v1"
    assert manifest["provider"] == "alpaca"
    assert manifest["snapshot_id"] == f"alpaca-assets-20260928T123015123456Z-{page.sha256[:12]}"
    assert manifest["observed_at"] == "2026-09-28T12:30:15.123456Z"
    assert manifest["page"] == {"endpoint": ASSETS_PATH, "params": ASSETS_PARAMS, "sha256": page.sha256}
    assert (manifest["asset_count"], manifest["active_count"], manifest["inactive_count"]) == (4, 2, 1)
    assert manifest["other_status_count"] == 1
    assert manifest["redistributable"] is False
    assert manifest["records_hash"].startswith("sha256:")
    assert manifest["manifest_hash"] == hash_without(manifest, "manifest_hash")
    assert verify_asset_manifest(manifest, page) == assets_from_page(page)
    # the same observation in another zone is the same snapshot
    zone = timezone(timedelta(hours=-7))
    assert make_asset_manifest(page, OBSERVED.astimezone(zone)) == manifest
    assert (
        make_asset_manifest(page, OBSERVED + timedelta(microseconds=1))["snapshot_id"]
        != manifest["snapshot_id"]
    )
    changed = _page([*_rows()[:3], _row("00000000-0000-4000-8000-00000000000d", symbol="EEE")])
    assert make_asset_manifest(changed, OBSERVED)["records_hash"] != manifest["records_hash"]
    _fail(
        "DATA_MANIFEST_INVALID", "observation time", lambda: make_asset_manifest(page, datetime(2026, 9, 28))
    )


def _rehash(manifest: dict[str, Any]) -> dict[str, Any]:
    manifest["manifest_hash"] = hash_without(manifest, "manifest_hash")
    return manifest


@pytest.mark.parametrize(
    "edit",
    [
        lambda m: m.update(schema="other"),
        lambda m: m.update(provider="other"),
        lambda m: m.update(page={**m["page"], "sha256": "0" * 64}),
        lambda m: m.update(page={**m["page"], "endpoint": "/v2/other"}),
        lambda m: m.update(page={**m["page"], "params": {}}),
        lambda m: m.update(manifest_hash="sha256:" + "0" * 64),
        lambda m: m.pop("manifest_hash"),
    ],
)
def test_a_manifest_that_does_not_describe_this_page_is_refused(
    edit: Callable[[dict[str, Any]], None],
) -> None:
    page = _page(_rows())
    manifest = make_asset_manifest(page, OBSERVED)
    edit(manifest)
    _fail("DATA_MANIFEST_INVALID", "asset snapshot", lambda: verify_asset_manifest(manifest, page))


@pytest.mark.parametrize(
    "edit",
    [
        lambda m: m.update(observed_at="2026-09-28T12:30:15.123456"),
        lambda m: m.update(observed_at="2026-09-28T14:30:15.123456+02:00"),
        lambda m: m.update(observed_at="nonsense"),
        lambda m: m.update(observed_at=5),
        lambda m: m.pop("observed_at"),
    ],
)
def test_the_observation_time_must_be_utc(edit: Callable[[dict[str, Any]], None]) -> None:
    page = _page(_rows())
    manifest = make_asset_manifest(page, OBSERVED)
    edit(manifest)
    _rehash(manifest)
    _fail("DATA_MANIFEST_INVALID", "asset observation time", lambda: verify_asset_manifest(manifest, page))


@pytest.mark.parametrize(
    "edit",
    [
        lambda m: m.update(snapshot_id="alpaca-assets-other"),
        lambda m: m.update(asset_count=m["asset_count"] + 1),
        lambda m: m.update(active_count=m["active_count"] + 1),
        lambda m: m.update(inactive_count=m["inactive_count"] + 1),
        lambda m: m.update(other_status_count=m["other_status_count"] + 1),
        lambda m: m.update(records_hash="sha256:" + "0" * 64),
        lambda m: m.update(redistributable=True),
        lambda m: m.update(redistributable=None),
        lambda m: m.pop("redistributable"),
    ],
)
def test_contents_that_disagree_with_the_page_are_refused(edit: Callable[[dict[str, Any]], None]) -> None:
    page = _page(_rows())
    manifest = make_asset_manifest(page, OBSERVED)
    edit(manifest)
    _rehash(manifest)
    _fail("DATA_MANIFEST_INVALID", "asset snapshot contents", lambda: verify_asset_manifest(manifest, page))


def _store(tmp_path: Path) -> AssetSnapshotStore:
    return AssetSnapshotStore(tmp_path / "cache", tmp_path / "manifests")


@pytest.mark.parametrize("value", [5, None, "", "A" * 64, "a" * 63, "a" * 65, "g" * 64, "../" + "a" * 61])
def test_page_paths_accept_only_lowercase_sha256_hex(tmp_path: Path, value: object) -> None:
    _fail("DATA_MANIFEST_INVALID", "asset page hash", lambda: _store(tmp_path).page_path(value))  # type: ignore[arg-type]
    digest = "ab" * 32
    assert _store(tmp_path).page_path(digest) == tmp_path / "cache" / "pages" / f"{digest}.json.gz"


def test_store_writes_the_gzip_page_and_the_sorted_manifest_once(tmp_path: Path) -> None:
    store = _store(tmp_path)
    page = _page(_rows())
    manifest = make_asset_manifest(page, OBSERVED)
    path = store.store(page, manifest)
    assert path == tmp_path / "manifests" / f"{manifest['snapshot_id']}.json"
    assert path.read_text(encoding="utf-8") == json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    cached = store.page_path(page.sha256)
    assert gzip.decompress(cached.read_bytes()) == page.body
    assert not list(cached.parent.glob("*.tmp"))
    first_bytes = cached.read_bytes()
    assert store.store(page, manifest) == path
    assert cached.read_bytes() == first_bytes
    assert store.load(manifest) == assets_from_page(page)


def test_store_refuses_an_unverifiable_manifest_before_writing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    page = _page(_rows())
    manifest = make_asset_manifest(page, OBSERVED)
    manifest["asset_count"] = 99
    _rehash(manifest)
    _fail("DATA_MANIFEST_INVALID", "asset snapshot contents", lambda: store.store(page, manifest))
    assert not (tmp_path / "cache").exists() and not (tmp_path / "manifests").exists()


def test_store_detects_a_corrupt_or_different_cached_page_and_a_manifest_collision(tmp_path: Path) -> None:
    store = _store(tmp_path)
    page = _page(_rows())
    manifest = make_asset_manifest(page, OBSERVED)
    cached = store.page_path(page.sha256)
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"not gzip")
    _fail("DATA_PAGE_CORRUPT", page.sha256, lambda: store.store(page, manifest))
    cached.write_bytes(gzip.compress(b"[]"))
    _fail("DATA_PAGE_CORRUPT", page.sha256, lambda: store.store(page, manifest))
    cached.unlink()
    path = store.store(page, manifest)
    path.write_text("{}\n", encoding="utf-8")
    _fail("DATA_MANIFEST_INVALID", "snapshot collision", lambda: store.store(page, manifest))


def test_load_reports_a_bad_reference_a_missing_page_and_a_corrupt_page(tmp_path: Path) -> None:
    store = _store(tmp_path)
    page = _page(_rows())
    manifest = make_asset_manifest(page, OBSERVED)
    for bad in ({}, {"page": None}, {"page": {"endpoint": "x"}}, None):
        _fail("DATA_MANIFEST_INVALID", "asset page reference", lambda bad=bad: store.load(bad))  # type: ignore[misc]
    _fail("DATA_PAGE_MISSING", "asset snapshot", lambda: store.load(manifest))
    cached = store.page_path(page.sha256)
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"garbage")
    _fail("DATA_PAGE_CORRUPT", "asset snapshot", lambda: store.load(manifest))
    cached.write_bytes(gzip.compress(page.body)[:-6])
    _fail("DATA_PAGE_CORRUPT", "asset snapshot", lambda: store.load(manifest))
    outside = {**manifest, "page": {**manifest["page"], "sha256": "../../x"}}
    _fail("DATA_MANIFEST_INVALID", "asset page hash", lambda: store.load(outside))


def test_manifest_listing_is_sorted_and_fails_closed_on_junk(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.manifests() == []
    store.manifest_dir.mkdir(parents=True)
    (store.manifest_dir / "b.json").write_text('{"n": 2}', encoding="utf-8")
    (store.manifest_dir / "a.json").write_text('{"n": 1}', encoding="utf-8")
    (store.manifest_dir / "ignored.txt").write_text("x", encoding="utf-8")
    assert store.manifests() == [{"n": 1}, {"n": 2}]
    (store.manifest_dir / "c.json").write_text("[1]", encoding="utf-8")
    _fail("DATA_MANIFEST_INVALID", "c.json", store.manifests)
    (store.manifest_dir / "c.json").write_text("{broken", encoding="utf-8")
    _fail("DATA_MANIFEST_INVALID", "c.json", store.manifests)


def _snapshots(store: AssetSnapshotStore, *moments: datetime, variant: bool = False) -> list[dict[str, Any]]:
    manifests = []
    for index, moment in enumerate(moments):
        rows = (
            _rows() if not (variant and index) else [*_rows(), _row("00000000-0000-4000-8000-0000000000e0")]
        )
        page = _page(rows)
        manifest = make_asset_manifest(page, moment)
        store.store(page, manifest)
        manifests.append(manifest)
    return manifests


def test_as_of_picks_the_latest_snapshot_captured_by_the_cutoff_and_within_the_age_limit(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    old, mid, late = (
        OBSERVED - timedelta(days=40),
        OBSERVED - timedelta(days=10),
        OBSERVED + timedelta(days=2),
    )
    manifests = _snapshots(store, old, mid, late)
    assert store.as_of(OBSERVED)["snapshot_id"] == manifests[1]["snapshot_id"]
    assert store.as_of(late)["snapshot_id"] == manifests[2]["snapshot_id"]
    assert store.as_of(late - timedelta(microseconds=1))["snapshot_id"] == manifests[1]["snapshot_id"]
    with pytest.raises(LibraryError, match="^UNIVERSE_ASOF_UNAVAILABLE$"):
        store.as_of(old - timedelta(days=1))


def test_as_of_age_limit_is_inclusive_and_configurable(tmp_path: Path) -> None:
    store = _store(tmp_path)
    (only,) = _snapshots(store, OBSERVED)
    assert store.as_of(OBSERVED + timedelta(days=31))["snapshot_id"] == only["snapshot_id"]
    with pytest.raises(LibraryError, match="^UNIVERSE_ASOF_UNAVAILABLE$"):
        store.as_of(OBSERVED + timedelta(days=31, microseconds=1))
    assert (
        store.as_of(OBSERVED + timedelta(days=3), max_age=timedelta(days=3))["snapshot_id"]
        == only["snapshot_id"]
    )
    with pytest.raises(LibraryError, match="^UNIVERSE_ASOF_UNAVAILABLE$"):
        store.as_of(OBSERVED + timedelta(days=3, microseconds=1), max_age=timedelta(days=3))
    with pytest.raises(LibraryError, match="^UNIVERSE_ASOF_UNAVAILABLE$"):
        store.as_of(OBSERVED - timedelta(microseconds=1))


def test_as_of_requires_an_aware_cutoff_and_a_positive_age_and_breaks_ties_by_snapshot_id(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _fail("DATA_MANIFEST_INVALID", "asset cutoff", lambda: store.as_of(datetime(2026, 9, 28)))
    _fail("DATA_MANIFEST_INVALID", "asset cutoff", lambda: store.as_of(OBSERVED, max_age=timedelta(0)))
    _fail(
        "DATA_MANIFEST_INVALID", "asset cutoff", lambda: store.as_of(OBSERVED, max_age=timedelta(seconds=-1))
    )
    manifests = _snapshots(store, OBSERVED, OBSERVED, variant=True)
    assert manifests[0]["snapshot_id"] != manifests[1]["snapshot_id"]
    expected = max(manifests, key=lambda item: item["snapshot_id"])
    assert store.as_of(OBSERVED)["snapshot_id"] == expected["snapshot_id"]
    assert (
        store.as_of(OBSERVED.astimezone(timezone(timedelta(hours=3))))["snapshot_id"]
        == expected["snapshot_id"]
    )


def test_as_of_verifies_each_snapshot_it_considers(tmp_path: Path) -> None:
    store = _store(tmp_path)
    manifests = _snapshots(store, OBSERVED - timedelta(days=1))
    store.page_path(manifests[0]["page"]["sha256"]).write_bytes(b"corrupt")
    _fail("DATA_PAGE_CORRUPT", "asset snapshot", lambda: store.as_of(OBSERVED))
