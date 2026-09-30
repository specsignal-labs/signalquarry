# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from signalquarry._internal.data.library import LibraryError
from signalquarry._internal.data.universe import (
    ASSETS_PARAMS,
    ASSETS_PATH,
    AssetPage,
    AssetSnapshotStore,
    assets_from_page,
    make_asset_manifest,
)
from signalquarry.api import init, universe_as_of, universe_snapshot, universe_verify
from signalquarry.cli.main import main

CAPTURED = datetime(2026, 9, 28, 12, tzinfo=UTC)
FIRST_ID = "00000000-0000-4000-8000-000000000001"
SECOND_ID = "00000000-0000-4000-8000-000000000002"


def _page(rows: list[dict]) -> AssetPage:
    body = json.dumps(rows, sort_keys=True).encode()
    return AssetPage(ASSETS_PATH, dict(ASSETS_PARAMS), body, hashlib.sha256(body).hexdigest())


def _row(asset_id: str, status: str, *, symbol: str = "SAME") -> dict:
    return {
        "id": asset_id,
        "class": "us_equity",
        "symbol": symbol,
        "exchange": "NYSE",
        "status": status,
        "tradable": status == "active",
        "name": "provider text stays in the local raw cache",
    }


@dataclass
class Source:
    page: AssetPage
    calls: int = 0

    def asset_snapshot(self) -> AssetPage:
        self.calls += 1
        return self.page


def test_snapshot_includes_inactive_identity_and_asof_is_prospective(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "lab"
    assert init(project, package="lab").status == "ok"
    monkeypatch.setenv("SIGNALQUARRY_CACHE_DIR", str(tmp_path / "cache"))
    page = _page([_row(FIRST_ID, "inactive"), _row(SECOND_ID, "active")])
    source = Source(page)
    result = universe_snapshot(project=project, source=source)
    assert result.status == "ok" and source.calls == 1
    assert result.data["asset_count"] == 2
    assert result.data["active_count"] == result.data["inactive_count"] == 1
    assert result.data["other_status_count"] == 0
    captured = datetime.fromisoformat(result.data["observed_at"].replace("Z", "+00:00"))
    assert abs((datetime.now(UTC) - captured).total_seconds()) < 10
    manifest_path = project / result.data["manifest"]
    manifest_text = manifest_path.read_text()
    assert "provider text" not in manifest_text and FIRST_ID not in manifest_text
    assert universe_verify(project=project).status == "ok"

    store = AssetSnapshotStore(tmp_path / "cache", project / "data/universe/snapshots")
    manifest = store.as_of(captured + timedelta(minutes=1))
    assets = store.load(manifest)
    assert len(assets) == 2 and assets[0].symbol == assets[1].symbol == "SAME"
    assert {item.status for item in assets} == {"active", "inactive"}
    assert universe_as_of(known_at=captured + timedelta(days=1), project=project).status == "ok"
    assert universe_as_of(known_at=captured - timedelta(seconds=1), project=project).reason_codes == [
        "UNIVERSE_ASOF_UNAVAILABLE"
    ]
    assert universe_as_of(known_at=captured + timedelta(days=32), project=project).reason_codes == [
        "UNIVERSE_ASOF_UNAVAILABLE"
    ]


def test_universe_cli_exposes_verification_and_timezone_aware_cutoff(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    project = tmp_path / "lab"
    assert init(project, package="lab").status == "ok"
    monkeypatch.setenv("SIGNALQUARRY_CACHE_DIR", str(tmp_path / "cache"))
    assert universe_snapshot(project=project, source=Source(_page([_row(FIRST_ID, "active")]))).status == "ok"
    assert main(["--json", "universe", "verify", "--project", str(project)]) == 0
    assert json.loads(capsys.readouterr().out)["data"]["snapshots"][0]["ok"] is True
    assert (
        main(
            [
                "--json",
                "universe",
                "as-of",
                "--known-at",
                "2020-01-01T00:00:00+00:00",
                "--project",
                str(project),
            ]
        )
        == 69
    )
    assert json.loads(capsys.readouterr().out)["reason_codes"] == ["UNIVERSE_ASOF_UNAVAILABLE"]


def test_snapshot_is_idempotent_and_detects_corrupted_cache(tmp_path: Path) -> None:
    page = _page([_row(FIRST_ID, "active")])
    store = AssetSnapshotStore(tmp_path / "cache", tmp_path / "manifests")
    manifest = make_asset_manifest(page, CAPTURED)
    path = store.store(page, manifest)
    assert store.store(page, manifest) == path
    cached = store.page_path(page.sha256)
    cached.write_bytes(b"corrupt")
    with pytest.raises(LibraryError, match="DATA_PAGE_CORRUPT"):
        store.store(page, manifest)
    with pytest.raises(LibraryError, match="DATA_PAGE_CORRUPT"):
        store.load(manifest)


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [_row(FIRST_ID, "active"), _row(FIRST_ID, "inactive")],
        [{**_row(FIRST_ID, "active"), "class": "crypto"}],
        [{**_row(FIRST_ID, "active"), "id": "not-a-uuid"}],
        [{**_row(FIRST_ID, "active"), "tradable": "true"}],
    ],
)
def test_invalid_or_incomplete_asset_list_fails_closed(rows: list[dict]) -> None:
    with pytest.raises(LibraryError, match="PROVIDER_RESPONSE_INVALID"):
        assets_from_page(_page(rows))


def test_snapshot_rejects_filtered_status_and_naive_time() -> None:
    page = _page([_row(FIRST_ID, "active")])
    filtered = AssetPage(
        page.endpoint, {"asset_class": "us_equity", "status": "active"}, page.body, page.sha256
    )
    with pytest.raises(LibraryError, match="PROVIDER_RESPONSE_INVALID"):
        assets_from_page(filtered)
    with pytest.raises(LibraryError, match="DATA_MANIFEST_INVALID"):
        make_asset_manifest(page, datetime(2026, 9, 28))


def test_manifest_page_hash_cannot_escape_cache(tmp_path: Path) -> None:
    store = AssetSnapshotStore(tmp_path / "cache", tmp_path / "manifests")
    with pytest.raises(LibraryError, match="DATA_MANIFEST_INVALID"):
        store.page_path("../../other")


def test_other_asset_status_is_counted_and_malformed_manifest_blocks(tmp_path: Path) -> None:
    page = _page([_row(FIRST_ID, "pending")])
    store = AssetSnapshotStore(tmp_path / "cache", tmp_path / "manifests")
    manifest = make_asset_manifest(page, CAPTURED)
    assert manifest["other_status_count"] == 1
    path = store.store(page, manifest)
    assert store.load(manifest)[0].status == "pending"
    path.write_text("not JSON", encoding="utf-8")
    with pytest.raises(LibraryError, match="DATA_MANIFEST_INVALID"):
        store.manifests()
