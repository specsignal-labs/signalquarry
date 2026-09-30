# SPDX-License-Identifier: Apache-2.0
"""Prospective, content-addressed Alpaca asset-list observations.

An asset snapshot says what the API returned when captured. It cannot
reconstruct earlier membership, distinguish common shares from every other US
equity instrument, or establish a historical listing date by itself.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

from signalquarry._internal.canonical import canonical_hash, hash_without, to_canonical
from signalquarry._internal.data.library import LibraryError

ASSETS_PATH = "/v2/assets"
ASSETS_PARAMS = {"asset_class": "us_equity"}  # no status filter: include active and inactive
SNAPSHOT_SCHEMA = "signalquarry.asset-snapshot/v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class AssetPage:
    endpoint: str
    params: dict[str, str]
    body: bytes
    sha256: str


@dataclass(frozen=True)
class ObservedAsset:
    asset_id: str  # provider-scoped Alpaca UUID, not a ticker
    symbol: str
    exchange: str
    status: str
    tradable: bool


def assets_from_page(page: AssetPage) -> tuple[ObservedAsset, ...]:
    """Validate the full asset response; ticker reuse across IDs is allowed."""
    if page.endpoint != ASSETS_PATH or page.params != ASSETS_PARAMS:
        raise LibraryError("PROVIDER_RESPONSE_INVALID", "asset request")
    if hashlib.sha256(page.body).hexdigest() != page.sha256:
        raise LibraryError("DATA_PAGE_CORRUPT", "asset page hash")
    try:
        rows = json.loads(page.body)
    except (ValueError, UnicodeDecodeError) as exc:
        raise LibraryError("PROVIDER_RESPONSE_INVALID", "asset JSON") from exc
    if not isinstance(rows, list) or not rows:
        raise LibraryError("PROVIDER_RESPONSE_INVALID", "asset list")
    assets: dict[str, ObservedAsset] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise LibraryError("PROVIDER_RESPONSE_INVALID", "asset row")
        try:
            raw_id = row["id"]
            symbol = row["symbol"]
            exchange = row["exchange"]
            status = row["status"]
            asset_class = row["class"]
            tradable = row["tradable"]
            if (
                not isinstance(raw_id, str)
                or not isinstance(symbol, str)
                or not symbol
                or symbol != symbol.strip()
                or not isinstance(exchange, str)
                or not exchange
                or not isinstance(status, str)
                or not status
                or asset_class != "us_equity"
                or type(tradable) is not bool
            ):
                raise ValueError("asset fields")
            asset_id = str(UUID(raw_id))
        except (KeyError, ValueError, TypeError) as exc:
            raise LibraryError("PROVIDER_RESPONSE_INVALID", "asset fields") from exc
        if asset_id in assets:
            raise LibraryError("PROVIDER_RESPONSE_INVALID", "duplicate asset id")
        assets[asset_id] = ObservedAsset(asset_id, symbol, exchange, status, tradable)
    return tuple(assets[asset_id] for asset_id in sorted(assets))


def _records_hash(assets: tuple[ObservedAsset, ...]) -> str:
    return canonical_hash(
        [
            {
                "asset_id": item.asset_id,
                "symbol": item.symbol,
                "exchange": item.exchange,
                "status": item.status,
                "tradable": item.tradable,
            }
            for item in assets
        ]
    )


def make_asset_manifest(page: AssetPage, observed_at: datetime) -> dict[str, Any]:
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise LibraryError("DATA_MANIFEST_INVALID", "observation time")
    observed_at = observed_at.astimezone(UTC)
    assets = assets_from_page(page)
    body = {
        "schema": SNAPSHOT_SCHEMA,
        "snapshot_id": f"alpaca-assets-{observed_at:%Y%m%dT%H%M%S%fZ}-{page.sha256[:12]}",
        "provider": "alpaca",
        "observed_at": observed_at,
        "page": {"endpoint": page.endpoint, "params": page.params, "sha256": page.sha256},
        "asset_count": len(assets),
        "active_count": sum(item.status == "active" for item in assets),
        "inactive_count": sum(item.status == "inactive" for item in assets),
        "other_status_count": sum(item.status not in ("active", "inactive") for item in assets),
        "records_hash": _records_hash(assets),
        "redistributable": False,
    }
    body = to_canonical(body)
    body["manifest_hash"] = canonical_hash(body)
    return body


@dataclass(frozen=True)
class AssetSnapshotStore:
    cache_dir: Path
    manifest_dir: Path

    def page_path(self, sha256: str) -> Path:
        if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
            raise LibraryError("DATA_MANIFEST_INVALID", "asset page hash")
        return self.cache_dir / "pages" / f"{sha256}.json.gz"

    def store(self, page: AssetPage, manifest: dict[str, Any]) -> Path:
        verify_asset_manifest(manifest, page)
        cached = self.page_path(page.sha256)
        cached.parent.mkdir(parents=True, exist_ok=True)
        if cached.exists():
            try:
                existing = gzip.decompress(cached.read_bytes())
            except OSError as exc:
                raise LibraryError("DATA_PAGE_CORRUPT", page.sha256) from exc
            if existing != page.body:
                raise LibraryError("DATA_PAGE_CORRUPT", page.sha256)
        else:
            temporary = cached.with_suffix(".tmp")
            temporary.write_bytes(gzip.compress(page.body, mtime=0))
            temporary.replace(cached)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        path = self.manifest_dir / f"{manifest['snapshot_id']}.json"
        content = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise LibraryError("DATA_MANIFEST_INVALID", "snapshot collision")
        if not path.exists():
            path.write_text(content, encoding="utf-8")
        return path

    def load(self, manifest: dict[str, Any]) -> tuple[ObservedAsset, ...]:
        try:
            reference = manifest["page"]
            endpoint, params, sha256 = reference["endpoint"], reference["params"], reference["sha256"]
        except (KeyError, TypeError) as exc:
            raise LibraryError("DATA_MANIFEST_INVALID", "asset page reference") from exc
        try:
            raw = self.page_path(sha256).read_bytes()
        except OSError as exc:
            raise LibraryError("DATA_PAGE_MISSING", "asset snapshot") from exc
        try:
            body = gzip.decompress(raw)
        except (OSError, EOFError) as exc:
            raise LibraryError("DATA_PAGE_CORRUPT", "asset snapshot") from exc
        page = AssetPage(endpoint, params, body, sha256)
        return verify_asset_manifest(manifest, page)

    def manifests(self) -> list[dict[str, Any]]:
        if not self.manifest_dir.is_dir():
            return []
        out: list[dict[str, Any]] = []
        for path in sorted(self.manifest_dir.glob("*.json")):
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise LibraryError("DATA_MANIFEST_INVALID", path.name) from exc
            if not isinstance(manifest, dict):
                raise LibraryError("DATA_MANIFEST_INVALID", path.name)
            out.append(manifest)
        return out

    def as_of(self, known_at: datetime, *, max_age: timedelta = timedelta(days=31)) -> dict[str, Any]:
        """Last verified snapshot captured by cutoff, within an explicit freshness limit."""
        if known_at.tzinfo is None or known_at.utcoffset() is None or max_age <= timedelta(0):
            raise LibraryError("DATA_MANIFEST_INVALID", "asset cutoff")
        cutoff = known_at.astimezone(UTC)
        candidates = []
        for manifest in self.manifests():
            self.load(manifest)
            observed = datetime.fromisoformat(manifest["observed_at"].replace("Z", "+00:00"))
            if observed <= cutoff and cutoff - observed <= max_age:
                candidates.append((observed, manifest["snapshot_id"], manifest))
        if not candidates:
            raise LibraryError("UNIVERSE_ASOF_UNAVAILABLE")
        return max(candidates, key=lambda item: (item[0], item[1]))[2]


def verify_asset_manifest(manifest: dict[str, Any], page: AssetPage) -> tuple[ObservedAsset, ...]:
    if (
        manifest.get("schema") != SNAPSHOT_SCHEMA
        or manifest.get("provider") != "alpaca"
        or hash_without(manifest, "manifest_hash") != manifest.get("manifest_hash")
        or manifest.get("page") != {"endpoint": page.endpoint, "params": page.params, "sha256": page.sha256}
    ):
        raise LibraryError("DATA_MANIFEST_INVALID", "asset snapshot")
    try:
        observed = datetime.fromisoformat(manifest["observed_at"].replace("Z", "+00:00"))
        if observed.tzinfo is None or observed.utcoffset() != timedelta(0):
            raise ValueError("observation time")
    except (KeyError, TypeError, ValueError) as exc:
        raise LibraryError("DATA_MANIFEST_INVALID", "asset observation time") from exc
    assets = assets_from_page(page)
    if (
        manifest.get("snapshot_id") != f"alpaca-assets-{observed:%Y%m%dT%H%M%S%fZ}-{page.sha256[:12]}"
        or manifest.get("asset_count") != len(assets)
        or manifest.get("active_count") != sum(item.status == "active" for item in assets)
        or manifest.get("inactive_count") != sum(item.status == "inactive" for item in assets)
        or manifest.get("other_status_count")
        != sum(item.status not in ("active", "inactive") for item in assets)
        or manifest.get("records_hash") != _records_hash(assets)
        or manifest.get("redistributable") is not False
    ):
        raise LibraryError("DATA_MANIFEST_INVALID", "asset snapshot contents")
    return assets
