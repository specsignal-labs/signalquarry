# SPDX-License-Identifier: Apache-2.0
"""Prospective asset-list capture and verification for later universe research."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from signalquarry._internal.data.credentials import load_data_credentials
from signalquarry._internal.data.library import LibraryError
from signalquarry._internal.data.universe import AssetPage, AssetSnapshotStore, make_asset_manifest
from signalquarry._internal.paper.brokers.alpaca_paper import AlpacaPaperBroker
from signalquarry._internal.paper.models import PaperError
from signalquarry._internal.project.project import ProjectError, find_root
from signalquarry.api.data import library_for
from signalquarry.api.envelope import Envelope


class AssetSource(Protocol):
    def asset_snapshot(self) -> AssetPage: ...


def _store(root: Path) -> AssetSnapshotStore:
    return AssetSnapshotStore(library_for(root).cache_dir, root / "data" / "universe" / "snapshots")


def universe_snapshot(*, project: Path | None = None, source: AssetSource | None = None) -> Envelope:
    """Record a full current snapshot; never claim it existed before capture."""
    envelope = Envelope(command="universe snapshot")
    try:
        root = find_root(project)
    except ProjectError as exc:
        return Envelope(
            command=envelope.command, status="invalid", reason_codes=[exc.code], summary=exc.detail
        )
    if source is None:
        credentials = load_data_credentials()
        if credentials is None:
            return Envelope(
                command=envelope.command,
                status="unavailable",
                reason_codes=["DATA_CREDENTIALS_MISSING"],
                summary="Alpaca credentials are required for the read-only asset snapshot",
            )
        source = AlpacaPaperBroker(credentials.key_id, credentials.secret_key)
    try:
        page = source.asset_snapshot()
        captured = datetime.now(UTC)
        manifest = make_asset_manifest(page, captured)
        path = _store(root).store(page, manifest)
    except (PaperError, LibraryError) as exc:
        code = exc.code
        envelope.status = (
            "unavailable" if code in ("BROKER_UNAVAILABLE", "BROKER_CREDENTIALS_REJECTED") else "invalid"
        )
        envelope.reason_codes, envelope.summary = [code], str(exc)
        return envelope
    envelope.summary = f"captured {manifest['asset_count']} US-equity asset records"
    envelope.data = {
        "snapshot_id": manifest["snapshot_id"],
        "observed_at": manifest["observed_at"],
        "asset_count": manifest["asset_count"],
        "active_count": manifest["active_count"],
        "inactive_count": manifest["inactive_count"],
        "other_status_count": manifest["other_status_count"],
        "manifest": str(path.relative_to(root)),
        "historical_asof_before_capture": False,
    }
    envelope.artifacts = [
        {
            "path": str(path.relative_to(root)),
            "sha256": manifest["manifest_hash"].removeprefix("sha256:"),
            "kind": "asset_snapshot_manifest",
        }
    ]
    return envelope


def universe_verify(*, project: Path | None = None) -> Envelope:
    envelope = Envelope(command="universe verify")
    try:
        root = find_root(project)
        store = _store(root)
        manifests = store.manifests()
    except (ProjectError, LibraryError, OSError, ValueError) as exc:
        code = getattr(exc, "code", "DATA_MANIFEST_INVALID")
        return Envelope(command=envelope.command, status="invalid", reason_codes=[code], summary=str(exc))
    results: list[dict[str, Any]] = []
    for manifest in manifests:
        try:
            store.load(manifest)
            results.append({"snapshot_id": manifest.get("snapshot_id"), "ok": True})
        except (LibraryError, KeyError, TypeError, ValueError) as exc:
            results.append({"snapshot_id": manifest.get("snapshot_id"), "ok": False, "error": str(exc)})
    failed = [item for item in results if not item["ok"]]
    envelope.data = {"snapshots": results}
    envelope.summary = f"{len(results) - len(failed)} of {len(results)} asset snapshots verified"
    if failed:
        envelope.status = "blocked"
        envelope.reason_codes = sorted({item["error"].split(":", 1)[0] for item in failed})
    return envelope


def universe_as_of(*, known_at: datetime, project: Path | None = None) -> Envelope:
    envelope = Envelope(command="universe as-of")
    try:
        root = find_root(project)
        manifest = _store(root).as_of(known_at)
    except ProjectError as exc:
        return Envelope(
            command=envelope.command, status="invalid", reason_codes=[exc.code], summary=exc.detail
        )
    except (LibraryError, KeyError, TypeError, ValueError) as exc:
        code = getattr(exc, "code", "DATA_MANIFEST_INVALID")
        return Envelope(command=envelope.command, status="unavailable", reason_codes=[code], summary=str(exc))
    envelope.data = {
        key: manifest[key]
        for key in (
            "snapshot_id",
            "observed_at",
            "asset_count",
            "active_count",
            "inactive_count",
            "other_status_count",
            "records_hash",
        )
    }
    envelope.summary = f"{manifest['snapshot_id']} was captured by the requested cutoff"
    return envelope
