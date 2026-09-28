# SPDX-License-Identifier: Apache-2.0
"""Prospective asset-list capture and verification for later universe research."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol, cast

from signalquarry._internal.data.credentials import load_data_credentials
from signalquarry._internal.data.library import LibraryError, dataset_from_manifest
from signalquarry._internal.data.panel import PanelStore
from signalquarry._internal.data.universe import AssetPage, AssetSnapshotStore, make_asset_manifest
from signalquarry._internal.data.universe_build import (
    UniverseBuildPolicy,
    load_classification_snapshot,
    make_universe_manifest,
    store_classification_snapshot,
    verify_universe_manifest,
)
from signalquarry._internal.paper.brokers.alpaca_paper import AlpacaPaperBroker
from signalquarry._internal.paper.models import PaperError
from signalquarry._internal.project.project import ProjectError, find_root
from signalquarry.api.data import library_for
from signalquarry.api.envelope import Envelope


class AssetSource(Protocol):
    def asset_snapshot(self) -> AssetPage: ...


def _store(root: Path) -> AssetSnapshotStore:
    return AssetSnapshotStore(library_for(root).cache_dir, root / "data" / "universe" / "snapshots")


def _replay_universe_build(root: Path, manifest: dict[str, Any]) -> None:
    """Recompute a stored membership record from its original cached inputs."""
    library = library_for(root)
    asset_store = _store(root)
    asset_manifest = next(
        (
            item
            for item in asset_store.manifests()
            if item.get("snapshot_id") == manifest.get("asset_snapshot_id")
            and item.get("manifest_hash") == manifest.get("asset_snapshot_hash")
        ),
        None,
    )
    if asset_manifest is None:
        raise LibraryError("UNIVERSE_INPUT_UNAVAILABLE", "asset snapshot for build missing")
    assets = asset_store.load(asset_manifest)
    dataset_manifest = next(
        (
            item
            for item in library.manifests()
            if item.get("dataset_id") == manifest.get("dataset_id")
            and item.get("manifest_hash") == manifest.get("dataset_manifest_hash")
            and item.get("dataset_identity") == manifest.get("dataset_identity")
        ),
        None,
    )
    if dataset_manifest is None:
        raise LibraryError("UNIVERSE_INPUT_UNAVAILABLE", "dataset manifest for build missing")
    classification = load_classification_snapshot(library.cache_dir, manifest["classification_snapshot_hash"])
    dataset = dataset_from_manifest(library, dataset_manifest)
    panels = PanelStore(library.cache_dir)
    panels.build(dataset)
    panel = panels.load(dataset.identity()).window(
        decision_session=date.fromisoformat(manifest["decision_session"]), lookback=20
    )
    policy_data = manifest["policy"]
    expected = make_universe_manifest(
        assets=assets,
        asset_manifest=asset_manifest,
        classification_snapshot=classification,
        dataset_manifest=dataset_manifest,
        panel=panel,
        decision_session=date.fromisoformat(manifest["decision_session"]),
        known_at=datetime.fromisoformat(manifest["known_at"].replace("Z", "+00:00")),
        policy=UniverseBuildPolicy(
            minimum_price=Decimal(policy_data["minimum_price"]),
            minimum_listing_age_days=policy_data["minimum_listing_age_days"],
            minimum_dollar_volume_percentile=Decimal(policy_data["minimum_dollar_volume_percentile"]),
        ),
    )
    if expected != manifest:
        raise LibraryError("DATA_MANIFEST_INVALID", "universe build replay mismatch")


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
    build_dir = root / "data" / "universe" / "builds"
    builds: list[dict[str, Any]] = []
    if build_dir.is_dir():
        for path in sorted(build_dir.glob("*.json")):
            try:
                raw_manifest = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(raw_manifest, dict):
                    raise ValueError("manifest must be an object")
                manifest = cast(dict[str, Any], raw_manifest)
                verify_universe_manifest(manifest)
                _replay_universe_build(root, manifest)
                builds.append({"manifest_hash": manifest.get("manifest_hash"), "ok": True})
            except (OSError, ValueError, TypeError, KeyError, LibraryError) as exc:
                builds.append({"manifest": path.name, "ok": False, "error": str(exc)})
    failed = [item for item in [*results, *builds] if not item["ok"]]
    envelope.data = {"snapshots": results, "builds": builds}
    envelope.summary = (
        f"{sum(item['ok'] for item in results)} of {len(results)} asset snapshots and "
        f"{sum(item['ok'] for item in builds)} of {len(builds)} universe builds verified"
    )
    if failed:
        envelope.status = "blocked"
        envelope.reason_codes = sorted({item["error"].split(":", 1)[0] for item in failed})
    return envelope


def universe_build(
    *,
    decision_session: date,
    known_at: datetime,
    dataset_id: str,
    classification_file: Path,
    minimum_price: Decimal,
    minimum_listing_age_days: int,
    minimum_dollar_volume_percentile: Decimal,
    project: Path | None = None,
) -> Envelope:
    """Build one monthly universe from complete observations available by cutoff."""
    envelope = Envelope(command="universe build")
    try:
        root = find_root(project)
    except ProjectError as exc:
        return Envelope(
            command=envelope.command, status="invalid", reason_codes=[exc.code], summary=exc.detail
        )
    try:
        raw_classification = json.loads(classification_file.read_text(encoding="utf-8"))
        if not isinstance(raw_classification, dict):
            raise ValueError("classification snapshot must be an object")
        classification_snapshot = cast(dict[str, Any], raw_classification)
        store = _store(root)
        asset_manifest = store.as_of(known_at)
        assets = store.load(asset_manifest)
        library = library_for(root)
        dataset_manifest = next(
            (item for item in library.manifests() if item.get("dataset_id") == dataset_id), None
        )
        if dataset_manifest is None:
            raise LibraryError("DATA_MANIFEST_INVALID", "dataset not found")
        dataset = dataset_from_manifest(library, dataset_manifest)
        panel_store = PanelStore(library.cache_dir)
        panel_store.build(dataset)
        panel = panel_store.load(dataset.identity()).window(decision_session=decision_session, lookback=20)
        manifest = make_universe_manifest(
            assets=assets,
            asset_manifest=asset_manifest,
            classification_snapshot=classification_snapshot,
            dataset_manifest=dataset_manifest,
            panel=panel,
            decision_session=decision_session,
            known_at=known_at,
            policy=UniverseBuildPolicy(
                minimum_price=minimum_price,
                minimum_listing_age_days=minimum_listing_age_days,
                minimum_dollar_volume_percentile=minimum_dollar_volume_percentile,
            ),
        )
        verify_universe_manifest(manifest)
        store_classification_snapshot(
            library.cache_dir, classification_snapshot, manifest["classification_snapshot_hash"]
        )
        build_dir = root / "data" / "universe" / "builds"
        build_dir.mkdir(parents=True, exist_ok=True)
        path = build_dir / f"{manifest['manifest_hash'].removeprefix('sha256:')}.json"
        content = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise LibraryError("DATA_MANIFEST_INVALID", "universe build collision")
        if not path.exists():
            path.write_text(content, encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, LibraryError, PaperError) as exc:
        code = getattr(exc, "code", "UNIVERSE_CLASSIFICATION_INVALID")
        status = (
            "unavailable"
            if code in ("UNIVERSE_ASOF_UNAVAILABLE", "UNIVERSE_INPUT_UNAVAILABLE")
            else "invalid"
        )
        return Envelope(command=envelope.command, status=status, reason_codes=[code], summary=str(exc))

    envelope.data = {
        "decision_session": manifest["decision_session"],
        "known_at": manifest["known_at"],
        "dataset_id": manifest["dataset_id"],
        "asset_snapshot_id": manifest["asset_snapshot_id"],
        "classification_snapshot_hash": manifest["classification_snapshot_hash"],
        "counts": manifest["counts"],
        "manifest": str(path.relative_to(root)),
        "manifest_hash": manifest["manifest_hash"],
    }
    envelope.summary = f"built {manifest['counts']['members']} point-in-time common-stock members"
    envelope.artifacts = [
        {
            "path": str(path.relative_to(root)),
            "sha256": manifest["manifest_hash"].removeprefix("sha256:"),
            "kind": "point_in_time_universe_manifest",
        }
    ]
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
