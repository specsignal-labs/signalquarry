# SPDX-License-Identifier: Apache-2.0
"""Point-in-time equity-universe construction from dated local evidence."""

from __future__ import annotations

import gzip
import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

from signalquarry._internal.canonical import canonical_hash, hash_without, to_canonical
from signalquarry._internal.data.library import LibraryError
from signalquarry._internal.data.panel import PanelWindow
from signalquarry._internal.data.universe import ObservedAsset

CLASSIFICATION_SCHEMA = "signalquarry.instrument-classification-snapshot/v1"
UNIVERSE_BUILD_SCHEMA = "signalquarry.point-in-time-universe/v1"
LOOKBACK_SESSIONS = 20
MICRO = 1_000_000
DAILY_BAR_SAFE_AFTER = time(16, 15)
NEW_YORK = ZoneInfo("America/New_York")
_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class UniverseBuildPolicy:
    """Required filters for one monthly, pre-session membership decision."""

    minimum_price: Decimal
    minimum_listing_age_days: int
    minimum_dollar_volume_percentile: Decimal

    def validate(self) -> None:
        if (
            not isinstance(self.minimum_price, Decimal)
            or not self.minimum_price.is_finite()
            or self.minimum_price <= 0
        ):
            raise LibraryError("USAGE_INVALID", "minimum price")
        if type(self.minimum_listing_age_days) is not int or self.minimum_listing_age_days < 0:
            raise LibraryError("USAGE_INVALID", "minimum listing age")
        if (
            not isinstance(self.minimum_dollar_volume_percentile, Decimal)
            or not self.minimum_dollar_volume_percentile.is_finite()
            or not Decimal(0) <= self.minimum_dollar_volume_percentile <= Decimal(100)
        ):
            raise LibraryError("USAGE_INVALID", "minimum dollar-volume percentile")


def _timestamp(value: Any, field: str, *, error_code: str = "UNIVERSE_CLASSIFICATION_INVALID") -> datetime:
    if not isinstance(value, str):
        raise LibraryError(error_code, field)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("timezone required")
        return parsed.astimezone(UTC)
    except ValueError as exc:
        raise LibraryError(error_code, field) from exc


def _date(value: Any, field: str) -> date:
    if not isinstance(value, str):
        raise LibraryError("UNIVERSE_CLASSIFICATION_INVALID", field)
    try:
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError("date format")
        return parsed
    except ValueError as exc:
        raise LibraryError("UNIVERSE_CLASSIFICATION_INVALID", field) from exc


def verify_classification_snapshot(
    snapshot: dict[str, Any],
    assets: tuple[ObservedAsset, ...],
    *,
    known_at: datetime,
    max_age: timedelta = timedelta(days=31),
) -> tuple[dict[str, dict[str, Any]], str]:
    """Validate a complete, dated security-master snapshot against asset IDs."""
    if known_at.tzinfo is None or known_at.utcoffset() is None or max_age <= timedelta(0):
        raise LibraryError("DATA_MANIFEST_INVALID", "classification cutoff")
    if not isinstance(snapshot, dict) or set(snapshot) != {"schema", "source", "observed_at", "records"}:
        raise LibraryError("UNIVERSE_CLASSIFICATION_INVALID", "snapshot fields")
    if (
        snapshot.get("schema") != CLASSIFICATION_SCHEMA
        or not isinstance(snapshot.get("source"), str)
        or not snapshot["source"].strip()
        or not isinstance(snapshot.get("records"), list)
    ):
        raise LibraryError("UNIVERSE_CLASSIFICATION_INVALID", "snapshot header")
    observed_at = _timestamp(snapshot.get("observed_at"), "observed_at")
    cutoff = known_at.astimezone(UTC)
    if observed_at > cutoff or cutoff - observed_at > max_age:
        raise LibraryError("UNIVERSE_INPUT_UNAVAILABLE", "classification snapshot outside cutoff")

    expected_ids = {asset.asset_id for asset in assets}
    assets_by_symbol = {asset.symbol: asset.asset_id for asset in assets}
    if len(assets_by_symbol) != len(assets):
        assets_by_symbol = {}
    records: dict[str, dict[str, Any]] = {}
    key_kind: str | None = None
    record_keys: list[str] = []
    for raw in snapshot["records"]:
        if not isinstance(raw, dict) or set(raw) not in (
            {"asset_id", "security_type", "listed_at"},
            {"symbol", "security_type", "listed_at"},
        ):
            raise LibraryError("UNIVERSE_CLASSIFICATION_INVALID", "record fields")
        kind = "asset_id" if "asset_id" in raw else "symbol"
        if key_kind is not None and key_kind != kind:
            raise LibraryError("UNIVERSE_CLASSIFICATION_INVALID", "mixed record keys")
        key_kind = kind
        raw_key, security_type, listed = raw[kind], raw["security_type"], raw["listed_at"]
        if not isinstance(raw_key, str) or not isinstance(security_type, str):
            raise LibraryError("UNIVERSE_CLASSIFICATION_INVALID", "record values")
        if kind == "asset_id":
            try:
                asset_id = str(UUID(raw_key))
            except ValueError as exc:
                raise LibraryError("UNIVERSE_CLASSIFICATION_INVALID", "asset_id") from exc
            if raw_key != asset_id:
                raise LibraryError("UNIVERSE_CLASSIFICATION_INVALID", "asset_id")
        else:
            if not raw_key or raw_key != raw_key.strip() or not assets_by_symbol:
                raise LibraryError("UNIVERSE_CLASSIFICATION_INVALID", "ambiguous symbol key")
            asset_id = assets_by_symbol.get(raw_key)
            if asset_id is None:
                raise LibraryError("UNIVERSE_CLASSIFICATION_INVALID", "unknown symbol")
        if asset_id in records or security_type not in ("common_stock", "other"):
            raise LibraryError("UNIVERSE_CLASSIFICATION_INVALID", "record identity")
        listed_at = None if listed is None else _date(listed, "listed_at")
        if security_type == "common_stock" and listed_at is None:
            raise LibraryError("UNIVERSE_CLASSIFICATION_INVALID", "common stock listing date")
        records[asset_id] = {"security_type": security_type, "listed_at": listed_at}
        record_keys.append(raw_key)
    if set(records) != expected_ids:
        raise LibraryError("UNIVERSE_CLASSIFICATION_INVALID", "incomplete asset coverage")
    if record_keys != sorted(record_keys):
        raise LibraryError("UNIVERSE_CLASSIFICATION_INVALID", "record order")
    return records, canonical_hash(snapshot)


def _median(values: list[Decimal]) -> Decimal:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / Decimal(2)


def _classification_cache_path(cache_dir: Path, snapshot_hash: str) -> Path:
    if not isinstance(snapshot_hash, str) or not _HASH.fullmatch(snapshot_hash):
        raise LibraryError("DATA_MANIFEST_INVALID", "classification hash")
    return cache_dir / "universe" / "classifications" / f"{snapshot_hash.removeprefix('sha256:')}.json.gz"


def store_classification_snapshot(cache_dir: Path, snapshot: dict[str, Any], snapshot_hash: str) -> Path:
    """Keep the verified source snapshot in a private local cache for replay."""
    if canonical_hash(snapshot) != snapshot_hash:
        raise LibraryError("UNIVERSE_CLASSIFICATION_INVALID", "classification hash")
    path = _classification_cache_path(cache_dir, snapshot_hash)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    body = json.dumps(
        to_canonical(snapshot), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    compressed = gzip.compress(body, mtime=0)
    if path.exists():
        try:
            existing = gzip.decompress(path.read_bytes())
        except (OSError, EOFError) as exc:
            raise LibraryError("DATA_PAGE_CORRUPT", "classification snapshot") from exc
        if existing != body:
            raise LibraryError("DATA_PAGE_CORRUPT", "classification snapshot collision")
    else:
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(compressed)
        os.chmod(temporary, 0o600)
        temporary.replace(path)
    os.chmod(path, 0o600)
    return path


def load_classification_snapshot(cache_dir: Path, snapshot_hash: str) -> dict[str, Any]:
    """Load and re-hash a private classification snapshot by its content ID."""
    path = _classification_cache_path(cache_dir, snapshot_hash)
    try:
        body = gzip.decompress(path.read_bytes())
        snapshot = json.loads(body)
    except FileNotFoundError as exc:
        raise LibraryError("UNIVERSE_INPUT_UNAVAILABLE", "classification snapshot cache missing") from exc
    except (OSError, EOFError, ValueError) as exc:
        raise LibraryError("DATA_PAGE_CORRUPT", "classification snapshot") from exc
    if not isinstance(snapshot, dict) or canonical_hash(snapshot) != snapshot_hash:
        raise LibraryError("DATA_PAGE_CORRUPT", "classification snapshot hash")
    return snapshot


def _dollar_volume(close_micro: int, volume: Any) -> Decimal:
    try:
        volume_value = Decimal(str(float(volume)))
    except (TypeError, ValueError, ArithmeticError) as exc:
        raise LibraryError("DATA_PANEL_INVALID", "volume") from exc
    if not volume_value.is_finite() or volume_value < 0:
        raise LibraryError("DATA_PANEL_INVALID", "volume")
    return Decimal(close_micro) * volume_value / Decimal(MICRO)


def make_universe_manifest(
    *,
    assets: tuple[ObservedAsset, ...],
    asset_manifest: dict[str, Any],
    classification_snapshot: dict[str, Any],
    dataset_manifest: dict[str, Any],
    panel: PanelWindow,
    decision_session: date,
    known_at: datetime,
    policy: UniverseBuildPolicy,
) -> dict[str, Any]:
    """Build one reproducible membership manifest using only pre-cutoff inputs."""
    policy.validate()
    if known_at.tzinfo is None or known_at.utcoffset() is None:
        raise LibraryError("DATA_MANIFEST_INVALID", "decision cutoff must precede session")
    cutoff = known_at.astimezone(UTC)
    if cutoff.date() >= decision_session:
        raise LibraryError("DATA_MANIFEST_INVALID", "decision cutoff must precede session")
    observed = _timestamp(
        asset_manifest.get("observed_at"), "asset observed_at", error_code="DATA_MANIFEST_INVALID"
    )
    if observed > cutoff or cutoff - observed > timedelta(days=31):
        raise LibraryError("UNIVERSE_INPUT_UNAVAILABLE", "asset snapshot outside cutoff")
    if (
        not isinstance(asset_manifest.get("manifest_hash"), str)
        or not _HASH.fullmatch(asset_manifest["manifest_hash"])
        or hash_without(asset_manifest, "manifest_hash") != asset_manifest["manifest_hash"]
    ):
        raise LibraryError("DATA_MANIFEST_INVALID", "asset snapshot identity")
    classifications, classification_hash = verify_classification_snapshot(
        classification_snapshot, assets, known_at=cutoff
    )
    fetched_at = _timestamp(
        dataset_manifest.get("fetched_at"), "dataset fetched_at", error_code="DATA_MANIFEST_INVALID"
    )
    if fetched_at > cutoff:
        raise LibraryError("UNIVERSE_INPUT_UNAVAILABLE", "dataset fetched after cutoff")
    symbols = dataset_manifest.get("symbols")
    if (
        dataset_manifest.get("schema") != "signalquarry.dataset-manifest/v1"
        or dataset_manifest.get("provider") != "alpaca"
        or dataset_manifest.get("adjustment") != "raw"
        or not isinstance(symbols, list)
        or not all(isinstance(symbol, str) and symbol for symbol in symbols)
        or symbols != sorted(set(symbols))
        or not isinstance(dataset_manifest.get("manifest_hash"), str)
        or not _HASH.fullmatch(dataset_manifest["manifest_hash"])
        or hash_without(dataset_manifest, "manifest_hash") != dataset_manifest["manifest_hash"]
        or not isinstance(dataset_manifest.get("dataset_identity"), str)
        or not _HASH.fullmatch(dataset_manifest["dataset_identity"])
        or dataset_manifest.get("dataset_identity") != panel.dataset_identity
    ):
        raise LibraryError("DATA_MANIFEST_INVALID", "dataset identity")
    try:
        dataset_end = _date(dataset_manifest.get("end"), "dataset end")
    except LibraryError as exc:
        raise LibraryError("DATA_MANIFEST_INVALID", "dataset end") from exc
    if dataset_end >= decision_session or dataset_end > cutoff.date():
        raise LibraryError("UNIVERSE_INPUT_UNAVAILABLE", "dataset includes sessions after cutoff")
    if len(panel.sessions) != LOOKBACK_SESSIONS or panel.sessions[-1] >= decision_session:
        raise LibraryError("UNIVERSE_INPUT_UNAVAILABLE", "20 complete prior sessions are required")
    latest_bar_available_at = datetime.combine(
        panel.sessions[-1], DAILY_BAR_SAFE_AFTER, tzinfo=NEW_YORK
    ).astimezone(UTC)
    if fetched_at < latest_bar_available_at:
        raise LibraryError("UNIVERSE_INPUT_UNAVAILABLE", "latest daily bar fetched before session completion")
    if (
        panel.present.shape != (LOOKBACK_SESSIONS, len(panel.symbols))
        or panel.volume.shape != (LOOKBACK_SESSIONS, len(panel.symbols))
        or panel.micro["close"].shape != (LOOKBACK_SESSIONS, len(panel.symbols))
        or len(panel.symbols) != len(set(panel.symbols))
    ):
        raise LibraryError("DATA_PANEL_INVALID", "presence shape")

    common = [asset for asset in assets if classifications[asset.asset_id]["security_type"] == "common_stock"]
    common_symbols = [asset.symbol for asset in common]
    if len(common_symbols) != len(set(common_symbols)):
        raise LibraryError("UNIVERSE_CLASSIFICATION_INVALID", "duplicate common-stock symbol")
    symbol_index = {symbol: index for index, symbol in enumerate(panel.symbols)}
    if set(panel.symbols) != set(dataset_manifest.get("symbols", [])):
        raise LibraryError("DATA_PANEL_INVALID", "panel symbols differ from dataset manifest")

    eligible: list[tuple[str, Decimal]] = []
    excluded_age = 0
    excluded_history = 0
    excluded_price = 0
    for asset in common:
        listed_at = classifications[asset.asset_id]["listed_at"]
        if listed_at > decision_session:
            raise LibraryError("UNIVERSE_CLASSIFICATION_INVALID", "listing date after decision")
        if (decision_session - listed_at).days < policy.minimum_listing_age_days:
            excluded_age += 1
            continue
        index = symbol_index.get(asset.symbol)
        if index is None or not bool(panel.present[:, index].all()):
            excluded_history += 1
            continue
        closes = panel.micro["close"][:, index]
        volumes = panel.volume[:, index]
        if len(closes) != LOOKBACK_SESSIONS or any(int(close) <= 0 for close in closes):
            excluded_history += 1
            continue
        last_price = Decimal(int(closes[-1])) / Decimal(MICRO)
        if last_price < policy.minimum_price:
            excluded_price += 1
            continue
        daily = [_dollar_volume(int(close), volume) for close, volume in zip(closes, volumes, strict=True)]
        eligible.append((asset.symbol, _median(daily)))

    # Average-rank percentiles put tied values at the same threshold and map a
    # unique maximum to 100. The rank universe is the price/history cohort.
    ranked = []
    for symbol, median_volume in eligible:
        lower = sum(other < median_volume for _, other in eligible)
        equal = sum(other == median_volume for _, other in eligible)
        percentile = (
            Decimal(100)
            if len(eligible) == 1
            else Decimal(100)
            * (Decimal(lower) + (Decimal(equal) - Decimal(1)) / Decimal(2))
            / Decimal(len(eligible) - 1)
        )
        ranked.append((symbol, percentile))
    members = sorted(
        symbol for symbol, percentile in ranked if percentile >= policy.minimum_dollar_volume_percentile
    )
    body = {
        "schema": UNIVERSE_BUILD_SCHEMA,
        "decision_session": decision_session,
        "known_at": cutoff,
        "asset_snapshot_id": asset_manifest.get("snapshot_id"),
        "asset_snapshot_hash": asset_manifest["manifest_hash"],
        "classification_snapshot_hash": classification_hash,
        "dataset_id": dataset_manifest.get("dataset_id"),
        "dataset_identity": dataset_manifest["dataset_identity"],
        "dataset_manifest_hash": dataset_manifest["manifest_hash"],
        "lookback_sessions": LOOKBACK_SESSIONS,
        "policy": {
            "minimum_price": policy.minimum_price,
            "minimum_listing_age_days": policy.minimum_listing_age_days,
            "minimum_dollar_volume_percentile": policy.minimum_dollar_volume_percentile,
        },
        "counts": {
            "common_stocks": len(common),
            "excluded_listing_age": excluded_age,
            "excluded_history": excluded_history,
            "excluded_price": excluded_price,
            "dollar_volume_ranked": len(eligible),
            "excluded_dollar_volume_rank": len(eligible) - len(members),
            "members": len(members),
        },
        "member_symbols": members,
        "members_hash": canonical_hash(members),
        "redistributable": False,
    }
    result = to_canonical(body)
    result["manifest_hash"] = canonical_hash(result)
    return result


def verify_universe_manifest(manifest: dict[str, Any]) -> None:
    """Verify the immutable member set and content-addressed manifest."""
    if not isinstance(manifest, dict):
        raise LibraryError("DATA_MANIFEST_INVALID", "point-in-time universe")
    members = manifest.get("member_symbols")
    if (
        manifest.get("schema") != UNIVERSE_BUILD_SCHEMA
        or not isinstance(members, list)
        or not all(isinstance(symbol, str) and symbol for symbol in members)
        or members != sorted(set(members))
        or manifest.get("members_hash") != canonical_hash(members)
        or manifest.get("redistributable") is not False
        or not isinstance(manifest.get("manifest_hash"), str)
        or not _HASH.fullmatch(manifest["manifest_hash"])
        or hash_without(manifest, "manifest_hash") != manifest.get("manifest_hash")
    ):
        raise LibraryError("DATA_MANIFEST_INVALID", "point-in-time universe")
