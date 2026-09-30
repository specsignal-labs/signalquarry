# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from signalquarry._internal.canonical import canonical_hash, hash_without
from signalquarry._internal.data.library import LibraryError
from signalquarry._internal.data.panel import PanelWindow
from signalquarry._internal.data.synthetic import weekday_sessions
from signalquarry._internal.data.universe import (
    ASSETS_PARAMS,
    ASSETS_PATH,
    AssetPage,
    assets_from_page,
    make_asset_manifest,
)
from signalquarry._internal.data.universe_build import (
    CLASSIFICATION_SCHEMA,
    UniverseBuildPolicy,
    _median,
    load_classification_snapshot,
    make_universe_manifest,
    store_classification_snapshot,
    verify_classification_snapshot,
    verify_universe_manifest,
)

ASSET_IDS = tuple(f"00000000-0000-4000-8000-{number:012d}" for number in range(1, 6))
SYMBOLS = ("AAA", "BBB", "CHEAP", "ETF", "NEW")
DECISION_SESSION = date(2026, 9, 29)
KNOWN_AT = datetime(2026, 9, 28, 21, tzinfo=UTC)
OBSERVED_AT = KNOWN_AT - timedelta(hours=1)
DATASET_IDENTITY = "sha256:" + "a" * 64


def _inputs(*, fetched_at: datetime = KNOWN_AT - timedelta(minutes=30)):
    rows = [
        {
            "id": asset_id,
            "class": "us_equity",
            "symbol": symbol,
            "exchange": "NYSE",
            "status": "active",
            "tradable": True,
        }
        for asset_id, symbol in zip(ASSET_IDS, SYMBOLS, strict=True)
    ]
    body = json.dumps(rows, sort_keys=True).encode()
    page = AssetPage(ASSETS_PATH, dict(ASSETS_PARAMS), body, hashlib.sha256(body).hexdigest())
    assets = assets_from_page(page)
    asset_manifest = make_asset_manifest(page, OBSERVED_AT)
    listing_dates = {
        "AAA": "2020-01-01",
        "BBB": "2020-01-01",
        "CHEAP": "2020-01-01",
        "ETF": None,
        "NEW": "2026-09-20",
    }
    types = {symbol: "other" if symbol == "ETF" else "common_stock" for symbol in SYMBOLS}
    classification = {
        "schema": CLASSIFICATION_SCHEMA,
        "source": "synthetic-test-source",
        "observed_at": OBSERVED_AT.isoformat().replace("+00:00", "Z"),
        "records": [
            {
                "asset_id": asset.asset_id,
                "security_type": types[asset.symbol],
                "listed_at": listing_dates[asset.symbol],
            }
            for asset in assets
        ],
    }
    panel_symbols = SYMBOLS
    prices = np.array([20, 30, 4, 50, 40], dtype=np.int64) * 1_000_000
    closes = np.tile(prices, (20, 1))
    volumes = np.tile(np.array([1000, 10, 1000, 1000, 1000], dtype=np.float64), (20, 1))
    panel = PanelWindow(
        dataset_identity=DATASET_IDENTITY,
        decision_session=DECISION_SESSION,
        sessions=weekday_sessions(date(2026, 8, 31), date(2026, 9, 28))[-20:],
        symbols=panel_symbols,
        micro={"close": closes},
        volume=volumes,
        present=np.ones((20, len(panel_symbols)), dtype=bool),
    )
    dataset_manifest = {
        "schema": "signalquarry.dataset-manifest/v1",
        "provider": "alpaca",
        "feed": "sip",
        "adjustment": "raw",
        "dataset_id": "alpaca-sip-test",
        "dataset_identity": DATASET_IDENTITY,
        "symbols": list(panel_symbols),
        "start": "2026-08-31",
        "end": "2026-09-28",
        "fetched_at": fetched_at.isoformat().replace("+00:00", "Z"),
    }
    dataset_manifest["manifest_hash"] = canonical_hash(dataset_manifest)
    return assets, asset_manifest, classification, dataset_manifest, panel


def _build(*, fetched_at: datetime = KNOWN_AT - timedelta(minutes=30)) -> dict:
    assets, asset_manifest, classification, dataset_manifest, panel = _inputs(fetched_at=fetched_at)
    return make_universe_manifest(
        assets=assets,
        asset_manifest=asset_manifest,
        classification_snapshot=classification,
        dataset_manifest=dataset_manifest,
        panel=panel,
        decision_session=DECISION_SESSION,
        known_at=KNOWN_AT,
        policy=UniverseBuildPolicy(
            minimum_price=Decimal("10"),
            minimum_listing_age_days=30,
            minimum_dollar_volume_percentile=Decimal("50"),
        ),
    )


def _make_manifest(
    assets,
    asset_manifest,
    classification,
    dataset_manifest,
    panel,
    *,
    policy: UniverseBuildPolicy | None = None,
    known_at: datetime = KNOWN_AT,
    decision_session: date = DECISION_SESSION,
) -> dict:
    return make_universe_manifest(
        assets=assets,
        asset_manifest=asset_manifest,
        classification_snapshot=classification,
        dataset_manifest=dataset_manifest,
        panel=panel,
        decision_session=decision_session,
        known_at=known_at,
        policy=policy or UniverseBuildPolicy(Decimal("10"), 30, Decimal("50")),
    )


def test_universe_build_applies_dated_filters_and_verifies_manifest() -> None:
    manifest = _build()
    assert manifest["member_symbols"] == ["AAA"]
    assert manifest["counts"] == {
        "common_stocks": 4,
        "excluded_listing_age": 1,
        "excluded_history": 0,
        "excluded_price": 1,
        "dollar_volume_ranked": 2,
        "excluded_dollar_volume_rank": 1,
        "members": 1,
    }
    verify_universe_manifest(manifest)
    manifest["member_symbols"].append("FUTURE")
    with pytest.raises(LibraryError, match="DATA_MANIFEST_INVALID"):
        verify_universe_manifest(manifest)


def test_universe_build_rejects_unavailable_and_incomplete_inputs() -> None:
    with pytest.raises(LibraryError, match="UNIVERSE_INPUT_UNAVAILABLE"):
        _build(fetched_at=KNOWN_AT + timedelta(seconds=1))
    with pytest.raises(LibraryError, match="UNIVERSE_INPUT_UNAVAILABLE"):
        _build(fetched_at=datetime(2026, 9, 28, 20, 14, tzinfo=UTC))
    assert _build(fetched_at=datetime(2026, 9, 28, 20, 15, tzinfo=UTC))["member_symbols"] == ["AAA"]

    assets, asset_manifest, classification, dataset_manifest, panel = _inputs()
    classification["records"].pop()
    with pytest.raises(LibraryError, match="UNIVERSE_CLASSIFICATION_INVALID"):
        make_universe_manifest(
            assets=assets,
            asset_manifest=asset_manifest,
            classification_snapshot=classification,
            dataset_manifest=dataset_manifest,
            panel=panel,
            decision_session=DECISION_SESSION,
            known_at=KNOWN_AT,
            policy=UniverseBuildPolicy(Decimal("1"), 0, Decimal("0")),
        )


def test_universe_build_requires_twenty_complete_prior_sessions() -> None:
    assets, asset_manifest, classification, dataset_manifest, panel = _inputs()
    short_panel = PanelWindow(
        dataset_identity=panel.dataset_identity,
        decision_session=panel.decision_session,
        sessions=panel.sessions[:-1],
        symbols=panel.symbols,
        micro={"close": panel.micro["close"][:-1]},
        volume=panel.volume[:-1],
        present=panel.present[:-1],
    )
    with pytest.raises(LibraryError, match="UNIVERSE_INPUT_UNAVAILABLE"):
        make_universe_manifest(
            assets=assets,
            asset_manifest=asset_manifest,
            classification_snapshot=classification,
            dataset_manifest=dataset_manifest,
            panel=short_panel,
            decision_session=DECISION_SESSION,
            known_at=KNOWN_AT,
            policy=UniverseBuildPolicy(Decimal("1"), 0, Decimal("0")),
        )


def test_universe_api_persists_hash_only_build(monkeypatch, tmp_path: Path) -> None:
    from signalquarry.api import init
    from signalquarry.api import universe as api_universe

    project = tmp_path / "project"
    assert init(project, package="universe_test").status == "ok"
    assets, asset_manifest, classification, dataset_manifest, panel = _inputs()
    classification_path = tmp_path / "classification.json"
    classification_path.write_text(json.dumps(classification), encoding="utf-8")

    class SnapshotStore:
        def as_of(self, _known_at):
            return asset_manifest

        def load(self, _manifest):
            return assets

        def manifests(self):
            return [asset_manifest]

    class FakePanelStore:
        def __init__(self, _cache_dir):
            pass

        def build(self, _dataset):
            return None

        def load(self, _identity):
            return SimpleNamespace(window=lambda **_kwargs: panel)

    monkeypatch.setattr(api_universe, "_store", lambda _root: SnapshotStore())
    monkeypatch.setattr(
        api_universe,
        "library_for",
        lambda _root: SimpleNamespace(cache_dir=tmp_path / "cache", manifests=lambda: [dataset_manifest]),
    )
    monkeypatch.setattr(
        api_universe,
        "dataset_from_manifest",
        lambda _library, _manifest: SimpleNamespace(identity=lambda: DATASET_IDENTITY),
    )
    monkeypatch.setattr(api_universe, "PanelStore", FakePanelStore)

    result = api_universe.universe_build(
        decision_session=DECISION_SESSION,
        known_at=KNOWN_AT,
        dataset_id=dataset_manifest["dataset_id"],
        classification_file=classification_path,
        minimum_price=Decimal("10"),
        minimum_listing_age_days=30,
        minimum_dollar_volume_percentile=Decimal("50"),
        project=project,
    )
    assert result.status == "ok"
    assert result.data["counts"]["members"] == 1
    assert result.data["classification_provenance_verified"] is False
    assert "classification provenance unverified" in result.summary
    path = project / result.data["manifest"]
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["member_symbols"] == ["AAA"]
    assert "synthetic-test-source" not in json.dumps(result.as_dict())
    assert result.artifacts[0]["sha256"] == stored["manifest_hash"].removeprefix("sha256:")
    verified = api_universe.universe_verify(project=project)
    assert verified.status == "ok"
    assert verified.data["builds"] == [{"manifest_hash": stored["manifest_hash"], "ok": True}]


def test_universe_cli_build_forwards_explicit_policy(monkeypatch, capsys, tmp_path: Path) -> None:
    from signalquarry.api.envelope import Envelope
    from signalquarry.cli import main as cli

    calls: dict = {}

    def build(**kwargs):
        calls.update(kwargs)
        return Envelope(command="universe build", summary="synthetic build")

    monkeypatch.setattr(cli.api, "universe_build", build)
    classification_path = tmp_path / "classification.json"
    result = cli.main(
        [
            "--json",
            "universe",
            "build",
            "--session",
            "2026-09-29",
            "--known-at",
            "2026-09-28T21:00:00+00:00",
            "--dataset-id",
            "alpaca-sip-test",
            "--classification-file",
            str(classification_path),
            "--minimum-price",
            "10",
            "--minimum-listing-age-days",
            "30",
            "--minimum-dollar-volume-percentile",
            "50",
            "--project",
            str(tmp_path),
        ]
    )
    assert result == 0
    capsys.readouterr()
    assert calls["decision_session"] == DECISION_SESSION
    assert calls["known_at"] == KNOWN_AT
    assert calls["minimum_price"] == Decimal("10")
    assert calls["minimum_listing_age_days"] == 30
    assert calls["minimum_dollar_volume_percentile"] == Decimal("50")
    assert calls["classification_file"] == classification_path


@pytest.mark.parametrize(
    "policy",
    [
        UniverseBuildPolicy(Decimal("0"), 0, Decimal("0")),
        UniverseBuildPolicy(Decimal("NaN"), 0, Decimal("0")),
        UniverseBuildPolicy(1, 0, Decimal("0")),
        UniverseBuildPolicy(Decimal("1"), -1, Decimal("0")),
        UniverseBuildPolicy(Decimal("1"), True, Decimal("0")),
        UniverseBuildPolicy(Decimal("1"), 0, Decimal("NaN")),
        UniverseBuildPolicy(Decimal("1"), 0, Decimal("-0.1")),
        UniverseBuildPolicy(Decimal("1"), 0, Decimal("100.1")),
        UniverseBuildPolicy(Decimal("1"), 0, 50),
    ],
)
def test_universe_policy_rejects_nonfinite_or_out_of_range_values(policy) -> None:
    with pytest.raises(LibraryError, match="USAGE_INVALID"):
        policy.validate()


def test_classification_snapshot_accepts_sorted_symbol_keys() -> None:
    assets, _, classification, _, _ = _inputs()
    by_asset_id = {record["asset_id"]: record for record in classification["records"]}
    classification["records"] = sorted(
        [
            {
                "symbol": asset.symbol,
                "security_type": by_asset_id[asset.asset_id]["security_type"],
                "listed_at": by_asset_id[asset.asset_id]["listed_at"],
            }
            for asset in assets
        ],
        key=lambda record: record["symbol"],
    )
    records, snapshot_hash = verify_classification_snapshot(classification, assets, known_at=KNOWN_AT)
    assert len(records) == len(assets)
    assert snapshot_hash == canonical_hash(classification)


@pytest.mark.parametrize(
    ("change", "code"),
    [
        (lambda snapshot: snapshot.update(extra=True), "UNIVERSE_CLASSIFICATION_INVALID"),
        (lambda snapshot: snapshot.update(schema="unknown"), "UNIVERSE_CLASSIFICATION_INVALID"),
        (lambda snapshot: snapshot.update(source="  "), "UNIVERSE_CLASSIFICATION_INVALID"),
        (lambda snapshot: snapshot.update(observed_at="not-a-timestamp"), "UNIVERSE_CLASSIFICATION_INVALID"),
        (lambda snapshot: snapshot.update(observed_at=123), "UNIVERSE_CLASSIFICATION_INVALID"),
        (
            lambda snapshot: snapshot.update(observed_at=(KNOWN_AT + timedelta(seconds=1)).isoformat()),
            "UNIVERSE_INPUT_UNAVAILABLE",
        ),
    ],
)
def test_classification_snapshot_rejects_bad_headers_and_observation_times(change, code) -> None:
    assets, _, classification, _, _ = _inputs()
    change(classification)
    with pytest.raises(LibraryError, match=code):
        verify_classification_snapshot(classification, assets, known_at=KNOWN_AT)


def test_classification_snapshot_rejects_invalid_cutoff_and_stale_snapshot() -> None:
    assets, _, classification, _, _ = _inputs()
    with pytest.raises(LibraryError, match="DATA_MANIFEST_INVALID"):
        verify_classification_snapshot(classification, assets, known_at=KNOWN_AT.replace(tzinfo=None))
    with pytest.raises(LibraryError, match="DATA_MANIFEST_INVALID"):
        verify_classification_snapshot(classification, assets, known_at=KNOWN_AT, max_age=timedelta(0))
    classification["observed_at"] = (KNOWN_AT - timedelta(days=32)).isoformat()
    with pytest.raises(LibraryError, match="UNIVERSE_INPUT_UNAVAILABLE"):
        verify_classification_snapshot(classification, assets, known_at=KNOWN_AT)

    _, _, classification, _, _ = _inputs()
    classification["observed_at"] = "2026-09-28T20:00:00"
    with pytest.raises(LibraryError, match="UNIVERSE_CLASSIFICATION_INVALID"):
        verify_classification_snapshot(classification, assets, known_at=KNOWN_AT)


@pytest.mark.parametrize(
    ("change", "code"),
    [
        (lambda row: row.pop("listed_at"), "UNIVERSE_CLASSIFICATION_INVALID"),
        (lambda row: row.update(asset_id="not-a-uuid"), "UNIVERSE_CLASSIFICATION_INVALID"),
        (
            lambda row: row.update(asset_id=ASSET_IDS[0].replace("00000000", "abcdef00").upper()),
            "UNIVERSE_CLASSIFICATION_INVALID",
        ),
        (lambda row: row.update(security_type="preferred_stock"), "UNIVERSE_CLASSIFICATION_INVALID"),
        (lambda row: row.update(listed_at="2020-1-1"), "UNIVERSE_CLASSIFICATION_INVALID"),
        (lambda row: row.update(listed_at="20200101"), "UNIVERSE_CLASSIFICATION_INVALID"),
        (lambda row: row.update(listed_at="2020-01-01T00:00:00Z"), "UNIVERSE_CLASSIFICATION_INVALID"),
        (lambda row: row.update(listed_at=20200101), "UNIVERSE_CLASSIFICATION_INVALID"),
        (
            lambda row: row.update(security_type="common_stock", listed_at=None),
            "UNIVERSE_CLASSIFICATION_INVALID",
        ),
    ],
)
def test_classification_snapshot_rejects_malformed_records(change, code) -> None:
    assets, _, classification, _, _ = _inputs()
    change(classification["records"][0])
    with pytest.raises(LibraryError, match=code):
        verify_classification_snapshot(classification, assets, known_at=KNOWN_AT)


def test_classification_snapshot_rejects_duplicate_mixed_unknown_and_unsorted_records() -> None:
    assets, _, classification, _, _ = _inputs()
    duplicate = dict(classification)
    duplicate["records"] = [*classification["records"], dict(classification["records"][0])]
    with pytest.raises(LibraryError, match="UNIVERSE_CLASSIFICATION_INVALID"):
        verify_classification_snapshot(duplicate, assets, known_at=KNOWN_AT)

    mixed = dict(classification)
    second = dict(classification["records"][1])
    second.pop("asset_id")
    second["symbol"] = "BBB"
    mixed["records"] = [classification["records"][0], second]
    with pytest.raises(LibraryError, match="UNIVERSE_CLASSIFICATION_INVALID"):
        verify_classification_snapshot(mixed, assets, known_at=KNOWN_AT)

    symbol_rows = [{"symbol": asset.symbol, "security_type": "other", "listed_at": None} for asset in assets]
    symbol_rows[0]["symbol"] = "UNKNOWN"
    unknown = {**classification, "records": sorted(symbol_rows, key=lambda row: row["symbol"])}
    with pytest.raises(LibraryError, match="UNIVERSE_CLASSIFICATION_INVALID"):
        verify_classification_snapshot(unknown, assets, known_at=KNOWN_AT)

    unsorted = dict(classification)
    unsorted["records"] = list(reversed(classification["records"]))
    with pytest.raises(LibraryError, match="UNIVERSE_CLASSIFICATION_INVALID"):
        verify_classification_snapshot(unsorted, assets, known_at=KNOWN_AT)

    wrong_value_type = dict(classification)
    wrong_value_type["records"] = [dict(classification["records"][0], asset_id=123)]
    with pytest.raises(LibraryError, match="UNIVERSE_CLASSIFICATION_INVALID"):
        verify_classification_snapshot(wrong_value_type, assets, known_at=KNOWN_AT)


def test_classification_snapshot_rejects_ambiguous_symbol_keys() -> None:
    assets, _, classification, _, _ = _inputs()
    from dataclasses import replace

    duplicate_symbols = (assets[0], replace(assets[1], symbol=assets[0].symbol), *assets[2:])
    symbol_rows = [
        {"symbol": asset.symbol, "security_type": "other", "listed_at": None} for asset in duplicate_symbols
    ]
    symbol_rows.sort(key=lambda row: row["symbol"])
    symbol_snapshot = {**classification, "records": symbol_rows}
    with pytest.raises(LibraryError, match="UNIVERSE_CLASSIFICATION_INVALID"):
        verify_classification_snapshot(symbol_snapshot, duplicate_symbols, known_at=KNOWN_AT)

    whitespace = {
        **classification,
        "records": [{"symbol": " AAA ", "security_type": "other", "listed_at": None}],
    }
    with pytest.raises(LibraryError, match="UNIVERSE_CLASSIFICATION_INVALID"):
        verify_classification_snapshot(whitespace, assets, known_at=KNOWN_AT)


def test_classification_cache_round_trip_checks_hash_permissions_and_corruption(tmp_path: Path) -> None:
    _, _, classification, _, _ = _inputs()
    snapshot_hash = canonical_hash(classification)
    path = store_classification_snapshot(tmp_path, classification, snapshot_hash)
    assert load_classification_snapshot(tmp_path, snapshot_hash) == classification
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert store_classification_snapshot(tmp_path, classification, snapshot_hash) == path

    with pytest.raises(LibraryError, match="DATA_MANIFEST_INVALID"):
        load_classification_snapshot(tmp_path, "invalid-hash")
    with pytest.raises(LibraryError, match="UNIVERSE_INPUT_UNAVAILABLE"):
        load_classification_snapshot(tmp_path, "sha256:" + "b" * 64)
    with pytest.raises(LibraryError, match="UNIVERSE_CLASSIFICATION_INVALID"):
        store_classification_snapshot(tmp_path, classification, "sha256:" + "b" * 64)

    path.write_bytes(b"broken gzip")
    with pytest.raises(LibraryError, match="DATA_PAGE_CORRUPT"):
        load_classification_snapshot(tmp_path, snapshot_hash)
    with pytest.raises(LibraryError, match="DATA_PAGE_CORRUPT"):
        store_classification_snapshot(tmp_path, classification, snapshot_hash)


def test_classification_cache_rejects_hash_mismatch_after_decompression(tmp_path: Path) -> None:
    _, _, classification, _, _ = _inputs()
    snapshot_hash = canonical_hash(classification)
    path = store_classification_snapshot(tmp_path, classification, snapshot_hash)
    path.write_bytes(gzip.compress(b"{}", mtime=0))
    with pytest.raises(LibraryError, match="DATA_PAGE_CORRUPT"):
        load_classification_snapshot(tmp_path, snapshot_hash)


def test_classification_cache_rejects_existing_content_collision(tmp_path: Path) -> None:
    _, _, classification, _, _ = _inputs()
    snapshot_hash = canonical_hash(classification)
    path = store_classification_snapshot(tmp_path, classification, snapshot_hash)
    path.write_bytes(gzip.compress(b"different body", mtime=0))
    with pytest.raises(LibraryError, match="DATA_PAGE_CORRUPT"):
        store_classification_snapshot(tmp_path, classification, snapshot_hash)


def test_universe_build_rejects_invalid_cutoffs_and_asset_snapshots() -> None:
    inputs = _inputs()
    with pytest.raises(LibraryError, match="DATA_MANIFEST_INVALID"):
        _make_manifest(*inputs, known_at=KNOWN_AT.replace(tzinfo=None))
    with pytest.raises(LibraryError, match="DATA_MANIFEST_INVALID"):
        _make_manifest(*inputs, decision_session=KNOWN_AT.date())

    assets, asset_manifest, classification, dataset_manifest, panel = inputs
    for observed_at in (
        (KNOWN_AT + timedelta(seconds=1)).isoformat(),
        (KNOWN_AT - timedelta(days=32)).isoformat(),
    ):
        bad_asset_manifest = dict(asset_manifest, observed_at=observed_at)
        with pytest.raises(LibraryError, match="UNIVERSE_INPUT_UNAVAILABLE"):
            _make_manifest(assets, bad_asset_manifest, classification, dataset_manifest, panel)

    malformed_time = dict(asset_manifest, observed_at="bad-time")
    with pytest.raises(LibraryError, match="DATA_MANIFEST_INVALID"):
        _make_manifest(assets, malformed_time, classification, dataset_manifest, panel)

    bad_identity = dict(asset_manifest, manifest_hash="sha256:" + "f" * 64)
    with pytest.raises(LibraryError, match="DATA_MANIFEST_INVALID"):
        _make_manifest(assets, bad_identity, classification, dataset_manifest, panel)


@pytest.mark.parametrize(
    "change",
    [
        lambda manifest: manifest.update(schema="unknown"),
        lambda manifest: manifest.update(provider="other"),
        lambda manifest: manifest.update(adjustment="split_adjusted"),
        lambda manifest: manifest.update(symbols=list(reversed(manifest["symbols"]))),
        lambda manifest: manifest.update(symbols=[*manifest["symbols"], manifest["symbols"][0]]),
        lambda manifest: manifest.update(symbols=[{}, *manifest["symbols"][1:]]),
        lambda manifest: manifest.update(manifest_hash="sha256:" + "f" * 64),
        lambda manifest: manifest.update(dataset_identity="sha256:" + "f" * 64),
    ],
)
def test_universe_build_rejects_dataset_manifest_identity_errors(change) -> None:
    assets, asset_manifest, classification, dataset_manifest, panel = _inputs()
    change(dataset_manifest)
    with pytest.raises(LibraryError, match="DATA_MANIFEST_INVALID"):
        _make_manifest(assets, asset_manifest, classification, dataset_manifest, panel)


def test_universe_build_rejects_malformed_or_future_dataset_end() -> None:
    assets, asset_manifest, classification, dataset_manifest, panel = _inputs()
    malformed = dict(dataset_manifest, end="2026-9-28")
    malformed["manifest_hash"] = hash_without(malformed, "manifest_hash")
    with pytest.raises(LibraryError, match="DATA_MANIFEST_INVALID"):
        _make_manifest(assets, asset_manifest, classification, malformed, panel)

    future = dict(dataset_manifest, end=DECISION_SESSION.isoformat())
    future["manifest_hash"] = hash_without(future, "manifest_hash")
    with pytest.raises(LibraryError, match="UNIVERSE_INPUT_UNAVAILABLE"):
        _make_manifest(assets, asset_manifest, classification, future, panel)


def test_universe_build_rejects_invalid_panel_shapes_and_symbols() -> None:
    assets, asset_manifest, classification, dataset_manifest, panel = _inputs()
    with pytest.raises(LibraryError, match="UNIVERSE_INPUT_UNAVAILABLE"):
        _make_manifest(
            assets,
            asset_manifest,
            classification,
            dataset_manifest,
            replace(panel, sessions=(*panel.sessions[:-1], DECISION_SESSION)),
        )
    with pytest.raises(LibraryError, match="DATA_PANEL_INVALID"):
        _make_manifest(
            assets, asset_manifest, classification, dataset_manifest, replace(panel, volume=panel.volume[:-1])
        )
    with pytest.raises(LibraryError, match="DATA_PANEL_INVALID"):
        _make_manifest(
            assets,
            asset_manifest,
            classification,
            dataset_manifest,
            replace(panel, symbols=("AAA", "AAA", "CHEAP", "ETF", "NEW")),
        )

    different_symbols = replace(panel, symbols=("AAA", "BBB", "CHEAP", "ETF", "OTHER"))
    with pytest.raises(LibraryError, match="DATA_PANEL_INVALID"):
        _make_manifest(assets, asset_manifest, classification, dataset_manifest, different_symbols)

    missing_symbol = replace(
        panel,
        symbols=panel.symbols[1:],
        micro={"close": panel.micro["close"][:, 1:]},
        volume=panel.volume[:, 1:],
        present=panel.present[:, 1:],
    )
    reduced_manifest = dict(dataset_manifest, symbols=list(panel.symbols[1:]))
    reduced_manifest["manifest_hash"] = hash_without(reduced_manifest, "manifest_hash")
    result = _make_manifest(assets, asset_manifest, classification, reduced_manifest, missing_symbol)
    assert result["counts"]["excluded_history"] == 1


def test_universe_build_rejects_listing_dates_after_decision() -> None:
    assets, asset_manifest, classification, dataset_manifest, panel = _inputs()
    classification["records"][0]["listed_at"] = "2026-09-30"
    with pytest.raises(LibraryError, match="UNIVERSE_CLASSIFICATION_INVALID"):
        _make_manifest(assets, asset_manifest, classification, dataset_manifest, panel)


def test_universe_build_rejects_duplicate_common_stock_symbols() -> None:
    assets, asset_manifest, classification, dataset_manifest, panel = _inputs()
    duplicate_symbols = (assets[0], replace(assets[1], symbol=assets[0].symbol), *assets[2:])
    with pytest.raises(LibraryError, match="UNIVERSE_CLASSIFICATION_INVALID"):
        _make_manifest(duplicate_symbols, asset_manifest, classification, dataset_manifest, panel)


def test_universe_build_counts_missing_history_and_nonpositive_closes() -> None:
    assets, asset_manifest, classification, dataset_manifest, panel = _inputs()
    presence = panel.present.copy()
    presence[:, 1] = False
    closes = panel.micro["close"].copy()
    closes[0, 2] = 0
    result = _make_manifest(
        assets,
        asset_manifest,
        classification,
        dataset_manifest,
        replace(panel, present=presence, micro={"close": closes}),
        policy=UniverseBuildPolicy(Decimal("1"), 0, Decimal("0")),
    )
    assert result["counts"]["excluded_history"] == 2
    assert result["counts"]["members"] == 2


def test_universe_build_rejects_invalid_volume_values() -> None:
    assets, asset_manifest, classification, dataset_manifest, panel = _inputs()
    volume = panel.volume.astype(object)
    volume[0, 0] = "not-a-number"
    with pytest.raises(LibraryError, match="DATA_PANEL_INVALID"):
        _make_manifest(
            assets, asset_manifest, classification, dataset_manifest, replace(panel, volume=volume)
        )

    volume = panel.volume.copy()
    volume[0, 0] = np.inf
    with pytest.raises(LibraryError, match="DATA_PANEL_INVALID"):
        _make_manifest(
            assets, asset_manifest, classification, dataset_manifest, replace(panel, volume=volume)
        )

    volume[0, 0] = -1
    with pytest.raises(LibraryError, match="DATA_PANEL_INVALID"):
        _make_manifest(
            assets, asset_manifest, classification, dataset_manifest, replace(panel, volume=volume)
        )


def test_universe_build_handles_single_member_and_empty_price_cohort() -> None:
    assets, asset_manifest, classification, dataset_manifest, panel = _inputs()
    single = _make_manifest(
        assets,
        asset_manifest,
        classification,
        dataset_manifest,
        panel,
        policy=UniverseBuildPolicy(Decimal("25"), 30, Decimal("100")),
    )
    assert single["member_symbols"] == ["BBB"]
    assert single["counts"]["dollar_volume_ranked"] == 1

    empty = _make_manifest(
        assets,
        asset_manifest,
        classification,
        dataset_manifest,
        panel,
        policy=UniverseBuildPolicy(Decimal("1000"), 30, Decimal("100")),
    )
    assert empty["member_symbols"] == []
    assert empty["counts"]["dollar_volume_ranked"] == 0


def test_universe_build_gives_tied_median_volumes_the_same_percentile() -> None:
    assets, asset_manifest, classification, dataset_manifest, panel = _inputs()
    volume = panel.volume.copy()
    volume[:, 0] = 1500
    volume[:, 1] = 1000
    tied = _make_manifest(
        assets,
        asset_manifest,
        classification,
        dataset_manifest,
        replace(panel, volume=volume),
        policy=UniverseBuildPolicy(Decimal("1"), 0, Decimal("50")),
    )
    assert tied["member_symbols"] == ["AAA", "BBB", "NEW"]


def test_median_returns_the_center_value_for_odd_samples() -> None:
    assert _median([Decimal("3"), Decimal("1"), Decimal("2")]) == Decimal("2")


@pytest.mark.parametrize(
    "change",
    [
        lambda manifest: manifest.update(schema="unknown"),
        lambda manifest: manifest.update(member_symbols=["AAA", "AAA"]),
        lambda manifest: manifest.update(member_symbols=["ZZZ", "AAA"]),
        lambda manifest: manifest.update(member_symbols=[1]),
        lambda manifest: manifest.update(members_hash="sha256:" + "0" * 64),
        lambda manifest: manifest.update(redistributable=True),
        lambda manifest: manifest.update(manifest_hash="invalid"),
    ],
)
def test_universe_manifest_verification_rejects_mutated_fields(change) -> None:
    manifest = _build()
    change(manifest)
    with pytest.raises(LibraryError, match="DATA_MANIFEST_INVALID"):
        verify_universe_manifest(manifest)


def test_universe_manifest_verification_rejects_non_object() -> None:
    with pytest.raises(LibraryError, match="DATA_MANIFEST_INVALID"):
        verify_universe_manifest(None)
