# SPDX-License-Identifier: Apache-2.0
"""Cached-page universe replay, successful T1 search and read-only refusals."""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from signalquarry import api
from signalquarry._internal.data.library import dataset_from_manifest
from signalquarry._internal.data.universe import ASSETS_PARAMS, ASSETS_PATH, AssetPage, make_asset_manifest
from signalquarry._internal.data.universe_build import CLASSIFICATION_SCHEMA
from signalquarry._internal.factors.training import formula_training_input
from signalquarry._internal.validation import ledger
from signalquarry.api.universe import _store
from signalquarry.cli.main import main
from tests.unit.factor_replay import build_universe, record_dataset
from tests.unit.test_factor_search_api import search_project  # noqa: F401

from .test_factor_holdout import project  # noqa: F401


def _protected_bytes(root):
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for base in (root / "evidence", root / "families", root / ".signalquarry")
        if base.exists()
        for path in (base, *base.rglob("*"))
    }


def _build(root: Path, dataset_id: str) -> Path:
    asset_id = "00000000-0000-4000-8000-000000000001"
    observed = datetime(2024, 4, 1, 20, tzinfo=UTC)
    body = json.dumps(
        [
            {
                "id": asset_id,
                "class": "us_equity",
                "symbol": "SYNA",
                "exchange": "NYSE",
                "status": "active",
                "tradable": True,
            }
        ]
    ).encode()
    page = AssetPage(ASSETS_PATH, dict(ASSETS_PARAMS), body, hashlib.sha256(body).hexdigest())
    _store(root).store(page, make_asset_manifest(page, observed))
    classification = root / "classification.json"
    classification.write_text(
        json.dumps(
            {
                "schema": CLASSIFICATION_SCHEMA,
                "source": "synthetic-test-source",
                "observed_at": observed.isoformat(),
                "records": [
                    {"asset_id": asset_id, "security_type": "common_stock", "listed_at": "2020-01-01"}
                ],
            }
        )
    )
    result = api.universe_build(
        decision_session=date(2024, 4, 2),
        known_at=datetime(2024, 4, 1, 21, tzinfo=UTC),
        dataset_id=dataset_id,
        classification_file=classification,
        minimum_price=Decimal(1),
        minimum_listing_age_days=0,
        minimum_dollar_volume_percentile=Decimal(0),
        project=root,
    )
    assert result.status == "ok", result.as_dict()
    return Path(result.data["manifest"])


@pytest.mark.parametrize(
    "damage",
    [
        "none",
        "page",
        "missing-page",
        "universe",
        "wrong-dataset",
        "asset",
        "classification",
        "dataset-object",
    ],
)
def test_factor_search_real_replay_refusals_leave_ledger_and_artifacts_untouched(project, damage) -> None:  # noqa: F811
    root, dataset_id = project
    assert api.factor_holdout_seal("alpha", dataset_id, project=root).status == "ok"
    path = _build(root, dataset_id)
    manifest = json.loads((root / path).read_text())
    library = api.factor.library_for(root)
    expected = "FACTOR_SEARCH_INPUT_UNVERIFIED"
    if damage == "dataset-object":
        (root / "data/manifests" / f"{dataset_id}.json").write_text("[]")
        expected = "DATA_MANIFEST_INVALID"
    elif damage in {"page", "missing-page"}:
        data = next(row for row in library.manifests() if row["dataset_id"] == dataset_id)
        cached = library.page_path(data["bar_pages"][0]["sha256"])
        if damage == "page":
            cached.write_bytes(gzip.compress(b"tampered"))
            expected = "DATA_PAGE_CORRUPT"
        else:
            cached.unlink()
            expected = "DATA_PAGE_MISSING"
    elif damage == "universe":
        manifest["member_symbols"] = []
        (root / path).write_text(json.dumps(manifest))
        expected = "DATA_MANIFEST_INVALID"
    elif damage == "wrong-dataset":
        manifest["dataset_id"] = "wrong"
        from signalquarry._internal.canonical import canonical_hash

        manifest["manifest_hash"] = canonical_hash(
            {k: v for k, v in manifest.items() if k != "manifest_hash"}
        )
        (root / path).write_text(json.dumps(manifest))
        expected = "UNIVERSE_INPUT_UNAVAILABLE"
    elif damage == "asset":
        store = _store(root)
        stored = store.manifests()[0]
        store.page_path(stored["page"]["sha256"]).unlink()
        expected = "DATA_PAGE_MISSING"
    elif damage == "classification":
        from signalquarry._internal.data.universe_build import _classification_cache_path

        _classification_cache_path(library.cache_dir, manifest["classification_snapshot_hash"]).unlink()
        expected = "UNIVERSE_INPUT_UNAVAILABLE"
    before = _protected_bytes(root)
    result = api.factor_search(
        "alpha",
        dataset_id,
        universe_manifests=[path],
        training_cutoff=date(2023, 11, 30),
        horizon=1,
        seed=7,
        budget=5,
        project=root,
    )
    assert result.reason_codes == [expected], result.as_dict()
    assert _protected_bytes(root) == before


def test_factor_search_mcp_tool_is_absent():
    pytest.importorskip("mcp")
    from mcp import Client

    from signalquarry.mcp.server import mcp

    async def exercise():
        async with Client(mcp) as client:
            names = {item.name for item in (await client.list_tools()).tools}
            assert "sqy_factor_ls" in names
            assert not any("factor_search" in name or "factor_emit" in name for name in names)

    asyncio.run(exercise())


def test_factor_search_cli_succeeds_with_sequence_of_genuinely_replayed_prefixes(search_project, capsys):  # noqa: F811
    root, args = search_project
    builds = [json.loads((root / path).read_text()) for path in args["universe_manifests"]]
    assert len({build["dataset_id"] for build in builds}) == 3
    assert all(build["dataset_id"] != args["dataset_id"] for build in builds)
    assert api.universe_verify(project=root).status == "ok"
    argv = ["--json", "factor", "search", "--project", str(root)]
    for key, value in args.items():
        if key == "universe_manifests":
            for path in value:
                argv.extend(["--universe-manifest", str(path)])
        else:
            argv.extend(["--" + key.replace("_", "-"), str(value)])
    assert main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["data"]["scope"] == "exploratory"
    assert payload["data"]["evidence_grade"] == "none"
    assert payload["data"]["holdout_accessed"] is False
    report = json.loads((root / payload["artifacts"][0]["path"]).read_text())
    planted = next(
        row for row in report["candidates"] if row["expression"] == "rank((volume / ts_mean(volume, 5)))"
    )
    assert planted["discovery"] and planted["mean_ic"] > 0.2
    assert len(ledger.factor_trials(root).entries()) == args["budget"]
    before = _protected_bytes(root)
    assert main(argv) == 0
    repeated = json.loads(capsys.readouterr().out)
    assert repeated["data"] == payload["data"] and repeated["artifacts"] == payload["artifacts"]
    assert _protected_bytes(root) == before


@pytest.mark.parametrize("command", ["evaluate", "report"])
def test_factor_recorded_diagnostics_accept_prefix_builds_and_keep_unverified_labels(search_project, command):  # noqa: F811
    root, args = search_project
    before = ledger.factor_trials(root).entries(), ledger.factor_holdouts(root).entries()
    result = getattr(api, "factor_" + command)(
        "price-momentum", args["dataset_id"], universe_manifests=args["universe_manifests"], project=root
    )
    assert result.status == "ok", result.as_dict()
    assert result.data["scope"] == "unverified"
    assert result.data["horizons"]
    if command == "evaluate":
        assert result.data["authority"] == {
            "provenance_verified": False,
            "trial_recorded": False,
            "evidence_grade": None,
            "holdout_accessed": False,
        }
        assert len(result.data["decision_sessions"]) == 3
    else:
        report = (root / result.artifacts[0]["path"]).read_text()
        assert "unverified" in report
        assert "No trial was recorded, no evidence grade assigned and no holdout accessed" in report
    assert before == (ledger.factor_trials(root).entries(), ledger.factor_holdouts(root).entries())


@pytest.mark.parametrize("damage", ["bar", "missing-bar", "source", "split", "dividend", "pay-date"])
@pytest.mark.parametrize("command", ["search", "evaluate", "report"])
def test_non_prefix_recorded_dataset_refuses_before_any_output(search_project, damage, command):  # noqa: F811
    root, args = search_project
    library = api.factor.library_for(root)
    manifest = next(row for row in library.manifests() if row["dataset_id"] == args["dataset_id"])
    dataset = dataset_from_manifest(library, manifest)
    item = dataset.series[dataset.symbols[0]]
    if damage == "bar":
        item.micro["close"][0] += 1_000_000
    elif damage == "missing-bar":
        item.present[0] = False
    elif damage == "split":
        dataset.splits = (replace(dataset.splits[0], ratio=Decimal(3)),)
    elif damage == "dividend":
        dataset.dividends = (replace(dataset.dividends[0], amount=Decimal(1)),)
    elif damage == "pay-date":
        dataset.dividends = (replace(dataset.dividends[0], pay_date=dataset.sessions[-1]),)
    changed = record_dataset(root, dataset, dataset.sessions[-1], feed="iex" if damage == "source" else "sip")
    before = _protected_bytes(root)
    if command == "search":
        result = api.factor_search(project=root, **{**args, "dataset_id": changed["dataset_id"]})
    else:
        result = getattr(api, "factor_" + command)(
            "price-momentum",
            changed["dataset_id"],
            universe_manifests=args["universe_manifests"],
            project=root,
        )
    assert result.reason_codes == ["DATA_MANIFEST_INVALID"], result.as_dict()
    assert _protected_bytes(root) == before


def test_prefix_build_membership_is_unchanged_by_post_cutoff_bars(search_project):  # noqa: F811
    root, args = search_project
    library = api.factor.library_for(root)
    manifest = next(row for row in library.manifests() if row["dataset_id"] == args["dataset_id"])
    dataset = dataset_from_manifest(library, manifest)
    paths = args["universe_manifests"][:1]
    first = api.factor._load_evaluation_membership(root, paths, manifest, dataset)
    cutoff = first[0].decision_cutoff.date()
    for index, session in enumerate(dataset.sessions):
        if session > cutoff:
            for item in dataset.series.values():
                for values in (*item.micro.values(), item.volume):
                    values[index] *= 2
    changed = record_dataset(root, dataset, dataset.sessions[-1])
    replayed = dataset_from_manifest(library, changed)
    second = api.factor._load_evaluation_membership(root, paths, changed, replayed)
    assert first == second
    data = formula_training_input(replayed, second, training_cutoff=args["training_cutoff"], horizon=1)
    first_index = data.context.sessions.index(first[0].session)
    assert not data.eligible[:first_index].any()
    assert data.eligible[first_index:].all()


def test_genuine_prefix_membership_uses_latest_build_and_never_backfills(search_project):  # noqa: F811
    root, args = search_project
    library = api.factor.library_for(root)
    manifests = {row["dataset_id"]: row for row in library.manifests()}
    selected = manifests[args["dataset_id"]]
    dataset = dataset_from_manifest(library, selected)
    paths = []
    for path, percentile in zip(args["universe_manifests"], (50, 0, 75), strict=True):
        build = json.loads((root / path).read_text())
        paths.append(
            build_universe(
                root,
                manifests[build["dataset_id"]],
                date.fromisoformat(build["decision_session"]),
                percentile=percentile,
            )
        )
    members = api.factor._load_evaluation_membership(root, paths[::-1], selected, dataset)
    assert len(members[0].symbols) < len(members[1].symbols)
    assert len(members[2].symbols) < len(members[1].symbols)
    data = formula_training_input(dataset, members, training_cutoff=args["training_cutoff"], horizon=1)
    for row, session in enumerate(data.context.sessions):
        applicable = [member for member in members if member.session <= session]
        expected = set(applicable[-1].symbols) if applicable else set()
        assert data.eligible[row].tolist() == [symbol in expected for symbol in data.context.universe]
