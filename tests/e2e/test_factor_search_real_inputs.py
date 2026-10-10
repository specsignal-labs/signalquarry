# SPDX-License-Identifier: Apache-2.0
"""Actual cached-page and universe replay refusal coverage; no substituted verifiers."""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from signalquarry import api
from signalquarry._internal.data.universe import ASSETS_PARAMS, ASSETS_PATH, AssetPage, make_asset_manifest
from signalquarry._internal.data.universe_build import CLASSIFICATION_SCHEMA
from signalquarry.api.universe import _store

from .test_factor_holdout import project  # noqa: F401


def _protected_bytes(root):
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for base in (root / "evidence", root / "families", root / ".signalquarry/factor_searches")
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
