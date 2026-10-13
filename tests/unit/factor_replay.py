# SPDX-License-Identifier: Apache-2.0
"""Record dated FakeAlpaca pages and construct universes without verifier stubs."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path

from signalquarry import api
from signalquarry._internal.data.alpaca import AlpacaDataClient, RateLimiter
from signalquarry._internal.data.dataset import Dataset
from signalquarry._internal.data.library import build_dataset, make_manifest
from signalquarry._internal.data.universe import ASSETS_PARAMS, ASSETS_PATH, AssetPage, make_asset_manifest
from signalquarry._internal.data.universe_build import CLASSIFICATION_SCHEMA
from signalquarry.api.universe import _store
from tests.fakes import FakeAlpaca


def record_dataset(root: Path, source: Dataset, end: date, *, feed: str = "sip") -> dict:
    client = AlpacaDataClient(
        "key", "secret", transport=FakeAlpaca(source), limiter=RateLimiter(sleep=lambda _: None)
    )
    start = source.sessions[0]
    bars = client.daily_bars(source.symbols, start, end, feed)
    actions = client.corporate_actions(source.symbols, start, end)
    dataset = build_dataset(bars, actions, source=f"alpaca:{feed}")
    library = api.factor.library_for(root)
    for page in (*bars, *actions):
        library.store_page(page)
    manifest = make_manifest(
        provider="alpaca",
        feed=feed,
        symbols=source.symbols,
        start=start,
        end=end,
        bar_pages=bars,
        action_pages=actions,
        dataset=dataset,
        fetched_at=datetime.combine(end, time(22), UTC),
    )
    library.write_manifest(manifest)
    return manifest


def build_universe(root: Path, dataset_manifest: dict, decision: date, *, percentile: int = 0) -> Path:
    observed = datetime.combine(date.fromisoformat(dataset_manifest["end"]), time(22, 30), UTC)
    assets = [
        {
            "id": f"00000000-0000-4000-8000-{index:012d}",
            "class": "us_equity",
            "symbol": symbol,
            "exchange": "NYSE",
            "status": "active",
            "tradable": True,
        }
        for index, symbol in enumerate(dataset_manifest["symbols"], 1)
    ]
    body = json.dumps(assets).encode()
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
                    {"asset_id": asset["id"], "security_type": "common_stock", "listed_at": "2020-01-01"}
                    for asset in assets
                ],
            }
        )
    )
    result = api.universe_build(
        decision_session=decision,
        known_at=observed.replace(hour=23),
        dataset_id=dataset_manifest["dataset_id"],
        classification_file=classification,
        minimum_price=Decimal("0.01"),
        minimum_listing_age_days=0,
        minimum_dollar_volume_percentile=Decimal(percentile),
        project=root,
    )
    assert result.status == "ok", result.as_dict()
    return Path(result.data["manifest"])
