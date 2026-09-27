# SPDX-License-Identifier: Apache-2.0
"""Local data library: content-addressed raw pages, committed manifests, rebuildable datasets.

* ``$SIGNALQUARRY_CACHE_DIR/pages/<sha256>.json.gz`` — raw provider pages (source of truth).
* ``<project>/data/manifests/<dataset_id>.json`` — ``signalquarry.dataset-manifest/v1``:
  request parameters, page hashes, coverage and the dataset identity. No prices,
  ``redistributable: false``. Safe to commit.
* Datasets are rebuilt from the pages; ``verify`` proves the pages still hash to
  the manifest and rebuild to the same dataset identity.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from signalquarry._internal.canonical import canonical_hash, hash_without, to_canonical
from signalquarry._internal.data.alpaca import RawPage, actions_from_pages, bars_from_pages
from signalquarry._internal.data.dataset import FIELDS, MICRO, Dataset, SymbolSeries

MANIFEST_SCHEMA = "signalquarry.dataset-manifest/v1"


class LibraryError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}:{detail}" if detail else code)
        self.code = code


@dataclass(frozen=True)
class Library:
    cache_dir: Path
    manifest_dir: Path

    def page_path(self, sha256: str) -> Path:
        return self.cache_dir / "pages" / f"{sha256}.json.gz"

    def store_page(self, page: RawPage) -> None:
        path = self.page_path(page.sha256)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(gzip.compress(page.body, mtime=0))
            temporary.replace(path)

    def load_page(self, reference: dict[str, Any]) -> RawPage:
        path = self.page_path(reference["sha256"])
        try:
            body = gzip.decompress(path.read_bytes())
        except OSError as exc:
            raise LibraryError("DATA_PAGE_MISSING", reference["sha256"]) from exc
        if hashlib.sha256(body).hexdigest() != reference["sha256"]:
            raise LibraryError("DATA_PAGE_CORRUPT", reference["sha256"])
        return RawPage(
            reference["endpoint"], reference["params"], body, reference["sha256"], json.loads(body)
        )

    def manifests(self) -> list[dict[str, Any]]:
        if not self.manifest_dir.is_dir():
            return []
        return [
            json.loads(path.read_text(encoding="utf-8")) for path in sorted(self.manifest_dir.glob("*.json"))
        ]

    def write_manifest(self, manifest: dict[str, Any]) -> Path:
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        path = self.manifest_dir / f"{manifest['dataset_id']}.json"
        path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return path


def build_dataset(bar_pages: list[RawPage], action_pages: list[RawPage], *, source: str) -> Dataset:
    bars = bars_from_pages(bar_pages)
    if not bars:
        raise LibraryError("DATA_EMPTY")
    splits, dividends = actions_from_pages(action_pages)
    sessions = tuple(sorted({row["session"] for rows in bars.values() for row in rows}))
    index = {session: i for i, session in enumerate(sessions)}
    series: dict[str, SymbolSeries] = {}
    for symbol, rows in sorted(bars.items()):
        micro = {name: np.zeros(len(sessions), dtype=np.int64) for name in FIELDS}
        volume = np.zeros(len(sessions))
        present = np.zeros(len(sessions), dtype=bool)
        for row in rows:
            i = index[row["session"]]
            for name in FIELDS:
                micro[name][i] = int((row[name] * MICRO).to_integral_value())
            volume[i] = row["volume"]
            present[i] = True
        series[symbol] = SymbolSeries(micro, volume, present)
    return Dataset(sessions, series, splits, dividends, source=source)


def make_manifest(
    *,
    provider: str,
    feed: str,
    symbols: tuple[str, ...],
    start: date,
    end: date,
    bar_pages: list[RawPage],
    action_pages: list[RawPage],
    dataset: Dataset,
    fetched_at: datetime,
) -> dict[str, Any]:
    rows = {}
    for symbol in symbols:
        item = dataset.series.get(symbol)
        present = np.flatnonzero(item.present) if item is not None else np.array([], dtype=int)
        rows[symbol] = {
            "count": int(len(present)),
            "first": dataset.sessions[int(present[0])].isoformat() if len(present) else None,
            "last": dataset.sessions[int(present[-1])].isoformat() if len(present) else None,
        }
    body = {
        "schema": MANIFEST_SCHEMA,
        "dataset_id": f"{provider}-{feed}-1day-{dataset.identity()[7:19]}",
        "provider": provider,
        "feed": feed,
        "timeframe": "1Day",
        "adjustment": "raw",
        "symbols": list(symbols),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "fetched_at": fetched_at.astimezone(UTC),
        "bar_pages": [{"endpoint": p.endpoint, "params": p.params, "sha256": p.sha256} for p in bar_pages],
        "action_pages": [
            {"endpoint": p.endpoint, "params": p.params, "sha256": p.sha256} for p in action_pages
        ],
        "coverage": rows,
        "splits": len(dataset.splits),
        "dividends": len(dataset.dividends),
        "sessions": len(dataset.sessions),
        "dataset_identity": dataset.identity(),
        "redistributable": False,
    }
    body = to_canonical(body)
    body["manifest_hash"] = canonical_hash(body)
    return body


def dataset_from_manifest(library: Library, manifest: dict[str, Any]) -> Dataset:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise LibraryError("DATA_MANIFEST_INVALID", "schema")
    if hash_without(manifest, "manifest_hash") != manifest.get("manifest_hash"):
        raise LibraryError("DATA_MANIFEST_INVALID", "manifest_hash")
    bar_pages = [library.load_page(ref) for ref in manifest["bar_pages"]]
    action_pages = [library.load_page(ref) for ref in manifest["action_pages"]]
    dataset = build_dataset(bar_pages, action_pages, source=f"{manifest['provider']}:{manifest['feed']}")
    if dataset.identity() != manifest["dataset_identity"]:
        raise LibraryError("DATA_IDENTITY_MISMATCH", manifest["dataset_id"])
    return dataset


def select_manifest(library: Library, symbols: tuple[str, ...], feed: str) -> dict[str, Any] | None:
    candidates = [
        m for m in library.manifests() if m.get("feed") == feed and set(symbols) <= set(m.get("symbols", []))
    ]
    return max(candidates, key=lambda m: (m["end"], m["fetched_at"]), default=None)
