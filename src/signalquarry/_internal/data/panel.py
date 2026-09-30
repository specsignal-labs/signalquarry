# SPDX-License-Identifier: Apache-2.0
"""Derived, read-only daily research panels outside the project tree.

The source of truth remains the raw-page cache and its dataset manifest. A
panel can be deleted and rebuilt without changing the dataset identity.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from signalquarry._internal.data.dataset import FIELDS, Dataset
from signalquarry._internal.data.library import Library, LibraryError, dataset_from_manifest

PANEL_SCHEMA = "signalquarry.research-panel/v1"
PANEL_FIELDS = (*FIELDS, "volume", "present")
_IDENTITY = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _immutable(values: np.ndarray) -> np.ndarray:
    # A bytes-backed array cannot be made writeable by toggling NumPy flags.
    return np.frombuffer(values.tobytes(order="C"), dtype=values.dtype).reshape(values.shape)


@dataclass(frozen=True)
class PanelWindow:
    """Observed daily bars strictly before ``decision_session``.

    Arrays have shape ``(sessions, symbols)``. Missing bars are distinguished
    by ``present``; zero-filled values at those locations must never be used.
    Prices are raw int64 micro-units, without corporate-action adjustment.
    """

    dataset_identity: str
    decision_session: date
    sessions: tuple[date, ...]
    symbols: tuple[str, ...]
    micro: Mapping[str, np.ndarray]
    volume: np.ndarray
    present: np.ndarray


@dataclass(frozen=True)
class LoadedPanel:
    """Verified full panel for engine use only; never pass it to factor code."""

    dataset_identity: str
    sessions: tuple[date, ...]
    symbols: tuple[str, ...]
    micro: Mapping[str, np.ndarray]
    volume: np.ndarray
    present: np.ndarray

    def window(self, *, decision_session: date, lookback: int | None = None) -> PanelWindow:
        if lookback is not None and (type(lookback) is not int or lookback < 1):
            raise ValueError("DATA_PANEL_LOOKBACK_INVALID")
        stop = bisect_left(self.sessions, decision_session)
        start = max(0, stop - lookback) if lookback is not None else 0
        return PanelWindow(
            dataset_identity=self.dataset_identity,
            decision_session=decision_session,
            sessions=self.sessions[start:stop],
            symbols=self.symbols,
            micro=MappingProxyType(
                {field_name: _immutable(values[start:stop]) for field_name, values in self.micro.items()}
            ),
            volume=_immutable(self.volume[start:stop]),
            present=_immutable(self.present[start:stop]),
        )


@dataclass(frozen=True)
class PanelStore:
    cache_dir: Path

    def path(self, dataset_identity: str) -> Path:
        if not _IDENTITY.fullmatch(dataset_identity):
            raise LibraryError("DATA_PANEL_INVALID", "dataset_identity")
        return self.cache_dir / "panels" / dataset_identity.removeprefix("sha256:")

    def build(self, dataset: Dataset) -> Path:
        """Create a rebuildable Parquet panel, publishing only complete output."""
        dataset_identity = dataset.identity()
        destination = self.path(dataset_identity)
        if destination.exists():
            self._metadata(dataset_identity)
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".panel-", dir=destination.parent))
        try:
            symbols = dataset.symbols
            if not symbols:
                raise LibraryError("DATA_PANEL_INVALID", "empty_symbols")
            hashes = {}
            for field_name in PANEL_FIELDS:
                columns = {}
                for symbol in symbols:
                    item = dataset.series[symbol]
                    values = item.micro[field_name] if field_name in FIELDS else getattr(item, field_name)
                    columns[symbol] = pa.array(values)
                path = temporary / f"{field_name}.parquet"
                pq.write_table(pa.table(columns), path, compression="zstd", row_group_size=256)
                hashes[field_name] = _digest(path)
            metadata = {
                "schema": PANEL_SCHEMA,
                "dataset_identity": dataset_identity,
                "source": dataset.source,
                "sessions": [session.isoformat() for session in dataset.sessions],
                "symbols": list(symbols),
                "files_sha256": hashes,
            }
            (temporary / "metadata.json").write_text(
                json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            try:
                temporary.rename(destination)
            except OSError:
                # Another writer published the same immutable identity first.
                if not destination.exists():
                    raise
                self._metadata(dataset_identity)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return destination

    def build_from_manifest(self, library: Library, manifest: dict) -> Path:
        """Derive a panel only after raw pages and dataset identity verify."""
        if library.cache_dir != self.cache_dir:
            raise LibraryError("DATA_PANEL_INVALID", "cache_dir")
        return self.build(dataset_from_manifest(library, manifest))

    def _metadata(self, dataset_identity: str) -> dict:
        directory = self.path(dataset_identity)
        try:
            metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            sessions = [date.fromisoformat(value) for value in metadata["sessions"]]
            symbols = metadata["symbols"]
            if (
                metadata["schema"] != PANEL_SCHEMA
                or metadata["dataset_identity"] != dataset_identity
                or sessions != sorted(set(sessions))
                or symbols != sorted(set(symbols))
                or not all(isinstance(symbol, str) and symbol for symbol in symbols)
                or set(metadata["files_sha256"]) != set(PANEL_FIELDS)
            ):
                raise ValueError("metadata")
            for field_name in PANEL_FIELDS:
                path = directory / f"{field_name}.parquet"
                if _digest(path) != metadata["files_sha256"][field_name]:
                    raise ValueError(field_name)
            return metadata
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LibraryError("DATA_PANEL_INVALID", dataset_identity) from exc

    def load(
        self,
        dataset_identity: str,
        *,
        symbols: tuple[str, ...] | None = None,
    ) -> LoadedPanel:
        """Verify the cache once and load selected symbols for repeated decisions."""
        metadata = self._metadata(dataset_identity)
        available = tuple(metadata["symbols"])
        selected = available if symbols is None else tuple(symbols)
        if len(selected) != len(set(selected)) or not set(selected) <= set(available):
            raise LibraryError("DATA_PANEL_SYMBOL_INVALID")
        sessions = tuple(date.fromisoformat(value) for value in metadata["sessions"])
        directory = self.path(dataset_identity)

        def read(field_name: str, dtype: np.dtype) -> np.ndarray:
            try:
                if not selected:
                    return np.empty((len(sessions), 0), dtype=dtype)
                table = pq.read_table(directory / f"{field_name}.parquet", columns=list(selected))
                if table.num_rows != len(sessions):
                    raise ValueError("row_count")
                values = np.column_stack(
                    [table.column(symbol).to_numpy(zero_copy_only=False) for symbol in selected]
                ).astype(dtype, copy=False)
                values.flags.writeable = False
                return values
            except (OSError, KeyError, ValueError, pa.ArrowException) as exc:
                raise LibraryError("DATA_PANEL_INVALID", field_name) from exc

        micro = MappingProxyType({field_name: read(field_name, np.dtype("int64")) for field_name in FIELDS})
        return LoadedPanel(
            dataset_identity=dataset_identity,
            sessions=sessions,
            symbols=selected,
            micro=micro,
            volume=read("volume", np.dtype("float64")),
            present=read("present", np.dtype("bool")),
        )

    def window(
        self,
        dataset_identity: str,
        *,
        decision_session: date,
        symbols: tuple[str, ...] | None = None,
        lookback: int | None = None,
    ) -> PanelWindow:
        """Convenience read for one decision; use ``load`` for repeated dates."""
        return self.load(dataset_identity, symbols=symbols).window(
            decision_session=decision_session, lookback=lookback
        )
