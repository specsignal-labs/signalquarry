# SPDX-License-Identifier: Apache-2.0
"""Layout, integrity checks and windowing rules of the derived research panel."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from signalquarry._internal.data.dataset import FIELDS, Dataset
from signalquarry._internal.data.library import Library, LibraryError
from signalquarry._internal.data.panel import PANEL_FIELDS, PANEL_SCHEMA, LoadedPanel, PanelStore
from signalquarry._internal.data.synthetic import synthetic_dataset

SYMBOLS = ("SYNA", "SYNB", "SYNC")


def _dataset() -> Dataset:
    dataset = synthetic_dataset(date(2024, 1, 2), date(2024, 1, 12), symbols=SYMBOLS)
    dataset.series["SYNB"].present[2] = False
    return dataset


def _built(tmp_path: Path) -> tuple[PanelStore, Dataset, Path]:
    store = PanelStore(tmp_path / "cache")
    dataset = _dataset()
    return store, dataset, store.build(dataset)


def _rewrite_metadata(path: Path, edit: Callable[[dict[str, Any]], None]) -> None:
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    edit(metadata)
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


@pytest.mark.parametrize(
    "identity",
    [
        "sha256:" + "A" * 64,
        "sha256:" + "a" * 63,
        "sha256:" + "a" * 65,
        "a" * 64,
        "sha256:",
        "../x",
        "sha256:" + "g" * 64,
    ],
)
def test_only_lowercase_sha256_identities_have_a_panel_path(tmp_path: Path, identity: str) -> None:
    with pytest.raises(LibraryError, match="^DATA_PANEL_INVALID:dataset_identity$"):
        PanelStore(tmp_path).path(identity)
    digest = "ab" * 32
    assert PanelStore(tmp_path).path(f"sha256:{digest}") == tmp_path / "panels" / digest


def test_a_built_panel_has_one_parquet_per_field_and_compact_metadata(tmp_path: Path) -> None:
    store, dataset, path = _built(tmp_path)
    identity = dataset.identity()
    assert path == tmp_path / "cache" / "panels" / identity.removeprefix("sha256:")
    assert sorted(item.name for item in path.iterdir()) == sorted(
        [*(f"{name}.parquet" for name in PANEL_FIELDS), "metadata.json"]
    )
    assert PANEL_FIELDS == (*FIELDS, "volume", "present")
    text = (path / "metadata.json").read_text(encoding="utf-8")
    assert text.endswith("}\n") and ": " not in text and ", " not in text
    metadata = json.loads(text)
    assert metadata["schema"] == PANEL_SCHEMA
    assert metadata["dataset_identity"] == identity
    assert metadata["source"] == dataset.source
    assert metadata["sessions"] == [session.isoformat() for session in dataset.sessions]
    assert metadata["symbols"] == list(dataset.symbols) == sorted(SYMBOLS)
    assert set(metadata["files_sha256"]) == set(PANEL_FIELDS)
    for name, digest in metadata["files_sha256"].items():
        assert hashlib.sha256((path / f"{name}.parquet").read_bytes()).hexdigest() == digest
    assert not [item for item in path.parent.iterdir() if item.name.startswith(".panel-")]
    assert store.build(dataset) == path


def test_parquet_columns_hold_the_series_values_in_symbol_order(tmp_path: Path) -> None:
    _, dataset, path = _built(tmp_path)
    table = pq.read_table(path / "close.parquet")
    assert table.column_names == list(dataset.symbols)
    for symbol in dataset.symbols:
        assert table.column(symbol).to_pylist() == dataset.series[symbol].micro["close"].tolist()
    assert pq.read_table(path / "present.parquet").column("SYNB").to_pylist()[2] is False
    assert pq.read_table(path / "volume.parquet").schema.field("SYNA").type == pa.float64()


def test_an_existing_panel_is_verified_not_rebuilt(tmp_path: Path) -> None:
    store, dataset, path = _built(tmp_path)
    before = {item.name: item.stat().st_mtime_ns for item in path.iterdir()}
    store.build(dataset)
    assert {item.name: item.stat().st_mtime_ns for item in path.iterdir()} == before
    _rewrite_metadata(path, lambda metadata: metadata.update(schema="other"))
    with pytest.raises(LibraryError, match=f"^DATA_PANEL_INVALID:{dataset.identity()}$"):
        store.build(dataset)


def test_a_dataset_without_symbols_cannot_become_a_panel(tmp_path: Path) -> None:
    dataset = _dataset()
    empty = Dataset(dataset.sessions, {}, source=dataset.source)
    store = PanelStore(tmp_path / "cache")
    with pytest.raises(LibraryError, match="^DATA_PANEL_INVALID:empty_symbols$"):
        store.build(empty)
    assert not [item for item in (tmp_path / "cache" / "panels").iterdir() if item.name.startswith(".panel-")]
    assert not (store.path(empty.identity())).exists()


def test_losing_a_publish_race_to_a_valid_panel_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, dataset, path = _built(tmp_path)
    saved = tmp_path / "saved"
    shutil.copytree(path, saved)
    shutil.rmtree(path)
    real = Path.rename

    def raced(self: Path, target: Path) -> Path:
        if self.name.startswith(".panel-"):
            shutil.copytree(saved, target)
            raise OSError("lost the race")
        return real(self, target)

    monkeypatch.setattr(Path, "rename", raced)
    assert store.build(dataset) == path
    assert not [item for item in path.parent.iterdir() if item.name.startswith(".panel-")]


def test_losing_a_publish_race_to_an_invalid_panel_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, dataset, path = _built(tmp_path)
    shutil.rmtree(path)
    real = Path.rename

    def raced(self: Path, target: Path) -> Path:
        if self.name.startswith(".panel-"):
            target.mkdir()
            raise OSError("lost the race")
        return real(self, target)

    monkeypatch.setattr(Path, "rename", raced)
    with pytest.raises(LibraryError, match=f"^DATA_PANEL_INVALID:{dataset.identity()}$"):
        store.build(dataset)


def test_a_failed_publish_that_left_nothing_behind_reraises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, dataset, path = _built(tmp_path)
    shutil.rmtree(path)

    def broken(self: Path, target: Path) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "rename", broken)
    with pytest.raises(OSError, match="disk full"):
        store.build(dataset)
    assert not [item for item in path.parent.iterdir() if item.name.startswith(".panel-")]
    assert not path.exists()


@pytest.mark.parametrize(
    "edit",
    [
        lambda m: m.update(schema="other"),
        lambda m: m.update(dataset_identity="sha256:" + "0" * 64),
        lambda m: m.update(sessions=list(reversed(m["sessions"]))),
        lambda m: m.update(sessions=[*m["sessions"], m["sessions"][0]]),
        lambda m: m.update(sessions=["not-a-date"]),
        lambda m: m.update(symbols=list(reversed(m["symbols"]))),
        lambda m: m.update(symbols=[*m["symbols"], m["symbols"][0]]),
        lambda m: m.update(symbols=["", *m["symbols"]]),
        lambda m: m.update(symbols=[1, 2]),
        lambda m: m["files_sha256"].pop("volume"),
        lambda m: m["files_sha256"].update(extra="0" * 64),
        lambda m: m["files_sha256"].update(close="0" * 64),
        lambda m: m["files_sha256"].update(present="0" * 64),
        lambda m: m.pop("sessions"),
        lambda m: m.pop("symbols"),
        lambda m: m.pop("schema"),
        lambda m: m.pop("files_sha256"),
    ],
)
def test_tampered_metadata_is_rejected_with_the_identity_as_detail(
    tmp_path: Path, edit: Callable[[dict[str, Any]], None]
) -> None:
    store, dataset, path = _built(tmp_path)
    _rewrite_metadata(path, edit)
    with pytest.raises(LibraryError, match=f"^DATA_PANEL_INVALID:{dataset.identity()}$"):
        store.load(dataset.identity())


def test_a_missing_or_unreadable_metadata_or_data_file_is_rejected(tmp_path: Path) -> None:
    store, dataset, path = _built(tmp_path)
    expected = f"^DATA_PANEL_INVALID:{dataset.identity()}$"
    (path / "volume.parquet").unlink()
    with pytest.raises(LibraryError, match=expected):
        store.load(dataset.identity())
    store2, dataset2, path2 = _built(tmp_path / "second")
    (path2 / "metadata.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(LibraryError, match=f"^DATA_PANEL_INVALID:{dataset2.identity()}$"):
        store2.load(dataset2.identity())
    (path2 / "metadata.json").unlink()
    with pytest.raises(LibraryError, match=f"^DATA_PANEL_INVALID:{dataset2.identity()}$"):
        store2.load(dataset2.identity())


def test_load_returns_typed_read_only_arrays_in_the_requested_symbol_order(tmp_path: Path) -> None:
    store, dataset, _ = _built(tmp_path)
    everything = store.load(dataset.identity())
    assert everything.symbols == tuple(sorted(SYMBOLS))
    assert everything.sessions == dataset.sessions
    assert everything.dataset_identity == dataset.identity()
    picked = store.load(dataset.identity(), symbols=("SYNC", "SYNA"))
    assert picked.symbols == ("SYNC", "SYNA")
    rows = len(dataset.sessions)
    for name in FIELDS:
        assert picked.micro[name].dtype == np.int64 and picked.micro[name].shape == (rows, 2)
        assert not picked.micro[name].flags.writeable
        np.testing.assert_array_equal(picked.micro[name][:, 0], dataset.series["SYNC"].micro[name])
        np.testing.assert_array_equal(picked.micro[name][:, 1], dataset.series["SYNA"].micro[name])
    assert picked.volume.dtype == np.float64 and picked.present.dtype == np.bool_
    assert not picked.volume.flags.writeable and not picked.present.flags.writeable
    np.testing.assert_array_equal(everything.present[:, 1], dataset.series["SYNB"].present)
    assert not everything.present[2, 1]
    with pytest.raises(TypeError):
        picked.micro["close"] = np.zeros(1)  # type: ignore[index]


def test_an_empty_selection_has_typed_zero_width_arrays(tmp_path: Path) -> None:
    store, dataset, _ = _built(tmp_path)
    empty = store.load(dataset.identity(), symbols=())
    rows = len(dataset.sessions)
    assert empty.symbols == ()
    assert empty.micro["close"].shape == (rows, 0) and empty.micro["close"].dtype == np.int64
    assert empty.volume.shape == (rows, 0) and empty.volume.dtype == np.float64
    assert empty.present.shape == (rows, 0) and empty.present.dtype == np.bool_


@pytest.mark.parametrize("symbols", [("SYNA", "SYNA"), ("NOPE",), ("SYNA", "NOPE")])
def test_duplicate_or_unknown_symbols_are_refused(tmp_path: Path, symbols: tuple[str, ...]) -> None:
    store, dataset, _ = _built(tmp_path)
    with pytest.raises(LibraryError, match="^DATA_PANEL_SYMBOL_INVALID$"):
        store.load(dataset.identity(), symbols=symbols)


def test_a_parquet_with_the_wrong_row_count_is_rejected_even_if_its_digest_is_recorded(
    tmp_path: Path,
) -> None:
    store, dataset, path = _built(tmp_path)
    short = pa.table({symbol: pa.array([1, 2], type=pa.int64()) for symbol in dataset.symbols})
    pq.write_table(short, path / "close.parquet")
    digest = hashlib.sha256((path / "close.parquet").read_bytes()).hexdigest()
    _rewrite_metadata(path, lambda metadata: metadata["files_sha256"].update(close=digest))
    with pytest.raises(LibraryError, match="^DATA_PANEL_INVALID:close$"):
        store.load(dataset.identity())


def test_a_selected_column_missing_from_a_parquet_is_rejected(tmp_path: Path) -> None:
    store, dataset, path = _built(tmp_path)
    partial = pa.table({"SYNA": pa.array([0] * len(dataset.sessions), type=pa.int64())})
    pq.write_table(partial, path / "open.parquet")
    digest = hashlib.sha256((path / "open.parquet").read_bytes()).hexdigest()
    _rewrite_metadata(path, lambda metadata: metadata["files_sha256"].update(open=digest))
    with pytest.raises(LibraryError, match="^DATA_PANEL_INVALID:open$"):
        store.load(dataset.identity(), symbols=("SYNB",))
    assert store.load(dataset.identity(), symbols=("SYNA",)).micro["open"].shape[1] == 1


def _loaded(tmp_path: Path) -> tuple[LoadedPanel, Dataset]:
    store, dataset, _ = _built(tmp_path)
    return store.load(dataset.identity()), dataset


def test_a_window_holds_only_sessions_strictly_before_the_decision(tmp_path: Path) -> None:
    panel, dataset = _loaded(tmp_path)
    sessions = dataset.sessions
    cut = sessions[4]
    window = panel.window(decision_session=cut)
    assert window.sessions == sessions[:4]
    assert window.decision_session == cut and window.dataset_identity == panel.dataset_identity
    assert window.symbols == panel.symbols
    assert (
        window.micro["close"].shape == (4, 3)
        and window.volume.shape == (4, 3)
        and window.present.shape == (4, 3)
    )
    # a decision between sessions includes everything before it
    midday = date.fromordinal(cut.toordinal() - 1)
    assert panel.window(decision_session=midday).sessions == tuple(s for s in sessions if s < midday)
    assert panel.window(decision_session=sessions[0]).sessions == ()
    assert panel.window(decision_session=sessions[0]).micro["close"].shape == (0, 3)
    assert panel.window(decision_session=date(2030, 1, 1)).sessions == sessions
    assert panel.window(decision_session=sessions[-1]).sessions == sessions[:-1]


def test_a_lookback_keeps_only_the_latest_sessions_and_may_exceed_the_history(tmp_path: Path) -> None:
    panel, dataset = _loaded(tmp_path)
    sessions = dataset.sessions
    cut = sessions[5]
    assert panel.window(decision_session=cut, lookback=1).sessions == (sessions[4],)
    assert panel.window(decision_session=cut, lookback=3).sessions == sessions[2:5]
    assert panel.window(decision_session=cut, lookback=5).sessions == sessions[:5]
    assert panel.window(decision_session=cut, lookback=99).sessions == sessions[:5]
    assert panel.window(decision_session=sessions[0], lookback=3).sessions == ()
    values = panel.window(decision_session=cut, lookback=2)
    np.testing.assert_array_equal(values.micro["close"], panel.micro["close"][3:5])
    np.testing.assert_array_equal(values.present, panel.present[3:5])


@pytest.mark.parametrize("lookback", [0, -1, True, 2.0, "3"])
def test_an_invalid_lookback_is_refused(tmp_path: Path, lookback: object) -> None:
    panel, dataset = _loaded(tmp_path)
    with pytest.raises(ValueError, match="^DATA_PANEL_LOOKBACK_INVALID$"):
        panel.window(decision_session=dataset.sessions[3], lookback=lookback)  # type: ignore[arg-type]


def test_window_values_cannot_be_made_writeable(tmp_path: Path) -> None:
    panel, dataset = _loaded(tmp_path)
    window = panel.window(decision_session=dataset.sessions[3])
    for array in (*window.micro.values(), window.volume, window.present):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flags.writeable = True
    with pytest.raises(TypeError):
        window.micro["close"] = np.zeros(1)  # type: ignore[index]


def test_a_manifest_panel_requires_the_library_to_share_the_cache(tmp_path: Path) -> None:
    store = PanelStore(tmp_path / "cache")
    other = Library(tmp_path / "elsewhere", tmp_path / "manifests")
    with pytest.raises(LibraryError, match="^DATA_PANEL_INVALID:cache_dir$"):
        store.build_from_manifest(other, {})
